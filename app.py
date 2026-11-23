"""
Smart Traffic Violation Detection  (Streamlit + YOLOv8 + ByteTrack)
Detects:  1) Red-light crossing   2) Riding without helmet
Upload a traffic video -> live annotated video -> violation photos stored below.

Run:  streamlit run app.py
"""
import os
import time
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(page_title="Smart Traffic Violation Detection", layout="wide")

SAVE_DIR = "violations"
os.makedirs(SAVE_DIR, exist_ok=True)
DEMO_VIDEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_video.mp4")

# COCO class ids used by the pretrained YOLOv8 model
VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
MOTORCYCLE = 3
TRAFFIC_LIGHT = 9
FONT = cv2.FONT_HERSHEY_SIMPLEX


@st.cache_resource
def load_model(path: str):
    return YOLO(path)


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def light_color(crop):
    """Return RED / YELLOW / GREEN from a traffic-light crop using HSV colour."""
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = cv2.countNonZero(cv2.inRange(hsv, (0, 100, 120), (10, 255, 255))) + \
          cv2.countNonZero(cv2.inRange(hsv, (160, 100, 120), (180, 255, 255)))
    green = cv2.countNonZero(cv2.inRange(hsv, (40, 80, 100), (90, 255, 255)))
    yellow = cv2.countNonZero(cv2.inRange(hsv, (15, 100, 120), (35, 255, 255)))
    counts = [red, green, yellow]
    if max(counts) < 10:
        return None
    return ["RED", "GREEN", "YELLOW"][int(np.argmax(counts))]


def save_violation(frame, box, vtype, tid):
    """Crop the offending vehicle, stamp the violation name, save as JPG."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad = 25
    crop = frame[max(0, y1 - pad):min(h, y2 + pad),
                 max(0, x1 - pad):min(w, x2 + pad)].copy()
    cv2.rectangle(crop, (0, 0), (crop.shape[1], 26), (0, 0, 255), -1)
    cv2.putText(crop, vtype, (5, 19), FONT, 0.55, (255, 255, 255), 2)
    name = f"{vtype.replace(' ', '_')}_{tid}_{int(time.time() * 1000)}.jpg"
    path = os.path.join(SAVE_DIR, name)
    cv2.imwrite(path, crop)
    return path


def render_gallery(placeholder, violations):
    """Show all captured violation photos in a grid (below the video)."""
    with placeholder.container():
        st.subheader(f"📸 Captured Violations ({len(violations)})")
        if not violations:
            st.info("No violations captured yet.")
            return
        cols = st.columns(4)
        for i, v in enumerate(violations):
            with cols[i % 4]:
                st.image(v["file"], use_container_width=True,
                         caption=f"{v['type']} | ID {v['vehicle_id']} | {v['time_sec']}s")


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🚦 Smart Traffic Violation Detection")
st.caption("Red-light crossing + Helmet violation | YOLOv8 + ByteTrack + Streamlit")

with st.sidebar:
    st.header("⚙️ Settings")
    vehicle_weights = st.text_input("Vehicle model (COCO)", "yolov8n.pt")
    helmet_weights = st.text_input(
        "Helmet model (custom .pt, optional)", "best.pt",
        help="YOLO model trained on a helmet dataset (Kaggle). "
             "Leave a non-existing path to disable helmet detection.")
    no_helmet_names = st.text_input(
        "'No helmet' class names (comma separated)",
        "no helmet,without helmet,no_helmet,nohelmet,no-helmet,without_helmet")
    conf = st.slider("Vehicle confidence", 0.1, 0.9, 0.35, 0.05)
    helmet_conf = st.slider("Helmet confidence", 0.1, 0.9, 0.40, 0.05)
    skip = st.slider("Process every Nth frame", 1, 5, 2)

    st.subheader("🔴 Red light settings")
    line_pos = st.slider("Stop-line position (% of frame height)", 10, 95, 60)
    crossing_dir = st.radio("Vehicles move", ["Top → Bottom (towards camera)",
                                              "Bottom → Top (away from camera)"])
    signal_mode = st.selectbox("Signal source", [
        "Auto-detect from video (traffic light visible)",
        "Always RED",
        "Cycle (simulated)"])
    if signal_mode == "Cycle (simulated)":
        red_sec = st.number_input("Red seconds", 1, 120, 10)
        green_sec = st.number_input("Green seconds", 1, 120, 10)

    detect_red = False
    detect_helmet = st.checkbox("Detect helmet violation", True)

video_source = st.radio("Video source", ["Upload a video", "Try the built-in demo"])
video_file = None
if video_source == "Upload a video":
    video_file = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])
    video_available = video_file is not None
else:
    video_available = os.path.isfile(DEMO_VIDEO)
    if video_available:
        st.video(DEMO_VIDEO)
    else:
        st.warning("The built-in demo video (demo_video.mp4) was not found.")

start = st.button("▶️ Start Detection", type="primary", disabled=not video_available)

video_ph = st.empty()          # annotated video frames
status_ph = st.empty()         # progress / signal status
gallery_ph = st.empty()        # violation photos (below the visual)

if "violations" not in st.session_state:
    st.session_state.violations = []

# --------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------
if start and video_available:
    st.session_state.violations = []
    violations = st.session_state.violations

    temp_video_path = None
    if video_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(video_file.read())
        tmp.close()
        temp_video_path = tmp.name
        video_path = temp_video_path
    else:
        video_path = DEMO_VIDEO

    vmodel = load_model(vehicle_weights)
    hmodel = None
    if detect_helmet and os.path.exists(helmet_weights):
        hmodel = load_model(helmet_weights)
    elif detect_helmet:
        st.warning(f"Helmet model '{helmet_weights}' not found - helmet detection disabled.")

    no_helmet_set = {n.strip().lower() for n in no_helmet_names.split(",")}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(H * line_pos / 100)
    top_to_bottom = crossing_dir.startswith("Top")

    prev_y = {}                # track id -> previous y of vehicle bottom
    red_done, helmet_done = set(), set()
    signal = "UNKNOWN"
    frame_no = 0
    progress = st.progress(0.0)
    render_gallery(gallery_ph, violations)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % skip:
            continue
        t_sec = round(frame_no / fps, 1)

        # ---- 1. detect + track vehicles and traffic lights ----------------
        res = vmodel.track(frame, persist=True, conf=conf, verbose=False,
                           classes=list(VEHICLES) + [TRAFFIC_LIGHT],
                           tracker="bytetrack.yaml")[0]
        vehicles = []          # (track_id, class_id, (x1,y1,x2,y2))
        for b in res.boxes:
            cls = int(b.cls[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            if cls == TRAFFIC_LIGHT:
                if signal_mode.startswith("Auto"):
                    c = light_color(frame[y1:y2, x1:x2])
                    if c:
                        signal = c
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            else:
                tid = int(b.id[0]) if b.id is not None else None
                vehicles.append((tid, cls, (x1, y1, x2, y2)))

        # ---- signal state for non-auto modes -------------------------------
        if signal_mode == "Always RED":
            signal = "RED"
        elif signal_mode == "Cycle (simulated)":
            signal = "RED" if (t_sec % (red_sec + green_sec)) < red_sec else "GREEN"

        # ---- 2. red-light crossing -----------------------------------------
        for tid, cls, box in vehicles:
            x1, y1, x2, y2 = box
            color = (0, 200, 0)
            if tid is not None:
                py = prev_y.get(tid)
                if detect_red and signal == "RED" and py is not None and tid not in red_done:
                    crossed = (py < line_y <= y2) if top_to_bottom else (py > line_y >= y2)
                    if crossed:
                        red_done.add(tid)
                        path = save_violation(frame, box, "Red Light Crossing", tid)
                        violations.append({"type": "Red Light Crossing", "vehicle_id": tid,
                                           "vehicle": VEHICLES[cls], "time_sec": t_sec,
                                           "file": path})
                        render_gallery(gallery_ph, violations)
                prev_y[tid] = y2
                if tid in red_done:
                    color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{VEHICLES[cls]} {tid if tid is not None else ''}",
                        (x1, max(15, y1 - 6)), FONT, 0.5, color, 2)

        # ---- 3. helmet violation ---------------------------------------------
        if hmodel is not None:
            hres = hmodel(frame, conf=helmet_conf, verbose=False)[0]
            bikes = [(t, b) for t, c, b in vehicles if c == MOTORCYCLE and t is not None]
            for hb in hres.boxes:
                label = hres.names[int(hb.cls[0])].lower()
                hx1, hy1, hx2, hy2 = map(int, hb.xyxy[0])
                is_bad = label in no_helmet_set
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2),
                              (0, 0, 255) if is_bad else (255, 128, 0), 2)
                cv2.putText(frame, label, (hx1, max(15, hy1 - 6)), FONT, 0.5,
                            (0, 0, 255) if is_bad else (255, 128, 0), 2)
                if not is_bad:
                    continue
                cx, cy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
                # match the no-helmet head to a tracked motorcycle (box extended upwards for rider)
                for tid, (bx1, by1, bx2, by2) in bikes:
                    ext_y1 = by1 - int(0.8 * (by2 - by1))
                    if bx1 <= cx <= bx2 and ext_y1 <= cy <= by2 and tid not in helmet_done:
                        helmet_done.add(tid)
                        box = (min(bx1, hx1), min(by1, hy1), max(bx2, hx2), max(by2, hy2))
                        path = save_violation(frame, box, "No Helmet", tid)
                        violations.append({"type": "No Helmet", "vehicle_id": tid,
                                           "vehicle": "motorcycle", "time_sec": t_sec,
                                           "file": path})
                        render_gallery(gallery_ph, violations)
                        break

        # ---- 4. draw stop line + signal + show frame -------------------------
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
        sig_col = {"RED": (0, 0, 255), "GREEN": (0, 200, 0), "YELLOW": (0, 255, 255)}.get(signal, (200, 200, 200))
        cv2.circle(frame, (30, 30), 14, sig_col, -1)
        cv2.putText(frame, f"Signal: {signal}", (55, 38), FONT, 0.7, sig_col, 2)
        video_ph.image(frame, channels="BGR", use_container_width=True)
        progress.progress(min(frame_no / total, 1.0))
        status_ph.markdown(f"**Violations so far:** {len(violations)}  |  "
                           f"Red light: {len(red_done)}  |  No helmet: {len(helmet_done)}")

    cap.release()
    if temp_video_path is not None:
        os.unlink(temp_video_path)
    progress.progress(1.0)
    st.success("Processing finished ✅")

# --------------------------------------------------------------------------
# Final results (also shown after reruns, e.g. when clicking download)
# --------------------------------------------------------------------------
render_gallery(gallery_ph, st.session_state.violations)

if st.session_state.violations:
    df = pd.DataFrame(st.session_state.violations)[["type", "vehicle", "vehicle_id", "time_sec", "file"]]
    st.subheader("📋 Violation Log")
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Download log (CSV)", df.to_csv(index=False).encode(),
                       "violations_log.csv", "text/csv")
