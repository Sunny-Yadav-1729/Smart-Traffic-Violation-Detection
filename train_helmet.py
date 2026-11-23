"""
Train the helmet model -> produces best.pt

The script downloads the labeled Kaggle Helmet Detection dataset, converts its
Pascal VOC XML annotations to YOLO format, then trains YOLOv8.
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import kagglehub
from ultralytics import YOLO


CLASSES = ["with helmet", "without helmet"]
DATASET_ID = "andrewmvd/helmet-detection"


def convert_dataset(source_dir: Path, dataset_dir: Path) -> int:
    image_paths = [
        path
        for path in source_dir.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    images = {path.name: path for path in image_paths}
    annotations = list(source_dir.rglob("*.xml"))
    records = []

    for annotation in annotations:
        root = ET.parse(annotation).getroot()
        filename = root.findtext("filename")
        image_path = images.get(Path(filename).name) if filename else None
        if image_path is None:
            continue

        width = int(root.findtext("size/width"))
        height = int(root.findtext("size/height"))
        lines = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name not in CLASSES:
                continue
            box = obj.find("bndbox")
            if box is None:
                continue
            x1 = float(box.findtext("xmin"))
            y1 = float(box.findtext("ymin"))
            x2 = float(box.findtext("xmax"))
            y2 = float(box.findtext("ymax"))
            lines.append(
                f"{CLASSES.index(name)} {(x1 + x2) / 2 / width:.6f} "
                f"{(y1 + y2) / 2 / height:.6f} "
                f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
            )
        if lines:
            records.append((image_path, lines))

    if len(records) < 2:
        raise RuntimeError(
            f"Expected at least two annotated images, found {len(records)}."
        )

    random.Random(42).shuffle(records)
    split = max(1, min(len(records) - 1, int(0.85 * len(records))))
    for subdir in ("images/train", "images/val", "labels/train", "labels/val"):
        (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)

    for index, (image_path, lines) in enumerate(records):
        part = "train" if index < split else "val"
        shutil.copy2(image_path, dataset_dir / "images" / part / image_path.name)
        label_path = dataset_dir / "labels" / part / f"{image_path.stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")

    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(
        f'path: "{dataset_dir.resolve().as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: with helmet\n  1: without helmet\n",
        encoding="utf-8",
    )
    print(f"Converted annotated images: {len(records)}")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the helmet detector.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="runs/helmet/weights/last.pt",
        metavar="CHECKPOINT",
        help="Resume from a checkpoint (defaults to runs/helmet/weights/last.pt).",
    )
    args = parser.parse_args()

    source_dir = Path(kagglehub.dataset_download(DATASET_ID))
    dataset_dir = Path("dataset")
    convert_dataset(source_dir, dataset_dir)

    if args.resume:
        checkpoint = Path(args.resume)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training checkpoint not found: {checkpoint}")
        model = YOLO(str(checkpoint))
        model.train(resume=True, device=0, batch=8)
    else:
        model = YOLO("yolov8n.pt")
        model.train(
            data=str(dataset_dir / "data.yaml"),
            epochs=50,
            imgsz=640,
            batch=8,
            workers=4,
            project=str(Path("runs").resolve()),
            name="helmet",
        )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training completed without producing {best}.")
    output = Path("best.pt")
    shutil.copy2(best, output)
    print(f"Done! Model saved at: {output.resolve()}")

    try:
        from google.colab import files
    except ImportError:
        return
    files.download(str(output))


if __name__ == "__main__":
    main()
