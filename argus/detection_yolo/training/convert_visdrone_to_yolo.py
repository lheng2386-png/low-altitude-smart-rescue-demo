"""Convert VisDrone DET annotations into the Argus fused YOLO class space.

Expected input layout:

VisDrone2019-DET-train/
  images/
  annotations/
VisDrone2019-DET-val/
  images/
  annotations/
VisDrone2019-DET-test-dev/
  images/
  annotations/

VisDrone DET classes:
0 ignored, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck,
7 tricycle, 8 awning-tricycle, 9 bus, 10 motor, 11 others.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


UNIFIED_CLASSES = [
    "cat",
    "civilian",
    "cow",
    "dog",
    "horse",
    "rescuer",
    "vehicle",
    "fire",
    "smoke",
    "boat",
    "flood_water",
    "building",
    "road",
    "tree",
    "vegetation",
    "bridge",
    "aircraft",
]

VISDRONE_TO_UNIFIED = {
    1: "civilian",
    2: "civilian",
    3: "vehicle",
    4: "vehicle",
    5: "vehicle",
    6: "vehicle",
    7: "vehicle",
    8: "vehicle",
    9: "vehicle",
    10: "vehicle",
}

SPLIT_DIRS = {
    "train": ["VisDrone2019-DET-train", "VisDrone2018-DET-train"],
    "valid": ["VisDrone2019-DET-val", "VisDrone2018-DET-val"],
    "test": ["VisDrone2019-DET-test-dev", "VisDrone2018-DET-test-dev"],
}


def write_data_yaml(output_dir: Path) -> None:
    names = ", ".join(f"'{name}'" for name in UNIFIED_CLASSES)
    (output_dir / "data.yaml").write_text(
        "\n".join(
            [
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "",
                f"nc: {len(UNIFIED_CLASSES)}",
                f"names: [{names}]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def find_split_dir(source_root: Path, split: str) -> Path | None:
    for candidate in SPLIT_DIRS[split]:
        path = source_root / candidate
        if path.exists():
            return path
    return None


def link_or_copy(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
    else:
        target.symlink_to(source.resolve())


def convert_box(line: str, width: int, height: int) -> str | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 6:
        return None
    x, y, w, h = [float(value) for value in parts[:4]]
    category_id = int(float(parts[5]))
    target_class = VISDRONE_TO_UNIFIED.get(category_id)
    if target_class is None or w <= 1 or h <= 1:
        return None

    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 1 or box_h <= 1:
        return None

    cls_id = UNIFIED_CLASSES.index(target_class)
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    return f"{cls_id} {cx:.8f} {cy:.8f} {box_w / width:.8f} {box_h / height:.8f}"


def convert_split(source_root: Path, output_dir: Path, split: str, image_mode: str) -> dict:
    split_dir = find_split_dir(source_root, split)
    if split_dir is None:
        return {"split": split, "status": "missing", "images": 0, "labels": 0}

    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"
    out_images = output_dir / split / "images"
    out_labels = output_dir / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    images = 0
    labels = 0
    skipped_boxes = 0
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        with Image.open(image_path) as img:
            width, height = img.size

        target_stem = f"visdrone_{image_path.stem}"
        target_image = out_images / f"{target_stem}{image_path.suffix.lower()}"
        target_label = out_labels / f"{target_stem}.txt"
        link_or_copy(image_path, target_image, image_mode)
        images += 1

        source_label = annotations_dir / f"{image_path.stem}.txt"
        lines = []
        if source_label.exists():
            for raw_line in source_label.read_text(encoding="utf-8").splitlines():
                converted = convert_box(raw_line, width, height)
                if converted is None:
                    skipped_boxes += 1
                    continue
                lines.append(converted)
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        labels += len(lines)

    return {"split": split, "status": "ok", "images": images, "labels": labels, "skipped_boxes": skipped_boxes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VisDrone DET to Argus fused YOLO format.")
    parser.add_argument("--source", required=True, help="Directory containing VisDrone2019-DET-* folders.")
    parser.add_argument("--output", default="../datasets/visdrone_argus_yolo")
    parser.add_argument("--image-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source).resolve()
    output_dir = Path(args.output).resolve()
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source": str(source_root),
        "output_dir": str(output_dir),
        "classes": UNIFIED_CLASSES,
        "splits": [
            convert_split(source_root, output_dir, split, args.image_mode)
            for split in ("train", "valid", "test")
        ],
        "note": "VisDrone contributes civilian and vehicle classes only; ignored/others categories are skipped.",
    }
    write_data_yaml(output_dir)
    (output_dir / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
