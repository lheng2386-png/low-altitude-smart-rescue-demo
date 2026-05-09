"""Convert DOTA-style oriented boxes to Argus fused YOLO axis-aligned boxes."""

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

DOTA_TO_UNIFIED = {
    "small-vehicle": "vehicle",
    "large-vehicle": "vehicle",
    "ship": "boat",
    "harbor": "boat",
    "plane": "aircraft",
    "helicopter": "aircraft",
    "bridge": "bridge",
    "roundabout": "road",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


def link_or_copy(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
    else:
        target.symlink_to(source.resolve())


def convert_dota_line(line: str, width: int, height: int) -> str | None:
    parts = line.strip().split()
    if len(parts) < 9 or parts[0].lower().startswith("imagesource"):
        return None
    try:
        coords = [float(value) for value in parts[:8]]
    except ValueError:
        return None
    source_label = parts[8].strip().lower()
    target_label = DOTA_TO_UNIFIED.get(source_label)
    if target_label not in UNIFIED_CLASSES:
        return None

    xs = coords[0::2]
    ys = coords[1::2]
    x1 = max(0.0, min(float(width), min(xs)))
    y1 = max(0.0, min(float(height), min(ys)))
    x2 = max(0.0, min(float(width), max(xs)))
    y2 = max(0.0, min(float(height), max(ys)))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 1 or box_h <= 1:
        return None

    cls_id = UNIFIED_CLASSES.index(target_label)
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    return f"{cls_id} {cx:.8f} {cy:.8f} {box_w / width:.8f} {box_h / height:.8f}"


def find_split_dir(source_root: Path, split: str) -> Path | None:
    candidates = {
        "train": ["train", "Train", "DOTA-v1.0_train", "DOTA-v1.5_train", "DOTA-v2.0_train"],
        "valid": ["val", "valid", "Val", "DOTA-v1.0_val", "DOTA-v1.5_val", "DOTA-v2.0_val"],
        "test": ["test", "test-dev", "Test", "DOTA-v1.0_test", "DOTA-v1.5_test", "DOTA-v2.0_test"],
    }
    for name in candidates[split]:
        candidate = source_root / name
        if candidate.exists():
            return candidate
    return None


def find_child(parent: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = parent / name
        if candidate.exists():
            return candidate
    return None


def convert_split(source_root: Path, output_dir: Path, split: str, image_mode: str) -> dict:
    split_dir = find_split_dir(source_root, split)
    if split_dir is None:
        return {"split": split, "status": "missing", "images": 0, "labels": 0}

    images_dir = find_child(split_dir, ["images", "JPEGImages", "Images"]) or split_dir
    labels_dir = find_child(split_dir, ["labelTxt", "annotations", "labelTxt-v1.0", "labelTxt-v1.5"])
    if labels_dir is None:
        return {"split": split, "status": "missing_labels", "images": 0, "labels": 0}

    out_images = output_dir / split / "images"
    out_labels = output_dir / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    converted_images = 0
    converted_boxes = 0
    skipped_boxes = 0
    missing_labels = 0
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        target_stem = f"dota_{image_path.stem}"
        link_or_copy(image_path, out_images / f"{target_stem}{image_path.suffix.lower()}", image_mode)
        lines = []
        for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            converted = convert_dota_line(raw_line, width, height)
            if converted is None:
                skipped_boxes += 1
                continue
            lines.append(converted)
        (out_labels / f"{target_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        converted_images += 1
        converted_boxes += len(lines)

    return {
        "split": split,
        "status": "ok",
        "source_dir": str(split_dir),
        "images": converted_images,
        "boxes": converted_boxes,
        "missing_labels": missing_labels,
        "skipped_boxes": skipped_boxes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DOTA labels to Argus fused YOLO format.")
    parser.add_argument("--source", required=True, help="Directory containing DOTA split folders.")
    parser.add_argument("--output", default="../datasets/dota_argus_yolo")
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
        "splits": [convert_split(source_root, output_dir, split, args.image_mode) for split in ("train", "valid", "test")],
        "note": "DOTA oriented boxes are converted to axis-aligned YOLO boxes; fine for detection, not for rotated-box evaluation.",
    }
    write_data_yaml(output_dir)
    (output_dir / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
