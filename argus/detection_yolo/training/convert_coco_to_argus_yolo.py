"""Convert COCO-style object annotations to the Argus fused YOLO class space."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


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

LABEL_ALIASES = {
    "cat": "cat",
    "civilian": "civilian",
    "person": "civilian",
    "people": "civilian",
    "pedestrian": "civilian",
    "swimmer": "civilian",
    "human": "civilian",
    "human_target": "civilian",
    "rescuer": "rescuer",
    "rescue_worker": "rescuer",
    "firefighter": "rescuer",
    "cow": "cow",
    "dog": "dog",
    "horse": "horse",
    "vehicle": "vehicle",
    "car": "vehicle",
    "van": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
    "boat": "boat",
    "ship": "boat",
    "vessel": "boat",
    "watercraft": "boat",
    "fire": "fire",
    "flame": "fire",
    "smoke": "smoke",
    "water": "flood_water",
    "flood": "flood_water",
    "flood_water": "flood_water",
    "road_flooded": "flood_water",
    "building_flooded": "flood_water",
    "pool": "flood_water",
    "building": "building",
    "building_non_flooded": "building",
    "house": "building",
    "roof": "building",
    "structure": "building",
    "road": "road",
    "street": "road",
    "highway": "road",
    "paved_road": "road",
    "tree": "tree",
    "trees": "tree",
    "vegetation": "vegetation",
    "low_vegetation": "vegetation",
    "grass": "vegetation",
    "forest": "vegetation",
    "bridge": "bridge",
    "aircraft": "aircraft",
    "plane": "aircraft",
    "airplane": "aircraft",
    "helicopter": "aircraft",
}


def normalize_name(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


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


def load_coco(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_image(image_root: Path, file_name: str) -> Path | None:
    candidate = image_root / file_name
    if candidate.exists():
        return candidate
    matches = list(image_root.rglob(Path(file_name).name))
    return matches[0] if matches else None


def convert_annotation(annotation: dict[str, Any], image: dict[str, Any], category_map: dict[int, str]) -> str | None:
    target_label = category_map.get(int(annotation.get("category_id")))
    if target_label not in UNIFIED_CLASSES:
        return None
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    width = float(image.get("width") or 0)
    height = float(image.get("height") or 0)
    if width <= 0 or height <= 0:
        return None
    x, y, w, h = [float(value) for value in bbox[:4]]
    if w <= 1 or h <= 1:
        return None

    x1 = max(0.0, min(width, x))
    y1 = max(0.0, min(height, y))
    x2 = max(0.0, min(width, x + w))
    y2 = max(0.0, min(height, y + h))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 1 or box_h <= 1:
        return None

    class_id = UNIFIED_CLASSES.index(target_label)
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    return f"{class_id} {cx:.8f} {cy:.8f} {box_w / width:.8f} {box_h / height:.8f}"


def convert_split(
    annotation_path: Path,
    image_root: Path,
    output_dir: Path,
    split: str,
    image_mode: str,
    source_prefix: str,
) -> dict[str, Any]:
    coco = load_coco(annotation_path)
    image_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    category_map = {}
    skipped_categories = []
    for category in coco.get("categories", []):
        source_name = normalize_name(category.get("name", ""))
        target_name = LABEL_ALIASES.get(source_name)
        if target_name in UNIFIED_CLASSES:
            category_map[int(category["id"])] = target_name
        else:
            skipped_categories.append(category.get("name", ""))

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        if annotation.get("iscrowd") == 1:
            continue
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    out_images = output_dir / split / "images"
    out_labels = output_dir / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    converted_boxes = 0
    skipped_boxes = 0
    missing_images = 0
    for image_id, image in sorted(image_by_id.items()):
        source_image = find_image(image_root, image["file_name"])
        if source_image is None:
            missing_images += 1
            continue
        suffix = source_image.suffix.lower() or ".jpg"
        target_stem = f"{source_prefix}_{Path(image['file_name']).stem}_{image_id}"
        link_or_copy(source_image, out_images / f"{target_stem}{suffix}", image_mode)
        copied_images += 1

        lines = []
        for annotation in annotations_by_image.get(image_id, []):
            line = convert_annotation(annotation, image, category_map)
            if line is None:
                skipped_boxes += 1
                continue
            lines.append(line)
        converted_boxes += len(lines)
        (out_labels / f"{target_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return {
        "split": split,
        "annotation_path": str(annotation_path),
        "image_root": str(image_root),
        "images": copied_images,
        "boxes": converted_boxes,
        "missing_images": missing_images,
        "skipped_boxes": skipped_boxes,
        "skipped_categories": sorted(set(str(name) for name in skipped_categories)),
    }


def parse_split_arg(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected SPLIT=ANNOTATIONS_JSON=IMAGE_ROOT")
    split = "valid" if parts[0] == "val" else parts[0]
    if split not in {"train", "valid", "test"}:
        raise argparse.ArgumentTypeError("Split must be train, valid/val, or test")
    return split, Path(parts[1]).resolve(), Path(parts[2]).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COCO annotations to Argus fused YOLO format.")
    parser.add_argument(
        "--split",
        action="append",
        type=parse_split_arg,
        required=True,
        help="SPLIT=ANNOTATIONS_JSON=IMAGE_ROOT. Repeat for train/valid/test.",
    )
    parser.add_argument("--output", default="../datasets/coco_argus_yolo")
    parser.add_argument("--source-prefix", default="coco")
    parser.add_argument("--image-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "output_dir": str(output_dir),
        "classes": UNIFIED_CLASSES,
        "splits": [
            convert_split(annotation_path, image_root, output_dir, split, args.image_mode, args.source_prefix)
            for split, annotation_path, image_root in args.split
        ],
    }
    write_data_yaml(output_dir)
    (output_dir / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
