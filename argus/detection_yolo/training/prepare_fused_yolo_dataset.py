"""Build a unified YOLO dataset for improving Argus detection.

The main source is the AeroRescue YOLO dataset. Optional Argus/VisDrone-style
YOLO datasets can be added later and remapped into the same class space.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path


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
    "pedestrian": "civilian",
    "people": "civilian",
    "human": "civilian",
    "human_candidate": "civilian",
    "cow": "cow",
    "dog": "dog",
    "horse": "horse",
    "rescuer": "rescuer",
    "rescue_worker": "rescuer",
    "firefighter": "rescuer",
    "vehicle": "vehicle",
    "car": "vehicle",
    "van": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "tractor": "vehicle",
    "trailer": "vehicle",
    "tricycle": "vehicle",
    "awning-tricycle": "vehicle",
    "motor": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
    "fire": "fire",
    "flame": "fire",
    "smoke": "smoke",
    "boat": "boat",
    "ship": "boat",
    "vessel": "boat",
    "water": "flood_water",
    "flood": "flood_water",
    "flood_water": "flood_water",
    "standing_water": "flood_water",
    "inundation": "flood_water",
    "building": "building",
    "house": "building",
    "roof": "building",
    "structure": "building",
    "residential_building": "building",
    "commercial_building": "building",
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_names_from_yaml(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            value = stripped.split(":", 1)[1].strip()
            names = ast.literal_eval(value)
            if isinstance(names, dict):
                return [names[key] for key in sorted(names)]
            return list(names)
    raise ValueError(f"Could not find YOLO names in {path}")


def resolve_yolo_path(dataset_root: Path, yaml_path: Path, key: str) -> Path:
    text = yaml_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            value = stripped.split(":", 1)[1].strip().strip("'\"")
            configured = (yaml_path.parent / value).resolve()
            if configured.exists():
                return configured
            local_split = "valid" if key == "val" else key
            local_path = (dataset_root / local_split / "images").resolve()
            if local_path.exists():
                return local_path
            return configured
    local_split = "valid" if key == "val" else key
    return (dataset_root / local_split / "images").resolve()


def write_dataset_yaml(output_dir: Path) -> None:
    names = ", ".join(f"'{name}'" for name in UNIFIED_CLASSES)
    data_yaml = "\n".join(
        [
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "",
            f"nc: {len(UNIFIED_CLASSES)}",
            f"names: [{names}]",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


def write_direct_dataset_yaml(output_dir: Path, source_root: Path, source_yaml: Path) -> dict:
    """Write a fused data.yaml that points to the original dataset in place.

    This is valid for the AeroRescue dataset because its six classes are the
    first six classes in UNIFIED_CLASSES, so existing label ids remain correct.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = resolve_yolo_path(source_root, source_yaml, "train")
    valid_path = resolve_yolo_path(source_root, source_yaml, "val")
    test_path = resolve_yolo_path(source_root, source_yaml, "test")
    names = ", ".join(f"'{name}'" for name in UNIFIED_CLASSES)
    data_yaml = "\n".join(
        [
            f"train: {train_path}",
            f"val: {valid_path}",
            f"test: {test_path}",
            "",
            f"nc: {len(UNIFIED_CLASSES)}",
            f"names: [{names}]",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    summary = {
        "output_dir": str(output_dir),
        "mode": "direct",
        "source": str(source_root),
        "classes": UNIFIED_CLASSES,
        "note": "Uses the AeroRescue dataset in place. Existing labels remain valid because its six classes are the first six fused classes.",
        "paths": {"train": str(train_path), "valid": str(valid_path), "test": str(test_path)},
    }
    (output_dir / "fusion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def link_or_copy_image(source: Path, target: Path, mode: str) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
    else:
        target.symlink_to(source.resolve())


def copy_and_remap_split(
    source_root: Path,
    source_yaml: Path,
    split: str,
    source_name: str,
    output_dir: Path,
    image_mode: str,
) -> dict:
    source_names = parse_names_from_yaml(source_yaml)
    source_to_target = {}
    skipped_classes = []
    for idx, name in enumerate(source_names):
        normalized = str(name).strip().lower().replace(" ", "_")
        target_name = LABEL_ALIASES.get(normalized)
        if target_name in UNIFIED_CLASSES:
            source_to_target[idx] = UNIFIED_CLASSES.index(target_name)
        else:
            skipped_classes.append(name)

    images_dir = resolve_yolo_path(source_root, source_yaml, "val" if split == "valid" else split)
    labels_dir = Path(str(images_dir).replace("/images", "/labels"))
    out_images = output_dir / split / "images"
    out_labels = output_dir / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_labels = 0
    skipped_annotations = 0
    missing_labels = 0

    for image_path in sorted(images_dir.iterdir()) if images_dir.exists() else []:
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        target_stem = f"{source_name}_{image_path.stem}"
        target_image = out_images / f"{target_stem}{image_path.suffix.lower()}"
        link_or_copy_image(image_path, target_image, image_mode)
        copied_images += 1

        label_path = labels_dir / f"{image_path.stem}.txt"
        target_label = out_labels / f"{target_stem}.txt"
        if not label_path.exists():
            target_label.write_text("", encoding="utf-8")
            missing_labels += 1
            continue

        remapped_lines = []
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            source_class = int(float(parts[0]))
            if source_class not in source_to_target:
                skipped_annotations += 1
                continue
            remapped_lines.append(" ".join([str(source_to_target[source_class]), *parts[1:5]]))
        target_label.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")
        copied_labels += 1

    return {
        "source": source_name,
        "split": split,
        "images": copied_images,
        "labels": copied_labels,
        "missing_labels": missing_labels,
        "skipped_annotations": skipped_annotations,
        "skipped_classes": sorted(set(skipped_classes)),
    }


def build_dataset(sources: list[tuple[str, Path]], output_dir: Path, image_mode: str = "symlink") -> dict:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    summary = {"output_dir": str(output_dir), "classes": UNIFIED_CLASSES, "splits": []}
    for source_name, source_root in sources:
        source_yaml = source_root / "data.yaml"
        if not source_yaml.exists():
            raise FileNotFoundError(f"Missing YOLO data.yaml for {source_name}: {source_yaml}")
        for split in ("train", "valid", "test"):
            summary["splits"].append(
                copy_and_remap_split(
                    source_root,
                    source_yaml,
                    split,
                    source_name,
                    output_dir,
                    image_mode=image_mode,
                )
            )

    write_dataset_yaml(output_dir)
    (output_dir / "fusion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a unified Argus YOLO training dataset.")
    parser.add_argument(
        "--aerorescue-dataset",
        default="../../../urban-disaster-monitor/dataset",
        help="Path to the existing AeroRescue YOLO dataset.",
    )
    parser.add_argument(
        "--argus-yolo-dataset",
        action="append",
        default=[],
        help="Optional additional YOLO dataset root to merge, e.g. VisDrone converted to YOLO.",
    )
    parser.add_argument("--output", default="../datasets/argus_fused_yolo", help="Output dataset directory.")
    parser.add_argument(
        "--image-mode",
        choices=["direct", "symlink", "copy"],
        default="direct",
        help="direct writes data.yaml to the source dataset; symlink/copy builds a remapped merged dataset.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    aerorescue_root = (script_dir / args.aerorescue_dataset).resolve()
    sources = [("aerorescue", aerorescue_root)]
    for index, dataset in enumerate(args.argus_yolo_dataset, start=1):
        sources.append((f"argus{index}", Path(dataset).resolve()))

    output_dir = (script_dir / args.output).resolve()
    if args.image_mode == "direct" and not args.argus_yolo_dataset:
        summary = write_direct_dataset_yaml(output_dir, aerorescue_root, aerorescue_root / "data.yaml")
    else:
        image_mode = "symlink" if args.image_mode == "direct" else args.image_mode
        summary = build_dataset(sources, output_dir, image_mode=image_mode)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
