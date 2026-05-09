"""Export Argus report images and detection prelabels for YOLO review.

The generated labels are preannotations, not ground truth. Review and correct
them in a labeling tool before using the dataset for training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import urllib.error
import urllib.request
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
    "human": "civilian",
    "human_target": "civilian",
    "rescuer": "rescuer",
    "rescue_worker": "rescuer",
    "firefighter": "rescuer",
    "cow": "cow",
    "dog": "dog",
    "horse": "horse",
    "vehicle": "vehicle",
    "land_vehicle": "vehicle",
    "small_vehicle": "vehicle",
    "car": "vehicle",
    "van": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "tractor": "vehicle",
    "trailer": "vehicle",
    "tricycle": "vehicle",
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


def fetch_json(api_base: str, path: str) -> Any:
    url = f"{api_base.rstrip('/')}/{path.lstrip('/')}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(api_base: str, relative_url: str, target: Path) -> None:
    url = f"{api_base.rstrip('/')}/{relative_url.lstrip('/')}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        target.write_bytes(response.read())


def copy_or_link_local(local_reports_data: Path, relative_url: str, target: Path, mode: str) -> bool:
    source = local_reports_data / relative_url.removeprefix("reports_data/").lstrip("/")
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
    else:
        target.symlink_to(source.resolve())
    return True


def discover_report_ids(api_base: str) -> list[int]:
    groups = fetch_json(api_base, "/groups/")
    report_ids: list[int] = []
    for group in groups:
        for report in group.get("reports", []) or []:
            if report.get("mapping_report") and report.get("report_id") is not None:
                report_ids.append(int(report["report_id"]))
    return sorted(set(report_ids))


def normalize_label(label: str | None) -> str | None:
    key = str(label or "").strip().lower().replace(" ", "_")
    return LABEL_ALIASES.get(key)


def yolo_line(det: dict[str, Any], width: int, height: int) -> str | None:
    target_label = normalize_label(det.get("class_name") or det.get("category_name"))
    if target_label not in UNIFIED_CLASSES:
        return None
    bbox = det.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    x, y, w, h = [float(v) for v in bbox[:4]]
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        return None

    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 1 or box_h <= 1:
        return None

    cls_id = UNIFIED_CLASSES.index(target_label)
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    nw = box_w / width
    nh = box_h / height
    return f"{cls_id} {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}"


def choose_split(index: int, total: int, train_ratio: float, valid_ratio: float) -> str:
    if total <= 1:
        return "train"
    fraction = index / total
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + valid_ratio:
        return "valid"
    return "test"


def write_data_yaml(output_dir: Path) -> None:
    names = ", ".join(f"'{name}'" for name in UNIFIED_CLASSES)
    text = "\n".join(
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
    (output_dir / "data.yaml").write_text(text, encoding="utf-8")


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output).resolve()
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_ids = args.report_id or discover_report_ids(args.api_base)
    local_reports_data = Path(args.local_reports_data).resolve() if args.local_reports_data else None

    records: list[dict[str, Any]] = []
    detections_by_image: dict[int, list[dict[str, Any]]] = {}
    for report_id in report_ids:
        for det in fetch_json(args.api_base, f"/detections/r/{report_id}"):
            if args.verified_only and not det.get("manually_verified"):
                continue
            detections_by_image.setdefault(int(det["image_id"]), []).append(det)
        for image in fetch_json(args.api_base, f"/images/report/{report_id}"):
            if Path(str(image.get("url", ""))).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            records.append({"report_id": report_id, "image": image})

    random.Random(args.seed).shuffle(records)
    manifest: list[dict[str, Any]] = []
    stats = {
        "reports": report_ids,
        "images": 0,
        "prelabels": 0,
        "empty_label_files": 0,
        "download_failures": 0,
        "skipped_unmapped_detections": 0,
    }

    for index, record in enumerate(records):
        image = record["image"]
        split = choose_split(index, len(records), args.train_ratio, args.valid_ratio)
        image_id = int(image["id"])
        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        suffix = Path(image["url"]).suffix.lower() or ".jpg"
        filename = f"report{record['report_id']}_image{image_id}{suffix}"
        target_image = output_dir / split / "images" / filename
        target_label = output_dir / split / "labels" / f"{Path(filename).stem}.txt"

        try:
            if local_reports_data and copy_or_link_local(local_reports_data, image["url"], target_image, args.local_image_mode):
                pass
            else:
                download_file(args.api_base, image["url"], target_image)
        except (OSError, urllib.error.URLError) as exc:
            stats["download_failures"] += 1
            manifest.append({"image_id": image_id, "status": "download_failed", "error": str(exc)})
            continue

        lines = []
        for det in detections_by_image.get(image_id, []):
            line = yolo_line(det, width, height)
            if line is None:
                stats["skipped_unmapped_detections"] += 1
                continue
            lines.append(line)

        target_label.parent.mkdir(parents=True, exist_ok=True)
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["images"] += 1
        stats["prelabels"] += len(lines)
        if not lines:
            stats["empty_label_files"] += 1
        manifest.append(
            {
                "report_id": record["report_id"],
                "image_id": image_id,
                "split": split,
                "image": str(target_image.relative_to(output_dir)),
                "label": str(target_label.relative_to(output_dir)),
                "prelabels": len(lines),
                "prelabels_need_review": True,
                "source_url": image["url"],
            }
        )

    write_data_yaml(output_dir)
    summary = {
        **stats,
        "output_dir": str(output_dir),
        "classes": UNIFIED_CLASSES,
        "prelabels_need_review": True,
        "note": "Review and correct all prelabels before training; unreviewed model output is not ground truth.",
    }
    (output_dir / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Argus images and detection prelabels for YOLO review.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8008", help="Argus API base URL.")
    parser.add_argument("--report-id", type=int, action="append", default=[], help="Report id to export. Repeatable.")
    parser.add_argument("--output", default="../datasets/argus_review_yolo", help="Output YOLO dataset directory.")
    parser.add_argument("--local-reports-data", default="", help="Optional local reports_data directory for faster image copy/link.")
    parser.add_argument("--local-image-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--verified-only", action="store_true", help="Export only detections marked manually_verified=true.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2386)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.valid_ratio < 0 or args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("Expected 0 < train-ratio and train-ratio + valid-ratio < 1")

    print(json.dumps(export_dataset(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
