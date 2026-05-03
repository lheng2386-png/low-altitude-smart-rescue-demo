"""Lightweight local detection model comparison scaffold.

This script checks local detector weights and can run a small inference summary on
an image folder. It does not download models, train models, or fabricate metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT_DIR / "model_comparison" / "model_registry.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_registry(path: Path) -> list[dict]:
    """Load the model registry JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _model_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return round(path.stat().st_size / (1024 * 1024), 2)


def _list_images(image_folder: Path, max_images: int) -> list[Path]:
    if not image_folder.exists():
        return []
    images = [p for p in sorted(image_folder.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if max_images > 0:
        images = images[:max_images]
    return images


def evaluate_registry(registry_path: Path, image_folder: Path, output_csv: Path, max_images: int) -> list[dict]:
    """Run a friendly local inference summary for available YOLO weights."""
    registry = load_registry(registry_path)
    images = _list_images(image_folder, max_images)
    rows = []

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Ultralytics is unavailable: {exc}")
        print("Only weight-existence checks will be written.")
        YOLO = None

    for item in registry:
        weights_path = ROOT_DIR / item.get("weights", "")
        row = {
            "model_name": item.get("name", "unknown"),
            "source": "AeroRescue-AI" if item.get("type") == "ultralytics_yolo" else "Detection-Models reference",
            "result_type": "local_inference_summary" if item.get("type") == "ultralytics_yolo" else "reference_structure",
            "precision": "",
            "recall": "",
            "map50": "",
            "fps": "",
            "latency_ms": "",
            "status": "",
            "notes": "",
        }

        if not weights_path.exists():
            row["status"] = "missing_weights"
            row["notes"] = "Weights not found locally; no inference was run."
            rows.append(row)
            continue

        if item.get("type") != "ultralytics_yolo":
            row["status"] = "not_implemented"
            row["notes"] = "Optional detector scaffold only; no local evaluator implemented yet."
            rows.append(row)
            continue

        if not images:
            row["status"] = "available_no_images"
            row["notes"] = "Weights exist, but no images were found for inference summary."
            rows.append(row)
            continue

        if YOLO is None:
            row["status"] = "available_not_run"
            row["notes"] = "Weights exist, but ultralytics could not be imported."
            rows.append(row)
            continue

        try:
            model = YOLO(str(weights_path))
            start = time.perf_counter()
            detections = 0
            for image_path in images:
                results = model(str(image_path), verbose=False)
                for result in results:
                    detections += len(result.boxes) if result.boxes is not None else 0
            elapsed = max(time.perf_counter() - start, 1e-6)
            fps = len(images) / elapsed
            row["fps"] = round(fps, 2)
            row["latency_ms"] = round((elapsed / max(len(images), 1)) * 1000, 2)
            row["status"] = "inference_summary_only"
            row["notes"] = f"Ran {len(images)} images and produced {detections} detections. Labels were not evaluated, so mAP/precision/recall are blank."
        except Exception as exc:
            row["status"] = "failed"
            row["notes"] = f"Inference failed: {exc}"

        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AeroRescue-AI local detection model comparison scaffold.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Path to model_registry.json.")
    parser.add_argument("--image-folder", default=str(ROOT_DIR / "app" / "examples"), help="Local image folder for inference summary.")
    parser.add_argument("--output", default=str(ROOT_DIR / "model_comparison" / "local_inference_summary.csv"), help="Output CSV path.")
    parser.add_argument("--max-images", type=int, default=5, help="Maximum number of images to run. Use 0 for all.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = evaluate_registry(
            Path(args.registry),
            Path(args.image_folder),
            Path(args.output),
            args.max_images,
        )
    except Exception as exc:
        print(f"Model comparison scaffold failed: {exc}")
        return 1

    print(f"Model comparison scaffold wrote {len(rows)} rows to {args.output}")
    print("No mAP, precision, or recall values are reported unless real labeled evaluation is implemented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
