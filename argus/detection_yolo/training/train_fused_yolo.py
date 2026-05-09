"""Train a single fused YOLO model for the Argus YOLO worker."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Argus fused YOLOv11 detector.")
    parser.add_argument("--data", default="../datasets/argus_fused_yolo/data.yaml")
    parser.add_argument(
        "--base-model",
        default="../../../urban-disaster-monitor/models/yolov11s/best.pt",
        help="Local .pt path or an Ultralytics model name such as yolo11x.pt.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="argus_fused_yolov11")
    parser.add_argument("--export", default="../models/argus-fused-yolov11-best.pt")
    args = parser.parse_args()

    from ultralytics import YOLO

    script_dir = Path(__file__).resolve().parent
    data_yaml = (script_dir / args.data).resolve()
    base_model_path = (script_dir / args.base_model).resolve()
    export_path = (script_dir / args.export).resolve()

    if not data_yaml.exists():
        raise FileNotFoundError(f"Prepared fused data.yaml not found: {data_yaml}")
    base_model = str(base_model_path) if base_model_path.exists() else args.base_model

    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(script_dir.parent / "runs" / "train"),
        name=args.name,
        exist_ok=True,
    )

    run_dir = Path(getattr(results, "save_dir", script_dir.parent / "runs" / "train" / args.name))
    best_path = run_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best_path}")

    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, export_path)
    summary = {
        "run_dir": str(run_dir),
        "best_path": str(best_path),
        "export_path": str(export_path),
        "data": str(data_yaml),
        "base_model": str(base_model),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
    }
    (export_path.parent / "argus-fused-yolov11-training-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
