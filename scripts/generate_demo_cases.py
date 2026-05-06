"""Generate offline 灾情感知及影响评估 demo case outputs.

The script uses local repository images and local YOLO weights when available.
Generated segmentation masks are manually prepared demo masks for decision-layer
demonstration, not automatic segmentation predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from path_planner import (  # noqa: E402
    compare_path_plans,
    create_dual_path_overlay,
    create_path_overlay,
    plan_baseline_path,
    plan_risk_aware_path,
)
from priority_ranker import rank_targets  # noqa: E402
from report_generator import generate_report  # noqa: E402
from segmentation_engine import (  # noqa: E402
    create_segmentation_overlay,
    summarize_segmentation,
    validate_segmentation_mask,
)
from segmentation_engine import get_environment_context_for_target  # noqa: E402
from terp_engine import rank_targets_by_terp  # noqa: E402


CASE_DIR = ROOT_DIR / "demo_cases"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "static" / "images" / "showcase"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
YOLO_WEIGHT_CANDIDATES = [
    ROOT_DIR / "models" / "yolov11m" / "best.pt",
    ROOT_DIR / "models" / "yolov11s" / "best.pt",
    ROOT_DIR / "models" / "yolov11n" / "best.pt",
    ROOT_DIR / "models" / "yolov11l" / "best.pt",
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_configs(selected_cases: list[str] | None = None) -> list[dict]:
    configs = []
    for path in sorted(CASE_DIR.glob("case_*/case_config.json")):
        config = _read_json(path)
        if selected_cases and config["case_id"] not in selected_cases:
            continue
        config["_config_path"] = str(path)
        configs.append(config)
    return configs


def _resolve_first_existing(paths: list[str]) -> Path | None:
    for item in paths:
        path = ROOT_DIR / item
        if path.exists():
            return path
    return None


def _select_weights(model_variant: str | None = None) -> Path | None:
    if model_variant:
        candidate = ROOT_DIR / "models" / model_variant / "best.pt"
        if candidate.exists():
            return candidate
    for candidate in YOLO_WEIGHT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _load_yolo_model(weights_path: Path | None):
    if weights_path is None:
        return None, "No local YOLO weights found; detection overlay will use fallback behavior."
    try:
        from ultralytics import YOLO

        return YOLO(str(weights_path)), f"Loaded YOLO weights: {weights_path.relative_to(ROOT_DIR)}"
    except Exception as exc:
        return None, f"Could not load YOLO model: {exc}"


def _extract_targets(results) -> list[dict]:
    targets = []
    if not results:
        return targets

    result = results[0]
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return targets

    for index, box in enumerate(boxes, start=1):
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
        confidence = float(box.conf[0].detach().cpu().item())
        class_id = int(box.cls[0].detach().cpu().item())
        class_name = str(names.get(class_id, class_id))
        x1, y1, x2, y2 = xyxy
        targets.append(
            {
                "id": f"T{index:03d}",
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": [round(v, 2) for v in xyxy],
                "center": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
                "area": round(max(0.0, x2 - x1) * max(0.0, y2 - y1), 2),
            }
        )
    return targets


def _draw_detection_overlay(image_rgb: np.ndarray, targets: list[dict]) -> np.ndarray:
    overlay = image_rgb.copy()
    if not CV2_AVAILABLE:
        pil = Image.fromarray(overlay.astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        for target in targets:
            x1, y1, x2, y2 = [int(round(v)) for v in target["bbox"]]
            label = f"{target['id']} {target['class_name']} {target['confidence']:.2f}"
            draw.rectangle((x1, y1, x2, y2), outline=(255, 230, 0), width=2)
            draw.text((x1, max(2, y1 - 14)), label, fill=(255, 230, 0))
        return np.asarray(pil)
    for target in targets:
        x1, y1, x2, y2 = [int(round(v)) for v in target["bbox"]]
        label = f"{target['id']} {target['class_name']} {target['confidence']:.2f}"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 230, 0), 2)
        cv2.putText(
            overlay,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 230, 0),
            2,
            cv2.LINE_AA,
        )
    return overlay


def _manual_demo_mask(width: int, height: int, mask_type: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)

    if mask_type == "flood_water":
        mask[int(height * 0.42) :, :] = 1
        mask[int(height * 0.62) : int(height * 0.75), int(width * 0.05) : int(width * 0.95)] = 7
        mask[int(height * 0.52) : int(height * 0.64), int(width * 0.38) : int(width * 0.62)] = 8
    elif mask_type == "building_damage":
        mask[int(height * 0.18) : int(height * 0.62), int(width * 0.15) : int(width * 0.55)] = 4
        mask[int(height * 0.36) : int(height * 0.86), int(width * 0.48) : int(width * 0.86)] = 5
        mask[int(height * 0.72) : int(height * 0.84), int(width * 0.05) : int(width * 0.95)] = 7
    elif mask_type == "road_blocked":
        mask[int(height * 0.58) : int(height * 0.72), :] = 7
        mask[int(height * 0.48) : int(height * 0.82), int(width * 0.42) : int(width * 0.58)] = 8
        mask[int(height * 0.18) : int(height * 0.42), int(width * 0.65) : int(width * 0.88)] = 1
    elif mask_type == "mixed_environment":
        mask[int(height * 0.48) :, : int(width * 0.36)] = 1
        mask[int(height * 0.58) : int(height * 0.72), int(width * 0.2) : int(width * 0.98)] = 7
        mask[int(height * 0.18) : int(height * 0.48), int(width * 0.45) : int(width * 0.72)] = 4
        mask[int(height * 0.12) : int(height * 0.32), int(width * 0.05) : int(width * 0.28)] = 9
        mask[int(height * 0.42) : int(height * 0.55), int(width * 0.72) : int(width * 0.9)] = 6
    elif mask_type == "background_only":
        mask[:, :] = 0
    else:
        mask[:, :] = 0

    return mask


def _save_image(path: Path, array: np.ndarray | Image.Image | None) -> bool:
    if array is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(array, Image.Image):
        array.save(path)
    else:
        Image.fromarray(array.astype(np.uint8)).save(path)
    return True


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _path_json_safe(comparison: dict) -> dict:
    return {
        "baseline_length": comparison.get("baseline_length", 0),
        "baseline_cost": comparison.get("baseline_cost", 0.0),
        "risk_aware_length": comparison.get("risk_aware_length", 0),
        "risk_aware_cost": comparison.get("risk_aware_cost", 0.0),
        "baseline_environment_risk": comparison.get("baseline_environment_risk", 0.0),
        "risk_aware_environment_risk": comparison.get("risk_aware_environment_risk", 0.0),
        "risk_reduction": comparison.get("risk_reduction", 0.0),
        "message": comparison.get("message", ""),
    }


def _generate_one_case(config: dict, model, model_status: str, output_root: Path) -> dict:
    case_id = config["case_id"]
    output_dir = output_root / case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = _resolve_first_existing(config.get("image_candidates", []))
    if image_path is None:
        summary = {
            "case_id": case_id,
            "status": "missing_input",
            "message": "No input image candidate exists.",
            "artifacts": [],
        }
        (output_dir / "case_summary.md").write_text(_case_summary_md(config, summary), encoding="utf-8")
        return summary

    image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image)
    height, width = image_rgb.shape[:2]
    _save_image(output_dir / "input.jpg", image)

    targets = []
    detection_message = model_status
    if model is not None:
        try:
            results = model(image_rgb[:, :, ::-1], conf=float(config.get("conf_threshold", 0.3)), verbose=False)
            targets = _extract_targets(results)
            detection_message += f"; detected {len(targets)} targets."
        except Exception as exc:
            detection_message += f"; detection failed: {exc}"
            targets = []
    else:
        detection_message += "; no detection was run."

    detection_overlay = _draw_detection_overlay(image_rgb, targets)
    _save_image(output_dir / "detection_overlay.png", detection_overlay)

    mask = _manual_demo_mask(width, height, config.get("mask_type", "background_only"))
    _save_image(output_dir / "demo_mask.png", mask)
    validation = validate_segmentation_mask(mask)
    segmentation_summary = summarize_segmentation(mask) if validation.get("valid") else {}
    segmentation_overlay = create_segmentation_overlay(image_rgb, mask) if validation.get("valid") else None
    _save_image(output_dir / "segmentation_overlay.png", segmentation_overlay)

    ranked_targets = rank_targets(targets, width, height, mask if validation.get("valid") else None)
    start_x, start_y = config.get("start_point", [20, -1])
    if start_y is None or start_y < 0:
        start_y = height - 20

    baseline_path = plan_baseline_path(ranked_targets, width, height, start_point=(start_x, start_y))
    risk_aware_path = plan_risk_aware_path(ranked_targets, mask if validation.get("valid") else None, width, height, start_point=(start_x, start_y))
    comparison = compare_path_plans(baseline_path, risk_aware_path, mask if validation.get("valid") else None)

    base_path_image = segmentation_overlay if segmentation_overlay is not None else image_rgb
    risk_overlay = create_path_overlay(base_path_image, risk_aware_path)
    dual_overlay = create_dual_path_overlay(base_path_image, baseline_path, risk_aware_path)
    _save_image(output_dir / "risk_aware_path_overlay.png", risk_overlay)
    _save_image(output_dir / "dual_path_overlay.png", dual_overlay)

    environment_contexts = {}
    target_path_results = {}
    for target in targets:
        target_id = target.get("id")
        if validation.get("valid"):
            environment_contexts[target_id] = get_environment_context_for_target(target, mask)
        target_path_results[target_id] = plan_risk_aware_path([target], mask if validation.get("valid") else None, width, height, start_point=(start_x, start_y))

    terp_rankings = rank_targets_by_terp(targets, width, height, environment_contexts=environment_contexts, path_results=target_path_results)
    report = generate_report(targets, ranked_targets, segmentation_summary, risk_aware_path, terp_rankings, comparison)

    _write_csv(
        output_dir / "target_table.csv",
        targets,
        ["id", "class_name", "confidence", "bbox", "center", "area"],
    )
    _write_csv(
        output_dir / "terp_ranking.csv",
        terp_rankings,
        ["rank", "target_id", "class_name", "target_score", "environment_score", "accessibility_score", "terp_score", "terp_level", "reason"],
    )
    (output_dir / "path_comparison.json").write_text(json.dumps(_path_json_safe(comparison), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "rescue_report.txt").write_text(report, encoding="utf-8")

    artifacts = [
        "input.jpg",
        "demo_mask.png",
        "detection_overlay.png",
        "segmentation_overlay.png",
        "risk_aware_path_overlay.png",
        "dual_path_overlay.png",
        "target_table.csv",
        "terp_ranking.csv",
        "path_comparison.json",
        "rescue_report.txt",
        "case_summary.md",
    ]
    summary = {
        "case_id": case_id,
        "title": config.get("title", case_id),
        "status": "generated",
        "input": str(image_path.relative_to(ROOT_DIR)),
        "target_count": len(targets),
        "mask_validation": validation,
        "detection_message": detection_message,
        "manual_mask": True,
        "auto_segmentation_checkpoint_used": False,
        "artifacts": artifacts,
    }
    (output_dir / "case_summary.md").write_text(_case_summary_md(config, summary), encoding="utf-8")
    return summary


def _case_summary_md(config: dict, summary: dict) -> str:
    artifacts = summary.get("artifacts", [])
    artifact_lines = "\n".join(f"- `{item}`" for item in artifacts) if artifacts else "- No artifacts generated."
    validation = summary.get("mask_validation", {})
    return f"""# {config.get('title', summary.get('case_id', 'Demo Case'))}

Scenario: {config.get('scenario', '')}

Status: {summary.get('status')}

Input: `{summary.get('input', 'not_available')}`

Target count: {summary.get('target_count', 0)}

Detection status: {summary.get('detection_message', summary.get('message', 'not_available'))}

Mask policy: This mask is manually prepared for decision-layer demonstration. It is not an automatic segmentation prediction.

Mask validation: {validation.get('message', 'not_available')}

Auto segmentation checkpoint used: {summary.get('auto_segmentation_checkpoint_used', False)}

Artifacts:

{artifact_lines}

Current limitations:

- Demo masks are manually prepared and should not be described as automatic model predictions.
- Path planning is an image-plane reference route, not a GPS route.
- No UAV flight control or real road network is connected.
"""


def generate_demo_cases(output_dir: Path, selected_cases: list[str] | None = None, model_variant: str | None = None) -> list[dict]:
    """Generate all selected demo cases into the showcase output directory."""
    configs = _case_configs(selected_cases)
    weights = _select_weights(model_variant)
    model, model_status = _load_yolo_model(weights)
    summaries = []
    for config in configs:
        print(f"Generating {config['case_id']}...")
        summaries.append(_generate_one_case(config, model, model_status, output_dir))
    (output_dir / "demo_case_index.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 灾情感知及影响评估 offline showcase demo cases.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for generated case artifacts.")
    parser.add_argument("--case", action="append", help="Generate only a specific case_id. Can be used multiple times.")
    parser.add_argument("--model", default=None, help="Preferred YOLO variant, such as yolov11m, yolov11s, or yolov11n.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = generate_demo_cases(Path(args.output_dir), selected_cases=args.case, model_variant=args.model)
    print(f"Generated {len(summaries)} demo cases.")
    print("All demo masks are manually prepared for decision-layer demonstration, not automatic segmentation predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
