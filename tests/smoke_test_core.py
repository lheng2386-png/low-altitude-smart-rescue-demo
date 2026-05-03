import sys
import subprocess
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from path_planner import (  # noqa: E402
    build_cost_map,
    compare_path_plans,
    plan_baseline_path,
    plan_rescue_path,
    plan_risk_aware_path,
)
from segmentation_engine import (  # noqa: E402
    create_segmentation_overlay,
    summarize_segmentation,
    validate_segmentation_mask,
)
from segmentation_model import get_segmentation_model_status  # noqa: E402
from scene_applicability_gate import evaluate_scene_applicability  # noqa: E402
from terp_engine import calculate_terp, rank_targets_by_terp  # noqa: E402
from scripts.generate_demo_cases import _manual_demo_mask  # noqa: E402
from model_comparison.evaluate_detection_models import load_registry  # noqa: E402


def main():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :] = (90, 110, 130)

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    mask[24:44, 8:44] = 7
    mask[44:60, 32:60] = 8

    validation = validate_segmentation_mask(mask)
    assert validation["valid"] is True
    assert validation["unique_class_ids"] == [0, 1, 7, 8]
    assert validation["unknown_class_ids"] == []

    invalid_mask = mask.copy()
    invalid_mask[0, 0] = 99
    invalid_validation = validate_segmentation_mask(invalid_mask)
    assert invalid_validation["valid"] is False
    assert invalid_validation["unknown_class_ids"] == [99]

    summary = summarize_segmentation(mask)
    assert "water" in summary
    assert "road_clear" in summary
    assert "road_blocked" in summary

    overlay = create_segmentation_overlay(image, mask)
    assert overlay is not None
    assert overlay.shape == image.shape

    cost_map = build_cost_map(mask, 64, 64)
    assert cost_map.shape == (64, 64)
    assert float(cost_map[16, 16]) == 100.0
    assert float(cost_map[30, 20]) == 1.0
    assert float(cost_map[50, 40]) == 80.0

    ranked_targets = [
        {
            "id": "T001",
            "target_id": "T001",
            "class_name": "civilian",
            "center": [56, 8],
            "bbox": [52, 4, 60, 12],
            "confidence": 0.92,
            "area": 64.0,
            "risk_score": 95.0,
        }
    ]
    path_result = plan_rescue_path(ranked_targets, mask, 64, 64, start_point=(4, 60))
    assert path_result["found"] is True
    assert path_result["path_length"] > 0

    baseline = plan_baseline_path(ranked_targets, 64, 64, start_point=(4, 60))
    risk_aware = plan_risk_aware_path(ranked_targets, mask, 64, 64, start_point=(4, 60))
    comparison = compare_path_plans(baseline, risk_aware, mask)
    assert comparison["baseline_length"] > 0
    assert comparison["risk_aware_length"] > 0
    assert "risk_reduction" in comparison

    no_mask_baseline = plan_baseline_path(ranked_targets, 64, 64, start_point=(4, 60))
    no_mask_risk_aware = plan_risk_aware_path(ranked_targets, None, 64, 64, start_point=(4, 60))
    no_mask_comparison = compare_path_plans(no_mask_baseline, no_mask_risk_aware, None)
    assert "语义分割掩码" in no_mask_comparison["message"]

    terp = calculate_terp(
        ranked_targets[0],
        64,
        64,
        environment_context={"environment_risk_score": 18.0, "environment_reason": "目标附近存在水域。"},
        path_result=risk_aware,
    )
    assert terp["terp_score"] > 0
    assert terp["terp_level"] in {"Low", "Medium", "High", "Critical"}

    terp_ranking = rank_targets_by_terp(
        ranked_targets,
        64,
        64,
        environment_contexts={"T001": {"environment_risk_score": 18.0, "environment_reason": "目标附近存在水域。"}},
        path_results={"T001": risk_aware},
    )
    assert terp_ranking[0]["target_id"] == "T001"

    multi_targets = [
        ranked_targets[0],
        {
            "id": "T002",
            "target_id": "T002",
            "class_name": "rescuer",
            "center": [12, 12],
            "bbox": [8, 8, 16, 16],
            "confidence": 0.88,
            "area": 64.0,
            "risk_score": 20.0,
        },
    ]
    multi_ranking = rank_targets_by_terp(
        multi_targets,
        64,
        64,
        environment_contexts={
            "T001": {"environment_risk_score": 18.0, "environment_reason": "目标附近存在水域。"},
            "T002": {"environment_risk_score": 0.0, "environment_reason": "环境风险较低。"},
        },
        path_results={"T001": risk_aware, "T002": baseline},
    )
    assert multi_ranking[0]["terp_score"] >= multi_ranking[1]["terp_score"]

    gate_full = evaluate_scene_applicability(ranked_targets, segmentation_mask=mask, segmentation_source="uploaded")
    assert gate_full["allow_environment_fusion"] is True
    gate_no_target = evaluate_scene_applicability([], segmentation_mask=mask, segmentation_source="uploaded")
    assert gate_no_target["allow_path_planning"] is False

    generated_mask = _manual_demo_mask(64, 64, "road_blocked")
    generated_validation = validate_segmentation_mask(generated_mask)
    assert generated_validation["valid"] is True
    assert 8 in generated_validation["unique_class_ids"]

    registry = load_registry(ROOT_DIR / "model_comparison" / "model_registry.json")
    assert any(item["name"] == "yolov11n" for item in registry)

    missing_status = get_segmentation_model_status(ROOT_DIR / "checkpoints" / "missing_smoke_test.pth")
    assert missing_status["available"] is False

    help_result = subprocess.run(
        [sys.executable, str(ROOT_DIR / "scripts" / "generate_demo_cases.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "Generate AeroRescue-AI offline showcase demo cases" in help_result.stdout

    print("AeroRescue-AI demo case and model comparison smoke test passed.")


if __name__ == "__main__":
    main()
