import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import path_planner  # noqa: E402
from path_planner import (  # noqa: E402
    build_cost_map,
    compare_path_plans,
    create_path_overlay,
    get_path_planning_dependency_status,
    plan_baseline_path,
    plan_risk_aware_path,
)


def _assert_path_truthfulness(result):
    assert result["path_type"] == "image_plane_path"
    assert result["is_gps_navigation"] is False
    assert result["is_gis_route"] is False
    assert result["human_review_required"] is True
    assert "not a GPS navigation route" in result["truthfulness_note"]


def main():
    status = get_path_planning_dependency_status()
    assert status["status"] in {"available", "dependency_missing"}
    if not path_planner.CV2_AVAILABLE:
        assert status["status"] == "dependency_missing"
        assert status["dependency"] == "opencv-python"
        assert status["can_support_decision"] is False

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 7
    cost_map = build_cost_map(mask, 64, 64)
    assert cost_map.shape == (64, 64)

    targets = [
        {
            "id": "T001",
            "target_id": "T001",
            "class_name": "civilian",
            "center": [58, 6],
            "bbox": [54, 2, 62, 10],
            "confidence": 0.9,
            "area": 64.0,
        }
    ]
    baseline = plan_baseline_path(targets, 64, 64, start_point=(4, 60))
    risk = plan_risk_aware_path(targets, mask, 64, 64, start_point=(4, 60))
    comparison = compare_path_plans(baseline, risk, mask)
    _assert_path_truthfulness(baseline)
    _assert_path_truthfulness(risk)
    _assert_path_truthfulness(comparison)

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    overlay = create_path_overlay(image, risk)
    assert overlay is not None
    assert overlay.shape == image.shape

    print("AeroRescue-AI optional cv2 path planner smoke test passed.")


if __name__ == "__main__":
    main()
