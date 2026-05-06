import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from decision_fusion_adapter import (  # noqa: E402
    build_decision_fusion_summary,
    compute_coverage_planning_score,
    compute_image_plane_search_priority_map,
    compute_segmentation_damage_impact_score,
)


def main():
    image_shape = (100, 100, 3)
    targets = [
        {
            "id": "T001",
            "class_name": "civilian",
            "bbox": [40, 40, 60, 60],
            "center": [50, 50],
            "confidence": 0.9,
        }
    ]
    search = compute_image_plane_search_priority_map(image_shape, targets=targets)
    assert search["success"] is True
    assert search["priority_map_shape"] == [100, 100]
    assert search["priority_map"].shape == (100, 100)
    assert search["priority_statistics"]["max_priority"] > search["priority_statistics"]["mean_priority"]
    assert search["priority_statistics"]["high_priority_area_ratio"] >= 0

    segmentation_summary = {
        "major_damage": 0.2,
        "destroyed_building": 0.1,
        "water": 0.15,
        "road_blocked": 0.1,
        "road_clear": 0.2,
    }
    impact = compute_segmentation_damage_impact_score(segmentation_summary=segmentation_summary)
    assert impact["success"] is True
    assert impact["impact_score"] > 0
    assert impact["impact_level"] in {"low", "medium", "high", "critical"}
    assert "不是 SKAI" in impact["truthfulness_note"] or "full GIS" in impact["truthfulness_note"]

    path_result = {
        "found": True,
        "path": [[0, 0], [10, 10], [20, 20], [30, 30], [40, 40], [50, 50]],
    }
    coverage = compute_coverage_planning_score(image_shape, path_result=path_result, priority_map=search["priority_map"])
    assert coverage["success"] is True
    assert coverage["coverage_score"] >= 0
    assert "image-plane" in coverage["truthfulness_note"] or "not a real UAV flight route" in coverage["truthfulness_note"]

    fusion = build_decision_fusion_summary(search, impact, coverage, {"consensus": {"consensus_summary": "辅助一致性"}})
    assert fusion["success"] is True
    assert fusion["decision_fusion_score"] >= 0
    assert isinstance(fusion["summary_markdown"], str)
    assert isinstance(fusion["recommended_actions"], list)
    assert fusion["human_review_required"] is True

    failed_impact = compute_segmentation_damage_impact_score(None, None)
    assert failed_impact["success"] is False

    print("灾情感知及影响评估 decision fusion adapter smoke test passed.")


if __name__ == "__main__":
    main()
