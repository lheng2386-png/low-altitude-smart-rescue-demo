"""Smoke tests for damage assessment and rescue-entry generation."""

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from damage_assessment_service import (  # noqa: E402
    assess_damage_and_entry,
    determine_scene_mode,
    generate_rescue_entry,
    summarize_damage,
)


def make_wide_area_mask():
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[:, 4:12] = 7  # Road-Clear component connected to image boundary.
    mask[35:43, 10:82] = 7
    mask[8:25, 30:48] = 2
    mask[8:25, 52:70] = 3
    mask[48:68, 35:55] = 4
    mask[48:68, 60:78] = 5
    mask[0:12, 82:99] = 1
    mask[30:42, 86:98] = 8
    return mask


def test_damage_summary_and_entry():
    mask = make_wide_area_mask()
    targets = [
        {
            "id": "T001",
            "class_name": "civilian",
            "bbox": [74, 50, 82, 60],
            "center": [78, 55],
            "area": 80,
        }
    ]

    summary = summarize_damage(mask)
    assert summary["building_damage"]["no_damage_area"] > 0
    assert summary["building_damage"]["major_damage_area"] > 0
    assert summary["road_stats"]["road_clear_ratio"] > 0
    assert summary["overall_damage_level"] in {"Superficial Damage", "Medium Damage", "Major Damage"}

    scene_mode, reason = determine_scene_mode(mask, targets=targets)
    assert scene_mode == "Wide-area Assessment", reason

    entry = generate_rescue_entry(mask, target=targets[0])
    assert entry["entry_found"] is True
    assert entry["entry_point_x"] is not None
    assert entry["entry_point_y"] is not None

    assessment = assess_damage_and_entry(mask, targets=targets, top_target=targets[0])
    assert assessment["scene_mode"] == "Wide-area Assessment"
    assert assessment["entry"]["entry_found"] is True
    assert assessment["path_planning_enabled"] is True


def test_no_road_returns_no_entry():
    mask = np.zeros((50, 60), dtype=np.uint8)
    mask[10:30, 20:45] = 4
    entry = generate_rescue_entry(mask)
    assert entry["entry_found"] is False

    assessment = assess_damage_and_entry(mask, targets=[])
    assert assessment["path_planning_enabled"] is False
    assert assessment["scene_mode"] == "Local Reconnaissance"


def test_large_target_is_local_reconnaissance():
    mask = make_wide_area_mask()
    targets = [
        {
            "id": "T900",
            "class_name": "civilian",
            "bbox": [10, 10, 88, 70],
            "center": [49, 40],
            "area": 4680,
        }
    ]
    scene_mode, reason = determine_scene_mode(mask, targets=targets)
    assert scene_mode == "Local Reconnaissance", reason

    assessment = assess_damage_and_entry(mask, targets=targets, top_target=targets[0])
    assert assessment["path_planning_enabled"] is False


if __name__ == "__main__":
    test_damage_summary_and_entry()
    test_no_road_returns_no_entry()
    test_large_target_is_local_reconnaissance()
    print("AeroRescue-AI damage assessment smoke test passed.")
