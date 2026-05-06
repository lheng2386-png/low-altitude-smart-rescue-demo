import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from scene_mode_and_entry_service import (  # noqa: E402
    analyze_scene_mode,
    build_path_planning_gate_result,
    find_rescue_entry_point,
)


def _image(width=100, height=80):
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))


def main():
    image = _image()
    large_target = [{"id": "T001", "bbox": [10, 10, 90, 70], "area": 80 * 60, "center": [50, 40]}]
    result = analyze_scene_mode(image, segmentation_mask=None, detections=large_target)
    assert result["scene_mode"] == "local_reconnaissance"
    assert result["path_planning_allowed"] is False

    result = analyze_scene_mode(image, segmentation_mask=None, detections=[])
    assert result["scene_mode"] == "unknown"
    assert result["path_planning_allowed"] is False

    small_target = [{"id": "T001", "bbox": [45, 35, 55, 45], "area": 100, "center": [50, 40]}]

    no_road_mask = np.zeros((80, 100), dtype=np.uint8)
    no_road_mask[20:50, 30:70] = 4
    result = analyze_scene_mode(image, segmentation_mask=no_road_mask, detections=small_target)
    entry = find_rescue_entry_point(no_road_mask, target_point=[50, 40])
    gate = build_path_planning_gate_result(result, entry)
    assert entry["entry_found"] is False
    assert gate["path_enabled"] is False

    road_mask = np.zeros((80, 100), dtype=np.uint8)
    road_mask[:, 8:18] = 7
    road_mask[20:60, 50:80] = 2
    result = analyze_scene_mode(image, segmentation_mask=road_mask, detections=small_target)
    assert result["scene_mode"] == "wide_area_assessment"
    assert result["path_planning_allowed"] is True

    entry = find_rescue_entry_point(road_mask, target_point=[50, 40])
    assert entry["entry_found"] is True
    assert entry["entry_point"] is not None
    assert entry["candidate_count"] >= 1

    gate = build_path_planning_gate_result(result, entry)
    assert gate["path_enabled"] is True
    assert gate["start_point"] == entry["entry_point"]
    assert gate["start_source"] == "auto_road_clear_entry"

    local_gate = build_path_planning_gate_result(
        {"scene_mode": "local_reconnaissance", "reason": "local close-up"},
        entry,
    )
    assert local_gate["path_enabled"] is False

    forced_gate = build_path_planning_gate_result(
        {"scene_mode": "local_reconnaissance", "reason": "local close-up"},
        entry,
        use_manual_start=True,
        manual_start_point=(5, 75),
        force_path_planning=True,
    )
    assert forced_gate["path_enabled"] is True
    assert forced_gate["start_source"] == "manual_force"

    print("灾情感知及影响评估 scene mode and rescue entry smoke test passed.")


if __name__ == "__main__":
    main()
