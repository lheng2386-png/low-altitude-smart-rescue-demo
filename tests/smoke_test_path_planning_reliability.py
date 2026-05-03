import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from scene_mode_and_entry_service import (  # noqa: E402
    build_path_planning_reliability_status,
    format_path_planning_reliability_status,
)


def main():
    disabled_gate = {"path_enabled": False, "gate_reason": "not suitable"}
    scene_local = {"scene_mode": "local_reconnaissance", "reason": "close-up"}
    no_entry = {"entry_found": False, "entry_reason": "no road"}
    status = build_path_planning_reliability_status(
        scene_local,
        no_entry,
        disabled_gate,
        segmentation_source_metadata={"source_type": "none", "source_label": "None"},
    )
    assert status["reliability_level"] == "not_applicable"
    assert status["is_real_gps_navigation"] is False
    assert status["scene_mode_method"] == "rule_based"
    assert status["reliability_note"]

    enabled_gate = {"path_enabled": True, "start_point": [1, 2], "force_path_planning": False}
    scene_wide = {"scene_mode": "wide_area_assessment", "reason": "road exists"}
    entry = {"entry_found": True, "entry_point": [1, 2]}

    demo_status = build_path_planning_reliability_status(
        scene_wide,
        entry,
        enabled_gate,
        segmentation_source_metadata={"source_type": "demo_fallback", "source_label": "Demo / Fallback"},
    )
    assert demo_status["reliability_level"] == "low"
    assert demo_status["human_review_required"] is True

    uploaded_status = build_path_planning_reliability_status(
        scene_wide,
        entry,
        enabled_gate,
        segmentation_source_metadata={"source_type": "uploaded_mask", "source_label": "Uploaded Mask"},
    )
    assert uploaded_status["reliability_level"] == "medium"
    assert uploaded_status["is_real_gps_navigation"] is False

    auto_status = build_path_planning_reliability_status(
        scene_wide,
        entry,
        enabled_gate,
        segmentation_source_metadata={
            "source_type": "auto_model",
            "source_label": "Auto Segmentation Model",
            "is_model_prediction": True,
            "prediction_success": True,
        },
    )
    assert auto_status["reliability_level"] == "medium"

    forced_status = build_path_planning_reliability_status(
        scene_local,
        entry,
        {"path_enabled": True, "force_path_planning": True},
        segmentation_source_metadata={"source_type": "uploaded_mask", "source_label": "Uploaded Mask"},
        force_path_planning=True,
    )
    assert forced_status["reliability_level"] == "low"
    assert forced_status["human_review_required"] is True

    text = format_path_planning_reliability_status(uploaded_status)
    assert "图像平面参考路径" in text
    assert "规则判断" in text
    assert "Mask" in text

    print("AeroRescue-AI path planning reliability smoke test passed.")


if __name__ == "__main__":
    main()
