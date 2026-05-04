import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from mission_demo_orchestrator import (  # noqa: E402
    DEMO_STAGE_STATUS,
    build_stage_result,
    format_mission_demo_summary,
    normalize_demo_image_input,
    run_one_click_mission_demo,
)


def _make_image(width=64, height=64):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = 80
    image[20:36, 20:36, :] = [180, 180, 180]
    return image


def _make_mask(width=64, height=64):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[0:8, :] = 7  # road_clear touching the boundary
    mask[20:36, 20:36] = 4  # major_damage
    mask[40:52, 8:24] = 1  # water
    mask[30:42, 42:55] = 8  # road_blocked
    return mask


def main():
    normalized = normalize_demo_image_input(_make_image())
    assert normalized["width"] == 64
    assert normalized["height"] == 64
    assert normalized["np_image"].shape == (64, 64, 3)

    success_stage = build_stage_result("demo", DEMO_STAGE_STATUS["success"], result={"ok": True})
    failed_stage = build_stage_result("demo", DEMO_STAGE_STATUS["failed"], error_code="X", message="failed")
    assert success_stage["success"] is True
    assert failed_stage["success"] is False
    assert failed_stage["error_code"] == "X"

    with tempfile.TemporaryDirectory() as tmp:
        result = run_one_click_mission_demo(
            _make_image(),
            detection_mode="yolo_rescue_targets",
            model_variant="missing_smoke_test_model",
            segmentation_source="none",
            thermal_mode="skip",
            generate_final_report=True,
            output_root=tmp,
        )
        assert isinstance(result, dict)
        assert "stages" in result
        assert result["stages"]["detection"]["status"] in {DEMO_STAGE_STATUS["failed"], DEMO_STAGE_STATUS["partial_success"]}
        assert result["stages"]["detection"]["error_code"] in {"YOLO_WEIGHTS_MISSING", "DETECTION_STAGE_FAILED", None}
        assert result["stages"]["thermal"]["status"] == DEMO_STAGE_STATUS["skipped"]
        assert isinstance(result["mission_summary_markdown"], str)
        assert "真实性边界" in result["mission_summary_markdown"]
        assert result["stages"]["report"]["status"] in {DEMO_STAGE_STATUS["success"], DEMO_STAGE_STATUS["partial_success"], DEMO_STAGE_STATUS["failed"]}
        if result.get("final_report_markdown_path"):
            assert Path(result["final_report_markdown_path"]).exists()

    with tempfile.TemporaryDirectory() as tmp:
        result = run_one_click_mission_demo(
            _make_image(),
            detection_mode="yolo_rescue_targets",
            model_variant="missing_smoke_test_model",
            segmentation_source="uploaded_mask",
            segmentation_mask=_make_mask(),
            thermal_mode="skip",
            generate_final_report=True,
            output_root=tmp,
        )
        segmentation_stage = result["stages"]["segmentation"]
        assert segmentation_stage["status"] == DEMO_STAGE_STATUS["success"]
        assert segmentation_stage["result"]["mask_available"] is True
        assert segmentation_stage["result"]["is_model_prediction"] is False
        assert (
            "不代表自动模型预测" in segmentation_stage["truthfulness_note"]
            or "not" in segmentation_stage["truthfulness_note"].lower()
        )
        assert any("segmentation_result.json" in artifact for artifact in segmentation_stage["artifacts"])

    fake_summary = format_mission_demo_summary(
        {
            "mission_id": "mission_test",
            "output_root": "/tmp/mission_test",
            "success": False,
            "partial_success": True,
            "stages": {
                "detection": build_stage_result("detection", DEMO_STAGE_STATUS["failed"], message="missing weights"),
                "segmentation": build_stage_result("segmentation", DEMO_STAGE_STATUS["skipped"], message="none"),
                "decision": build_stage_result("decision", DEMO_STAGE_STATUS["not_requested"], message="not requested"),
                "thermal": build_stage_result("thermal", DEMO_STAGE_STATUS["skipped"], message="skip"),
                "report": build_stage_result("report", DEMO_STAGE_STATUS["success"], message="ok"),
            },
            "artifacts": [],
        }
    )
    assert "mission_test" in fake_summary
    assert "真实性边界" in fake_summary

    print("AeroRescue-AI mission demo orchestrator smoke test passed.")


if __name__ == "__main__":
    main()
