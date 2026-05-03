"""Smoke test for segmentation source metadata tracking."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from segmentation_source_metadata import (  # noqa: E402
    build_segmentation_source_metadata,
    format_segmentation_source_status,
    segmentation_visualization_note,
)
from report_generator import generate_report  # noqa: E402


def assert_note(metadata):
    assert metadata["truthfulness_note"]
    assert format_segmentation_source_status(metadata)
    assert segmentation_visualization_note(metadata)


def main():
    auto = build_segmentation_source_metadata(
        "auto_model",
        checkpoint_path="outputs/segmentation_training/checkpoints/best.pth",
        model_available=True,
        prediction_success=True,
    )
    assert auto["source_type"] == "auto_model"
    assert auto["is_model_prediction"] is True
    assert auto["model_available"] is True
    assert auto["prediction_success"] is True
    assert "checkpoint" in auto["checkpoint_path"]
    assert_note(auto)

    uploaded = build_segmentation_source_metadata(
        "uploaded_mask",
        mask_path="demo_mask.png",
        prediction_success=True,
    )
    assert uploaded["source_type"] == "uploaded_mask"
    assert uploaded["is_model_prediction"] is False
    assert uploaded["mask_path"] == "demo_mask.png"
    assert "不代表自动模型预测" in uploaded["truthfulness_note"]
    assert_note(uploaded)

    demo = build_segmentation_source_metadata("demo_fallback", fallback_reason="流程演示")
    assert demo["source_type"] == "demo_fallback"
    assert demo["is_model_prediction"] is False
    assert "不代表真实模型输出" in demo["truthfulness_note"]
    assert_note(demo)

    none = build_segmentation_source_metadata("none")
    assert none["source_type"] == "none"
    assert none["is_model_prediction"] is False
    assert none["truthfulness_note"]
    assert_note(none)

    missing_auto = build_segmentation_source_metadata(
        "auto_model",
        checkpoint_path="missing.pth",
        model_available=False,
        prediction_success=False,
        fallback_reason="checkpoint missing",
    )
    assert missing_auto["is_model_prediction"] is False
    assert "不会生成假分割图" in missing_auto["truthfulness_note"]
    assert_note(missing_auto)

    report = generate_report(
        [],
        [],
        segmentation_summary={},
        segmentation_source_metadata=uploaded,
        language="zh",
    )
    assert "分割来源" in report
    assert "是否为模型自动预测" in report
    assert "真实性说明" in report

    print("AeroRescue-AI segmentation source tracking smoke test passed.")


if __name__ == "__main__":
    main()
