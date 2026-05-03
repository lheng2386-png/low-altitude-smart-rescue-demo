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


def main():
    auto_ok = build_segmentation_source_metadata(
        "auto_model",
        checkpoint_path="outputs/segmentation_training/checkpoints/best.pth",
        model_available=True,
        prediction_success=True,
    )
    assert auto_ok["is_model_prediction"] is True

    auto_missing = build_segmentation_source_metadata(
        "auto_model",
        checkpoint_path="missing.pth",
        model_available=False,
        prediction_success=False,
        fallback_reason="missing checkpoint",
    )
    assert auto_missing["is_model_prediction"] is False

    uploaded = build_segmentation_source_metadata(
        "uploaded_mask",
        mask_path="demo_mask.png",
        prediction_success=True,
    )
    assert uploaded["is_model_prediction"] is False

    demo = build_segmentation_source_metadata("demo_fallback", fallback_reason="demo only")
    assert demo["is_model_prediction"] is False

    none = build_segmentation_source_metadata("none")
    assert none["is_model_prediction"] is False

    for metadata in [auto_ok, auto_missing, uploaded, demo, none]:
        assert metadata["truthfulness_note"]
        assert format_segmentation_source_status(metadata)
        assert segmentation_visualization_note(metadata)

    text = format_segmentation_source_status(uploaded)
    assert text.strip()

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

    print("AeroRescue-AI segmentation source metadata smoke test passed.")


if __name__ == "__main__":
    main()
