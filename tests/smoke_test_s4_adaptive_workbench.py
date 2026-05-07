import sys
from pathlib import Path

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import app  # noqa: E402


def _update_value(update):
    return update.get("value") if isinstance(update, dict) else getattr(update, "value", None)


def main():
    image = Image.new("RGB", (160, 120), (210, 220, 230))
    result = app.s4_adaptive_detection_workbench(
        "本地上传",
        None,
        image,
        "rescuedet_deformable_detr",
        0.3,
        "missing_smoke_test_model",
    )
    assert len(result) == 27
    assert "route" in result[2]
    assert "selected_backend_combo" in result[2]
    assert "recommended_backend_combo" in result[2]
    assert "requested_backend_combo" in result[2]
    assert "selected_main_backend" not in result[2]
    assert "selected_auxiliary_backends" not in result[2]
    assert "skipped_backends" in result[2]
    assert "unavailable_backends" in result[2]
    assert "expected_outputs" in result[2]

    backend_rows = result[4]
    assert any(row[0] == "air_sar_detector" and row[1] == "adapter_unavailable" for row in backend_rows)
    assert any(row[0] == "qazi_disaster_detector" and row[1] == "adapter_unavailable" for row in backend_rows)
    assert "qazi0" in _update_value(result[11])

    for path in [result[12], result[13], result[14], result[18], result[19], result[20], result[21], result[22], result[23]]:
        assert Path(path).exists()

    visible_text = "\n".join(str(item) for item in result[:18])
    forbidden = ["confirmed civilian", "confirmed survivor", "已确认幸存者"]
    assert not any(term in visible_text for term in forbidden)
    app_source = (APP_DIR / "app.py").read_text(encoding="utf-8")
    assert "高级详情 / Developer Details" in app_source
    developer_section = app_source[
        app_source.index("高级详情 / Developer Details") : app_source.index("with gr.Group(visible=False) as video_detection_group")
    ]
    assert "execution_plan.json" in developer_section

    print("S4 adaptive multi-model workbench smoke test passed.")


if __name__ == "__main__":
    main()
