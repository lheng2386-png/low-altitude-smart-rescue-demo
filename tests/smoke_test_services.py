import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from orthomosaic_service import process_orthomosaic  # noqa: E402
from reconstruction_service import process_reconstruction  # noqa: E402
from report_export_service import export_final_report  # noqa: E402
from scene_description_service import generate_scene_description  # noqa: E402
from thermal_service import analyze_thermal  # noqa: E402


def _synthetic_image(path):
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(20, 220, image.shape[1], dtype=np.uint8)
    image[:, :, 1] = 120
    image[20:44, 34:62, :] = (245, 245, 245)
    Image.fromarray(image).save(path)


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = Path(tmp_dir) / "synthetic_rescue_scene.jpg"
        _synthetic_image(image_path)

        ortho_image, ortho_status, ortho_log_text = process_orthomosaic([str(image_path)])
        assert ortho_image and Path(ortho_image).exists()
        assert "单幅航测图像预览" in ortho_status
        ortho_log = json.loads(ortho_log_text)
        assert ortho_log["image_count"] == 1
        assert ortho_log["stitch_success"] is False

        heatmap, overlay, thermal_status, thermal_json_text = analyze_thermal(str(image_path))
        assert heatmap and Path(heatmap).exists()
        assert overlay and Path(overlay).exists()
        assert "模拟热红外分析" in thermal_status
        thermal_result = json.loads(thermal_json_text)
        assert thermal_result["is_simulated_temperature"] is True
        assert "risk_level" in thermal_result

    reconstruction_outputs = process_reconstruction(None)
    assert reconstruction_outputs[5] == "未上传视频。"

    scene_md, scene_file = generate_scene_description(
        "Service Smoke Test",
        "合成图像服务测试。",
        "目标检测与综合决策报告测试文本。",
        use_ollama=False,
    )
    assert "AI 灾情描述" in scene_md
    assert Path(scene_file).exists()

    report_status, md_path, html_path, preview = export_final_report()
    assert "综合报告已生成" in report_status
    assert Path(md_path).exists()
    assert Path(html_path).exists()
    assert "AeroRescue-AI 综合救援报告" in preview

    print("AeroRescue-AI service smoke test passed.")


if __name__ == "__main__":
    main()
