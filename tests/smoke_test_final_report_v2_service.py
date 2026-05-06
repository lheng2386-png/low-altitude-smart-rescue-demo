import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from final_report_v2_service import (  # noqa: E402
    build_final_report_v2,
    build_report_context,
    markdown_to_simple_html,
    save_final_report_v2,
)
from report_export_service import export_final_report_v2  # noqa: E402


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_base_outputs(root):
    (root / "outputs").mkdir(parents=True, exist_ok=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_base_outputs(root)

        empty_report = build_final_report_v2(root_dir=root)
        assert empty_report["success"] is True
        assert isinstance(empty_report["report_markdown"], str)
        assert "灾情感知及影响评估 证据链驱动综合救援报告" in empty_report["report_markdown"]
        assert "证据链总览" in empty_report["report_markdown"]
        assert "未执行模块" in empty_report["report_markdown"]
        assert "真实性边界说明" in empty_report["report_markdown"]

        _write_json(
            root / "outputs" / "detection" / "detection_result.json",
            {
                "success": True,
                "detection_mode": "yolo_rescue_targets",
                "is_model_output": True,
                "target_count": 2,
                "truthfulness_note": "YOLO local model output.",
            },
        )
        detection_report = build_final_report_v2(root_dir=root)
        assert "主要模型输出证据" in detection_report["report_markdown"]
        assert "目标检测" in detection_report["report_markdown"] or "detection" in detection_report["report_markdown"].lower()
        assert "人工复核" in detection_report["report_markdown"]

        _write_json(
            root / "outputs" / "thermal" / "thermal_result.json",
            {
                "success": True,
                "thermal_mode": "simulated",
                "is_real_temperature_measurement": False,
                "truthfulness_note": "Simulated thermal, not real temperature.",
            },
        )
        thermal_report = build_final_report_v2(root_dir=root)
        assert "模拟 / 预览结果" in thermal_report["report_markdown"]
        assert "不是真实热红外测温" in thermal_report["report_markdown"] or "Simulated Thermal" in thermal_report["report_markdown"]

        _write_json(
            root / "outputs" / "decision_fusion" / "decision_fusion_summary.json",
            {
                "success": True,
                "decision_fusion_score": 72.0,
                "truthfulness_note": "Image-plane lightweight decision fusion.",
            },
        )
        decision_report = build_final_report_v2(root_dir=root)
        assert "辅助决策证据" in decision_report["report_markdown"]
        assert "image-plane" in decision_report["report_markdown"].lower() or "图像平面" in decision_report["report_markdown"]

        _write_json(
            root / "outputs" / "detection" / "detection_result.json",
            {
                "success": False,
                "error_code": "MODEL_UNAVAILABLE",
                "message": "Model weights missing.",
            },
        )
        failed_report = build_final_report_v2(root_dir=root)
        assert "执行失败" in failed_report["report_markdown"] or "依赖缺失" in failed_report["report_markdown"]
        assert "执行失败 / 依赖缺失模块" in failed_report["report_markdown"] or "未执行模块" in failed_report["report_markdown"]

        temp_save = save_final_report_v2(report=failed_report, root_dir=root, output_dir=root / "outputs" / "reports")
        assert Path(temp_save["markdown_path"]).exists()
        assert Path(temp_save["html_path"]).exists()
        assert Path(temp_save["json_path"]).exists()
        json.loads(Path(temp_save["json_path"]).read_text(encoding="utf-8"))

        html_text = markdown_to_simple_html("# 标题")
        assert "<html" in html_text.lower()
        assert "标题" in html_text

        export_status, md_path, html_path, json_path, preview = export_final_report_v2(
            root_dir=root,
            output_dir=root / "outputs" / "reports" / "v2_export",
        )
        assert "已生成" in export_status or "Final Report 2.0" in export_status
        assert md_path and Path(md_path).exists()
        assert html_path and Path(html_path).exists()
        assert json_path and Path(json_path).exists()
        assert isinstance(preview, str)

        context = build_report_context(root_dir=root)
        assert context["success"] is True
        assert context["summary"] is not None

    print("灾情感知及影响评估 final report v2 smoke test passed.")


if __name__ == "__main__":
    main()
