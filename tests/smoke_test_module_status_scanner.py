import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from module_status_scanner import (  # noqa: E402
    format_module_status_markdown,
    scan_all_modules,
    scan_single_module,
    safe_read_json,
)


def _write(path, data, as_text=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if as_text:
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        empty_scan = scan_all_modules(root)
        assert isinstance(empty_scan, dict)
        assert empty_scan["summary"]["not_run_count"] >= 1

        _write(
            root / "outputs" / "detection" / "detection_result.json",
            {
                "success": True,
                "detection_mode": "yolo_rescue_targets",
                "is_model_output": True,
                "target_count": 2,
                "truthfulness_note": "YOLO local model output.",
            },
        )
        detection_scan = scan_single_module("detection", root)
        assert detection_scan["executed"] is True
        assert detection_scan["success"] is True
        assert detection_scan["status"] in {"real_model_output", "executed_success"}
        assert "model_detection_output" in detection_scan["capability_tags"]

        _write(
            root / "outputs" / "thermal" / "thermal_result.json",
            {
                "success": True,
                "thermal_mode": "simulated",
                "is_real_temperature_measurement": False,
                "truthfulness_note": "Simulated thermal, not real temperature.",
            },
        )
        thermal_scan = scan_single_module("thermal", root)
        assert thermal_scan["status"] == "simulated_result"
        assert "not_real_temperature" in thermal_scan["capability_tags"]

        temp_matrix_path = root / "outputs" / "thermal" / "temperature_matrix.npy"
        np.save(temp_matrix_path, np.array([[30.0, 31.0], [32.0, 40.0]], dtype=np.float32))
        _write(
            root / "outputs" / "thermal" / "thermal_result.json",
            {
                "success": True,
                "thermal_mode": "radiometric",
                "is_real_temperature_measurement": True,
                "temperature_matrix_path": str(temp_matrix_path),
            },
        )
        thermal_real_scan = scan_single_module("thermal", root)
        assert thermal_real_scan["status"] == "real_measurement"
        assert "real_temperature_matrix" in thermal_real_scan["capability_tags"]

        odm_tif = root / "outputs" / "odm" / "task1" / "odm_orthophoto" / "odm_orthophoto.tif"
        odm_tif.parent.mkdir(parents=True, exist_ok=True)
        odm_tif.write_bytes(b"fake")
        odm_scan = scan_single_module("odm", root)
        assert odm_scan["status"] == "executed_success"
        assert "real_odm_orthophoto" in odm_scan["capability_tags"]

        _write(
            root / "outputs" / "orthomosaic" / "processing_log.json",
            {
                "success": True,
                "mode": "fast_preview",
                "truthfulness_note": "Fast Preview is not a real ODM orthomosaic.",
            },
        )
        ortho_scan = scan_single_module("orthomosaic", root)
        assert ortho_scan["status"] == "preview_only"
        assert "not_real_orthomosaic" in ortho_scan["capability_tags"]

        _write(
            root / "outputs" / "detection" / "detection_result.json",
            {
                "success": False,
                "error_code": "MODEL_UNAVAILABLE",
                "message": "Model weights missing.",
            },
        )
        failed_detection = scan_single_module("detection", root)
        assert failed_detection["success"] is False
        assert failed_detection["status"] == "dependency_missing"
        assert failed_detection["error_code"] == "MODEL_UNAVAILABLE"

        broken = root / "outputs" / "detection" / "detection_result.json"
        broken.write_text("{not json", encoding="utf-8")
        broken_scan = scan_single_module("detection", root)
        assert broken_scan["status"] != "executed_success"
        assert broken_scan["raw_result_summary"] is not None
        assert "json" in str(broken_scan["message"]).lower() or "json" in str(broken_scan["raw_result_summary"]).lower()

        full_scan = scan_all_modules(root)
        markdown = format_module_status_markdown(full_scan)
        assert isinstance(markdown, str)
        assert "模块执行状态扫描" in markdown
        assert "根据 outputs/ 运行产物和 JSON 元数据推断" in markdown

        assert safe_read_json(root / "outputs" / "missing.json") is None

    print("灾情感知及影响评估 module status scanner smoke test passed.")


if __name__ == "__main__":
    main()
