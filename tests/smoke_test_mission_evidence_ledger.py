import json
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from mission_evidence_ledger import (  # noqa: E402
    build_evidence_record,
    build_mission_evidence_ledger,
    format_evidence_record_markdown,
    format_mission_evidence_ledger_markdown,
    get_decision_supporting_evidence,
    get_human_review_items,
    save_mission_evidence_ledger,
)
from module_status_scanner import scan_single_module  # noqa: E402


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        empty_ledger = build_mission_evidence_ledger(root_dir=root)
        assert empty_ledger["success"] is True
        assert empty_ledger["summary"]["none_count"] >= 1
        assert all(
            not record["can_support_decision"]
            for record in empty_ledger["evidence_records"].values()
            if record["evidence_level"] == "none"
        )

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
        _write_json(
            root / "outputs" / "thermal" / "thermal_result.json",
            {
                "success": True,
                "thermal_mode": "simulated",
                "is_real_temperature_measurement": False,
                "truthfulness_note": "Simulated thermal.",
            },
        )
        _write_json(
            root / "outputs" / "orthomosaic" / "processing_log.json",
            {
                "success": True,
                "mode": "fast_preview",
                "truthfulness_note": "Fast Preview is not real ODM.",
            },
        )
        _write_json(
            root / "outputs" / "decision_fusion" / "decision_fusion_summary.json",
            {
                "success": True,
                "decision_fusion_score": 72.0,
                "truthfulness_note": "Image-plane lightweight decision fusion.",
            },
        )
        _write_json(
            root / "outputs" / "decision_fusion" / "path_planning_result.json",
            {
                "success": True,
                "path": [[0, 0], [10, 10], [20, 20], [30, 30]],
                "truthfulness_note": "Image-plane reference path.",
            },
        )
        _write_json(
            root / "outputs" / "thermal" / "radiometric_thermal_result.json",
            {
                "success": True,
                "thermal_mode": "radiometric",
                "is_real_temperature_measurement": True,
                "temperature_matrix_path": str(root / "outputs" / "thermal" / "temperature_matrix.npy"),
                "truthfulness_note": "Radiometric thermal matrix parsed.",
            },
        )
        (root / "outputs" / "thermal" / "temperature_matrix.npy").parent.mkdir(parents=True, exist_ok=True)
        (root / "outputs" / "thermal" / "temperature_matrix.npy").write_bytes(b"fake-npy")
        (root / "outputs" / "odm" / "task1" / "odm_orthophoto").mkdir(parents=True, exist_ok=True)
        (root / "outputs" / "odm" / "task1" / "odm_orthophoto" / "odm_orthophoto.tif").write_bytes(b"fake")

        scan_result = build_mission_evidence_ledger(root_dir=root)
        detection_record = scan_result["evidence_records"]["detection"]
        assert detection_record["evidence_level"] == "strong"
        assert detection_record["evidence_type"] == "model_output"
        assert detection_record["can_support_decision"] is True
        assert detection_record["human_review_required"] is True

        thermal_record = scan_result["evidence_records"]["thermal"]
        assert thermal_record["evidence_level"] == "strong"
        assert thermal_record["evidence_type"] == "real_measurement"
        assert thermal_record["can_support_decision"] is True

        ortho_record = scan_result["evidence_records"]["orthomosaic"]
        assert ortho_record["evidence_level"] == "weak"
        assert ortho_record["can_support_decision"] is False
        assert any("preview" in item.lower() or "odm" in item.lower() for item in ortho_record["limitations"])

        decision_record = scan_result["evidence_records"]["decision_fusion"]
        assert decision_record["evidence_level"] == "medium"
        assert decision_record["evidence_type"] == "image_plane_decision"
        assert decision_record["can_support_decision"] is True

        path_record = scan_result["evidence_records"]["path_planning"]
        assert path_record["evidence_level"] == "medium"
        assert path_record["evidence_type"] == "image_plane_decision"
        assert path_record["can_support_decision"] is True

        ref_record = scan_result["evidence_records"]["detection_backend_registry"]
        assert ref_record["evidence_level"] == "none"
        assert ref_record["evidence_type"] == "reference_only"
        assert ref_record["can_support_decision"] is False

        support = get_decision_supporting_evidence(scan_result)
        assert support
        assert all(item["can_support_decision"] for item in support)
        levels = [item["evidence_level"] for item in support]
        assert levels == sorted(levels, key=lambda level: {"strong": 0, "medium": 1, "weak": 2, "none": 3}[level])

        review_items = get_human_review_items(scan_result)
        assert review_items
        assert any(item["module_key"] == "detection" for item in review_items)

        markdown = format_mission_evidence_ledger_markdown(scan_result)
        assert isinstance(markdown, str)
        assert "任务证据链总账" in markdown
        assert "主要模型输出证据" in markdown
        assert "全局真实性说明" in markdown
        assert "代码文件存在" in markdown or "不根据代码文件存在" in markdown

        detection_markdown = format_evidence_record_markdown(detection_record)
        assert "证据等级" in detection_markdown
        assert "真实性说明" in detection_markdown

        save_result = save_mission_evidence_ledger(scan_result, output_dir=root / "outputs" / "reports")
        assert Path(save_result["json_path"]).exists()
        assert Path(save_result["markdown_path"]).exists()

        scan_single = scan_single_module("detection", root)
        assert scan_single["status"] == "real_model_output"

    print("灾情感知及影响评估 mission evidence ledger smoke test passed.")


if __name__ == "__main__":
    main()
