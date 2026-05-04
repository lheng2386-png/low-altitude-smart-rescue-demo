import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from final_report_v2_service import build_final_report_v2  # noqa: E402
from mission_demo_orchestrator import run_decision_stage, run_segmentation_stage  # noqa: E402
from mission_evidence_ledger import build_mission_evidence_ledger  # noqa: E402
from module_status_scanner import scan_single_module  # noqa: E402


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[:, :] = [90, 110, 130]
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[70:90, :] = 7
    mask[20:42, 20:44] = 4
    mask[50:70, 40:60] = 8

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        segmentation_stage = run_segmentation_stage(
            image,
            segmentation_source="uploaded_mask",
            segmentation_mask=mask,
            output_dir=root / "outputs" / "segmentation_inference",
        )
        assert segmentation_stage["status"] == "success"

        detection_stage = {
            "stage_name": "detection",
            "status": "success",
            "success": True,
            "result": {
                "success": True,
                "is_model_output": True,
                "detection_mode": "dual_backend_compare",
                "targets": [
                    {
                        "id": "T001",
                        "target_id": "T001",
                        "class_name": "civilian",
                        "confidence": 0.88,
                        "bbox": [20, 20, 44, 52],
                        "center": [32, 36],
                        "area": 768,
                        "human_review_required": True,
                    },
                    {
                        "id": "TR002",
                        "target_id": "TR002",
                        "class_name": "human_candidate",
                        "confidence": 0.66,
                        "bbox": [60, 25, 82, 58],
                        "center": [71, 41],
                        "area": 726,
                        "source_backend": "transformer_rescuedet",
                        "human_review_required": True,
                    },
                ],
            },
            "artifacts": [],
            "truthfulness_note": "Synthetic runtime input for EC-TERP integration smoke test.",
        }

        decision_stage = run_decision_stage(
            image,
            detection_stage,
            segmentation_stage,
            start_point=(10, 86),
            output_dir=root / "outputs" / "decision_fusion",
        )
        assert decision_stage["status"] in {"success", "partial_success"}
        ec_path = root / "outputs" / "ec_terp" / "ec_terp_rankings.json"
        ui_path = root / "outputs" / "ui" / "ec_terp_summary.json"
        assert ec_path.exists()
        assert ui_path.exists()

        ec_payload = json.loads(ec_path.read_text(encoding="utf-8"))
        rankings = ec_payload["rankings"]
        assert rankings
        for item in rankings:
            assert {
                "target_id",
                "target_type",
                "rank",
                "ec_terp_score",
                "score_components",
                "evidence_level",
                "source_modules",
                "is_confirmed_rescue_target",
                "human_review_required",
                "recommendation_type",
                "explanation",
                "limitations",
                "truthfulness_note",
            }.issubset(item.keys())
            assert item["is_confirmed_rescue_target"] is False
            assert item["human_review_required"] is True
            if item["target_type"] == "human_candidate":
                assert item["is_confirmed_rescue_target"] is False
                assert any("not a confirmed civilian" in limitation for limitation in item["limitations"])

        ui_payload = json.loads(ui_path.read_text(encoding="utf-8"))
        assert ui_payload["module"] == "ec_terp"
        assert ui_payload["status"] == "executed_success"
        assert "visuals" in ui_payload
        assert "explainability" in ui_payload
        assert ui_payload["explainability"]["formula"] == "EC-TERP = αT + βE + γR + δC + λQ - μU"
        assert "Not GPS navigation" in ui_payload["truthfulness_badges"]
        assert "Not automatic rescue decision" in ui_payload["truthfulness_badges"]

        scan = scan_single_module("ec_terp_ranking", root_dir=root)
        assert scan["status"] == "executed_success"
        assert "assistive_priority_ranking" in scan["capability_tags"]

        ledger = build_mission_evidence_ledger(root_dir=root)
        ec_record = ledger["evidence_records"]["ec_terp_ranking"]
        assert ec_record["can_support_decision"] is True
        assert ec_record["human_review_required"] is True
        assert ec_record["evidence_level"] == "medium"
        assert ec_record["evidence_type"] == "assistive_decision_support"
        assert any("assistive priority" in item for item in ec_record["limitations"])

        report = build_final_report_v2(root_dir=root)
        markdown = report["report_markdown"]
        assert "EC-TERP 救援辅助优先级排序" in markdown
        assert "EC-TERP is an assistive priority ranking algorithm" in markdown
        assert "Image-plane path planning is not GPS navigation" in markdown
        assert "Synthetic demo cases are not real rescue data" in markdown

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_json(
            root / "outputs" / "ec_terp_evaluation" / "ec_terp_evaluation_summary.json",
            {
                "success": True,
                "case_type_note": "Built-in cases are synthetic demo cases, not real rescue benchmark data.",
                "truthfulness_note": "Synthetic evaluation only.",
            },
        )
        scan = scan_single_module("ec_terp_ranking", root_dir=root)
        assert scan["status"] == "simulated_result"
        ledger = build_mission_evidence_ledger(root_dir=root)
        assert ledger["evidence_records"]["ec_terp_ranking"]["evidence_level"] == "weak"

    print("AeroRescue-AI EC-TERP integration smoke test passed.")


if __name__ == "__main__":
    main()
