import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.evidence_ledger import load_ledger  # noqa: E402
from app.mission_schema import create_mission, save_mission  # noqa: E402
from app.stages.decision_fusion_stage import run_decision_fusion_stage  # noqa: E402
from app.workflow.workflow_orchestrator import initialize_rescue_workflow  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        mission_dir = Path(tmp) / "mission"
        mission = create_mission(mission_name="Phase 7 Decision Fusion")
        mission = initialize_rescue_workflow(mission)
        mission["evidence_ledger_path"] = str(mission_dir / "evidence" / "ledger.json")
        save_mission(mission, mission_dir)

        local_recon_result = {
            "candidates": [
                {
                    "candidate_id": "C001",
                    "target_id": "T001",
                    "area_id": "A",
                    "class_name": "human_candidate",
                    "confidence": 0.88,
                },
                {
                    "candidate_id": "C002",
                    "target_id": "T002",
                    "area_id": "A",
                    "class_name": "vehicle",
                    "confidence": 0.76,
                },
            ]
        }
        target_verification_result = {
            "verification_records": [
                {"candidate_id": "C001", "review_status": "confirmed_candidate"},
                {"candidate_id": "C002", "review_status": "need_review"},
            ]
        }
        thermal_check_result = {
            "thermal_records": [
                {"candidate_id": "C001", "thermal_support_level": "weak"},
            ]
        }
        macro_analysis_result = {"macro_zones": [{"risk_level": "High"}]}

        mission, result = run_decision_fusion_stage(
            mission,
            mission_dir,
            local_recon_result=local_recon_result,
            target_verification_result=target_verification_result,
            thermal_check_result=thermal_check_result,
            macro_analysis_result=macro_analysis_result,
        )
        assert result["stage_key"] == "decision_fusion"
        assert result["decision_summary"]["decision_candidate_count"] == 2
        assert result["decision_candidates"][0]["candidate_id"] == "C001"
        assert result["decision_candidates"][0]["is_confirmed_civilian"] is False
        assert "does not replace rescue command judgment" in result["truthfulness_note"]
        assert mission["workflow_state"]["stages"]["decision_fusion"]["status"] == "completed"
        ledger = load_ledger(mission["evidence_ledger_path"])
        assert ledger["entries"]
        assert "not confirmed civilians" in ledger["entries"][-1]["truthfulness_note"]

        mission, empty_result = run_decision_fusion_stage(mission, mission_dir, local_recon_result={"candidates": []})
        assert empty_result["status"] == "degraded"
        assert "must not invent priority rankings" in empty_result["truthfulness_note"]

    print("AeroRescue-AI phase 7 decision fusion smoke test passed.")


if __name__ == "__main__":
    main()
