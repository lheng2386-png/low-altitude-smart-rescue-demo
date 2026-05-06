import json
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.evidence_ledger import load_ledger, save_ledger  # noqa: E402
from app.mission_schema import create_mission, save_mission  # noqa: E402
from app.services.evidence_report_service import (  # noqa: E402
    build_evidence_summary,
    build_final_report_data,
    build_workflow_report_summary,
    render_final_report_markdown,
    safe_load_json,
    save_final_report_outputs,
)
from app.stages.evidence_report_stage import run_evidence_report_stage  # noqa: E402
from app.workflow.workflow_orchestrator import initialize_rescue_workflow  # noqa: E402


def _ledger():
    return {
        "entries": [
            {
                "evidence_id": "E0001",
                "module": "workflow:local_recon",
                "source_type": "mock_detection",
                "truthfulness_note": "AI candidates are not confirmed civilians.",
                "limitation": "Local detection is image-level evidence.",
                "human_review_required": True,
            },
            {
                "evidence_id": "E0002",
                "module": "workflow:thermal_check",
                "source_type": "simulated_thermal",
                "truthfulness_note": "Simulated Thermal is not real temperature measurement.",
                "limitation": "No real temperature matrix.",
                "human_review_required": True,
            },
        ]
    }


def _stage_results():
    return {
        "global_mapping": {},
        "macro_analysis": {},
        "area_tasking": {},
        "local_recon": {
            "stage_key": "local_recon",
            "status": "completed",
            "candidate_count": 2,
            "candidates": [
                {"candidate_id": "C001", "class_name": "human_candidate", "human_review_required": True},
                {"candidate_id": "C002", "class_name": "vehicle", "human_review_required": True},
            ],
        },
        "target_verification": {
            "stage_key": "target_verification",
            "status": "completed",
            "verification_summary": {"verification_count": 2, "confirmed_candidate_count": 1},
            "verification_records": [
                {
                    "verification_id": "V001",
                    "candidate_id": "C001",
                    "class_name": "human_candidate",
                    "review_status": "confirmed_candidate",
                    "human_review_required": True,
                    "truthfulness_note": "Target verification provides visual evidence for human review and is not a final rescue conclusion.",
                }
            ],
        },
        "thermal_check": {
            "stage_key": "thermal_check",
            "status": "completed",
            "thermal_summary": {"thermal_check_count": 1, "weak_support_count": 1},
            "thermal_records": [
                {
                    "thermal_check_id": "TH001",
                    "candidate_id": "C001",
                    "thermal_support_level": "weak",
                    "human_review_required": True,
                    "truthfulness_note": "Thermal support is auxiliary evidence and not confirmation of life.",
                }
            ],
        },
        "decision_fusion": {
            "stage_key": "decision_fusion",
            "status": "completed",
            "decision_summary": {"decision_candidate_count": 2},
            "top_priority_candidate": {"candidate_id": "C001", "ec_terp_score": 82, "priority_level": "High"},
            "decision_candidates": [
                {
                    "candidate_id": "C001",
                    "target_id": "T001",
                    "rank": 1,
                    "ec_terp_score": 82,
                    "priority_level": "High",
                    "human_review_required": True,
                }
            ],
        },
        "rescue_recommendation": {
            "stage_key": "rescue_recommendation",
            "status": "completed",
            "recommendation_summary": {"recommendation_count": 1, "gps_navigation_count": 0},
            "recommendations": [
                {
                    "recommendation_id": "R001",
                    "candidate_id": "C001",
                    "priority_level": "High",
                    "route_type": "image_plane_path",
                    "is_gps_navigation": False,
                    "human_review_required": True,
                    "truthfulness_note": "Image-plane path is not GPS navigation.",
                }
            ],
        },
    }


def main():
    assert safe_load_json("/path/that/does/not/exist.json", default={"ok": False}) == {"ok": False}

    evidence_summary = build_evidence_summary(_ledger())
    assert evidence_summary["total_evidence_count"] == 2
    assert evidence_summary["human_review_required_count"] == 2
    assert "workflow:local_recon" in evidence_summary["module_counts"]
    assert "simulated_thermal" in evidence_summary["source_type_counts"]
    assert evidence_summary["limitations"]

    mission = create_mission(mission_name="Phase 9 Evidence Report")
    mission = initialize_rescue_workflow(mission)
    mission["workflow_mode"] = "direct_local_recon"
    mission["global_context_available"] = False
    mission["map_registration_available"] = False
    workflow_report_summary = build_workflow_report_summary(mission, mission["workflow_state"])
    assert workflow_report_summary["workflow_mode"] == "direct_local_recon"
    assert any("No verified global map" in item for item in workflow_report_summary["missing_context_notes"])
    assert any("not georeferenced" in item for item in workflow_report_summary["missing_context_notes"])

    with tempfile.TemporaryDirectory() as tmp:
        mission_dir = Path(tmp) / "mission"
        mission["evidence_ledger_path"] = str(mission_dir / "evidence" / "ledger.json")
        save_mission(mission, mission_dir)
        report_data = build_final_report_data(
            mission,
            mission_dir,
            stage_results=_stage_results(),
            ledger=_ledger(),
        )
        assert report_data["report_title"] == "灾情感知及影响评估 Final Report 2.0"
        assert report_data["report_type"] == "AI-assisted decision-support report"
        assert report_data["priority_recommendations"]
        assert report_data["truthfulness_boundaries"]
        assert "not a final rescue conclusion" in report_data["final_notice"]

        markdown = render_final_report_markdown(report_data)
        assert "灾情感知及影响评估 Final Report 2.0" in markdown
        assert "任务基本信息" in markdown
        assert "EC-TERP" in markdown
        assert "真实性边界" in markdown
        assert "AI 辅助决策报告" in markdown
        assert "不构成最终救援结论" in markdown

        save_result = save_final_report_outputs(report_data, mission_dir / "outputs" / "reports")
        assert Path(save_result["report_json_path"]).exists()
        assert Path(save_result["report_markdown_path"]).exists()

    with tempfile.TemporaryDirectory() as tmp:
        mission_dir = Path(tmp) / "mission"
        mission = create_mission(mission_name="Phase 9 Stage")
        mission = initialize_rescue_workflow(mission)
        mission["evidence_ledger_path"] = str(mission_dir / "evidence" / "ledger.json")
        save_mission(mission, mission_dir)
        save_ledger({"mission_id": mission["mission_id"], "entries": _ledger()["entries"]}, mission["evidence_ledger_path"])

        mission, result = run_evidence_report_stage(
            mission,
            mission_dir,
            stage_results=_stage_results(),
        )
        assert result["stage_key"] == "evidence_report"
        assert result["report_type"] == "AI-assisted decision-support report"
        assert Path(result["report_json_path"]).exists()
        assert Path(result["report_markdown_path"]).exists()
        assert mission["workflow_state"]["stages"]["evidence_report"]["status"] == "completed"
        ledger = load_ledger(mission["evidence_ledger_path"])
        assert len(ledger["entries"]) >= 3
        assert "not a final rescue conclusion" in ledger["entries"][-1]["truthfulness_note"]
        assert "Human review and field commander judgment are required before field action." in ledger["entries"][-1][
            "truthfulness_note"
        ]

        mission, empty_result = run_evidence_report_stage(
            mission,
            mission_dir,
            stage_results={},
            ledger={"entries": []},
        )
        assert empty_result["status"] in {"completed", "degraded"}
        assert Path(empty_result["report_markdown_path"]).exists()
        empty_markdown = Path(empty_result["report_markdown_path"]).read_text(encoding="utf-8")
        assert "No stage evidence is available" in empty_markdown or "report is limited" in empty_markdown

    print("灾情感知及影响评估 phase 9 evidence report smoke test passed.")


if __name__ == "__main__":
    main()
