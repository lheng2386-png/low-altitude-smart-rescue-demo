import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.evidence_ledger import load_ledger  # noqa: E402
from app.mission_schema import create_mission, save_mission  # noqa: E402
from app.services.rescue_recommendation_service import (  # noqa: E402
    build_rescue_recommendations,
    build_route_context_notes,
    build_task_recommendation_for_candidate,
    select_recommendation_targets,
    summarize_rescue_recommendations,
    summarize_route_result,
)
from app.stages.rescue_recommendation_stage import run_rescue_recommendation_stage  # noqa: E402
from app.workflow.workflow_orchestrator import initialize_rescue_workflow  # noqa: E402


def _decision_candidates():
    return [
        {
            "candidate_id": "C001",
            "target_id": "T001",
            "area_id": "A",
            "rank": 1,
            "priority_level": "High",
            "ec_terp_score": 80,
            "recommended_action": "prioritize_human_review_and_field_verification",
        },
        {
            "candidate_id": "C002",
            "target_id": "T002",
            "area_id": "A",
            "rank": 2,
            "priority_level": "Medium",
            "ec_terp_score": 50,
        },
        {
            "candidate_id": "C003",
            "target_id": "T003",
            "area_id": "B",
            "rank": 3,
            "priority_level": "Critical",
            "ec_terp_score": 90,
            "review_status": "rejected_false_positive",
            "should_exclude_from_rescue_ranking": True,
        },
    ]


def main():
    selected = select_recommendation_targets(_decision_candidates(), max_targets=3)
    assert [item["candidate_id"] for item in selected] == ["C001", "C002"]

    notes = build_route_context_notes(
        global_context_available=False,
        map_registration_available=False,
        segmentation_available=False,
        start_point_available=False,
    )
    assert any("No verified global map" in item for item in notes)
    assert any("image-plane pixels" in item for item in notes)
    assert any("No segmentation-derived risk map" in item for item in notes)
    assert any("No verified rescue team start point" in item for item in notes)

    route_result = {
        "found": True,
        "path_type": "image_plane_path",
        "is_gps_navigation": False,
        "start": [1, 2],
        "goal": [10, 20],
        "path_length": 30,
        "total_cost": 88.5,
        "message": "A* path planning succeeded.",
    }
    route_summary = summarize_route_result(route_result)
    assert route_summary["route_found"] is True
    recommendation = build_task_recommendation_for_candidate(
        _decision_candidates()[0],
        route_result=route_result,
        missing_context_notes=notes,
    )
    assert recommendation["route_type"] == "image_plane_path"
    assert recommendation["is_gps_navigation"] is False
    assert recommendation["is_autonomous_rescue_route"] is False
    assert recommendation["human_review_required"] is True
    assert "not GPS navigation" in recommendation["truthfulness_note"]
    assert "decision-support reference" in recommendation["truthfulness_note"]

    route_results = {"C001": route_result}
    recommendations = build_rescue_recommendations(
        _decision_candidates(),
        route_results=route_results,
        global_context_available=False,
        map_registration_available=False,
        segmentation_available=False,
        start_point_available=False,
    )
    rec_summary = summarize_rescue_recommendations(recommendations)
    assert rec_summary["recommendation_count"] == 2
    assert recommendations[0]["candidate_id"] == "C001"
    assert recommendations[0]["route_found"] is True
    assert recommendations[1]["candidate_id"] == "C002"
    assert recommendations[1]["route_found"] is False
    assert all(item["candidate_id"] != "C003" for item in recommendations)
    assert rec_summary["gps_navigation_count"] == 0

    with tempfile.TemporaryDirectory() as tmp:
        mission_dir = Path(tmp) / "mission"
        mission = create_mission(mission_name="Phase 8 Rescue Recommendation")
        mission = initialize_rescue_workflow(mission)
        mission["evidence_ledger_path"] = str(mission_dir / "evidence" / "ledger.json")
        mission["global_context_available"] = False
        mission["map_registration_available"] = False
        save_mission(mission, mission_dir)

        decision_fusion_result = {"decision_candidates": _decision_candidates()}
        mission, result = run_rescue_recommendation_stage(
            mission,
            mission_dir,
            decision_fusion_result=decision_fusion_result,
            route_results=route_results,
        )
        assert result["stage_key"] == "rescue_recommendation"
        assert result["recommendation_summary"]["recommendation_count"] == 2
        assert result["recommendation_summary"]["gps_navigation_count"] == 0
        assert mission["workflow_state"]["stages"]["rescue_recommendation"]["status"] == "completed"
        assert (mission_dir / "outputs" / "path" / "rescue_recommendation_result.json").exists()
        ledger = load_ledger(mission["evidence_ledger_path"])
        assert len(ledger["entries"]) >= 1
        assert "not GPS navigation" in ledger["entries"][-1]["truthfulness_note"]
        assert "Rescue task suggestion requires commander and field-team review." in ledger["entries"][-1]["truthfulness_note"]

        mission, empty_result = run_rescue_recommendation_stage(
            mission,
            mission_dir,
            decision_fusion_result={"decision_candidates": []},
        )
        assert empty_result["status"] == "degraded"
        assert "must not invent reachable routes" in empty_result["truthfulness_note"]

    print("灾情感知及影响评估 phase 8 rescue recommendation smoke test passed.")


if __name__ == "__main__":
    main()
