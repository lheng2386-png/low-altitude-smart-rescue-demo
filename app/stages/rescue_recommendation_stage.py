"""S8 rescue_recommendation stage wrapper for route and task suggestions."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from ..services.rescue_recommendation_service import (
        NO_ROUTE_INVENTION_NOTE,
        RESCUE_RECOMMENDATION_TRUTHFULNESS_NOTE,
        build_rescue_recommendations,
        summarize_rescue_recommendations,
    )
    from ..workflow.workflow_orchestrator import initialize_rescue_workflow, record_stage_output
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from services.rescue_recommendation_service import (
        NO_ROUTE_INVENTION_NOTE,
        RESCUE_RECOMMENDATION_TRUTHFULNESS_NOTE,
        build_rescue_recommendations,
        summarize_rescue_recommendations,
    )
    from workflow.workflow_orchestrator import initialize_rescue_workflow, record_stage_output


def _output_dir(mission_dir):
    output_dir = Path(mission_dir) / "outputs" / "path"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_result(output_dir, result):
    result_path = Path(output_dir) / "rescue_recommendation_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(result_path)


def _top_confidence(recommendations):
    scores = [item.get("ec_terp_score") for item in recommendations or [] if item.get("ec_terp_score") is not None]
    if not scores:
        return None
    return max(float(score) for score in scores) / 100.0


def run_rescue_recommendation_stage(
    mission,
    mission_dir,
    decision_fusion_result=None,
    image=None,
    segmentation_mask=None,
    start_point=None,
    route_results=None,
    route_overlay_paths=None,
    path_comparisons=None,
    max_targets=3,
    run_path_planning=False,
):
    """Run S8 route and task recommendation generation."""
    mission = initialize_rescue_workflow(mission)
    mission_dir = Path(mission_dir)
    output_dir = _output_dir(mission_dir)
    decision_candidates = list((decision_fusion_result or {}).get("decision_candidates") or [])
    global_context_available = bool(mission.get("global_context_available", False))
    map_registration_available = bool(mission.get("map_registration_available", False))
    segmentation_available = segmentation_mask is not None
    start_point_available = start_point is not None

    if not decision_candidates:
        truthfulness_note = f"{RESCUE_RECOMMENDATION_TRUTHFULNESS_NOTE} {NO_ROUTE_INVENTION_NOTE}"
        result = {
            "stage_key": "rescue_recommendation",
            "status": "degraded",
            "workflow_mode": mission.get("workflow_mode", "standard"),
            "recommendations": [],
            "recommendation_summary": summarize_rescue_recommendations([]),
            "global_context_available": global_context_available,
            "map_registration_available": map_registration_available,
            "segmentation_available": segmentation_available,
            "start_point_available": start_point_available,
            "truthfulness_note": truthfulness_note,
            "human_review_required": True,
        }
        result_path = _save_result(output_dir, result)
        record_stage_output(
            mission,
            mission_dir,
            stage_key="rescue_recommendation",
            output_ref=result_path,
            result_type="no_decision_candidate_for_recommendation",
            source_type="no_candidate",
            confidence=None,
            score=0,
            truthfulness_note=truthfulness_note,
            limitation=truthfulness_note,
            human_review_required=True,
        )
        return mission, result

    # Path planning is deliberately optional; tests and default workflows use
    # provided route_results to avoid cv2/GIS/GPS dependencies.
    if run_path_planning and not route_results:
        route_results = {}

    recommendations = build_rescue_recommendations(
        decision_candidates,
        route_results=route_results,
        route_overlay_paths=route_overlay_paths,
        path_comparisons=path_comparisons,
        global_context_available=global_context_available,
        map_registration_available=map_registration_available,
        segmentation_available=segmentation_available,
        start_point_available=start_point_available,
        max_targets=max_targets,
    )
    summary = summarize_rescue_recommendations(recommendations)
    status = "completed" if recommendations else "degraded"
    truthfulness_note = RESCUE_RECOMMENDATION_TRUTHFULNESS_NOTE
    result = {
        "stage_key": "rescue_recommendation",
        "status": status,
        "workflow_mode": mission.get("workflow_mode", "standard"),
        "recommendations": recommendations,
        "recommendation_summary": summary,
        "global_context_available": global_context_available,
        "map_registration_available": map_registration_available,
        "segmentation_available": segmentation_available,
        "start_point_available": start_point_available,
        "truthfulness_note": truthfulness_note,
        "human_review_required": True,
    }
    result_path = _save_result(output_dir, result)
    record_stage_output(
        mission,
        mission_dir,
        stage_key="rescue_recommendation",
        output_ref=result_path,
        result_type="route_and_task_recommendation",
        source_type="image_plane_route_suggestion",
        confidence=_top_confidence(recommendations),
        score=summary.get("route_found_count") or summary.get("recommendation_count"),
        truthfulness_note=truthfulness_note,
        limitation=truthfulness_note,
        human_review_required=True,
    )
    return mission, result
