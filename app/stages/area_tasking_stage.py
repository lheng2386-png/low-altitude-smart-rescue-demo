"""S3 area_tasking stage wrapper for priority inspection areas."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from ..services.area_tasking_service import build_area_tasks_from_macro_zones
    from ..workflow.workflow_orchestrator import initialize_rescue_workflow
    from ._stage_recording import record_stage_evidence
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from services.area_tasking_service import build_area_tasks_from_macro_zones
    from workflow.workflow_orchestrator import initialize_rescue_workflow
    from stages._stage_recording import record_stage_evidence


AREA_TASKING_TRUTHFULNESS = (
    "Area tasking is an auxiliary recommendation and requires commander review. "
    "System outputs are decision-support results and not final rescue conclusions."
)


def _stage_output_dir(mission_dir):
    output_dir = Path(mission_dir) / "outputs" / "workflow"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_stage_result(mission_dir, result):
    result_path = _stage_output_dir(mission_dir) / "area_tasking_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(result_path)


def run_area_tasking_stage(
    mission,
    mission_dir,
    macro_analysis_result=None,
    max_areas=3,
):
    """Run S3 area tasking from S2 macro zones."""
    mission = initialize_rescue_workflow(mission)
    macro_zones = list((macro_analysis_result or {}).get("macro_zones") or [])
    area_tasks = build_area_tasks_from_macro_zones(macro_zones, max_areas=max_areas)
    status = "completed" if macro_zones else "degraded"
    result = {
        "stage_key": "area_tasking",
        "status": status,
        "area_tasks": area_tasks,
        "area_count": len(area_tasks),
        "truthfulness_note": AREA_TASKING_TRUTHFULNESS,
        "human_review_required": True,
    }
    result_path = _save_stage_result(mission_dir, result)
    record_stage_evidence(
        mission,
        mission_dir,
        "area_tasking",
        "completed",
        output_ref=result_path,
        result_type="area_tasking_recommendation",
        source_type="rule_based",
        truthfulness_note=AREA_TASKING_TRUTHFULNESS,
        limitation=AREA_TASKING_TRUTHFULNESS,
        human_review_required=True,
    )
    return mission, result
