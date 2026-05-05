"""Shared evidence recording helpers for workflow stage wrappers."""

from __future__ import annotations

try:
    from ..mission_orchestrator import record_module_result
    from ..mission_schema import save_mission
    from ..workflow.workflow_orchestrator import initialize_rescue_workflow, record_stage_output
    from ..workflow.workflow_state import skip_stage, update_stage_status
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from mission_orchestrator import record_module_result
    from mission_schema import save_mission
    from workflow.workflow_orchestrator import initialize_rescue_workflow, record_stage_output
    from workflow.workflow_state import skip_stage, update_stage_status


def record_stage_evidence(
    mission,
    mission_dir,
    stage_key,
    status,
    output_ref="",
    result_type="",
    source_type="rule_based",
    truthfulness_note="",
    limitation="",
    human_review_required=True,
):
    """Record evidence and update workflow state for one stage wrapper."""
    mission = initialize_rescue_workflow(mission)
    if status == "completed":
        return record_stage_output(
            mission,
            mission_dir,
            stage_key,
            output_ref=output_ref,
            result_type=result_type,
            source_type=source_type,
            truthfulness_note=truthfulness_note,
            limitation=limitation,
            human_review_required=human_review_required,
        )

    workflow_status = "skipped" if status == "skipped" else "failed"
    entry = record_module_result(
        mission,
        mission_dir,
        module=f"workflow:{stage_key}",
        input_ref=stage_key,
        output_ref=output_ref,
        result_type=result_type,
        source_type=source_type,
        truthfulness_note=truthfulness_note,
        limitation=limitation or truthfulness_note,
        human_review_required=human_review_required,
    )
    if workflow_status == "skipped":
        mission["workflow_state"] = update_stage_status(
            mission.get("workflow_state"),
            stage_key,
            "skipped",
            output_ref=output_ref,
            result_type=result_type,
            evidence_id=entry["evidence_id"],
        )
        mission["workflow_state"] = skip_stage(
            mission.get("workflow_state"),
            stage_key,
            reason=truthfulness_note,
            allow_next_ready=True,
        )
    else:
        mission["workflow_state"] = update_stage_status(
            mission.get("workflow_state"),
            stage_key,
            workflow_status,
            output_ref=output_ref,
            result_type=result_type,
            evidence_id=entry["evidence_id"],
            error=truthfulness_note,
        )
    save_mission(mission, mission_dir)
    return entry
