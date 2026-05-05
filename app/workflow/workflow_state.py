"""Runtime state helpers for the AeroRescue-AI rescue workflow."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime

try:
    from .stage_definitions import build_default_stage_state, list_stage_keys
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from workflow.stage_definitions import build_default_stage_state, list_stage_keys


WORKFLOW_STAGE_STATUSES = {
    "pending",
    "ready",
    "running",
    "completed",
    "skipped",
    "failed",
    "completed_external",
    "manual_required",
}


def _utc_timestamp():
    """Return a compact UTC timestamp for stage transitions."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def create_initial_workflow_state():
    """Create a fresh nine-stage rescue workflow state."""
    return {
        "current_stage_key": "global_mapping",
        "stages": build_default_stage_state(),
    }


def _mark_next_pending_ready(stages, stage_key):
    """Mark the next pending stage after stage_key as ready."""
    keys = list_stage_keys()
    stage_index = keys.index(stage_key)
    for next_key in keys[stage_index + 1 :]:
        if stages[next_key].get("status") == "pending":
            stages[next_key]["status"] = "ready"
            break


def _refresh_current_stage_key(workflow_state, fallback_stage_key):
    """Point current_stage_key to the first ready stage when one exists."""
    stages = workflow_state.get("stages", {})
    for key in list_stage_keys():
        if stages.get(key, {}).get("status") == "ready":
            workflow_state["current_stage_key"] = key
            return workflow_state
    workflow_state["current_stage_key"] = fallback_stage_key
    return workflow_state


def update_stage_status(
    workflow_state,
    stage_key,
    status,
    output_ref="",
    result_type="",
    evidence_id=None,
    error="",
):
    """Return workflow_state with one stage status updated."""
    if status not in WORKFLOW_STAGE_STATUSES:
        raise ValueError(f"Unsupported workflow stage status: {status}")

    updated = deepcopy(workflow_state or create_initial_workflow_state())
    stages = updated.setdefault("stages", build_default_stage_state())
    if stage_key not in stages:
        raise KeyError(f"Unknown rescue workflow stage: {stage_key}")

    stage = stages[stage_key]
    stage["status"] = status
    if output_ref:
        stage["output_ref"] = str(output_ref)
    if result_type:
        stage["result_type"] = str(result_type)
    if evidence_id:
        stage.setdefault("evidence_ids", [])
        if evidence_id not in stage["evidence_ids"]:
            stage["evidence_ids"].append(evidence_id)
    if error:
        stage["error"] = str(error)

    now = _utc_timestamp()
    if status == "running" and not stage.get("started_at"):
        stage["started_at"] = now
    if status in {"completed", "skipped", "failed", "completed_external", "manual_required"}:
        stage["completed_at"] = now

    if status in {"completed", "completed_external"}:
        _mark_next_pending_ready(stages, stage_key)

    return _refresh_current_stage_key(updated, stage_key)


def mark_stage_completed_external(
    workflow_state,
    stage_key,
    output_ref="",
    result_type="external_import",
    evidence_id=None,
    note="",
):
    """Mark a stage as completed by an external/user-provided artifact."""
    updated = update_stage_status(
        workflow_state,
        stage_key,
        "completed_external",
        output_ref=output_ref,
        result_type=result_type,
        evidence_id=evidence_id,
    )
    if note:
        updated["stages"][stage_key]["note"] = str(note)
    return updated


def skip_stage(
    workflow_state,
    stage_key,
    reason="",
    allow_next_ready=True,
):
    """Mark a stage as intentionally skipped and optionally ready the next stage."""
    updated = update_stage_status(workflow_state, stage_key, "skipped")
    if reason:
        updated["stages"][stage_key]["skip_reason"] = str(reason)
    if allow_next_ready:
        _mark_next_pending_ready(updated["stages"], stage_key)
    return _refresh_current_stage_key(updated, stage_key)


def start_workflow_from_stage(
    workflow_state,
    stage_key,
    skipped_reason="Started from a later stage with user-provided inputs.",
):
    """Allow a mission to begin at a later stage with missing context recorded."""
    updated = deepcopy(workflow_state or create_initial_workflow_state())
    stages = updated.setdefault("stages", build_default_stage_state())
    keys = list_stage_keys()
    if stage_key not in stages:
        raise KeyError(f"Unknown rescue workflow stage: {stage_key}")

    target_index = keys.index(stage_key)
    now = _utc_timestamp()
    for key in keys[:target_index]:
        stage = stages[key]
        stage["status"] = "skipped"
        stage["skip_reason"] = str(skipped_reason)
        stage["completed_at"] = stage.get("completed_at") or now
    for key in keys[target_index:]:
        if key == stage_key:
            stages[key]["status"] = "ready"
        elif stages[key].get("status") == "ready":
            stages[key]["status"] = "pending"
    updated["current_stage_key"] = stage_key
    return updated


def get_current_stage(workflow_state):
    """Return the current stage runtime dictionary."""
    state = workflow_state or create_initial_workflow_state()
    stage_key = state.get("current_stage_key") or "global_mapping"
    return state.get("stages", {}).get(stage_key, {})


def summarize_workflow_state(workflow_state):
    """Summarize workflow progress and human review burden."""
    state = workflow_state or create_initial_workflow_state()
    stages = state.get("stages", {})
    ready_stage_key = ""
    completed_count = 0
    failed_count = 0
    review_count = 0

    for key in list_stage_keys():
        stage = stages.get(key, {})
        status = stage.get("status")
        if status in {"completed", "completed_external"}:
            completed_count += 1
        if status == "failed":
            failed_count += 1
        if stage.get("human_review_required"):
            review_count += 1
        if not ready_stage_key and status == "ready":
            ready_stage_key = key

    return {
        "total_stage_count": len(list_stage_keys()),
        "completed_stage_count": completed_count,
        "ready_stage_key": ready_stage_key,
        "current_stage_key": state.get("current_stage_key", ""),
        "failed_stage_count": failed_count,
        "human_review_required_count": review_count,
    }


def summarize_workflow_context(workflow_state):
    """Summarize whether global/macro/tasking context is available."""
    state = workflow_state or create_initial_workflow_state()
    stages = state.get("stages", {})

    def _available(stage_key):
        return stages.get(stage_key, {}).get("status") in {"completed", "completed_external"}

    missing_context_notes = []
    if not _available("global_mapping"):
        missing_context_notes.append("No global map is connected for this mission.")
    if not _available("macro_analysis"):
        missing_context_notes.append("No macro risk analysis is connected for this mission.")
    if not _available("area_tasking") and stages.get("area_tasking", {}).get("status") != "manual_required":
        missing_context_notes.append("No area tasking result is connected for this mission.")

    direct_entry_stage = ""
    for key in list_stage_keys():
        stage = stages.get(key, {})
        if stage.get("status") == "ready" and any(
            stages.get(previous_key, {}).get("status") == "skipped"
            for previous_key in list_stage_keys()[: list_stage_keys().index(key)]
        ):
            direct_entry_stage = key
            break
    if direct_entry_stage in {"local_recon", "target_verification", "thermal_check", "decision_fusion", "rescue_recommendation", "evidence_report"}:
        missing_context_notes.append("Local detection results cannot be projected to a verified geospatial map.")

    return {
        "global_mapping_available": _available("global_mapping"),
        "macro_analysis_available": _available("macro_analysis"),
        "area_tasking_available": _available("area_tasking") or stages.get("area_tasking", {}).get("status") == "manual_required",
        "direct_entry_stage": direct_entry_stage,
        "missing_context_notes": missing_context_notes,
    }
