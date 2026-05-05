"""S9 evidence_report stage wrapper for Final Report 2.0."""

from __future__ import annotations

from pathlib import Path

try:
    from ..services.evidence_report_service import (
        FIELD_REVIEW_NOTE,
        FINAL_REPORT_NOTICE,
        NO_INVENTION_NOTE,
        build_final_report_data,
        collect_available_stage_results,
        safe_load_json,
        save_final_report_outputs,
    )
    from ..workflow.workflow_orchestrator import initialize_rescue_workflow, record_stage_output
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from services.evidence_report_service import (
        FIELD_REVIEW_NOTE,
        FINAL_REPORT_NOTICE,
        NO_INVENTION_NOTE,
        build_final_report_data,
        collect_available_stage_results,
        safe_load_json,
        save_final_report_outputs,
    )
    from workflow.workflow_orchestrator import initialize_rescue_workflow, record_stage_output


REPORT_STAGE_TRUTHFULNESS_NOTE = f"{FINAL_REPORT_NOTICE} {NO_INVENTION_NOTE} {FIELD_REVIEW_NOTE}"


def _output_dir(mission_dir):
    output_dir = Path(mission_dir) / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _has_stage_results(stage_results):
    return any(bool(value) for value in (stage_results or {}).values())


def run_evidence_report_stage(
    mission,
    mission_dir,
    stage_results=None,
    ledger=None,
):
    """Run S9 Evidence Ledger and Final Report 2.0 generation."""
    mission = initialize_rescue_workflow(mission)
    mission_dir = Path(mission_dir)
    output_dir = _output_dir(mission_dir)

    if stage_results is None:
        stage_results = collect_available_stage_results(mission, mission_dir)
    if ledger is None:
        ledger = safe_load_json(mission.get("evidence_ledger_path") or mission_dir / "evidence" / "ledger.json", default={})

    report_data = build_final_report_data(
        mission,
        mission_dir,
        stage_results=stage_results,
        ledger=ledger,
    )
    save_result = save_final_report_outputs(report_data, output_dir)
    evidence_summary = report_data.get("evidence_summary", {}) or {}
    limitations = list(report_data.get("limitations", []) or [])
    truthfulness_note = REPORT_STAGE_TRUTHFULNESS_NOTE
    if not _has_stage_results(stage_results) and evidence_summary.get("total_evidence_count", 0) == 0:
        if "No stage evidence is available; the report is limited." not in limitations:
            limitations.append("No stage evidence is available; the report is limited.")
        truthfulness_note = f"{truthfulness_note} No stage evidence is available; the report is limited."

    status = "completed"
    if not _has_stage_results(stage_results) and evidence_summary.get("total_evidence_count", 0) == 0:
        status = "degraded"

    result = {
        "stage_key": "evidence_report",
        "status": status,
        "report_type": "AI-assisted decision-support report",
        "report_json_path": save_result.get("report_json_path", ""),
        "report_markdown_path": save_result.get("report_markdown_path", ""),
        "evidence_summary": evidence_summary,
        "workflow_summary": report_data.get("workflow_summary", {}) or {},
        "truthfulness_boundaries": report_data.get("truthfulness_boundaries", []) or [],
        "limitations": limitations,
        "truthfulness_note": truthfulness_note,
        "human_review_required": True,
    }

    record_stage_output(
        mission,
        mission_dir,
        stage_key="evidence_report",
        output_ref=result["report_markdown_path"],
        result_type="final_report_v2",
        source_type="report",
        confidence=None,
        score=evidence_summary.get("total_evidence_count", 0),
        truthfulness_note=truthfulness_note,
        limitation="Final Report summarizes available evidence and missing context; it requires human review.",
        human_review_required=True,
    )
    return mission, result
