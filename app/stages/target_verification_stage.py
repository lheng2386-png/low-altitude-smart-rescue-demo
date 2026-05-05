"""S5 target_verification stage wrapper for low-altitude visual review."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from ..services.target_verification_service import (
        NO_CANDIDATE_TRUTHFULNESS_NOTE,
        TARGET_VERIFICATION_TRUTHFULNESS_NOTE,
        build_verification_records,
        summarize_verification_records,
    )
    from ..workflow.workflow_orchestrator import initialize_rescue_workflow, record_stage_output
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from services.target_verification_service import (
        NO_CANDIDATE_TRUTHFULNESS_NOTE,
        TARGET_VERIFICATION_TRUTHFULNESS_NOTE,
        build_verification_records,
        summarize_verification_records,
    )
    from workflow.workflow_orchestrator import initialize_rescue_workflow, record_stage_output


def _stage_output_dir(mission_dir):
    output_dir = Path(mission_dir) / "outputs" / "target_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_stage_result(output_dir, result):
    result_path = Path(output_dir) / "target_verification_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(result_path)


def _max_confidence(candidates):
    confidences = [candidate.get("confidence") for candidate in candidates or [] if candidate.get("confidence") is not None]
    return max(confidences) if confidences else None


def _normalize_close_view_count(close_view_images):
    if not close_view_images:
        return 0
    if isinstance(close_view_images, dict):
        return len([value for value in close_view_images.values() if value])
    if isinstance(close_view_images, (list, tuple)):
        return len([value for value in close_view_images if value])
    return 1


def run_target_verification_stage(
    mission,
    mission_dir,
    local_recon_result=None,
    candidates=None,
    close_view_images=None,
    review_actions=None,
):
    """Run S5 target verification evidence packaging and workflow recording."""
    mission = initialize_rescue_workflow(mission)
    mission_dir = Path(mission_dir)
    output_dir = _stage_output_dir(mission_dir)
    candidate_list = list(candidates if candidates is not None else (local_recon_result or {}).get("candidates") or [])

    if not candidate_list:
        truthfulness_note = (
            f"{TARGET_VERIFICATION_TRUTHFULNESS_NOTE} {NO_CANDIDATE_TRUTHFULNESS_NOTE}"
        )
        result = {
            "stage_key": "target_verification",
            "status": "degraded",
            "candidate_count": 0,
            "verification_records": [],
            "verification_summary": summarize_verification_records([]),
            "close_view_image_count": _normalize_close_view_count(close_view_images),
            "truthfulness_note": truthfulness_note,
            "human_review_required": True,
        }
        result_path = _save_stage_result(output_dir, result)
        record_stage_output(
            mission,
            mission_dir,
            stage_key="target_verification",
            output_ref=result_path,
            result_type="no_candidate_for_verification",
            source_type="no_candidate",
            confidence=None,
            score=0,
            truthfulness_note=truthfulness_note,
            limitation=truthfulness_note,
            human_review_required=True,
        )
        return mission, result

    try:
        records = build_verification_records(
            candidate_list,
            output_dir,
            close_view_images=close_view_images,
            review_actions=review_actions,
        )
        summary = summarize_verification_records(records)
        status = "completed" if records else "degraded"
        truthfulness_note = TARGET_VERIFICATION_TRUTHFULNESS_NOTE
    except Exception as exc:
        records = []
        summary = summarize_verification_records([])
        status = "failed"
        truthfulness_note = f"{TARGET_VERIFICATION_TRUTHFULNESS_NOTE} Target verification failed: {exc}"

    result = {
        "stage_key": "target_verification",
        "status": status,
        "candidate_count": len(candidate_list),
        "verification_records": records,
        "verification_summary": summary,
        "close_view_image_count": _normalize_close_view_count(close_view_images),
        "truthfulness_note": truthfulness_note,
        "human_review_required": True,
    }
    result_path = _save_stage_result(output_dir, result)
    record_stage_output(
        mission,
        mission_dir,
        stage_key="target_verification",
        output_ref=result_path,
        result_type="target_verification_evidence",
        source_type="visual_evidence_crop",
        confidence=_max_confidence(candidate_list),
        score=summary.get("verification_count"),
        truthfulness_note=truthfulness_note,
        limitation=truthfulness_note,
        human_review_required=True,
    )
    return mission, result
