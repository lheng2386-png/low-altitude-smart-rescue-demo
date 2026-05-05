"""S6 thermal_check stage wrapper for auxiliary thermal support evidence."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from ..services.thermal_support_service import (
        NO_THERMAL_INVENTION_NOTE,
        SIMULATED_THERMAL_NOTE,
        THERMAL_TRUTHFULNESS_NOTE,
        build_thermal_check_records,
        normalize_thermal_inputs,
        summarize_thermal_check_records,
    )
    from ..workflow.workflow_orchestrator import initialize_rescue_workflow, record_stage_output
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from services.thermal_support_service import (
        NO_THERMAL_INVENTION_NOTE,
        SIMULATED_THERMAL_NOTE,
        THERMAL_TRUTHFULNESS_NOTE,
        build_thermal_check_records,
        normalize_thermal_inputs,
        summarize_thermal_check_records,
    )
    from workflow.workflow_orchestrator import initialize_rescue_workflow, record_stage_output


def _stage_output_dir(mission_dir):
    output_dir = Path(mission_dir) / "outputs" / "thermal_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_stage_result(output_dir, result):
    result_path = Path(output_dir) / "thermal_check_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(result_path)


def _normalize_results(thermal_results):
    if thermal_results is None:
        return []
    if isinstance(thermal_results, (dict, str)):
        return [thermal_results]
    if isinstance(thermal_results, (list, tuple)):
        return list(thermal_results)
    return []


def _run_optional_thermal_analysis(thermal_paths, thermal_mode):
    results = []
    try:
        from ..thermal_service import analyze_thermal
    except Exception:
        try:
            from thermal_service import analyze_thermal
        except Exception as exc:
            return [{"thermal_mode": "unknown", "error": f"thermal_service unavailable: {exc}"}]

    for path in thermal_paths:
        try:
            _, _, _, result_json = analyze_thermal(path, mode=thermal_mode)
            results.append(result_json)
        except Exception as exc:
            results.append({"thermal_mode": "unknown", "error": str(exc)})
    return results


def _source_type_for_records(records, thermal_results, run_thermal_analysis):
    if not records:
        return "thermal_unavailable"
    if any(item.get("thermal_mode") == "radiometric" and item.get("is_real_temperature_measurement") for item in records):
        return "radiometric_thermal"
    if any(item.get("thermal_mode") == "simulated" for item in records):
        return "simulated_thermal"
    if thermal_results and not run_thermal_analysis:
        return "imported_thermal_result"
    return "thermal_unavailable"


def _truthfulness_note(source_type):
    note = THERMAL_TRUTHFULNESS_NOTE
    if source_type == "simulated_thermal":
        note = f"{note} {SIMULATED_THERMAL_NOTE}"
    if source_type == "thermal_unavailable":
        note = f"{note} {NO_THERMAL_INVENTION_NOTE}"
    return note


def _confidence_from_records(records):
    scores = [float(item.get("thermal_support_score") or 0.0) for item in records or []]
    return max(scores) / 20.0 if scores else None


def run_thermal_check_stage(
    mission,
    mission_dir,
    target_verification_result=None,
    verification_records=None,
    thermal_images=None,
    thermal_results=None,
    thermal_mode="Simulated Thermal / 模拟热红外",
    rgb_thermal_alignment="unregistered_or_approximate",
    run_thermal_analysis=False,
):
    """Run S6 thermal support evidence generation."""
    mission = initialize_rescue_workflow(mission)
    mission_dir = Path(mission_dir)
    output_dir = _stage_output_dir(mission_dir)
    records_in = list(
        verification_records
        if verification_records is not None
        else (target_verification_result or {}).get("verification_records") or []
    )
    thermal_paths = normalize_thermal_inputs(thermal_images)

    if not records_in:
        truthfulness_note = f"{THERMAL_TRUTHFULNESS_NOTE} {NO_THERMAL_INVENTION_NOTE}"
        result = {
            "stage_key": "thermal_check",
            "status": "degraded",
            "verification_count": 0,
            "thermal_target_count": 0,
            "thermal_image_count": len(thermal_paths),
            "thermal_records": [],
            "thermal_summary": summarize_thermal_check_records([]),
            "thermal_mode": thermal_mode,
            "rgb_thermal_alignment": rgb_thermal_alignment,
            "truthfulness_note": truthfulness_note,
            "human_review_required": True,
        }
        result_path = _save_stage_result(output_dir, result)
        record_stage_output(
            mission,
            mission_dir,
            stage_key="thermal_check",
            output_ref=result_path,
            result_type="no_verification_record_for_thermal_check",
            source_type="thermal_unavailable",
            confidence=None,
            score=0,
            truthfulness_note=truthfulness_note,
            limitation=truthfulness_note,
            human_review_required=True,
        )
        return mission, result

    thermal_targets = [item for item in records_in if item.get("thermal_check_required")]
    normalized_results = _normalize_results(thermal_results)
    if run_thermal_analysis and thermal_paths and not normalized_results:
        normalized_results = _run_optional_thermal_analysis(thermal_paths, thermal_mode)

    source_type_hint = "simulated_thermal" if str(thermal_mode).lower().startswith("simulated") else "radiometric_thermal"
    if not normalized_results and not run_thermal_analysis:
        source_type_hint = "thermal_unavailable"

    thermal_records = build_thermal_check_records(
        records_in,
        thermal_images=thermal_paths,
        thermal_results=normalized_results,
        rgb_thermal_alignment=rgb_thermal_alignment,
        source_type=source_type_hint,
    )
    summary = summarize_thermal_check_records(thermal_records)
    source_type = _source_type_for_records(thermal_records, normalized_results, run_thermal_analysis)
    truthfulness_note = _truthfulness_note(source_type)
    status = "completed" if thermal_records else "degraded"
    result_type = "thermal_support_evidence" if thermal_records else "no_thermal_target"

    result = {
        "stage_key": "thermal_check",
        "status": status,
        "verification_count": len(records_in),
        "thermal_target_count": len(thermal_targets),
        "thermal_image_count": len(thermal_paths),
        "thermal_records": thermal_records,
        "thermal_summary": summary,
        "thermal_mode": thermal_mode,
        "rgb_thermal_alignment": rgb_thermal_alignment,
        "truthfulness_note": truthfulness_note,
        "human_review_required": True,
    }
    result_path = _save_stage_result(output_dir, result)
    record_stage_output(
        mission,
        mission_dir,
        stage_key="thermal_check",
        output_ref=result_path,
        result_type=result_type,
        source_type=source_type,
        confidence=_confidence_from_records(thermal_records),
        score=summary.get("strong_support_count") or summary.get("thermal_check_count"),
        truthfulness_note=truthfulness_note,
        limitation=truthfulness_note,
        human_review_required=True,
    )
    return mission, result
