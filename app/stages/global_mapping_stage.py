"""S1 global_mapping stage wrapper for high-altitude mapping."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from ..workflow.workflow_orchestrator import initialize_rescue_workflow
    from ._stage_recording import record_stage_evidence
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from workflow.workflow_orchestrator import initialize_rescue_workflow
    from stages._stage_recording import record_stage_evidence


GLOBAL_MAPPING_TRUTHFULNESS = (
    "Fast Preview / OpenCV Stitch / ORB Homography is not a real ODM georeferenced orthomosaic. "
    "System outputs are decision-support results and not final rescue conclusions."
)
NO_MAPPING_INPUT_NOTE = (
    "No high-altitude overlapping RGB images were provided. "
    "System outputs are decision-support results and not final rescue conclusions."
)


def _as_path(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)


def _normalize_image_files(image_files):
    files = image_files or []
    if not isinstance(files, (list, tuple)):
        files = [files]
    paths = []
    for item in files:
        raw_path = _as_path(item)
        if raw_path:
            paths.append(raw_path)
    return paths


def _stage_output_dir(mission_dir):
    output_dir = Path(mission_dir) / "outputs" / "workflow"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_stage_result(mission_dir, result):
    result_path = _stage_output_dir(mission_dir) / "global_mapping_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(result_path)


def _try_fast_preview(image_files):
    try:
        from ..orthomosaic_service import process_orthomosaic
    except Exception:
        try:
            from orthomosaic_service import process_orthomosaic
        except Exception as exc:
            return "", f"Fast preview unavailable because orthomosaic_service could not be imported: {exc}", {}

    try:
        base_map_path, status_text, log_json = process_orthomosaic(image_files)
        try:
            processing_log = json.loads(log_json) if log_json else {}
        except Exception:
            processing_log = {"raw_log": log_json or ""}
        return base_map_path or "", status_text or "", processing_log
    except Exception as exc:
        return "", f"Fast preview failed without producing a map: {exc}", {"error": str(exc)}


def _try_real_odm(image_files, odm_task_name, max_images, fast_orthophoto):
    try:
        from ..odm_service import run_odm_task
    except Exception:
        try:
            from odm_service import run_odm_task
        except Exception as exc:
            return {}, f"ODM service unavailable: {exc}", ""

    try:
        preview_path, status_text, result_json, log_text = run_odm_task(
            image_files,
            task_name=odm_task_name,
            max_images=max_images,
            fast_orthophoto=fast_orthophoto,
        )
        try:
            odm_result = json.loads(result_json) if result_json else {}
        except Exception:
            odm_result = {"raw_result": result_json or ""}
        if preview_path and not odm_result.get("orthophoto_preview"):
            odm_result["orthophoto_preview"] = preview_path
        odm_result["status_text"] = status_text
        return odm_result, status_text or "", log_text or ""
    except Exception as exc:
        return {"error": str(exc)}, f"ODM execution failed without producing an orthophoto: {exc}", ""


def run_global_mapping_stage(
    mission,
    mission_dir,
    image_files=None,
    use_real_odm=False,
    odm_task_name="aerorescue_mapping",
    max_images=None,
    fast_orthophoto=True,
):
    """Run S1 high-altitude mapping with transparent ODM/preview boundaries."""
    mission = initialize_rescue_workflow(mission)
    image_paths = _normalize_image_files(image_files)
    mission_dir = Path(mission_dir)

    if not image_paths:
        result = {
            "stage_key": "global_mapping",
            "status": "skipped",
            "input_image_count": 0,
            "base_map_type": "none",
            "base_map_path": "",
            "processing_log": "No high-altitude overlapping RGB images were provided.",
            "odm_result": {},
            "truthfulness_note": NO_MAPPING_INPUT_NOTE,
            "human_review_required": True,
        }
        result_path = _save_stage_result(mission_dir, result)
        record_stage_evidence(
            mission,
            mission_dir,
            "global_mapping",
            "skipped",
            output_ref=result_path,
            result_type="none",
            source_type="rule_based",
            truthfulness_note=result["truthfulness_note"],
            limitation=result["truthfulness_note"],
            human_review_required=True,
        )
        return mission, result

    fast_preview_path, fast_preview_status, processing_log = _try_fast_preview(image_paths)
    base_map_path = fast_preview_path
    base_map_type = str(processing_log.get("mode") or "fast_preview") if fast_preview_path else "none"
    odm_result = {}
    odm_status = "Real ODM was not requested."
    odm_log_text = ""

    if use_real_odm:
        odm_result, odm_status, odm_log_text = _try_real_odm(image_paths, odm_task_name, max_images, fast_orthophoto)
        real_odm_output = odm_result.get("orthophoto_tif") or odm_result.get("orthophoto_preview")
        if real_odm_output:
            base_map_type = "real_odm"
            base_map_path = odm_result.get("orthophoto_preview") or odm_result.get("orthophoto_tif") or ""

    status = "completed" if base_map_path else "failed"
    processing_record = {
        "fast_preview_status": fast_preview_status,
        "fast_preview_log": processing_log,
        "odm_status": odm_status,
        "odm_log": odm_log_text,
        "real_odm_requested": bool(use_real_odm),
        "real_odm_succeeded": bool(odm_result.get("orthophoto_tif") or odm_result.get("orthophoto_preview")),
    }
    truthfulness_note = GLOBAL_MAPPING_TRUTHFULNESS
    if use_real_odm and base_map_type != "real_odm":
        truthfulness_note += " Real ODM was requested but no valid odm_orthophoto.tif or preview was produced."

    result = {
        "stage_key": "global_mapping",
        "status": status,
        "input_image_count": len(image_paths),
        "base_map_type": base_map_type,
        "base_map_path": base_map_path or "",
        "processing_log": json.dumps(processing_record, ensure_ascii=False, indent=2),
        "odm_result": odm_result,
        "real_odm_attempted": bool(use_real_odm),
        "real_odm_succeeded": bool(odm_result.get("orthophoto_tif") or odm_result.get("orthophoto_preview")),
        "truthfulness_note": truthfulness_note,
        "human_review_required": True,
    }
    result_path = _save_stage_result(mission_dir, result)
    record_stage_evidence(
        mission,
        mission_dir,
        "global_mapping",
        status,
        output_ref=result_path,
        result_type=base_map_type,
        source_type="real_odm" if base_map_type == "real_odm" else "orthomosaic_preview",
        truthfulness_note=truthfulness_note,
        limitation=truthfulness_note,
        human_review_required=True,
    )
    return mission, result
