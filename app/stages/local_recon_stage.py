"""S4 local_recon stage wrapper and context helpers.

This module intentionally does not run YOLO. It can accept imported/mock/future
model detections and standardize them into Rescue Candidates while recording
truthfulness boundaries in the workflow and Evidence Ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from ..services.local_recon_service import (
        LOCAL_RECON_NOT_GEOREFERENCED_NOTE,
        LOCAL_RECON_TRUTHFULNESS_NOTE,
        NO_DETECTION_TRUTHFULNESS_NOTE,
        build_no_detection_result,
        normalize_imported_detections,
        normalize_local_rgb_images,
        summarize_candidates,
    )
    from ..s4_reference_fusion import (
        annotate_targets_with_reference_policy,
        build_s4_reference_fusion_context,
    )
    from ..workflow.workflow_orchestrator import initialize_rescue_workflow, record_stage_output
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from services.local_recon_service import (
        LOCAL_RECON_NOT_GEOREFERENCED_NOTE,
        LOCAL_RECON_TRUTHFULNESS_NOTE,
        NO_DETECTION_TRUTHFULNESS_NOTE,
        build_no_detection_result,
        normalize_imported_detections,
        normalize_local_rgb_images,
        summarize_candidates,
    )
    from s4_reference_fusion import (
        annotate_targets_with_reference_policy,
        build_s4_reference_fusion_context,
    )
    from workflow.workflow_orchestrator import initialize_rescue_workflow, record_stage_output


DIRECT_LOCAL_RECON_TRUTHFULNESS = (
    "Direct local recon can run object detection on RGB imagery, but results are not georeferenced unless map registration is provided. "
    "Image-level target detection is decision-support evidence and requires human review."
)

IMPORTED_DETECTION_BOUNDARY = (
    "Mock/imported detections are not real model inference results. "
    "Imported/mock detections are for workflow testing only and are not automatic model results."
)


def _stage_output_dir(mission_dir):
    output_dir = Path(mission_dir) / "outputs" / "workflow"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_stage_result(mission_dir, result):
    result_path = _stage_output_dir(mission_dir) / "local_recon_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(result_path)


def _resolve_area_id(area_task, area_id):
    if area_task and area_task.get("area_id"):
        return str(area_task.get("area_id"))
    return str(area_id or "A")


def _resolve_global_context(mission, global_context_available):
    if global_context_available is not None:
        return bool(global_context_available)
    if (mission or {}).get("workflow_mode") == "direct_local_recon":
        return False
    if "global_context_available" in (mission or {}):
        return bool(mission.get("global_context_available"))
    return bool(mission.get("workflow_state", {}).get("stages", {}).get("global_mapping", {}).get("status") in {"completed", "completed_external"})


def _source_type_for_backend(detection_backend):
    backend = str(detection_backend or "none").lower()
    if backend == "mock":
        return "mock_detection"
    if backend in {"yolo", "yolo_optional"}:
        return "yolo_detection"
    if backend == "none":
        return "no_detection"
    return "imported_detection"


def _truthfulness_note(candidate_count, detection_backend, global_context_available, map_registration_available):
    note = LOCAL_RECON_TRUTHFULNESS_NOTE
    backend = str(detection_backend or "none").lower()
    if backend in {"imported", "mock", "manual", "demo"}:
        note = f"{note} {IMPORTED_DETECTION_BOUNDARY}"
    if not candidate_count:
        note = f"{note} {NO_DETECTION_TRUTHFULNESS_NOTE}"
    if not (global_context_available and map_registration_available):
        note = f"{note} {LOCAL_RECON_NOT_GEOREFERENCED_NOTE}"
    return note


def _max_confidence(candidates):
    confidences = [candidate.get("confidence") for candidate in candidates or [] if candidate.get("confidence") is not None]
    return max(confidences) if confidences else None


def prepare_direct_local_recon_context(
    mission,
    area_id="A",
    local_rgb_images=None,
):
    """Prepare S4 context for direct local RGB reconnaissance without detection."""
    files = local_rgb_images or []
    if not isinstance(files, (list, tuple)):
        files = [files]
    return {
        "stage_key": "local_recon",
        "workflow_mode": "direct_local_recon",
        "area_id": str(area_id or "A"),
        "local_rgb_image_count": len([item for item in files if item]),
        "global_context_available": False,
        "macro_context_available": False,
        "truthfulness_note": DIRECT_LOCAL_RECON_TRUTHFULNESS,
        "human_review_required": True,
    }


def run_local_recon_stage(
    mission,
    mission_dir,
    local_rgb_images=None,
    area_task=None,
    area_id="A",
    detections=None,
    detection_backend="imported",
    global_context_available=None,
    map_registration_available=False,
):
    """Run S4 local reconnaissance candidate normalization and evidence recording."""
    mission = initialize_rescue_workflow(mission)
    mission_dir = Path(mission_dir)
    workflow_mode = mission.get("workflow_mode", "standard")
    resolved_area_id = _resolve_area_id(area_task, area_id)
    image_paths = normalize_local_rgb_images(local_rgb_images)
    resolved_global_context = _resolve_global_context(mission, global_context_available)
    map_registration_available = bool(map_registration_available)
    source_type = _source_type_for_backend(detection_backend)
    s4_reference_fusion = build_s4_reference_fusion_context()

    if detections:
        source_image = image_paths[0] if image_paths else ""
        candidates = normalize_imported_detections(
            detections,
            image_path=source_image,
            area_id=resolved_area_id,
            source_type=source_type,
            global_context_available=resolved_global_context,
            map_registration_available=map_registration_available,
        )
        candidates = annotate_targets_with_reference_policy(candidates, s4_reference_fusion)
        candidate_summary = summarize_candidates(candidates)
        candidate_summary["s4_reference_fusion"] = {
            "adapter_version": s4_reference_fusion.get("adapter_version"),
            "reference_count": s4_reference_fusion.get("reference_count", 0),
            "person_detection_reference_count": s4_reference_fusion.get("person_detection_reference_count", 0),
        }
        status = "completed"
    else:
        no_detection = build_no_detection_result(
            area_id=resolved_area_id,
            local_rgb_images=image_paths,
            reason="No detection result was provided.",
        )
        candidates = []
        candidate_summary = summarize_candidates(candidates)
        candidate_summary["reason"] = no_detection["reason"]
        status = "degraded"

    if not image_paths and detections:
        status = "degraded"

    truthfulness_note = _truthfulness_note(
        len(candidates),
        detection_backend,
        resolved_global_context,
        map_registration_available,
    )
    result = {
        "stage_key": "local_recon",
        "status": status,
        "workflow_mode": workflow_mode,
        "area_id": resolved_area_id,
        "area_task": area_task or {},
        "local_rgb_image_count": len(image_paths),
        "local_rgb_images": image_paths,
        "detection_backend": str(detection_backend or "none"),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "candidate_summary": candidate_summary,
        "global_context_available": resolved_global_context,
        "map_registration_available": map_registration_available,
        "truthfulness_note": truthfulness_note,
        "s4_reference_fusion": s4_reference_fusion,
        "human_review_required": True,
    }
    result_path = _save_stage_result(mission_dir, result)
    record_stage_output(
        mission,
        mission_dir,
        stage_key="local_recon",
        output_ref=result_path,
        result_type="local_rgb_candidate_detection",
        source_type=source_type,
        confidence=_max_confidence(candidates),
        score=len(candidates) if candidates else None,
        truthfulness_note=truthfulness_note,
        limitation=truthfulness_note,
        human_review_required=True,
    )
    return mission, result
