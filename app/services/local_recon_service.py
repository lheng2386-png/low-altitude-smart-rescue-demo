"""Normalize local RGB reconnaissance detections into Rescue Candidates.

This service does not import or run YOLO. It only standardizes externally
provided, imported, manual, mock, or future model detections into a traceable
candidate schema with explicit truthfulness boundaries.
"""

from __future__ import annotations

from pathlib import Path


LOCAL_RECON_TRUTHFULNESS_NOTE = (
    "AI detections are candidates and not confirmed civilians. "
    "Local RGB detections are image-level evidence and require human review. "
    "Local RGB detections are not georeferenced unless map registration is provided."
)

LOCAL_RECON_NOT_GEOREFERENCED_NOTE = (
    "Local detection results cannot be treated as georeferenced rescue targets unless map registration is provided."
)

NO_DETECTION_TRUTHFULNESS_NOTE = (
    "No target candidate was generated. The system must not invent detections."
)

HUMAN_CLASS_NAMES = {"person", "people", "civilian", "human"}
VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "vehicle"}
CLASS_NAME_MAP = {
    "road": "road",
    "building": "building",
    "debris": "debris",
    "rubble": "debris",
    "fire": "fire_or_smoke",
    "smoke": "fire_or_smoke",
}


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


def normalize_local_rgb_images(local_rgb_images):
    """Return existing local RGB image paths from strings or Gradio file values."""
    files = local_rgb_images or []
    if not isinstance(files, (list, tuple)):
        files = [files]
    paths = []
    for item in files:
        raw_path = _as_path(item)
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.exists():
            paths.append(str(path))
    return paths


def normalize_candidate_class_name(class_name):
    """Normalize detector class labels to local-recon candidate classes."""
    original = str(class_name or "unknown").strip()
    normalized = original.lower()
    if normalized in HUMAN_CLASS_NAMES:
        return "human_candidate"
    if normalized in VEHICLE_CLASS_NAMES:
        return "vehicle"
    return CLASS_NAME_MAP.get(normalized, original)


def _numeric_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return []


def build_candidate_from_detection(
    detection,
    image_path="",
    area_id="A",
    candidate_index=1,
    source_type="imported_detection",
    global_context_available=False,
    map_registration_available=False,
):
    """Build one standardized Rescue Candidate from a detection dictionary."""
    detection = dict(detection or {})
    original_class_name = str(detection.get("class_name") or detection.get("label") or "unknown")
    confidence = detection.get("confidence", detection.get("score"))
    confidence = None if confidence is None else float(confidence)
    is_georeferenced = bool(global_context_available and map_registration_available)
    truthfulness_note = LOCAL_RECON_TRUTHFULNESS_NOTE
    if not is_georeferenced:
        truthfulness_note = f"{truthfulness_note} {LOCAL_RECON_NOT_GEOREFERENCED_NOTE}"

    return {
        "candidate_id": f"C{int(candidate_index):03d}",
        "target_id": f"T{int(candidate_index):03d}",
        "area_id": str(area_id or "A"),
        "source_image": str(image_path or detection.get("image_path") or detection.get("source_image") or ""),
        "class_name": normalize_candidate_class_name(original_class_name),
        "original_class_name": original_class_name,
        "confidence": confidence,
        "bbox": _numeric_list(detection.get("bbox")),
        "center": _numeric_list(detection.get("center")),
        "area": float(detection.get("area") or 0.0),
        "source_type": str(source_type or "imported_detection"),
        "review_status": "need_review",
        "human_review_required": True,
        "global_context_available": bool(global_context_available),
        "map_registration_available": bool(map_registration_available),
        "is_confirmed_civilian": False,
        "is_georeferenced": is_georeferenced,
        "truthfulness_note": truthfulness_note,
    }


def normalize_imported_detections(
    detections,
    image_path="",
    area_id="A",
    source_type="imported_detection",
    global_context_available=False,
    map_registration_available=False,
):
    """Normalize a list of imported/manual/mock detections into candidates."""
    return [
        build_candidate_from_detection(
            detection,
            image_path=image_path,
            area_id=area_id,
            candidate_index=index,
            source_type=source_type,
            global_context_available=global_context_available,
            map_registration_available=map_registration_available,
        )
        for index, detection in enumerate(detections or [], start=1)
    ]


def build_no_detection_result(
    area_id="A",
    local_rgb_images=None,
    reason="No detection result was provided.",
):
    """Return a transparent no-candidate result without inventing detections."""
    return {
        "candidate_count": 0,
        "candidates": [],
        "status": "degraded",
        "area_id": str(area_id or "A"),
        "local_rgb_image_count": len(normalize_local_rgb_images(local_rgb_images)),
        "reason": str(reason or "No detection result was provided."),
        "truthfulness_note": NO_DETECTION_TRUTHFULNESS_NOTE,
        "human_review_required": True,
    }


def summarize_candidates(candidates):
    """Summarize candidate counts, review burden, and class distribution."""
    class_counts = {}
    needs_review_count = 0
    has_georeferenced_candidate = False
    for candidate in candidates or []:
        class_name = str(candidate.get("class_name") or "unknown")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        if candidate.get("human_review_required"):
            needs_review_count += 1
        if candidate.get("is_georeferenced"):
            has_georeferenced_candidate = True

    return {
        "candidate_count": len(candidates or []),
        "human_candidate_count": class_counts.get("human_candidate", 0),
        "vehicle_count": class_counts.get("vehicle", 0),
        "needs_review_count": needs_review_count,
        "has_georeferenced_candidate": has_georeferenced_candidate,
        "class_counts": class_counts,
    }
