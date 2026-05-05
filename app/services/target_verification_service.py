"""Target verification evidence packaging for AeroRescue-AI.

S5 turns S4 Rescue Candidates into visual evidence records for human review.
It does not confirm civilians, rescued status, or final field findings.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


TARGET_VERIFICATION_TRUTHFULNESS_NOTE = (
    "Target verification provides visual evidence for human review and is not a final rescue conclusion. "
    "A human-reviewed candidate is still not equivalent to a confirmed rescued civilian. "
    "Cropped evidence is derived from image pixels and may miss context outside the crop."
)

NO_CANDIDATE_TRUTHFULNESS_NOTE = (
    "The system must not invent candidates or review decisions."
)

ALLOWED_REVIEW_STATUSES = {
    "need_review",
    "confirmed_candidate",
    "rejected_false_positive",
    "need_recheck",
    "urgent_review",
}


def clamp_bbox(bbox, image_width, image_height):
    """Clamp bbox to image bounds and return integer [x1, y1, x2, y2], or None."""
    if not bbox or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except Exception:
        return None
    if image_width <= 0 or image_height <= 0:
        return None

    max_x = int(image_width) - 1
    max_y = int(image_height) - 1
    left = max(0, min(max_x, int(round(min(x1, x2)))))
    right = max(0, min(max_x, int(round(max(x1, x2)))))
    top = max(0, min(max_y, int(round(min(y1, y2)))))
    bottom = max(0, min(max_y, int(round(max(y1, y2)))))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def expand_bbox(bbox, image_width, image_height, scale=1.8, min_padding=20):
    """Expand bbox for context evidence while staying inside image bounds."""
    clamped = clamp_bbox(bbox, image_width, image_height)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    target_w = max(width * float(scale or 1.8), width + 2 * float(min_padding or 0))
    target_h = max(height * float(scale or 1.8), height + 2 * float(min_padding or 0))
    expanded = [
        cx - target_w / 2.0,
        cy - target_h / 2.0,
        cx + target_w / 2.0,
        cy + target_h / 2.0,
    ]
    return clamp_bbox(expanded, image_width, image_height)


def _pil_crop_box(bbox):
    """Convert inclusive integer bbox to PIL's right/bottom-exclusive box."""
    x1, y1, x2, y2 = bbox
    return (x1, y1, x2 + 1, y2 + 1)


def crop_candidate_evidence(
    image_path,
    candidate,
    output_dir,
    verification_id="V001",
    context_scale=1.8,
):
    """Crop target and context evidence images for one candidate."""
    result = {
        "target_crop_path": "",
        "context_crop_path": "",
        "target_bbox": [],
        "context_bbox": [],
        "crop_success": False,
        "error": "",
    }
    if not image_path:
        result["error"] = "No image_path was provided for candidate evidence crop."
        return result
    path = Path(image_path)
    if not path.exists():
        result["error"] = f"Image path does not exist: {path}"
        return result

    try:
        image = Image.open(path).convert("RGB")
    except Exception as exc:
        result["error"] = f"Unable to read image for candidate evidence crop: {exc}"
        return result

    width, height = image.size
    target_bbox = clamp_bbox((candidate or {}).get("bbox"), width, height)
    if target_bbox is None:
        result["error"] = "Candidate bbox is invalid for evidence crop."
        return result
    context_bbox = expand_bbox(target_bbox, width, height, scale=context_scale)
    if context_bbox is None:
        result["error"] = "Candidate context bbox is invalid for evidence crop."
        return result

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_crop_path = output_dir / f"{verification_id}_target_crop.jpg"
    context_crop_path = output_dir / f"{verification_id}_context_crop.jpg"
    try:
        image.crop(_pil_crop_box(target_bbox)).save(target_crop_path, quality=92)
        image.crop(_pil_crop_box(context_bbox)).save(context_crop_path, quality=92)
    except Exception as exc:
        result["error"] = f"Unable to save candidate evidence crop: {exc}"
        return result

    result.update(
        {
            "target_crop_path": str(target_crop_path),
            "context_crop_path": str(context_crop_path),
            "target_bbox": target_bbox,
            "context_bbox": context_bbox,
            "crop_success": True,
            "error": "",
        }
    )
    return result


def _verification_flags(class_name, review_status):
    need_recheck = review_status == "need_recheck"
    thermal_check_required = False
    if review_status != "rejected_false_positive":
        thermal_check_required = class_name == "human_candidate" and review_status in {
            "need_review",
            "confirmed_candidate",
            "urgent_review",
        }
    return need_recheck, thermal_check_required


def build_verification_record(
    candidate,
    verification_index=1,
    crop_result=None,
    close_view_image="",
    evidence_source="source_image_crop",
    review_status="need_review",
    review_note="",
    reviewer="",
):
    """Build one target verification evidence record."""
    if review_status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(f"Unsupported review_status: {review_status}")
    candidate = dict(candidate or {})
    crop_result = crop_result or {}
    class_name = candidate.get("class_name", "")
    need_recheck, thermal_check_required = _verification_flags(class_name, review_status)
    return {
        "verification_id": f"V{int(verification_index):03d}",
        "candidate_id": candidate.get("candidate_id", ""),
        "target_id": candidate.get("target_id", ""),
        "area_id": candidate.get("area_id", ""),
        "class_name": class_name,
        "confidence": candidate.get("confidence"),
        "source_image": candidate.get("source_image", ""),
        "close_view_image": str(close_view_image or ""),
        "evidence_source": str(evidence_source or "source_image_crop"),
        "target_crop_path": crop_result.get("target_crop_path", ""),
        "context_crop_path": crop_result.get("context_crop_path", ""),
        "target_bbox": crop_result.get("target_bbox", []),
        "context_bbox": crop_result.get("context_bbox", []),
        "crop_success": bool(crop_result.get("crop_success", False)),
        "crop_error": crop_result.get("error", ""),
        "review_status": review_status,
        "review_note": str(review_note or ""),
        "reviewer": str(reviewer or ""),
        "need_recheck": need_recheck,
        "thermal_check_required": thermal_check_required,
        "human_review_required": True,
        "is_confirmed_civilian": False,
        "truthfulness_note": TARGET_VERIFICATION_TRUTHFULNESS_NOTE,
    }


def apply_review_action(record, review_status, review_note="", reviewer=""):
    """Apply a human review status without confirming civilian identity."""
    if review_status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(f"Unsupported review_status: {review_status}")
    updated = dict(record or {})
    updated["review_status"] = review_status
    updated["review_note"] = str(review_note or "")
    updated["reviewer"] = str(reviewer or "")
    need_recheck, thermal_check_required = _verification_flags(updated.get("class_name", ""), review_status)
    updated["need_recheck"] = need_recheck
    updated["thermal_check_required"] = thermal_check_required
    updated["human_review_required"] = True
    updated["is_confirmed_civilian"] = False
    updated["truthfulness_note"] = TARGET_VERIFICATION_TRUTHFULNESS_NOTE
    return updated


def _normalize_close_view_images(close_view_images):
    if not close_view_images:
        return {}
    if isinstance(close_view_images, dict):
        return {str(key): str(value) for key, value in close_view_images.items() if value}
    if not isinstance(close_view_images, (list, tuple)):
        close_view_images = [close_view_images]
    return {str(index): str(value) for index, value in enumerate(close_view_images, start=1) if value}


def _close_view_for_candidate(candidate, index, close_view_map):
    candidate_id = str(candidate.get("candidate_id", ""))
    if candidate_id and candidate_id in close_view_map:
        return close_view_map[candidate_id]
    return close_view_map.get(str(index), "")


def build_verification_records(
    candidates,
    output_dir,
    close_view_images=None,
    review_actions=None,
):
    """Build verification records and evidence crops for all candidates."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    close_view_map = _normalize_close_view_images(close_view_images)
    review_actions = review_actions or {}
    records = []

    for index, candidate in enumerate(candidates or [], start=1):
        verification_id = f"V{index:03d}"
        close_view_image = _close_view_for_candidate(candidate, index, close_view_map)
        image_path = close_view_image or candidate.get("source_image", "")
        evidence_source = "close_view_crop" if close_view_image else "fallback_source_image_crop"
        crop_result = crop_candidate_evidence(
            image_path,
            candidate,
            output_dir,
            verification_id=verification_id,
        )
        record = build_verification_record(
            candidate,
            verification_index=index,
            crop_result=crop_result,
            close_view_image=close_view_image,
            evidence_source=evidence_source,
        )
        action = review_actions.get(candidate.get("candidate_id", ""), {})
        if action:
            record = apply_review_action(
                record,
                action.get("review_status", record["review_status"]),
                review_note=action.get("review_note", ""),
                reviewer=action.get("reviewer", ""),
            )
        record["human_review_required"] = True
        record["is_confirmed_civilian"] = False
        records.append(record)
    return records


def summarize_verification_records(records):
    """Summarize target verification records for S6/S7 handoff."""
    records = list(records or [])
    return {
        "verification_count": len(records),
        "need_review_count": sum(1 for item in records if item.get("review_status") == "need_review"),
        "confirmed_candidate_count": sum(1 for item in records if item.get("review_status") == "confirmed_candidate"),
        "rejected_count": sum(1 for item in records if item.get("review_status") == "rejected_false_positive"),
        "need_recheck_count": sum(1 for item in records if item.get("need_recheck")),
        "urgent_review_count": sum(1 for item in records if item.get("review_status") == "urgent_review"),
        "thermal_check_required_count": sum(1 for item in records if item.get("thermal_check_required")),
        "human_candidate_count": sum(1 for item in records if item.get("class_name") == "human_candidate"),
    }
