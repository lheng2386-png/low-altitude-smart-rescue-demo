"""Evidence ledger utilities for AeroRescue-AI mission outputs.

The ledger records what each module consumed, produced, and can truthfully
claim. It is designed for human review and report traceability rather than
for model performance scoring.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


TRUTHFULNESS_NOTES = {
    "object_detection": "Detected targets are AI candidates and must not be treated as confirmed civilians.",
    "uploaded_mask": "Uploaded masks are user-provided and are not automatic model predictions.",
    "demo_fallback": "Demo fallback results are for workflow demonstration only.",
    "simulated_thermal": "Simulated thermal analysis is generated from RGB/gray intensity and is not real temperature measurement.",
    "radiometric_thermal": "Radiometric thermal results are valid only when a real temperature matrix is successfully parsed.",
    "orthomosaic_preview": "OpenCV/feature-based stitching preview is not a georeferenced ODM orthomosaic.",
    "reconstruction_preview": "ORB/keyframe/PLY preview is not a full SfM/MVS reconstruction.",
    "image_plane_path": "Image-plane path planning is not GPS navigation.",
    "report": "Generated report is an AI-assisted decision-support summary and requires human review.",
}

DEFAULT_LIMITATIONS = {
    "object_detection": "Candidate detections require field verification.",
    "uploaded_mask": "Mask quality depends on the user-provided source.",
    "demo_fallback": "Demo fallback cannot be used as operational evidence.",
    "simulated_thermal": "RGB/gray intensity cannot produce real temperature_matrix values.",
    "radiometric_thermal": "Requires a successfully parsed real thermal matrix and calibrated source data.",
    "orthomosaic_preview": "Preview stitching is not surveying-grade georeferencing.",
    "reconstruction_preview": "Preview point clouds and trajectories are not calibrated reconstruction outputs.",
    "image_plane_path": "Path coordinates are image-plane pixels, not GPS navigation waypoints.",
    "report": "Reports summarize evidence for decision support and do not replace rescue command judgment.",
}


def _utc_timestamp():
    """Return a compact UTC timestamp for evidence records."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def create_ledger(mission_id):
    """Create an empty evidence ledger for a mission."""
    return {
        "mission_id": str(mission_id),
        "created_at": _utc_timestamp(),
        "updated_at": _utc_timestamp(),
        "entries": [],
    }


def add_evidence_entry(
    ledger,
    module,
    input_ref="",
    output_ref="",
    result_type="",
    confidence=None,
    score=None,
    source_type="rule_based",
    truthfulness_note=None,
    limitation=None,
    human_review_required=True,
):
    """Append one evidence entry to a ledger and return the entry."""
    ledger.setdefault("entries", [])
    evidence_id = f"E{len(ledger['entries']) + 1:04d}"
    note_key = source_type if source_type in TRUTHFULNESS_NOTES else module
    truthfulness_note = truthfulness_note or TRUTHFULNESS_NOTES.get(note_key, "")
    limitation = limitation or DEFAULT_LIMITATIONS.get(note_key, DEFAULT_LIMITATIONS.get(module, "Requires human review."))
    entry = {
        "evidence_id": evidence_id,
        "mission_id": ledger.get("mission_id", ""),
        "module": str(module),
        "input_ref": str(input_ref or ""),
        "output_ref": str(output_ref or ""),
        "result_type": str(result_type or ""),
        "confidence": confidence,
        "score": score,
        "source_type": str(source_type or ""),
        "truthfulness_note": truthfulness_note,
        "limitation": limitation,
        "human_review_required": bool(human_review_required),
        "created_at": _utc_timestamp(),
    }
    ledger["entries"].append(entry)
    ledger["updated_at"] = _utc_timestamp()
    return entry


def save_ledger(ledger, ledger_path):
    """Persist ledger.json and return its path."""
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    return ledger_path


def load_ledger(ledger_path):
    """Load a ledger dictionary from JSON."""
    ledger_path = Path(ledger_path)
    return json.loads(ledger_path.read_text(encoding="utf-8"))


def summarize_ledger(ledger):
    """Summarize evidence totals, review burden, and module coverage."""
    entries = list((ledger or {}).get("entries", []))
    by_module = {}
    by_source_type = {}
    review_count = 0
    for entry in entries:
        module = entry.get("module", "unknown")
        source_type = entry.get("source_type", "unknown")
        by_module[module] = by_module.get(module, 0) + 1
        by_source_type[source_type] = by_source_type.get(source_type, 0) + 1
        if entry.get("human_review_required"):
            review_count += 1
    return {
        "mission_id": (ledger or {}).get("mission_id", ""),
        "total_evidence_count": len(entries),
        "human_review_required_count": review_count,
        "module_counts": by_module,
        "source_type_counts": by_source_type,
    }


def get_entries_requiring_review(ledger):
    """Return evidence entries explicitly marked for human review."""
    return [entry for entry in (ledger or {}).get("entries", []) if entry.get("human_review_required")]
