"""Lightweight Mission schema and JSON persistence for 灾情感知及影响评估.

The mission record is intentionally a plain Python dictionary so the current
Gradio prototype can adopt it gradually without adding a database or service
layer. Files are written with UTF-8 JSON to preserve bilingual notes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4


MISSION_STATUSES = {"created", "running", "completed", "failed"}

INPUT_SUBDIRS = ("rgb", "thermal", "masks", "metadata")
OUTPUT_SUBDIRS = (
    "detection",
    "segmentation",
    "thermal",
    "thermal_check",
    "orthomosaic",
    "reconstruction",
    "path",
    "priority",
    "target_verification",
    "reports",
)


def _utc_timestamp():
    """Return a stable ISO-8601 UTC timestamp for mission records."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def create_mission_id(created_at=None):
    """Create a human-readable mission id like M20260504_143012_ab12."""
    timestamp = created_at or datetime.utcnow()
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).replace(tzinfo=None)
    return f"M{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:4]}"


def ensure_mission_dirs(mission_dir):
    """Create and return the standard mission directory structure."""
    mission_dir = Path(mission_dir)
    paths = {
        "root": mission_dir,
        "input": mission_dir / "input",
        "outputs": mission_dir / "outputs",
        "evidence": mission_dir / "evidence",
    }

    mission_dir.mkdir(parents=True, exist_ok=True)
    for subdir in INPUT_SUBDIRS:
        path = mission_dir / "input" / subdir
        path.mkdir(parents=True, exist_ok=True)
        paths[f"input_{subdir}"] = path
    for subdir in OUTPUT_SUBDIRS:
        path = mission_dir / "outputs" / subdir
        path.mkdir(parents=True, exist_ok=True)
        paths[f"output_{subdir}"] = path
    paths["evidence"].mkdir(parents=True, exist_ok=True)
    return paths


def create_mission(
    mission_name=None,
    mission_id=None,
    input_summary=None,
    available_modules=None,
    disabled_modules=None,
    truthfulness_boundaries=None,
    outputs=None,
    evidence_ledger_path=None,
):
    """Build a new mission dictionary with explicit capability boundaries."""
    created_at = _utc_timestamp()
    mission_id = mission_id or create_mission_id(created_at)
    return {
        "mission_id": mission_id,
        "mission_name": mission_name or mission_id,
        "created_at": created_at,
        "updated_at": created_at,
        "status": "created",
        "input_summary": input_summary or {},
        "available_modules": list(available_modules or []),
        "disabled_modules": list(disabled_modules or []),
        "truthfulness_boundaries": list(truthfulness_boundaries or []),
        "outputs": outputs or {},
        "evidence_ledger_path": str(evidence_ledger_path or ""),
        "workflow_state": {},
    }


def save_mission(mission, mission_dir):
    """Persist mission.json and return its path."""
    mission_dir = Path(mission_dir)
    ensure_mission_dirs(mission_dir)
    mission = dict(mission)
    mission["updated_at"] = _utc_timestamp()
    mission_path = mission_dir / "mission.json"
    mission_path.write_text(json.dumps(mission, ensure_ascii=False, indent=2), encoding="utf-8")
    return mission_path


def load_mission(mission_json_path):
    """Load a mission dictionary from mission.json."""
    mission_json_path = Path(mission_json_path)
    return json.loads(mission_json_path.read_text(encoding="utf-8"))


def update_mission_status(mission, status, error=None):
    """Return a mission copy with an updated lifecycle status."""
    if status not in MISSION_STATUSES:
        raise ValueError(f"Unsupported mission status: {status}")
    updated = dict(mission)
    updated["status"] = status
    updated["updated_at"] = _utc_timestamp()
    if error:
        updated["error"] = str(error)
    return updated
