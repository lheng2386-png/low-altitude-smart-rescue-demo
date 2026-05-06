"""Mission orchestration skeleton for 灾情感知及影响评估.

This module wires the mission schema, input validator, and evidence ledger
together without replacing the existing Gradio processing flow. Existing
modules can call record_module_result as they are gradually adopted.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .evidence_ledger import add_evidence_entry, create_ledger, load_ledger, save_ledger, summarize_ledger
    from .input_validator import validate_mission_inputs
    from .mission_schema import create_mission, ensure_mission_dirs, save_mission, update_mission_status
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from evidence_ledger import add_evidence_entry, create_ledger, load_ledger, save_ledger, summarize_ledger
    from input_validator import validate_mission_inputs
    from mission_schema import create_mission, ensure_mission_dirs, save_mission, update_mission_status


def create_new_mission(missions_root, mission_name=None, mission_id=None):
    """Create a mission directory, mission.json, and empty evidence ledger."""
    mission = create_mission(mission_name=mission_name, mission_id=mission_id)
    mission_dir = Path(missions_root) / mission["mission_id"]
    paths = ensure_mission_dirs(mission_dir)
    ledger_path = paths["evidence"] / "ledger.json"
    mission["evidence_ledger_path"] = str(ledger_path)
    save_ledger(create_ledger(mission["mission_id"]), ledger_path)
    save_mission(mission, mission_dir)
    return mission, mission_dir


def initialize_mission_from_inputs(
    missions_root,
    mission_name=None,
    rgb_images=None,
    thermal_images=None,
    mask_files=None,
    video_file=None,
    odm_enabled=False,
    segmentation_checkpoint_path=None,
):
    """Create a mission and populate input capability classification."""
    input_summary = validate_mission_inputs(
        rgb_images=rgb_images,
        thermal_images=thermal_images,
        mask_files=mask_files,
        video_file=video_file,
        odm_enabled=odm_enabled,
        segmentation_checkpoint_path=segmentation_checkpoint_path,
    )
    mission = create_mission(
        mission_name=mission_name,
        input_summary=input_summary,
        available_modules=input_summary.get("available_modules", []),
        disabled_modules=input_summary.get("disabled_modules", []),
        truthfulness_boundaries=input_summary.get("truthfulness_boundaries", []),
    )
    mission_dir = Path(missions_root) / mission["mission_id"]
    paths = ensure_mission_dirs(mission_dir)
    ledger_path = paths["evidence"] / "ledger.json"
    mission["evidence_ledger_path"] = str(ledger_path)
    mission = update_mission_status(mission, "running")
    save_ledger(create_ledger(mission["mission_id"]), ledger_path)
    save_mission(mission, mission_dir)
    return mission, mission_dir


def record_module_result(
    mission,
    mission_dir,
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
    """Record a module result in the mission evidence ledger and mission outputs."""
    mission_dir = Path(mission_dir)
    ledger_path = Path(mission.get("evidence_ledger_path") or mission_dir / "evidence" / "ledger.json")
    ledger = load_ledger(ledger_path) if ledger_path.exists() else create_ledger(mission["mission_id"])
    entry = add_evidence_entry(
        ledger,
        module=module,
        input_ref=input_ref,
        output_ref=output_ref,
        result_type=result_type,
        confidence=confidence,
        score=score,
        source_type=source_type,
        truthfulness_note=truthfulness_note,
        limitation=limitation,
        human_review_required=human_review_required,
    )
    save_ledger(ledger, ledger_path)

    mission.setdefault("outputs", {})
    mission["outputs"][module] = {
        "output_ref": str(output_ref or ""),
        "result_type": str(result_type or ""),
        "evidence_id": entry["evidence_id"],
    }
    save_mission(mission, mission_dir)
    return entry


def build_mission_summary(mission, ledger=None):
    """Build a compact mission summary for reports and UI integration."""
    ledger_summary = summarize_ledger(ledger or {})
    return {
        "mission_id": mission.get("mission_id", ""),
        "mission_name": mission.get("mission_name", ""),
        "status": mission.get("status", ""),
        "available_modules": mission.get("available_modules", []),
        "disabled_modules": mission.get("disabled_modules", []),
        "truthfulness_boundaries": mission.get("truthfulness_boundaries", []),
        "outputs": mission.get("outputs", {}),
        "evidence_summary": ledger_summary,
    }


def finalize_mission(mission, mission_dir, status="completed"):
    """Finalize a mission, persist mission_summary.json, and return the summary."""
    mission_dir = Path(mission_dir)
    mission = update_mission_status(mission, status)
    ledger_path = Path(mission.get("evidence_ledger_path") or mission_dir / "evidence" / "ledger.json")
    ledger = load_ledger(ledger_path) if ledger_path.exists() else create_ledger(mission["mission_id"])
    summary = build_mission_summary(mission, ledger)
    summary_path = mission_dir / "mission_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_mission(mission, mission_dir)
    return summary
