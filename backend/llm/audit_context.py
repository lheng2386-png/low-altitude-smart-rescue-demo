import json
from pathlib import Path
from typing import Any

from .evidence_context import KNOWN_AUTHENTICITY_BOUNDARIES, build_mission_evidence_context


AUDIT_TARGETS = {"all", "llm_report", "final_report_v2", "copilot", "planner", "evidence_ledger"}


def build_audit_context(mission_id: str | None, audit_target: str = "all", root_dir: str | Path | None = None) -> dict[str, Any]:
    mission_id = str(mission_id or "current_mission").strip() or "current_mission"
    audit_target = str(audit_target or "all").strip().lower() or "all"
    if audit_target not in AUDIT_TARGETS:
        audit_target = "all"

    evidence_context = build_mission_evidence_context(mission_id, root_dir=root_dir)
    mission_root = Path(evidence_context.get("mission_root") or ".")
    evidence = evidence_context.get("evidence", {}) or {}

    return {
        "mission_id": mission_id,
        "audit_target": audit_target,
        "mission_root": str(mission_root),
        "mission_result": _unwrap_evidence(evidence.get("mission_result"), "No mission_result found for this mission."),
        "module_outputs": {
            "detection": _unwrap_evidence(evidence.get("detection_result"), "No detection result found for this mission."),
            "segmentation": _unwrap_evidence(evidence.get("segmentation_result"), "No segmentation result found for this mission."),
            "thermal": _unwrap_evidence(evidence.get("thermal_result"), "No thermal result found for this mission."),
            "path_planning": _unwrap_evidence(evidence.get("path_planning_result"), "No path planning result found for this mission."),
            "ec_terp": _unwrap_evidence(evidence.get("ec_terp_result"), "No EC-TERP result found for this mission."),
        },
        "evidence_ledger": _load_ledger(mission_root),
        "llm_report": _load_json_candidate(
            mission_root,
            [
                "outputs/reports/llm_mission_report.json",
                "outputs/reports/llm_report.json",
                "outputs/reports/mission_llm_report.json",
            ],
            "No saved LLM report found for this mission.",
        ),
        "copilot_answers": _load_json_candidate(
            mission_root,
            [
                "outputs/reports/mission_copilot_answers.json",
                "outputs/reports/mission_copilot_events.json",
            ],
            "No saved Mission Copilot answers found for this mission.",
        ),
        "planner_results": _load_json_candidate(
            mission_root,
            [
                "outputs/reports/mission_planner_result.json",
                "outputs/reports/mission_planner_events.json",
            ],
            "No saved Mission Planner result found for this mission.",
        ),
        "final_report_v2": _load_report_candidate(
            mission_root,
            [
                "outputs/reports/final_report_v2.md",
                "outputs/reports/final_report_v2.json",
                "outputs/reports/final_report_v2.html",
            ],
        ),
        "known_authenticity_boundaries": KNOWN_AUTHENTICITY_BOUNDARIES,
    }


def _unwrap_evidence(item: dict[str, Any] | None, note: str) -> dict[str, Any]:
    item = item or {}
    if item.get("status") == "available":
        return {
            "status": "available",
            "path": item.get("path", ""),
            "data": item.get("data"),
            "summary": item.get("summary", ""),
        }
    return {"status": "unavailable", "note": note, "data": None}


def _load_ledger(mission_root: Path) -> dict[str, Any]:
    ledger = _load_json_candidate(
        mission_root,
        [
            "outputs/reports/evidence_ledger.json",
            "outputs/reports/mission_evidence_ledger.json",
            "outputs/evidence/ledger.json",
        ],
        "No Evidence Ledger found for this mission.",
    )
    data = ledger.get("data")
    if isinstance(data, list):
        ledger["events"] = data
    elif isinstance(data, dict):
        ledger["events"] = data.get("events", [])
    else:
        ledger["events"] = []
    return ledger


def _load_json_candidate(mission_root: Path, candidates: list[str], note: str) -> dict[str, Any]:
    for relative_path in candidates:
        path = mission_root / relative_path
        if path.exists():
            try:
                return {
                    "status": "available",
                    "path": str(path),
                    "data": json.loads(path.read_text(encoding="utf-8")),
                }
            except Exception as exc:
                return {"status": "unavailable", "path": str(path), "note": f"Unable to read JSON: {exc}", "data": None}
    return {"status": "unavailable", "note": note, "data": None}


def _load_report_candidate(mission_root: Path, candidates: list[str]) -> dict[str, Any]:
    for relative_path in candidates:
        path = mission_root / relative_path
        if path.exists():
            try:
                if path.suffix.lower() == ".json":
                    data = json.loads(path.read_text(encoding="utf-8"))
                else:
                    data = path.read_text(encoding="utf-8", errors="ignore")
                return {"status": "available", "path": str(path), "data": data}
            except Exception as exc:
                return {"status": "unavailable", "path": str(path), "note": f"Unable to read report: {exc}", "data": None}
    return {"status": "unavailable", "note": "No Final Report V2 found for this mission.", "data": None}
