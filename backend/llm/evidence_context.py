import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]


KNOWN_AUTHENTICITY_BOUNDARIES = [
    "human_candidate is an unverified suspected human target and requires human review.",
    "simulated thermal is not radiometric temperature measurement.",
    "Fast Preview is not an ODM mapping artifact.",
    "image-plane path is not a navigation route.",
    "uploaded/demo mask is not automatic model segmentation.",
    "Do not invent GPS coordinates, temperature matrices, checkpoints, metrics, ODM outputs, or field rescue outcomes.",
]


EVIDENCE_CANDIDATES = {
    "mission_result": [
        "outputs/reports/mission_report_bundle.json",
        "outputs/mission_result.json",
        "outputs/demo_mission_result.json",
    ],
    "detection_result": [
        "outputs/detection/detection_result.json",
        "outputs/detection/dual_detection_result.json",
        "outputs/detection/transformer_detection_result.json",
    ],
    "segmentation_result": [
        "outputs/segmentation_inference/segmentation_result.json",
        "outputs/segmentation_inference/damage_summary.json",
        "outputs/segmentation_inference/segmentation_source.json",
    ],
    "thermal_result": [
        "outputs/thermal/radiometric_thermal_result.json",
        "outputs/thermal/thermal_result.json",
    ],
    "path_planning_result": [
        "outputs/decision_fusion/path_planning_result.json",
        "outputs/decision_fusion/path_comparison.json",
    ],
    "ec_terp_result": [
        "outputs/decision_fusion/ec_terp_ranking.json",
        "outputs/decision_fusion/ec_terp_comparison.json",
        "outputs/ec_terp_evaluation/ec_terp_evaluation_summary.json",
        "outputs/ec_terp_evaluation/terp_vs_ec_terp_comparison.json",
    ],
    "evidence_ledger": [
        "outputs/reports/mission_evidence_ledger.json",
        "outputs/evidence/ledger.json",
    ],
    "saved_llm_report": [
        "outputs/reports/llm_mission_report.json",
        "outputs/reports/llm_report.json",
        "outputs/reports/mission_llm_report.json",
    ],
}


def build_mission_evidence_context(mission_id: str | None, root_dir: str | Path | None = None) -> dict[str, Any]:
    mission_id = str(mission_id or "current_mission").strip() or "current_mission"
    mission_root = _resolve_mission_root(mission_id, root_dir=root_dir)
    limitations = []
    evidence = {}
    for key, candidates in EVIDENCE_CANDIDATES.items():
        item = _load_first_available(mission_root, candidates)
        evidence[key] = item
        if item["status"] != "available":
            limitations.append(f"{key} is unavailable for mission_id={mission_id}.")

    return {
        "mission_id": mission_id,
        "mission_root": str(mission_root),
        "evidence": evidence,
        "known_limitations": KNOWN_AUTHENTICITY_BOUNDARIES,
        "limitations": limitations + KNOWN_AUTHENTICITY_BOUNDARIES,
    }


def _resolve_mission_root(mission_id: str, root_dir: str | Path | None = None) -> Path:
    base = Path(root_dir).resolve() if root_dir else ROOT_DIR
    if not mission_id or mission_id in {"current", "current_mission", "default"}:
        return base
    candidates = [
        base / "outputs" / mission_id,
        base / "outputs" / "missions" / mission_id,
        base / "outputs" / "demo_cases" / mission_id,
        base / "demo_cases" / mission_id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base / "outputs" / "missions" / mission_id


def _load_first_available(root: Path, candidates: list[str]) -> dict[str, Any]:
    for relative_path in candidates:
        for path in _expand_candidate(root, relative_path):
            data = _safe_read(path)
            if data is not None:
                return {
                    "status": "available",
                    "path": str(path),
                    "summary": _summarize_data(data),
                    "data": data,
                }
    return {
        "status": "unavailable",
        "path": "",
        "summary": "Evidence item unavailable.",
        "data": None,
    }


def _expand_candidate(root: Path, relative_path: str) -> list[Path]:
    if any(token in relative_path for token in ["*", "?", "["]):
        return sorted(path for path in root.glob(relative_path) if path.exists())
    path = root / relative_path
    return [path] if path.exists() else []


def _safe_read(path: Path):
    try:
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return path.read_text(encoding="utf-8", errors="ignore")[:12000]
    except Exception as exc:
        return {"read_error": str(exc), "path": str(path)}


def _summarize_data(data: Any) -> str:
    if isinstance(data, dict):
        keys = ", ".join(list(data.keys())[:10])
        success = data.get("success")
        status = data.get("status") or data.get("message") or data.get("truthfulness_note")
        return f"dict keys: {keys}; success={success}; note={str(status)[:240]}"
    if isinstance(data, list):
        return f"list length={len(data)}"
    text = str(data).replace("\n", " ")
    return text[:300]
