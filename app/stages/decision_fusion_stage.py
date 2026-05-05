"""S7 decision_fusion stage wrapper for lightweight EC-TERP ranking.

The wrapper fuses already available S4/S5/S6/S2 outputs into a conservative
decision-support ranking. It does not replace rescue command judgment.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from ..workflow.workflow_orchestrator import initialize_rescue_workflow, record_stage_output
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from workflow.workflow_orchestrator import initialize_rescue_workflow, record_stage_output


DECISION_FUSION_TRUTHFULNESS_NOTE = (
    "EC-TERP provides decision-support priority ranking and does not replace rescue command judgment. "
    "AI candidates are not confirmed civilians. "
    "Thermal support is auxiliary evidence and not confirmation of life."
)
NO_DECISION_CANDIDATE_NOTE = "The system must not invent priority rankings when no candidate evidence is available."


def _output_dir(mission_dir):
    output_dir = Path(mission_dir) / "outputs" / "priority"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_result(output_dir, result):
    result_path = Path(output_dir) / "decision_fusion_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(result_path)


def _by_candidate_id(records):
    indexed = {}
    for record in records or []:
        candidate_id = record.get("candidate_id")
        if candidate_id:
            indexed[str(candidate_id)] = record
    return indexed


def _macro_risk_bonus(macro_analysis_result):
    zones = list((macro_analysis_result or {}).get("macro_zones") or [])
    if any(zone.get("risk_level") == "Critical" for zone in zones):
        return 10.0
    if any(zone.get("risk_level") == "High" for zone in zones):
        return 5.0
    if any(zone.get("risk_level") == "Medium" for zone in zones):
        return 2.5
    return 0.0


def _priority_level(score):
    if score >= 85:
        return "Critical"
    if score >= 65:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def _thermal_score(record):
    level = (record or {}).get("thermal_support_level")
    if level == "strong":
        return 20.0
    if level == "weak":
        return 10.0
    return 0.0


def _candidate_score(candidate, verification_record, thermal_record, macro_bonus):
    score = 0.0
    score += min(50.0, max(0.0, float(candidate.get("confidence") or 0.0) * 50.0))
    if candidate.get("class_name") == "human_candidate":
        score += 20.0
    if (verification_record or {}).get("review_status") in {"confirmed_candidate", "urgent_review"}:
        score += 10.0
    if (verification_record or {}).get("review_status") == "rejected_false_positive":
        score -= 100.0
    score += _thermal_score(thermal_record)
    score += macro_bonus
    return round(max(0.0, min(100.0, score)), 2)


def run_decision_fusion_stage(
    mission,
    mission_dir,
    local_recon_result=None,
    target_verification_result=None,
    thermal_check_result=None,
    macro_analysis_result=None,
):
    """Run S7 EC-TERP decision-support fusion from existing stage outputs."""
    mission = initialize_rescue_workflow(mission)
    mission_dir = Path(mission_dir)
    output_dir = _output_dir(mission_dir)
    candidates = list((local_recon_result or {}).get("candidates") or [])
    verification_by_id = _by_candidate_id((target_verification_result or {}).get("verification_records") or [])
    thermal_by_id = _by_candidate_id((thermal_check_result or {}).get("thermal_records") or [])

    if not candidates:
        truthfulness_note = f"{DECISION_FUSION_TRUTHFULNESS_NOTE} {NO_DECISION_CANDIDATE_NOTE}"
        result = {
            "stage_key": "decision_fusion",
            "status": "degraded",
            "decision_candidates": [],
            "decision_summary": {
                "decision_candidate_count": 0,
                "human_review_required_count": 0,
                "top_priority_level": "",
            },
            "top_priority_candidate": {},
            "truthfulness_note": truthfulness_note,
            "human_review_required": True,
        }
        result_path = _save_result(output_dir, result)
        record_stage_output(
            mission,
            mission_dir,
            stage_key="decision_fusion",
            output_ref=result_path,
            result_type="no_candidate_for_decision_fusion",
            source_type="rule_based_ec_terp",
            confidence=None,
            score=0,
            truthfulness_note=truthfulness_note,
            limitation=truthfulness_note,
            human_review_required=True,
        )
        return mission, result

    macro_bonus = _macro_risk_bonus(macro_analysis_result)
    decision_candidates = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", ""))
        verification_record = verification_by_id.get(candidate_id, {})
        thermal_record = thermal_by_id.get(candidate_id, {})
        score = _candidate_score(candidate, verification_record, thermal_record, macro_bonus)
        review_status = verification_record.get("review_status", candidate.get("review_status", "need_review"))
        decision_candidates.append(
            {
                "candidate_id": candidate_id,
                "target_id": candidate.get("target_id", ""),
                "area_id": candidate.get("area_id", ""),
                "class_name": candidate.get("class_name", ""),
                "confidence": candidate.get("confidence"),
                "review_status": review_status,
                "thermal_support_level": thermal_record.get("thermal_support_level", ""),
                "ec_terp_score": score,
                "priority_level": _priority_level(score),
                "recommended_action": "prioritize_human_review_and_field_verification",
                "should_exclude_from_rescue_ranking": review_status == "rejected_false_positive",
                "human_review_required": True,
                "is_confirmed_civilian": False,
                "truthfulness_note": DECISION_FUSION_TRUTHFULNESS_NOTE,
            }
        )
    decision_candidates.sort(key=lambda item: item.get("ec_terp_score", 0.0), reverse=True)
    for index, item in enumerate(decision_candidates, start=1):
        item["rank"] = index

    top = decision_candidates[0] if decision_candidates else {}
    result = {
        "stage_key": "decision_fusion",
        "status": "completed" if decision_candidates else "degraded",
        "decision_candidates": decision_candidates,
        "decision_summary": {
            "decision_candidate_count": len(decision_candidates),
            "human_review_required_count": sum(1 for item in decision_candidates if item.get("human_review_required")),
            "top_priority_level": top.get("priority_level", ""),
        },
        "top_priority_candidate": top,
        "truthfulness_note": DECISION_FUSION_TRUTHFULNESS_NOTE,
        "human_review_required": True,
    }
    result_path = _save_result(output_dir, result)
    record_stage_output(
        mission,
        mission_dir,
        stage_key="decision_fusion",
        output_ref=result_path,
        result_type="ec_terp_priority_ranking",
        source_type="rule_based_ec_terp",
        confidence=(top.get("ec_terp_score") / 100.0) if top else None,
        score=len(decision_candidates),
        truthfulness_note=DECISION_FUSION_TRUTHFULNESS_NOTE,
        limitation=DECISION_FUSION_TRUTHFULNESS_NOTE,
        human_review_required=True,
    )
    return mission, result
