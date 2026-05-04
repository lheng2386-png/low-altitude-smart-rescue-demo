import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .audit_context import AUDIT_TARGETS, build_audit_context
from .audit_schemas import AUDIT_JSON_SCHEMA, coerce_audit_result
from .prompts import AUDITOR_SYSTEM_PROMPT, build_auditor_user_prompt
from .report_assistant import FORBIDDEN_REPLACEMENTS


ROOT_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT_DIR / "outputs" / "reports"

UNSAFE_PHRASES = [
    "confirmed civilian",
    "confirmed survivor",
    "confirmed casualty",
    "victim confirmed",
    "rescued person",
    "measured temperature",
    "real temperature matrix",
    "actual body temperature",
    "real GPS route",
    "GPS navigation route",
    "georeferenced route",
    "real ODM orthomosaic",
    "survey-grade orthomosaic",
    "verified model-generated segmentation",
    "real rescue conclusion",
]

REQUIRED_LEDGER_EVENTS = {
    "detection": "detection_completed",
    "segmentation": "segmentation_loaded",
    "thermal": "thermal_simulation_completed",
    "path_planning": "path_planning_completed",
    "ec_terp": "ec_terp_priority_computed",
}


def run_evidence_audit(
    mission_id: str | None,
    audit_target: str = "all",
    root_dir: str | Path | None = None,
) -> dict[str, Any]:
    mission_id = str(mission_id or "current_mission").strip() or "current_mission"
    audit_target = str(audit_target or "all").strip().lower() or "all"
    if audit_target not in AUDIT_TARGETS:
        audit_target = "all"
    context = build_audit_context(mission_id, audit_target=audit_target, root_dir=root_dir)
    provider = _selected_provider_name()
    model = os.getenv("OPENAI_MODEL", "gpt-5.5")
    fallback_used = False
    fallback_reason = ""

    deterministic = _deterministic_audit(context)
    try:
        if provider == "openai":
            llm_result = _ask_openai(context, deterministic, model=model)
            result = _merge_audit_results(deterministic, llm_result)
        else:
            result = deterministic
            provider = "mock"
    except Exception as exc:
        fallback_used = True
        fallback_reason = str(exc)
        result = deterministic
        provider = "mock"

    result = validate_audit_result(result)
    saved_path = save_audit_result(mission_id, audit_target, result, provider, model if provider == "openai" else "", context)
    _append_audit_event(context, mission_id, audit_target, result)
    response = {
        "ok": True,
        "provider": provider,
        "model": model if provider == "openai" else "",
        "fallback_used": fallback_used,
        "result": result,
        "saved_path": str(saved_path),
    }
    if fallback_reason:
        response["fallback_reason"] = fallback_reason
    return response


def validate_audit_result(audit_result: dict[str, Any]) -> dict[str, Any]:
    result = coerce_audit_result(audit_result or {})
    result["human_review_required"] = True
    issues = result.get("issues", [])
    _assign_issue_ids(issues)
    for issue in issues:
        issue["auto_fix_allowed"] = False
    high = any(issue.get("severity") == "high" for issue in issues)
    medium = any(issue.get("severity") == "medium" for issue in issues)
    if high:
        result["audit_status"] = "fail"
        result["overall_risk_level"] = "high"
    elif medium or issues:
        result["audit_status"] = "warning"
        result["overall_risk_level"] = "medium" if medium else "low"
    else:
        result["audit_status"] = "pass"
        result["overall_risk_level"] = "low"
    return result


def save_audit_result(
    mission_id: str,
    audit_target: str,
    audit_result: dict[str, Any],
    provider: str,
    model: str,
    audit_context: dict[str, Any],
) -> Path:
    mission_root = Path(audit_context.get("mission_root") or ROOT_DIR)
    output_dir = mission_root / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mission_id": mission_id,
        "audit_target": audit_target,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        **audit_result,
    }
    path = output_dir / "llm_evidence_audit.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_latest_audit_result(mission_id: str | None, root_dir: str | Path | None = None) -> dict[str, Any]:
    context = build_audit_context(mission_id, root_dir=root_dir)
    path = Path(context.get("mission_root") or ROOT_DIR) / "outputs" / "reports" / "llm_evidence_audit.json"
    if not path.exists():
        return {"status": "unavailable", "note": "No saved evidence audit found for this mission."}
    return {"status": "available", "path": str(path), "data": json.loads(path.read_text(encoding="utf-8"))}


def _selected_provider_name() -> str:
    llm_enabled = os.getenv("LLM_ENABLE", "false").strip().lower() in {"1", "true", "yes", "on"}
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if not llm_enabled or provider != "openai" or not os.getenv("OPENAI_API_KEY"):
        return "mock"
    return "openai"


def _ask_openai(context: dict[str, Any], deterministic_result: dict[str, Any], model: str) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    payload = {
        "model": model,
        "instructions": AUDITOR_SYSTEM_PROMPT,
        "input": build_auditor_user_prompt({"audit_context": context, "deterministic_findings": deterministic_result}),
        "text": {
            "format": {
                "type": "json_schema",
                "name": AUDIT_JSON_SCHEMA["name"],
                "strict": AUDIT_JSON_SCHEMA["strict"],
                "schema": AUDIT_JSON_SCHEMA["schema"],
            }
        },
    }
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    return json.loads(_extract_output_text(response.json()))


def _extract_output_text(response_payload: dict[str, Any]) -> str:
    if isinstance(response_payload.get("output_text"), str):
        return response_payload["output_text"]
    for item in response_payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                return text
    raise ValueError("OpenAI Responses API returned no text output.")


def _merge_audit_results(deterministic: dict[str, Any], llm_result: dict[str, Any]) -> dict[str, Any]:
    merged = coerce_audit_result(llm_result or {})
    existing_keys = {_issue_key(issue) for issue in merged.get("issues", [])}
    for issue in deterministic.get("issues", []):
        if _issue_key(issue) not in existing_keys:
            merged.setdefault("issues", []).append(issue)
    merged.setdefault("safe_rewrite_suggestions", []).extend(deterministic.get("safe_rewrite_suggestions", []))
    for item in deterministic.get("missing_evidence", []):
        if item not in merged.setdefault("missing_evidence", []):
            merged["missing_evidence"].append(item)
    for item in deterministic.get("positive_checks", []):
        if item not in merged.setdefault("positive_checks", []):
            merged["positive_checks"].append(item)
    merged["human_review_required"] = True
    return merged


def _deterministic_audit(context: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    missing_evidence: list[str] = []
    positive_checks: list[str] = []
    target_docs = _target_documents(context)

    for location, payload in target_docs.items():
        text = _stringify(payload)
        for phrase in UNSAFE_PHRASES:
            if phrase.lower() in text.lower():
                issues.append(
                    _issue(
                        "high",
                        "unsafe_phrase",
                        location,
                        f"Unsafe phrase detected: {phrase}",
                        "Known authenticity boundary list.",
                        f"Replace with bounded evidence wording instead of '{phrase}'.",
                    )
                )
                suggestions.append(
                    {
                        "location": location,
                        "original_text": phrase,
                        "suggested_text": _safe_replacement(phrase),
                        "reason": "Unsafe wording exceeds evidence-supported decision-support boundaries.",
                    }
                )
        _check_human_review(payload, location, issues, text)

    _check_context_boundaries(context, target_docs, issues, suggestions)
    _check_missing_ledger_events(context, target_docs, issues, missing_evidence)
    _check_planner_tools(context, issues)

    if not issues:
        positive_checks.extend(
            [
                "No configured unsafe phrase was found in selected audit targets.",
                "Selected outputs preserve decision-support and human-review boundaries.",
                "No non-white-listed Mission Planner tool was found.",
            ]
        )

    return validate_audit_result(
        {
            "audit_status": "pass",
            "overall_risk_level": "low",
            "issues": issues,
            "safe_rewrite_suggestions": suggestions,
            "missing_evidence": missing_evidence,
            "positive_checks": positive_checks,
            "human_review_required": True,
        }
    )


def _target_documents(context: dict[str, Any]) -> dict[str, Any]:
    audit_target = context.get("audit_target", "all")
    docs = {}
    if audit_target in {"all", "llm_report"}:
        docs["LLM Report"] = context.get("llm_report")
    if audit_target in {"all", "final_report_v2"}:
        docs["Final Report V2"] = context.get("final_report_v2")
    if audit_target in {"all", "copilot"}:
        docs["Mission Copilot"] = context.get("copilot_answers")
    if audit_target in {"all", "planner"}:
        docs["Mission Planner"] = context.get("planner_results")
    if audit_target in {"all", "evidence_ledger"}:
        docs["Evidence Ledger"] = context.get("evidence_ledger")
    return docs


def _check_human_review(payload: Any, location: str, issues: list[dict[str, Any]], text: str):
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        if isinstance(data.get("events"), list):
            return
        nested_result = data.get("result")
        if isinstance(nested_result, dict) and nested_result.get("human_review_required") is True:
            return
        if data.get("human_review_required") is not True and "human_review_required" not in data:
            issues.append(
                _issue(
                    "medium",
                    "missing_human_review",
                    location,
                    "human_review_required is missing from this structured output.",
                    location,
                    "Add human_review_required: true and an explicit human review note.",
                )
            )
        elif data.get("human_review_required") is False:
            issues.append(
                _issue(
                    "medium",
                    "missing_human_review",
                    location,
                    "human_review_required is false, but all LLM outputs require review.",
                    location,
                    "Set human_review_required to true.",
                )
            )
    elif isinstance(data, str):
        lower = text.lower()
        if "human review" not in lower and "人工复核" not in lower and "requires review" not in lower:
            issues.append(
                _issue(
                    "medium",
                    "missing_human_review",
                    location,
                    "Text output does not include an explicit human review statement.",
                    location,
                    "Add a visible statement that all outputs require human review.",
                )
            )


def _check_context_boundaries(
    context: dict[str, Any],
    target_docs: dict[str, Any],
    issues: list[dict[str, Any]],
    suggestions: list[dict[str, Any]],
):
    text_by_location = {location: _stringify(payload).lower() for location, payload in target_docs.items()}
    detection_data = _module_data(context, "detection")
    thermal_data = _module_data(context, "thermal")
    path_data = _module_data(context, "path_planning")
    segmentation_data = _module_data(context, "segmentation")
    mission_result = (context.get("mission_result") or {}).get("data") or {}

    if _contains_human_candidate(detection_data) or _contains_human_candidate(mission_result):
        _boundary_check(text_by_location, ["confirmed survivor", "confirmed civilian"], "human_candidate is not confirmed person status.", issues, suggestions)
    if _thermal_is_simulated(thermal_data) or _thermal_is_simulated(mission_result):
        _boundary_check(text_by_location, ["measured temperature", "real temperature", "actual body temperature"], "simulated thermal is not real temperature measurement.", issues, suggestions)
    if _path_is_image_plane(path_data) or _path_is_image_plane(mission_result):
        _boundary_check(text_by_location, ["gps route", "gps navigation route", "georeferenced route"], "image-plane path is not GPS navigation.", issues, suggestions)
    if _segmentation_is_demo(segmentation_data) or _segmentation_is_demo(mission_result):
        _boundary_check(text_by_location, ["verified model-generated segmentation", "model-generated segmentation"], "demo/uploaded mask is not verified automatic segmentation.", issues, suggestions)
    if "fast_preview" in _stringify(mission_result).lower() or "fast preview" in _stringify(mission_result).lower():
        _boundary_check(text_by_location, ["real odm orthomosaic", "survey-grade orthomosaic"], "Fast Preview is not real ODM orthomosaic output.", issues, suggestions)


def _boundary_check(
    text_by_location: dict[str, str],
    phrases: list[str],
    evidence_reference: str,
    issues: list[dict[str, Any]],
    suggestions: list[dict[str, Any]],
):
    for location, text in text_by_location.items():
        for phrase in phrases:
            if phrase in text:
                issues.append(
                    _issue(
                        "high",
                        "authenticity_boundary",
                        location,
                        f"Boundary-violating wording detected: {phrase}",
                        evidence_reference,
                        f"Replace '{phrase}' with bounded decision-support wording.",
                    )
                )
                suggestions.append(
                    {
                        "location": location,
                        "original_text": phrase,
                        "suggested_text": _safe_replacement(phrase),
                        "reason": evidence_reference,
                    }
                )


def _check_missing_ledger_events(
    context: dict[str, Any],
    target_docs: dict[str, Any],
    issues: list[dict[str, Any]],
    missing_evidence: list[str],
):
    ledger_events = ((context.get("evidence_ledger") or {}).get("events") or [])
    event_types = {str(event.get("event_type", "")).lower() for event in ledger_events if isinstance(event, dict)}
    all_text = _stringify(target_docs).lower()
    for key, event_type in REQUIRED_LEDGER_EVENTS.items():
        if key in all_text and event_type.lower() not in event_types:
            message = f"Referenced {key} output but Evidence Ledger is missing {event_type}."
            missing_evidence.append(message)
            issues.append(
                _issue(
                    "medium",
                    "missing_evidence",
                    "Evidence Ledger",
                    message,
                    "Evidence Ledger event list.",
                    f"Add a bounded ledger event for {event_type} or remove unsupported reference.",
                )
            )


def _check_planner_tools(context: dict[str, Any], issues: list[dict[str, Any]]):
    planner = (context.get("planner_results") or {}).get("data")
    allowed = {
        "load_mission_result",
        "read_evidence_ledger",
        "load_llm_report",
        "load_ec_terp_result",
        "validate_authenticity_boundaries",
        "generate_mission_report",
        "ask_mission_copilot",
    }
    calls = []
    if isinstance(planner, dict):
        result = planner.get("result", planner)
        calls.extend(result.get("tool_plan", []) or [])
        calls.extend(result.get("executed_tools", []) or [])
    elif isinstance(planner, list):
        calls.extend(planner)
    for call in calls:
        if isinstance(call, dict):
            tool_name = call.get("tool_name")
            if tool_name and tool_name not in allowed:
                issues.append(
                    _issue(
                        "high",
                        "authenticity_boundary",
                        "Mission Planner",
                        f"Planner referenced non-white-listed tool: {tool_name}",
                        "Mission Planner white-listed tools.",
                        "Reject the tool plan and use only approved mission evidence tools.",
                    )
                )


def _append_audit_event(context: dict[str, Any], mission_id: str, audit_target: str, result: dict[str, Any]):
    mission_root = Path(context.get("mission_root") or ROOT_DIR)
    ledger_paths = [mission_root / "outputs" / "reports" / "mission_evidence_ledger.json"]
    loaded_path = ((context.get("evidence_ledger") or {}).get("path") or "").strip()
    if loaded_path:
        loaded = Path(loaded_path)
        if loaded not in ledger_paths:
            ledger_paths.append(loaded)
    event = _audit_event_payload(mission_id, audit_target, result)
    for ledger_path in ledger_paths:
        _append_event_to_ledger(ledger_path, mission_id, event)


def _audit_event_payload(mission_id: str, audit_target: str, result: dict[str, Any]) -> dict[str, Any]:
    issues = result.get("issues", []) or []
    return {
        "event_type": "llm_evidence_audit_completed",
        "mission_id": mission_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "LLM Evidence Auditor",
        "source_module": "LLM Evidence Auditor",
        "audit_target": audit_target,
        "audit_status": result.get("audit_status", "warning"),
        "overall_risk_level": result.get("overall_risk_level", "medium"),
        "issue_count": len(issues),
        "high_severity_issue_count": sum(1 for issue in issues if issue.get("severity") == "high"),
        "human_review_required": True,
        "message": "LLM Evidence Auditor reviewed mission outputs for evidence consistency and authenticity boundaries. Audit suggestions require human review.",
    }


def _append_event_to_ledger(ledger_path: Path, mission_id: str, event: dict[str, Any]):
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    except Exception:
        ledger = {}
    if isinstance(ledger, list):
        ledger = {"events": ledger}
    if not isinstance(ledger, dict):
        ledger = {}
    ledger.setdefault("mission_id", mission_id)
    events = ledger.setdefault("events", [])
    if not isinstance(events, list):
        events = []
        ledger["events"] = events
    events.append(dict(event))
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")


def _issue(severity: str, category: str, location: str, problem: str, evidence_reference: str, suggested_fix: str) -> dict[str, Any]:
    return {
        "issue_id": "AUDIT-000",
        "severity": severity,
        "category": category,
        "location": location,
        "problem": problem,
        "evidence_reference": evidence_reference,
        "suggested_fix": suggested_fix,
        "auto_fix_allowed": False,
    }


def _stringify(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _safe_replacement(phrase: str) -> str:
    text = phrase
    for pattern, replacement in FORBIDDEN_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    if text == phrase:
        return "bounded evidence-supported wording requiring human review"
    return text


def _module_data(context: dict[str, Any], key: str) -> Any:
    return ((context.get("module_outputs") or {}).get(key) or {}).get("data")


def _contains_human_candidate(data: Any) -> bool:
    return "human_candidate" in _stringify(data).lower()


def _thermal_is_simulated(data: Any) -> bool:
    text = _stringify(data).lower()
    return "simulated" in text and "thermal" in text


def _path_is_image_plane(data: Any) -> bool:
    text = _stringify(data).lower()
    return "image_plane_path" in text or "image-plane" in text


def _segmentation_is_demo(data: Any) -> bool:
    text = _stringify(data).lower()
    return "demo_mask" in text or "uploaded_mask" in text or "uploaded mask" in text


def _issue_key(issue: dict[str, Any]) -> tuple[str, str, str]:
    return (str(issue.get("category")), str(issue.get("location")), str(issue.get("problem")))


def _assign_issue_ids(issues: list[dict[str, Any]]):
    for index, issue in enumerate(issues, start=1):
        issue["issue_id"] = f"AUDIT-{index:03d}"
