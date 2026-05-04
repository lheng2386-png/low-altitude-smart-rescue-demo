import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .evidence_context import build_mission_evidence_context
from .guardrails import apply_copilot_guardrails, sanitize_text
from .prompts import PLANNER_SYSTEM_PROMPT, build_planner_user_prompt
from .schemas import TOOL_PLAN_JSON_SCHEMA, coerce_planner_result, coerce_tool_plan
from .tools import ALLOWED_TOOL_ARGUMENTS, WHITE_LISTED_TOOLS, execute_tool


ROOT_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT_DIR / "outputs" / "reports"
DANGEROUS_KEYWORDS = {
    "shell",
    "subprocess",
    "rm",
    "delete",
    "network",
    "curl",
    "wget",
    "arbitrary_file",
    "api_key",
    "env",
}


def execute_mission_planner(mission_id: str | None, user_goal: str | None, root_dir: str | Path | None = None) -> dict[str, Any]:
    mission_id = str(mission_id or "current_mission").strip() or "current_mission"
    raw_goal = str(user_goal or "").strip()
    user_goal = sanitize_text(raw_goal)
    mission_context = build_mission_evidence_context(mission_id, root_dir=root_dir)
    provider_name = _selected_provider_name()
    model = os.getenv("OPENAI_MODEL", "gpt-5.5")
    fallback_used = False
    fallback_reason = ""

    try:
        if provider_name == "openai":
            plan = _ask_openai_for_plan(user_goal, mission_context, model=model)
            provider = "openai"
        else:
            plan = _mock_plan(raw_goal, mission_id, mission_context)
            provider = "mock"
    except Exception as exc:
        fallback_used = True
        fallback_reason = str(exc)
        plan = _mock_plan(raw_goal, mission_id, mission_context)
        provider = "mock"

    validation = validate_tool_plan(plan, requested_mission_id=mission_id)
    if not validation["valid"]:
        result = _safe_rejection_result(plan, validation["errors"], mission_context)
        return {
            "ok": False,
            "provider": provider,
            "model": model if provider == "openai" else "",
            "fallback_used": fallback_used,
            "result": result,
            "error": "Tool plan validation failed.",
        }

    executed = []
    for call in plan.get("tool_plan", []):
        executed.append(
            execute_tool(
                call.get("tool_name", ""),
                call.get("arguments", {}),
                mission_id=mission_id,
                root_dir=root_dir,
            )
        )

    final = _build_final_result(plan, executed, mission_context, raw_goal)
    _append_planner_event(mission_id=mission_id, user_goal=user_goal, plan=plan, executed=executed, provider=provider)
    _append_evidence_ledger_event(
        mission_context=mission_context,
        mission_id=mission_id,
        user_goal=user_goal,
        plan=plan,
        executed=executed,
    )
    response = {
        "ok": True,
        "provider": provider,
        "model": model if provider == "openai" else "",
        "fallback_used": fallback_used,
        "result": final,
    }
    if fallback_reason:
        response["fallback_reason"] = fallback_reason
    return response


def validate_tool_plan(plan: dict[str, Any], requested_mission_id: str) -> dict[str, Any]:
    errors = []
    try:
        plan = coerce_tool_plan(plan)
    except Exception as exc:
        return {"valid": False, "errors": [f"Invalid tool plan schema: {exc}"], "plan": {}}

    calls = plan.get("tool_plan", [])
    if not calls:
        errors.append("tool_plan must not be empty.")

    plan_text = json.dumps(plan, ensure_ascii=False).lower()
    for keyword in DANGEROUS_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", plan_text):
            errors.append(f"Dangerous keyword rejected: {keyword}")

    for index, call in enumerate(calls):
        tool_name = call.get("tool_name", "")
        arguments = call.get("arguments", {}) or {}
        if tool_name not in WHITE_LISTED_TOOLS:
            errors.append(f"Unknown or non-white-listed tool at index {index}: {tool_name}")
            continue
        allowed_args = ALLOWED_TOOL_ARGUMENTS.get(tool_name, set())
        extra_args = set(arguments.keys()) - allowed_args
        if extra_args:
            errors.append(f"Tool {tool_name} has unsupported arguments: {sorted(extra_args)}")
        arg_mission_id = arguments.get("mission_id")
        if arg_mission_id is not None and str(arg_mission_id) != str(requested_mission_id):
            errors.append(f"Tool {tool_name} mission_id mismatch: {arg_mission_id}")

    return {"valid": not errors, "errors": errors, "plan": plan}


def _selected_provider_name() -> str:
    llm_enabled = os.getenv("LLM_ENABLE", "false").strip().lower() in {"1", "true", "yes", "on"}
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if not llm_enabled or provider != "openai" or not os.getenv("OPENAI_API_KEY"):
        return "mock"
    return "openai"


def _ask_openai_for_plan(user_goal: str, mission_context: dict[str, Any], model: str) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    payload = {
        "model": model,
        "instructions": PLANNER_SYSTEM_PROMPT,
        "input": build_planner_user_prompt(user_goal, mission_context, list(WHITE_LISTED_TOOLS)),
        "text": {
            "format": {
                "type": "json_schema",
                "name": TOOL_PLAN_JSON_SCHEMA["name"],
                "strict": TOOL_PLAN_JSON_SCHEMA["strict"],
                "schema": TOOL_PLAN_JSON_SCHEMA["schema"],
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


def _mock_plan(user_goal: str, mission_id: str, mission_context: dict[str, Any]) -> dict[str, Any]:
    lower = user_goal.lower()
    safe_goal = sanitize_text(user_goal)
    tool_plan = [
        {
            "tool_name": "load_mission_result",
            "arguments": {"mission_id": mission_id},
            "reason": "Load mission structured results.",
        },
        {
            "tool_name": "read_evidence_ledger",
            "arguments": {"mission_id": mission_id},
            "reason": "Check evidence events and review requirements.",
        },
        {
            "tool_name": "load_ec_terp_result",
            "arguments": {"mission_id": mission_id},
            "reason": "Load auxiliary priority evidence.",
        },
        {
            "tool_name": "validate_authenticity_boundaries",
            "arguments": {"text": safe_goal},
            "reason": "Validate user goal against authenticity boundaries.",
        },
    ]
    if "report" in lower:
        tool_plan.append(
            {
                "tool_name": "generate_mission_report",
                "arguments": {"mission_id": mission_id},
                "reason": "Generate an auxiliary mission report if requested.",
            }
        )
    if any(term in lower for term in ["survivor", "gps", "route", "temperature", "human_candidate", "why", "priority", "confirm"]):
        tool_plan.append(
            {
                "tool_name": "ask_mission_copilot",
                "arguments": {"mission_id": mission_id, "question": safe_goal},
                "reason": "Use evidence-grounded copilot for the user question.",
            }
        )
    return {
        "intent": _infer_intent(lower),
        "tool_plan": tool_plan,
        "expected_output": "Evidence-grounded mission planning summary for manual review.",
        "human_review_required": True,
    }


def _infer_intent(lower_goal: str) -> str:
    if "priority" in lower_goal or "manual review" in lower_goal:
        return "manual_review_prioritization"
    if "survivor" in lower_goal or "confirm" in lower_goal:
        return "authenticity_boundary_review"
    if "route" in lower_goal or "gps" in lower_goal:
        return "path_boundary_review"
    return "mission_evidence_review"


def _build_final_result(plan: dict[str, Any], executed: list[dict[str, Any]], mission_context: dict[str, Any], user_goal: str) -> dict[str, Any]:
    evidence_used = []
    for item in executed:
        if item.get("status") == "success":
            evidence_used.append({"source": item.get("tool_name", "unknown"), "summary": item.get("result_summary", "")})
    if not evidence_used:
        evidence_used.append({"source": "unavailable", "summary": "No tool evidence was available; manual review is required."})

    successful_tools = [item.get("tool_name") for item in executed if item.get("status") == "success"]
    final_response = (
        "Mission planner executed white-listed evidence tools for the requested goal. "
        f"Successful tools: {', '.join(successful_tools) if successful_tools else 'none'}. "
        "The result is auxiliary decision support only. Prioritize manual review using the cited mission evidence, especially EC-TERP, ledger, and candidate-target records when available. "
        "Evidence is insufficient for field confirmation or navigation without human review."
    )
    guarded = apply_copilot_guardrails(
        {
            "answer": final_response,
            "evidence_used": evidence_used,
            "limitations": mission_context.get("limitations", []),
            "human_review_required": True,
            "confidence_note": "Planner output is based only on white-listed tool results.",
        },
        evidence_context=mission_context,
    )
    return coerce_planner_result(
        {
            "intent": plan.get("intent", "mission_evidence_review"),
            "tool_plan": plan.get("tool_plan", []),
            "executed_tools": [
                {
                    "tool_name": item.get("tool_name", ""),
                    "status": item.get("status", "skipped"),
                    "result_summary": item.get("result_summary", ""),
                    "error": item.get("error"),
                }
                for item in executed
            ],
            "final_response": guarded.get("answer", ""),
            "evidence_used": guarded.get("evidence_used", []),
            "limitations": guarded.get("limitations", []),
            "human_review_required": True,
        }
    )


def _safe_rejection_result(plan: dict[str, Any], errors: list[str], mission_context: dict[str, Any]) -> dict[str, Any]:
    guarded = apply_copilot_guardrails(
        {
            "answer": "Tool plan validation failed. No tools were executed. Output remains decision support only and requires human review.",
            "evidence_used": [{"source": "tool_plan_validator", "summary": "; ".join(errors)}],
            "limitations": mission_context.get("limitations", []),
            "human_review_required": True,
            "confidence_note": "Planner rejected unsafe or invalid tool plan.",
        },
        evidence_context=mission_context,
    )
    return coerce_planner_result(
        {
            "intent": str((plan or {}).get("intent", "invalid_tool_plan")),
            "tool_plan": (plan or {}).get("tool_plan", []),
            "executed_tools": [],
            "final_response": guarded.get("answer", ""),
            "evidence_used": guarded.get("evidence_used", []),
            "limitations": guarded.get("limitations", []),
            "human_review_required": True,
        }
    )


def _append_planner_event(mission_id: str, user_goal: str, plan: dict[str, Any], executed: list[dict[str, Any]], provider: str):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    event_path = REPORT_DIR / "mission_planner_events.json"
    try:
        events = json.loads(event_path.read_text(encoding="utf-8")) if event_path.exists() else []
        if not isinstance(events, list):
            events = []
    except Exception:
        events = []
    events.append(
        {
            "event_type": "llm_mission_planner_executed",
            "mission_id": mission_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "LLM Tool-Orchestrated Mission Planner",
            "user_goal": sanitize_text(user_goal),
            "tools_requested": [call.get("tool_name") for call in plan.get("tool_plan", [])],
            "tools_executed": [item.get("tool_name") for item in executed if item.get("status") == "success"],
            "provider": provider,
            "human_review_required": True,
            "message": "LLM planner generated and executed a white-listed mission tool plan based on available evidence. Output is decision-support only and requires human review.",
        }
    )
    event_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_evidence_ledger_event(
    mission_context: dict[str, Any],
    mission_id: str,
    user_goal: str,
    plan: dict[str, Any],
    executed: list[dict[str, Any]],
):
    mission_root = Path(mission_context.get("mission_root") or ROOT_DIR)
    ledger_path = mission_root / "outputs" / "reports" / "mission_evidence_ledger.json"
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
    events.append(
        {
            "event_type": "llm_mission_planner_executed",
            "mission_id": mission_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "LLM Tool-Orchestrated Mission Planner",
            "source_module": "LLM Tool-Orchestrated Mission Planner",
            "user_goal": sanitize_text(user_goal),
            "tools_requested": [call.get("tool_name") for call in plan.get("tool_plan", [])],
            "tools_executed": [item.get("tool_name") for item in executed if item.get("status") == "success"],
            "human_review_required": True,
            "message": "LLM planner generated and executed a white-listed mission tool plan based on available evidence. Output is decision-support only and requires human review.",
        }
    )
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
