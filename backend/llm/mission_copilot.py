import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .evidence_context import build_mission_evidence_context
from .guardrails import apply_copilot_guardrails, sanitize_text
from .prompts import COPILOT_SYSTEM_PROMPT, build_copilot_user_prompt
from .schemas import COPILOT_JSON_SCHEMA, coerce_copilot_result


ROOT_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT_DIR / "outputs" / "reports"


def answer_mission_copilot_question(mission_id: str | None, question: str | None, root_dir: str | Path | None = None) -> dict[str, Any]:
    mission_id = str(mission_id or "current_mission").strip() or "current_mission"
    raw_question = str(question or "").strip()
    question = sanitize_text(raw_question).strip()
    evidence_context = build_mission_evidence_context(mission_id, root_dir=root_dir)
    provider_name = _selected_provider_name()
    model = os.getenv("OPENAI_MODEL", "gpt-5.5")
    fallback_used = False
    fallback_reason = ""

    try:
        if provider_name == "openai":
            result = _ask_openai(question, evidence_context, model=model)
            provider = "openai"
        else:
            result = _mock_answer(raw_question, evidence_context)
            provider = "mock"
    except Exception as exc:
        fallback_used = True
        fallback_reason = str(exc)
        result = _mock_answer(raw_question, evidence_context)
        provider = "mock"

    result = apply_copilot_guardrails(coerce_copilot_result(result), evidence_context=evidence_context)
    _append_copilot_event(
        mission_id=mission_id,
        question=question,
        result=result,
        provider=provider,
        fallback_used=fallback_used,
    )
    response = {
        "ok": True,
        "provider": provider,
        "model": model if provider == "openai" else "",
        "fallback_used": fallback_used,
        "result": result,
    }
    if fallback_reason:
        response["fallback_reason"] = fallback_reason
    return response


def _selected_provider_name() -> str:
    llm_enabled = os.getenv("LLM_ENABLE", "false").strip().lower() in {"1", "true", "yes", "on"}
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if not llm_enabled or provider != "openai" or not os.getenv("OPENAI_API_KEY"):
        return "mock"
    return "openai"


def _ask_openai(question: str, evidence_context: dict[str, Any], model: str) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    payload = {
        "model": model,
        "instructions": COPILOT_SYSTEM_PROMPT,
        "input": build_copilot_user_prompt(question, evidence_context),
        "text": {
            "format": {
                "type": "json_schema",
                "name": COPILOT_JSON_SCHEMA["name"],
                "strict": COPILOT_JSON_SCHEMA["strict"],
                "schema": COPILOT_JSON_SCHEMA["schema"],
            }
        },
    }
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
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


def _mock_answer(question: str, evidence_context: dict[str, Any]) -> dict[str, Any]:
    evidence = evidence_context.get("evidence", {}) or {}
    available = [
        (source, item)
        for source, item in evidence.items()
        if isinstance(item, dict) and item.get("status") == "available"
    ]
    evidence_used = [
        {"source": source, "summary": item.get("summary", "Available mission evidence.")}
        for source, item in available[:5]
    ]
    if not evidence_used:
        evidence_used = [
            {
                "source": "unavailable",
                "summary": "No specific mission artifact was available for this question.",
            }
        ]

    lower_question = question.lower()
    if any(term in lower_question for term in ["survivor", "civilian", "victim", "casualty"]):
        answer = (
            "Evidence is insufficient to verify a person status or field outcome. "
            "The current evidence can only support an unverified human_candidate or related target cue if that appears in the mission artifacts."
        )
    elif "temperature" in lower_question or "thermal" in lower_question:
        answer = (
            "Evidence is insufficient to provide a radiometric temperature unless radiometric thermal evidence is explicitly available. "
            "If the mission only contains simulated thermal output, it should be treated as a visualization cue."
        )
    elif "gps" in lower_question or "route" in lower_question or "path" in lower_question:
        answer = (
            "Evidence is insufficient to provide a real navigation route. "
            "Available path outputs, when present, are image-plane reference paths for review."
        )
    elif "real rescue target" in lower_question or ("conclude" in lower_question and "rescue" in lower_question):
        answer = (
            "The system output is decision-support only. Evidence is insufficient to verify a field rescue target or operational outcome without human review."
        )
    elif "high priority" in lower_question or "priority" in lower_question:
        answer = (
            "Priority can only be explained from available detection, risk ranking, TERP, EC-TERP, path, and ledger evidence. "
            "Review the cited evidence items and treat the answer as decision support."
        )
    else:
        answer = (
            "This answer is based only on the available mission evidence package. "
            "If the cited evidence is unavailable or incomplete, evidence is insufficient for a stronger conclusion."
        )

    return {
        "answer": answer,
        "evidence_used": evidence_used,
        "limitations": evidence_context.get("limitations", []),
        "human_review_required": True,
        "confidence_note": "Answer is based only on available mission evidence.",
    }


def _append_copilot_event(mission_id: str, question: str, result: dict[str, Any], provider: str, fallback_used: bool):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    event_path = REPORT_DIR / "mission_copilot_events.json"
    try:
        events = json.loads(event_path.read_text(encoding="utf-8")) if event_path.exists() else []
        if not isinstance(events, list):
            events = []
    except Exception:
        events = []
    events.append(
        {
            "event_type": "llm_copilot_query",
            "source": "Mission Evidence Copilot",
            "mission_id": mission_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "evidence_used": result.get("evidence_used", []),
            "human_review_required": True,
            "provider": provider,
            "fallback_used": bool(fallback_used),
            "message": "Mission Evidence Copilot answered a user question as auxiliary explanation based on available mission evidence; requires human review.",
        }
    )
    event_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
