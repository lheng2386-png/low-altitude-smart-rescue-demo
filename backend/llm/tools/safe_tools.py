from pathlib import Path
from typing import Any

from ..evidence_context import build_mission_evidence_context
from ..guardrails import apply_copilot_guardrails, sanitize_text
from ..mission_copilot import answer_mission_copilot_question
from ..report_assistant import generate_mission_report


ALLOWED_TOOL_ARGUMENTS = {
    "load_mission_result": {"mission_id"},
    "read_evidence_ledger": {"mission_id"},
    "load_llm_report": {"mission_id"},
    "load_ec_terp_result": {"mission_id"},
    "validate_authenticity_boundaries": {"text", "structured_result"},
    "generate_mission_report": {"mission_id", "mission_result"},
    "ask_mission_copilot": {"mission_id", "question"},
}


WHITE_LISTED_TOOLS = tuple(ALLOWED_TOOL_ARGUMENTS.keys())


def execute_tool(tool_name: str, arguments: dict[str, Any], mission_id: str, root_dir: str | Path | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    context = build_mission_evidence_context(mission_id, root_dir=root_dir)
    evidence = context.get("evidence", {})

    try:
        if tool_name == "load_mission_result":
            return _success(tool_name, _item_summary(evidence.get("mission_result")), evidence.get("mission_result"))
        if tool_name == "read_evidence_ledger":
            return _success(tool_name, _item_summary(evidence.get("evidence_ledger")), evidence.get("evidence_ledger"))
        if tool_name == "load_llm_report":
            return _success(tool_name, _item_summary(evidence.get("saved_llm_report")), evidence.get("saved_llm_report"))
        if tool_name == "load_ec_terp_result":
            return _success(tool_name, _item_summary(evidence.get("ec_terp_result")), evidence.get("ec_terp_result"))
        if tool_name == "validate_authenticity_boundaries":
            text = arguments.get("text") or arguments.get("structured_result") or ""
            checked = apply_copilot_guardrails(
                {
                    "answer": sanitize_text(text),
                    "evidence_used": [{"source": "validate_authenticity_boundaries", "summary": "Boundary validation executed."}],
                    "limitations": context.get("limitations", []),
                    "human_review_required": True,
                    "confidence_note": "Validation checks known authenticity boundaries.",
                },
                evidence_context=context,
            )
            unsafe = sanitize_text(text) != str(text or "")
            return _success(
                tool_name,
                "safe after guardrails" if not unsafe else "unsafe phrases corrected by guardrails",
                {"safe": True, "unsafe_phrases_corrected": unsafe, "corrected_text": checked.get("answer"), "limitations": checked.get("limitations", [])},
            )
        if tool_name == "generate_mission_report":
            mission_result = arguments.get("mission_result")
            if not isinstance(mission_result, dict):
                mission_result = (evidence.get("mission_result") or {}).get("data") or {"mission_id": mission_id, "evidence_context": context}
            report = generate_mission_report(mission_result)
            return _success(tool_name, "LLM mission report generated as auxiliary explanation.", report)
        if tool_name == "ask_mission_copilot":
            question = arguments.get("question") or "Summarize mission evidence for manual review."
            answer = answer_mission_copilot_question(mission_id, question, root_dir=root_dir)
            return _success(tool_name, "Mission Evidence Copilot answered using available evidence.", answer)
    except Exception as exc:
        return {
            "tool_name": tool_name,
            "status": "failed",
            "result_summary": "",
            "error": sanitize_text(str(exc)),
            "raw_result": None,
        }
    return {
        "tool_name": tool_name,
        "status": "skipped",
        "result_summary": "",
        "error": f"Tool {tool_name} is not implemented.",
        "raw_result": None,
    }


def _item_summary(item: dict[str, Any] | None) -> str:
    item = item or {}
    status = item.get("status", "unavailable")
    summary = item.get("summary", "Evidence item unavailable.")
    return f"{status}: {summary}"


def _success(tool_name: str, summary: str, raw_result: Any) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "status": "success",
        "result_summary": sanitize_text(summary),
        "error": None,
        "raw_result": raw_result,
    }
