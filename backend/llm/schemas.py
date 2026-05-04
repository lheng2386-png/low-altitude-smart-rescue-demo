from typing import Any

from pydantic import BaseModel, Field


class MissionReport(BaseModel):
    mission_summary: str = Field(default="")
    risk_interpretation: str = Field(default="")
    human_review_required: bool = Field(default=True)
    limitations: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    report_paragraph: str = Field(default="")


REPORT_JSON_SCHEMA: dict[str, Any] = {
    "name": "aerorescue_mission_report",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mission_summary": {"type": "string"},
            "risk_interpretation": {"type": "string"},
            "human_review_required": {"type": "boolean"},
            "limitations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "recommended_next_actions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "report_paragraph": {"type": "string"},
        },
        "required": [
            "mission_summary",
            "risk_interpretation",
            "human_review_required",
            "limitations",
            "recommended_next_actions",
            "report_paragraph",
        ],
    },
}


def coerce_report(payload: dict[str, Any]) -> dict[str, Any]:
    return MissionReport.model_validate(payload or {}).model_dump()


class CopilotEvidenceUsed(BaseModel):
    source: str = Field(default="unavailable")
    summary: str = Field(default="")


class MissionCopilotResult(BaseModel):
    answer: str = Field(default="")
    evidence_used: list[CopilotEvidenceUsed] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    human_review_required: bool = Field(default=True)
    confidence_note: str = Field(default="Answer is based only on available mission evidence.")


COPILOT_JSON_SCHEMA: dict[str, Any] = {
    "name": "aerorescue_mission_copilot_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string"},
            "evidence_used": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "source": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["source", "summary"],
                },
            },
            "limitations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "human_review_required": {"type": "boolean"},
            "confidence_note": {"type": "string"},
        },
        "required": [
            "answer",
            "evidence_used",
            "limitations",
            "human_review_required",
            "confidence_note",
        ],
    },
}


def coerce_copilot_result(payload: dict[str, Any]) -> dict[str, Any]:
    return MissionCopilotResult.model_validate(payload or {}).model_dump()


class LLMToolCall(BaseModel):
    tool_name: str = Field(default="")
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(default="")


class LLMToolPlan(BaseModel):
    intent: str = Field(default="mission_evidence_review")
    tool_plan: list[LLMToolCall] = Field(default_factory=list)
    expected_output: str = Field(default="Evidence-grounded decision-support summary.")
    human_review_required: bool = Field(default=True)


class LLMToolExecutionResult(BaseModel):
    tool_name: str = Field(default="")
    status: str = Field(default="skipped")
    result_summary: str = Field(default="")
    error: str | None = Field(default=None)


class MissionPlannerFinalResult(BaseModel):
    intent: str = Field(default="mission_evidence_review")
    tool_plan: list[LLMToolCall] = Field(default_factory=list)
    executed_tools: list[LLMToolExecutionResult] = Field(default_factory=list)
    final_response: str = Field(default="")
    evidence_used: list[CopilotEvidenceUsed] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    human_review_required: bool = Field(default=True)


TOOL_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "name": "aerorescue_llm_tool_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {"type": "string"},
            "tool_plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "tool_name": {"type": "string"},
                        "arguments": {"type": "object", "additionalProperties": True},
                        "reason": {"type": "string"},
                    },
                    "required": ["tool_name", "arguments", "reason"],
                },
            },
            "expected_output": {"type": "string"},
            "human_review_required": {"type": "boolean"},
        },
        "required": ["intent", "tool_plan", "expected_output", "human_review_required"],
    },
}


def coerce_tool_plan(payload: dict[str, Any]) -> dict[str, Any]:
    return LLMToolPlan.model_validate(payload or {}).model_dump()


def coerce_planner_result(payload: dict[str, Any]) -> dict[str, Any]:
    return MissionPlannerFinalResult.model_validate(payload or {}).model_dump()
