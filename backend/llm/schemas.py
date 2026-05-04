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
