from typing import Any

from pydantic import BaseModel, Field


class AuditIssue(BaseModel):
    issue_id: str = Field(default="AUDIT-000")
    severity: str = Field(default="low")
    category: str = Field(default="unsupported_claim")
    location: str = Field(default="unknown")
    problem: str = Field(default="")
    evidence_reference: str = Field(default="")
    suggested_fix: str = Field(default="")
    auto_fix_allowed: bool = Field(default=False)


class SafeRewriteSuggestion(BaseModel):
    location: str = Field(default="unknown")
    original_text: str = Field(default="")
    suggested_text: str = Field(default="")
    reason: str = Field(default="")


class EvidenceAuditResult(BaseModel):
    audit_status: str = Field(default="pass")
    overall_risk_level: str = Field(default="low")
    issues: list[AuditIssue] = Field(default_factory=list)
    safe_rewrite_suggestions: list[SafeRewriteSuggestion] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    positive_checks: list[str] = Field(default_factory=list)
    human_review_required: bool = Field(default=True)


AUDIT_JSON_SCHEMA: dict[str, Any] = {
    "name": "aerorescue_evidence_audit",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "audit_status": {"type": "string"},
            "overall_risk_level": {"type": "string"},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "issue_id": {"type": "string"},
                        "severity": {"type": "string"},
                        "category": {"type": "string"},
                        "location": {"type": "string"},
                        "problem": {"type": "string"},
                        "evidence_reference": {"type": "string"},
                        "suggested_fix": {"type": "string"},
                        "auto_fix_allowed": {"type": "boolean"},
                    },
                    "required": [
                        "issue_id",
                        "severity",
                        "category",
                        "location",
                        "problem",
                        "evidence_reference",
                        "suggested_fix",
                        "auto_fix_allowed",
                    ],
                },
            },
            "safe_rewrite_suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "location": {"type": "string"},
                        "original_text": {"type": "string"},
                        "suggested_text": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["location", "original_text", "suggested_text", "reason"],
                },
            },
            "missing_evidence": {"type": "array", "items": {"type": "string"}},
            "positive_checks": {"type": "array", "items": {"type": "string"}},
            "human_review_required": {"type": "boolean"},
        },
        "required": [
            "audit_status",
            "overall_risk_level",
            "issues",
            "safe_rewrite_suggestions",
            "missing_evidence",
            "positive_checks",
            "human_review_required",
        ],
    },
}


def coerce_audit_result(payload: dict[str, Any]) -> dict[str, Any]:
    return EvidenceAuditResult.model_validate(payload or {}).model_dump()
