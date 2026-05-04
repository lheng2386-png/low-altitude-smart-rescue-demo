import re
from typing import Any


DEFAULT_LIMITATIONS = [
    "Answer is based only on available mission evidence.",
    "Evidence boundaries prevent confirming field outcomes without human review.",
    "All outputs require human review before operational use.",
]


FORBIDDEN_PATTERNS = [
    (re.compile(r"confirmed\s+civilian", re.IGNORECASE), "unverified human candidate"),
    (re.compile(r"confirmed\s+survivor", re.IGNORECASE), "unverified human candidate"),
    (re.compile(r"confirmed\s+casualty", re.IGNORECASE), "unverified casualty-related conclusion"),
    (re.compile(r"victim\s+confirmed", re.IGNORECASE), "victim status unverified"),
    (re.compile(r"rescued\s+person", re.IGNORECASE), "unverified human target"),
    (re.compile(r"real\s+GPS\s+route", re.IGNORECASE), "image-plane reference path"),
    (re.compile(r"GPS\s+navigation\s+route", re.IGNORECASE), "image-plane reference path"),
    (re.compile(r"GPS\s+route", re.IGNORECASE), "image-plane reference path"),
    (re.compile(r"measured\s+temperature", re.IGNORECASE), "simulated thermal cue unless radiometric evidence is available"),
    (re.compile(r"actual\s+body\s+temperature", re.IGNORECASE), "unverified thermal cue"),
    (re.compile(r"real\s+temperature\s+matrix", re.IGNORECASE), "radiometric temperature matrix only if explicitly available"),
    (re.compile(r"real\s+temperature", re.IGNORECASE), "radiometric temperature only if explicitly available"),
    (re.compile(r"thermal\s+camera\s+confirmed", re.IGNORECASE), "thermal evidence requires review"),
    (re.compile(r"georeferenced\s+route", re.IGNORECASE), "image-plane reference path"),
    (re.compile(r"rescue\s+navigation\s+route", re.IGNORECASE), "image-plane reference path"),
    (re.compile(r"real\s+ODM\s+orthomosaic", re.IGNORECASE), "ODM output only if explicitly available"),
    (re.compile(r"georeferenced\s+orthomosaic", re.IGNORECASE), "georeferencing unverified unless ODM evidence is available"),
    (re.compile(r"survey-grade\s+orthomosaic", re.IGNORECASE), "preview orthomosaic"),
    (re.compile(r"mapping-grade\s+result", re.IGNORECASE), "preview result"),
    (re.compile(r"model-generated\s+segmentation", re.IGNORECASE), "supplied risk-area mask unless model evidence is available"),
    (re.compile(r"automatically\s+segmented\s+by\s+model", re.IGNORECASE), "segmentation source requires verification"),
    (re.compile(r"verified\s+segmentation\s+result", re.IGNORECASE), "segmentation result requiring review"),
    (re.compile(r"rescue\s+conclusion", re.IGNORECASE), "decision-support note"),
]


def sanitize_text(value: Any) -> str:
    text = str(value or "")
    for pattern, replacement in FORBIDDEN_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def sanitize_list(items: Any) -> list[str]:
    if not isinstance(items, list):
        items = [items] if items else []
    return [sanitize_text(item) for item in items if str(item or "").strip()]


def apply_copilot_guardrails(result: dict[str, Any], evidence_context: dict[str, Any] | None = None) -> dict[str, Any]:
    result = dict(result or {})
    result["answer"] = sanitize_text(result.get("answer", ""))
    result["human_review_required"] = True
    result["confidence_note"] = sanitize_text(
        result.get("confidence_note") or "Answer is based only on available mission evidence."
    )

    evidence_used = result.get("evidence_used")
    if not isinstance(evidence_used, list):
        evidence_used = []
    sanitized_evidence = []
    for item in evidence_used:
        if isinstance(item, dict):
            sanitized_evidence.append(
                {
                    "source": sanitize_text(item.get("source", "unknown")),
                    "summary": sanitize_text(item.get("summary", "")),
                }
            )
        elif str(item or "").strip():
            sanitized_evidence.append({"source": "unknown", "summary": sanitize_text(item)})
    if not sanitized_evidence:
        sanitized_evidence.append(
            {
                "source": "unavailable",
                "summary": "No specific evidence item was returned by the model; answer requires manual review.",
            }
        )
    result["evidence_used"] = sanitized_evidence

    limitations = sanitize_list(result.get("limitations"))
    for limitation in _context_limitations(evidence_context):
        if limitation not in limitations:
            limitations.append(limitation)
    if not limitations:
        limitations = list(DEFAULT_LIMITATIONS)
    result["limitations"] = limitations
    return result


def _context_limitations(evidence_context: dict[str, Any] | None) -> list[str]:
    limitations = []
    for item in (evidence_context or {}).get("limitations", []) or []:
        text = sanitize_text(item)
        if text and text not in limitations:
            limitations.append(text)
    if not limitations:
        limitations.extend(DEFAULT_LIMITATIONS)
    return limitations
