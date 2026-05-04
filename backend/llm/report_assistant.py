import os
import re
from typing import Any

from .mock_provider import MockProvider, build_truthfulness_limitations
from .openai_provider import OpenAIProvider
from .schemas import coerce_report


FORBIDDEN_REPLACEMENTS = [
    (re.compile(r"confirmed\s+civilian", re.IGNORECASE), "human_candidate requiring review"),
    (re.compile(r"\bis\s+(?:a\s+)?real\s+temperature\s+measurement\b", re.IGNORECASE), "is a simulated thermal visualization unless radiometric data explicitly proves otherwise"),
    (re.compile(r"属于真实测温|是真实测温"), "属于模拟热红外可视化，除非 radiometric 数据明确证明"),
    (re.compile(r"\bis\s+(?:a\s+)?real\s+ODM\s+orthomosaic\b", re.IGNORECASE), "is a Fast Preview / preview artifact unless ODM output is explicitly present"),
    (re.compile(r"是真实\s*ODM\s*正射|是专业正射影像"), "是 Fast Preview / 预览产物，除非明确存在 ODM 输出"),
    (re.compile(r"\bis\s+(?:a\s+)?GPS\s+navigation\s+route\b", re.IGNORECASE), "is an image-plane reference path"),
    (re.compile(r"是\s*GPS\s*导航路线"), "是图像平面参考路径"),
    (re.compile(r"\bis\s+(?:an?\s+)?automatic\s+model\s+segmentation\b", re.IGNORECASE), "is an uploaded/demo mask unless automatic model output is explicitly present"),
    (re.compile(r"是模型自动分割"), "是上传或演示 mask，除非明确存在自动模型输出"),
]


def generate_mission_report(mission_result: dict[str, Any]) -> dict[str, Any]:
    """Generate a bounded LLM-assisted mission report with mock fallback."""
    mission_result = mission_result or {}
    provider = _select_provider()
    fallback_used = False
    error = None
    try:
        report = provider.generate_mission_report(mission_result)
    except Exception as exc:
        fallback_used = True
        error = str(exc)
        provider = MockProvider()
        report = provider.generate_mission_report(mission_result)

    report = _apply_truthfulness_guardrails(coerce_report(report), mission_result)
    report["provider"] = provider.name
    report["fallback_used"] = fallback_used
    if error:
        report["fallback_reason"] = error
    report["truthfulness_guardrails"] = [
        "LLM is post-processing only, not a detection or decision engine.",
        "No invented checkpoints, metrics, GPS coordinates, temperature matrices, ODM outputs, or field rescue conclusions.",
    ]
    return report


def _select_provider():
    llm_enabled = os.getenv("LLM_ENABLE", "false").strip().lower() in {"1", "true", "yes", "on"}
    provider_name = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if not llm_enabled or provider_name != "openai" or not os.getenv("OPENAI_API_KEY"):
        return MockProvider()
    return OpenAIProvider()


def _apply_truthfulness_guardrails(report: dict[str, Any], mission_result: dict[str, Any]) -> dict[str, Any]:
    required_limitations = build_truthfulness_limitations(mission_result)
    existing = [str(item) for item in report.get("limitations", []) if str(item).strip()]
    lower_existing = "\n".join(existing).lower()
    for limitation in required_limitations:
        key_terms = [word for word in ["human_candidate", "simulated thermal", "fast preview", "image-plane", "uploaded", "post-processing"] if word in limitation.lower()]
        if not key_terms or not all(term in lower_existing for term in key_terms[:1]):
            existing.append(limitation)
            lower_existing += "\n" + limitation.lower()
    report["limitations"] = existing
    report["human_review_required"] = True

    for field in ["mission_summary", "risk_interpretation", "report_paragraph"]:
        report[field] = _sanitize_text(report.get(field, ""))
    report["limitations"] = [_sanitize_text(item) for item in report["limitations"]]
    report["recommended_next_actions"] = [_sanitize_text(item) for item in report.get("recommended_next_actions", [])]
    return report


def _sanitize_text(value: Any) -> str:
    text = str(value or "")
    for pattern, replacement in FORBIDDEN_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text
