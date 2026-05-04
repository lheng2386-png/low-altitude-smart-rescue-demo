import os
import re
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .mock_provider import MockProvider, build_truthfulness_limitations
from .openai_provider import OpenAIProvider
from .schemas import coerce_report
from .guardrails import sanitize_text


REPORT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "reports"


FORBIDDEN_REPLACEMENTS = [
    (re.compile(r"confirmed\s+civilian", re.IGNORECASE), "human_candidate requiring review"),
    (re.compile(r"confirmed\s+survivor", re.IGNORECASE), "human_candidate requiring review"),
    (re.compile(r"confirmed\s+casualty", re.IGNORECASE), "unverified casualty-related conclusion"),
    (re.compile(r"victim\s+confirmed", re.IGNORECASE), "victim status unverified"),
    (re.compile(r"rescued\s+person", re.IGNORECASE), "unverified human target"),
    (re.compile(r"measured\s+temperature", re.IGNORECASE), "simulated thermal cue unless radiometric evidence is available"),
    (re.compile(r"real\s+temperature\s+matrix", re.IGNORECASE), "radiometric temperature matrix only if explicitly available"),
    (re.compile(r"actual\s+body\s+temperature", re.IGNORECASE), "unverified thermal cue"),
    (re.compile(r"thermal\s+camera\s+confirmed", re.IGNORECASE), "thermal evidence requires review"),
    (re.compile(r"\bis\s+(?:a\s+)?real\s+temperature\s+measurement\b", re.IGNORECASE), "is a simulated thermal visualization unless radiometric data explicitly proves otherwise"),
    (re.compile(r"属于真实测温|是真实测温"), "属于模拟热红外可视化，除非 radiometric 数据明确证明"),
    (re.compile(r"\bis\s+(?:a\s+)?real\s+ODM\s+orthomosaic\b", re.IGNORECASE), "is a Fast Preview / preview artifact unless ODM output is explicitly present"),
    (re.compile(r"是真实\s*ODM\s*正射|是专业正射影像"), "是 Fast Preview / 预览产物，除非明确存在 ODM 输出"),
    (re.compile(r"\bis\s+(?:a\s+)?GPS\s+navigation\s+route\b", re.IGNORECASE), "is an image-plane reference path"),
    (re.compile(r"GPS\s+route", re.IGNORECASE), "image-plane reference path"),
    (re.compile(r"georeferenced\s+route", re.IGNORECASE), "image-plane reference path"),
    (re.compile(r"rescue\s+navigation\s+route", re.IGNORECASE), "image-plane reference path"),
    (re.compile(r"是\s*GPS\s*导航路线"), "是图像平面参考路径"),
    (re.compile(r"real\s+ODM\s+orthomosaic", re.IGNORECASE), "ODM output only if explicitly available"),
    (re.compile(r"georeferenced\s+orthomosaic", re.IGNORECASE), "georeferencing unverified unless ODM evidence is available"),
    (re.compile(r"survey-grade\s+orthomosaic", re.IGNORECASE), "preview orthomosaic"),
    (re.compile(r"mapping-grade\s+result", re.IGNORECASE), "preview result"),
    (re.compile(r"\bis\s+(?:an?\s+)?automatic\s+model\s+segmentation\b", re.IGNORECASE), "is an uploaded/demo mask unless automatic model output is explicitly present"),
    (re.compile(r"model-generated\s+segmentation", re.IGNORECASE), "supplied risk-area mask unless model evidence is available"),
    (re.compile(r"automatically\s+segmented\s+by\s+model", re.IGNORECASE), "segmentation source requires verification"),
    (re.compile(r"verified\s+segmentation\s+result", re.IGNORECASE), "segmentation result requiring review"),
    (re.compile(r"是模型自动分割"), "是上传或演示 mask，除非明确存在自动模型输出"),
    (re.compile(r"real\s+rescue\s+conclusion", re.IGNORECASE), "decision-support note"),
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
    if getattr(provider, "model", None):
        report["model"] = provider.model
    report["fallback_used"] = fallback_used
    if error:
        report["fallback_reason"] = error
    report["truthfulness_guardrails"] = [
        "LLM is post-processing only, not a detection or decision engine.",
        "No invented checkpoints, metrics, GPS coordinates, temperature matrices, ODM outputs, or field rescue conclusions.",
    ]
    _append_report_event(report)
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
    return sanitize_text(text)


def _append_report_event(report: dict[str, Any]):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    event_path = REPORT_DIR / "llm_report_events.json"
    try:
        events = json.loads(event_path.read_text(encoding="utf-8")) if event_path.exists() else []
        if not isinstance(events, list):
            events = []
    except Exception:
        events = []
    events.append(
        {
            "event_type": "llm_report_generated",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "LLM Mission Report Assistant",
            "human_review_required": True,
            "provider": report.get("provider", "unknown"),
            "fallback_used": bool(report.get("fallback_used")),
            "message": "LLM Mission Report Assistant generated an auxiliary explanation that requires human review.",
        }
    )
    event_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
