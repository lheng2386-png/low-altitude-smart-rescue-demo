from typing import Any

from .base import LLMProvider
from .schemas import coerce_report


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            parts.extend(_flatten_strings(key))
            parts.extend(_flatten_strings(item))
        return parts
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            parts.extend(_flatten_strings(item))
        return parts
    if value is None:
        return []
    return [str(value)]


def extract_evidence_flags(mission_result: dict[str, Any]) -> dict[str, bool]:
    text = " ".join(_flatten_strings(mission_result)).lower()
    return {
        "human_candidate": "human_candidate" in text,
        "simulated_thermal": "simulated thermal" in text or "simulated_thermal" in text or "thermal mode simulated" in text or "mode simulated" in text,
        "fast_preview": "fast preview" in text or "fast_preview" in text or "image_stitch_preview" in text or "opencv 拼接预览" in text,
        "image_plane_path": "image-plane" in text or "image plane" in text or "image_plane_path" in text or "图像平面" in text,
        "uploaded_or_demo_mask": "uploaded mask" in text or "uploaded_mask" in text or "demo mask" in text or "demo_mask" in text or "demo/uploaded mask" in text or "demo_fallback" in text,
        "auto_segmentation": "auto segmentation" in text or "auto_model" in text or "automatic segmentation" in text,
    }


def build_truthfulness_limitations(mission_result: dict[str, Any]) -> list[str]:
    flags = extract_evidence_flags(mission_result)
    limitations = [
        "This LLM report is a post-processing draft only; it does not replace detection, measurement, routing, or human command review."
    ]
    if flags["human_candidate"]:
        limitations.append("human_candidate is an auxiliary candidate and is not verified as a civilian or survivor without human review.")
    if flags["simulated_thermal"]:
        limitations.append("Simulated thermal output is based on image-derived visualization and is not a radiometric temperature measurement.")
    if flags["fast_preview"]:
        limitations.append("Fast Preview / OpenCV stitching is a preview artifact and is not an ODM mapping artifact.")
    if flags["image_plane_path"]:
        limitations.append("image-plane path output is a reference path in image coordinates and is not a navigation route.")
    if flags["uploaded_or_demo_mask"]:
        limitations.append("Demo/uploaded mask is a supplied risk-area mask and is not necessarily model-generated segmentation.")
    return limitations


class MockProvider(LLMProvider):
    name = "mock"

    def generate_mission_report(self, mission_result: dict[str, Any]) -> dict[str, Any]:
        flags = extract_evidence_flags(mission_result)
        target_count = mission_result.get("target_count")
        if target_count is None:
            targets = mission_result.get("targets")
            target_count = len(targets) if isinstance(targets, list) else "unknown"

        limitations = build_truthfulness_limitations(mission_result)
        actions = [
            "Review the original imagery and module JSON outputs before field use.",
            "Prioritize manual verification for high-risk targets and uncertain candidates.",
            "Record which modules were simulated, preview-only, failed, or not executed.",
        ]
        if flags["human_candidate"]:
            actions.insert(1, "Manually verify every human_candidate before using it as a rescue-priority fact.")
        if flags["simulated_thermal"]:
            actions.append("Use radiometric thermal data before making any temperature-based conclusion.")

        report = {
            "mission_summary": f"Mission result contains {target_count} target record(s) or candidate record(s) in the supplied payload.",
            "risk_interpretation": "Current risk interpretation should be treated as decision-support evidence derived from existing module outputs, not as an autonomous rescue conclusion.",
            "human_review_required": True,
            "limitations": limitations,
            "recommended_next_actions": actions,
            "report_paragraph": (
                "AeroRescue-AI produced an auxiliary mission report draft from the supplied result JSON. "
                "The draft highlights available targets, risk cues, and follow-up actions while preserving evidence boundaries. "
                "Any candidate, preview, simulated, uploaded, or image-plane result requires operator review before operational use."
            ),
        }
        return coerce_report(report)
