import json
import os
from typing import Any

import gradio as gr
import requests


TRUTHFULNESS_NOTICE = """
### Truthfulness Boundaries

- human_candidate is not confirmed civilian
- simulated thermal is not real temperature measurement
- Fast Preview is not a real ODM orthomosaic
- image-plane path is not a GPS navigation route
- all outputs require human review
""".strip()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict(orient="records")
        except TypeError:
            return value.to_dict()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def build_mission_result_from_ui(
    rescue_report_text,
    transformer_summary,
    segmentation_status,
    scene_gate_status,
    damage_summary,
    scene_mode,
    rescue_entry,
    path_gate,
    path_reliability,
    target_details,
    segmentation_summary,
    risk_ranking,
    terp_ranking,
    path_summary,
    path_comparison,
):
    return {
        "source": "gradio_target_detection_result_page",
        "rescue_report_text": rescue_report_text or "",
        "transformer_summary": transformer_summary or "",
        "segmentation_status": segmentation_status or "",
        "scene_gate_status": scene_gate_status or "",
        "damage_summary": damage_summary or "",
        "scene_mode": scene_mode or "",
        "rescue_entry": rescue_entry or "",
        "path_gate": path_gate or "",
        "path_reliability": path_reliability or "",
        "targets": _jsonable(target_details),
        "segmentation_summary": _jsonable(segmentation_summary),
        "risk_ranking": _jsonable(risk_ranking),
        "terp_ranking": _jsonable(terp_ranking),
        "path_summary": path_summary or "",
        "path_comparison": path_comparison or "",
        "truthfulness_boundary": TRUTHFULNESS_NOTICE,
    }


def request_llm_mission_report(*ui_values):
    mission_result = build_mission_result_from_ui(*ui_values)
    if not any(str(value or "").strip() for value in mission_result.values() if not isinstance(value, (list, dict))):
        return _empty_outputs("Please run a mission analysis first, then generate the AI mission report.")

    api_url = os.getenv("LLM_REPORT_API_URL", "http://127.0.0.1:7860/api/llm/mission-report")
    try:
        response = requests.post(api_url, json={"mission_result": mission_result}, timeout=60)
        response.raise_for_status()
        report = response.json()
    except Exception:
        return _empty_outputs("LLM report unavailable. Please check backend or API key.")

    provider = report.get("provider", "unknown")
    model = report.get("model") or report.get("openai_model") or ""
    provider_text = f"Provider: {provider}" + (f"\nModel: {model}" if model else "")
    fallback_note = ""
    if provider == "mock" or report.get("fallback_used"):
        fallback_note = "Mock/fallback report is displayed. This is template-based decision-support text, not model evidence."
        if report.get("fallback_reason"):
            fallback_note += f"\nFallback reason: {report.get('fallback_reason')}"

    limitations = _warning_list(report.get("limitations", []), fallback_note=fallback_note)
    actions = _bullet_list(report.get("recommended_next_actions", []))
    review = "Human Review Required" if report.get("human_review_required", True) else "Human review status not flagged by backend"
    included = "Included in Report Draft: available for Final Report V2 integration."

    return (
        "AI mission report generated.",
        provider_text,
        review,
        report.get("mission_summary", ""),
        report.get("risk_interpretation", ""),
        limitations,
        actions,
        report.get("report_paragraph", ""),
        included,
        json.dumps(report, ensure_ascii=False, indent=2),
    )


def _empty_outputs(message):
    return (
        message,
        "",
        "",
        "",
        "",
        _warning_list([]),
        "",
        "",
        "Included in Report Draft: not available until a report is generated.",
        "{}",
    )


def _bullet_list(items):
    if not items:
        return ""
    return "\n".join(f"- {item}" for item in items)


def _warning_list(items, fallback_note=""):
    lines = [TRUTHFULNESS_NOTICE]
    if fallback_note:
        lines.append(f"\n### Fallback Notice\n\n{fallback_note}")
    if items:
        lines.append("\n### Backend Limitations\n")
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def attach_llm_report_panel(inputs):
    with gr.Group():
        gr.Markdown("## AI Mission Report Assistant")
        gr.Markdown(TRUTHFULNESS_NOTICE)
        generate_btn = gr.Button("Generate AI Mission Report", variant="secondary")
        with gr.Row():
            with gr.Column():
                status = gr.Textbox(label="Status", lines=2)
                provider = gr.Textbox(label="Provider / Model", lines=2)
                human_review = gr.Textbox(label="Review Gate", lines=1)
                included_status = gr.Textbox(label="Final Report V2", lines=1, value="Included in Report Draft: not generated yet.")
            with gr.Column():
                mission_summary = gr.Textbox(label="mission_summary", lines=4)
                risk_interpretation = gr.Textbox(label="risk_interpretation", lines=4)
        limitations = gr.Markdown(label="limitations")
        recommended_next_actions = gr.Markdown(label="recommended_next_actions")
        report_paragraph = gr.Textbox(label="report_paragraph", lines=6)
        raw_json = gr.Code(label="Structured LLM Report JSON", language="json", lines=14)

    generate_btn.click(
        fn=request_llm_mission_report,
        inputs=inputs,
        outputs=[
            status,
            provider,
            human_review,
            mission_summary,
            risk_interpretation,
            limitations,
            recommended_next_actions,
            report_paragraph,
            included_status,
            raw_json,
        ],
    )
