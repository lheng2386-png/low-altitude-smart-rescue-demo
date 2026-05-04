import json
import os

import gradio as gr
import requests


COPILOT_BOUNDARIES = """
Mission Evidence Copilot answers only from available mission evidence.
It cannot verify field outcomes, real temperatures, or GPS navigation.
All outputs require human review.
""".strip()


def ask_mission_copilot(mission_id, question):
    mission_id = str(mission_id or "current_mission").strip() or "current_mission"
    question = str(question or "").strip()
    if not question:
        return _empty_outputs("Please enter a question about current mission evidence.")

    api_url = os.getenv("LLM_COPILOT_API_URL", "http://127.0.0.1:7860/api/llm/mission-copilot")
    try:
        response = requests.post(
            api_url,
            json={"mission_id": mission_id, "question": question},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return _empty_outputs("Mission copilot unavailable. Please check backend or API key.")

    result = payload.get("result", {}) or {}
    provider_text = f"Provider: {payload.get('provider', 'unknown')}"
    if payload.get("model"):
        provider_text += f"\nModel: {payload.get('model')}"
    if payload.get("fallback_used"):
        provider_text += "\nFallback: mock response was used."
        if payload.get("fallback_reason"):
            provider_text += f"\nReason: {payload.get('fallback_reason')}"

    limitations = result.get("limitations", []) or []
    answer = result.get("answer", "")
    status = "Evidence insufficient or unavailable." if _is_insufficient(answer, limitations) else "Copilot answer generated."
    return (
        status,
        provider_text,
        "Human Review Required" if result.get("human_review_required", True) else "Human review status forced by backend",
        answer,
        _format_evidence_used(result.get("evidence_used", [])),
        _format_limitations(limitations),
        result.get("confidence_note", ""),
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def _empty_outputs(message):
    return (
        message,
        "",
        "",
        "",
        "",
        _format_limitations([COPILOT_BOUNDARIES]),
        "",
        "{}",
    )


def _is_insufficient(answer, limitations):
    text = (str(answer or "") + "\n" + "\n".join(str(item) for item in limitations)).lower()
    return "insufficient" in text or "unavailable" in text


def _format_evidence_used(items):
    if not items:
        return "- unavailable: No specific evidence item was returned; manual review required."
    lines = []
    for item in items:
        if isinstance(item, dict):
            lines.append(f"- {item.get('source', 'unknown')}: {item.get('summary', '')}")
        else:
            lines.append(f"- {item}")
    return "\n".join(lines)


def _format_limitations(items):
    lines = [f"### Caution\n\n{COPILOT_BOUNDARIES}"]
    if items:
        lines.append("\n### Limitations\n")
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def set_example_question(question):
    return question


def attach_mission_copilot_panel():
    with gr.Group():
        gr.Markdown("## Mission Evidence Copilot")
        gr.Markdown(COPILOT_BOUNDARIES)
        mission_id = gr.Textbox(label="Mission ID", value="current_mission", lines=1)
        question = gr.Textbox(
            label="Ask about current mission evidence...",
            lines=3,
            placeholder="Ask about current mission evidence...",
        )
        with gr.Row():
            q_priority = gr.Button("Why is this area high priority?")
            q_human = gr.Button("What evidence supports the human_candidate?")
            q_thermal = gr.Button("What are the limitations of the thermal result?")
            q_review = gr.Button("Does this report require human review?")
        ask_btn = gr.Button("Ask", variant="secondary")
        with gr.Row():
            with gr.Column():
                status = gr.Textbox(label="Status", lines=2)
                provider = gr.Textbox(label="Provider / Model / Fallback", lines=4)
                human_review = gr.Textbox(label="Review Gate", lines=1)
                confidence_note = gr.Textbox(label="confidence_note", lines=2)
            with gr.Column():
                answer = gr.Textbox(label="answer", lines=8)
                evidence_used = gr.Markdown(label="evidence_used")
        limitations = gr.Markdown(label="limitations")
        raw_json = gr.Code(label="Mission Copilot JSON", language="json", lines=14)

    for button, example in [
        (q_priority, "Why is this area high priority?"),
        (q_human, "What evidence supports the human_candidate?"),
        (q_thermal, "What are the limitations of the thermal result?"),
        (q_review, "Does this report require human review?"),
    ]:
        button.click(fn=set_example_question, inputs=[gr.State(example)], outputs=[question])

    ask_btn.click(
        fn=ask_mission_copilot,
        inputs=[mission_id, question],
        outputs=[status, provider, human_review, answer, evidence_used, limitations, confidence_note, raw_json],
    )
