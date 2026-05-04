import json
import os

import gradio as gr
import requests


def run_mission_planner(mission_id, user_goal):
    mission_id = str(mission_id or "current_mission").strip() or "current_mission"
    user_goal = str(user_goal or "").strip()
    if not user_goal:
        return "Please enter a mission planning goal.", "", "", "", "{}"
    api_url = os.getenv("LLM_PLANNER_API_URL", "http://127.0.0.1:7860/api/llm/mission-planner")
    try:
        response = requests.post(api_url, json={"mission_id": mission_id, "user_goal": user_goal}, timeout=60)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return "Mission planner unavailable. Please check backend or API key.", "", "", "", "{}"
    result = payload.get("result", {}) or {}
    tool_plan = "\n".join(f"- {item.get('tool_name')}: {item.get('reason', '')}" for item in result.get("tool_plan", []))
    executed = "\n".join(f"- {item.get('tool_name')}: {item.get('status')} - {item.get('result_summary', '')}" for item in result.get("executed_tools", []))
    provider = f"Provider: {payload.get('provider', 'unknown')}"
    if payload.get("model"):
        provider += f"\nModel: {payload.get('model')}"
    if payload.get("fallback_used"):
        provider += "\nFallback: mock response was used."
    return (
        "Mission planner completed." if payload.get("ok") else "Mission planner returned a safe error.",
        provider,
        tool_plan,
        executed + "\n\nFinal response:\n" + result.get("final_response", ""),
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def attach_mission_planner_panel():
    with gr.Group():
        gr.Markdown("## LLM Tool-Orchestrated Mission Planner")
        gr.Markdown("Planner can only execute white-listed backend tools. Output is decision support and requires human review.")
        mission_id = gr.Textbox(label="Mission ID", value="current_mission")
        user_goal = gr.Textbox(
            label="Mission planning goal",
            value="Analyze this mission and identify which area should be prioritized for manual review.",
            lines=3,
        )
        run_btn = gr.Button("Run Mission Planner", variant="secondary")
        status = gr.Textbox(label="Status", lines=2)
        provider = gr.Textbox(label="Provider / Fallback", lines=3)
        tool_plan = gr.Markdown(label="tool_plan")
        executed = gr.Textbox(label="executed_tools / final_response", lines=10)
        raw_json = gr.Code(label="Mission Planner JSON", language="json", lines=14)
    run_btn.click(
        fn=run_mission_planner,
        inputs=[mission_id, user_goal],
        outputs=[status, provider, tool_plan, executed, raw_json],
    )
