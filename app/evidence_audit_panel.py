import json
import os

import gradio as gr
import requests


def run_evidence_audit_panel(mission_id, audit_target):
    mission_id = str(mission_id or "current_mission").strip() or "current_mission"
    audit_target = str(audit_target or "all").strip() or "all"
    api_url = os.getenv("LLM_AUDIT_API_URL", "http://127.0.0.1:7860/api/llm/evidence-audit")
    try:
        response = requests.post(api_url, json={"mission_id": mission_id, "audit_target": audit_target}, timeout=60)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return "Evidence audit unavailable. Please check backend or API key.", "", "", "[]", "{}"
    result = payload.get("result", {}) or {}
    issues = result.get("issues", []) or []
    issue_lines = [
        f"- {item.get('issue_id')}: {item.get('severity')} / {item.get('category')} / {item.get('location')} - {item.get('problem')}"
        for item in issues
    ]
    summary = (
        f"Audit status: {result.get('audit_status', 'unknown')}\n"
        f"Risk level: {result.get('overall_risk_level', 'unknown')}\n"
        f"Issues: {len(issues)}\n"
        "Human Review Required"
    )
    provider = f"Provider: {payload.get('provider', 'unknown')}"
    if payload.get("model"):
        provider += f"\nModel: {payload.get('model')}"
    if payload.get("fallback_used"):
        provider += "\nFallback: deterministic/mock audit was used."
    return (
        "Evidence audit completed." if payload.get("ok") else "Evidence audit returned a safe error.",
        summary,
        provider,
        "\n".join(issue_lines) if issue_lines else "No issues found.",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def attach_evidence_audit_panel():
    with gr.Group():
        gr.Markdown("## LLM Evidence Auditor / Consistency Critic")
        gr.Markdown("Audits evidence consistency and authenticity boundaries. Suggestions require human review and do not overwrite source outputs.")
        mission_id = gr.Textbox(label="Mission ID", value="current_mission")
        audit_target = gr.Dropdown(
            ["all", "llm_report", "final_report_v2", "copilot", "planner", "evidence_ledger"],
            label="Audit target",
            value="all",
        )
        run_btn = gr.Button("Run Evidence Audit", variant="secondary")
        status = gr.Textbox(label="Status", lines=1)
        summary = gr.Textbox(label="Audit Summary", lines=5)
        provider = gr.Textbox(label="Provider / Fallback", lines=3)
        issues = gr.Textbox(label="Issues / Suggested Fixes", lines=10)
        raw_json = gr.Code(label="Evidence Audit JSON", language="json", lines=14)
    run_btn.click(
        fn=run_evidence_audit_panel,
        inputs=[mission_id, audit_target],
        outputs=[status, summary, provider, issues, raw_json],
    )
