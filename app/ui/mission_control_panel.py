"""Mission Control panel for one-click S1-S9 demo missions.

The panel runs the demo orchestrator and formats mission outputs for Gradio.
It does not require YOLO weights, ODM/Docker, real thermal hardware, or LLM
APIs. Demo outputs are explicitly marked as workflow demonstration artifacts.
"""

from __future__ import annotations

from pathlib import Path


MISSION_CONTROL_INTRO = """
## Mission Control / 一键任务演示

AeroRescue-AI Mission Control 用于展示完整的低空无人机灾情智能辅助决策流程。该一键演示使用 demo/mock/imported 数据跑通 S1-S9 工作流，仅用于流程验证和比赛展示，不代表真实救援现场结论。
""".strip()

FINAL_REPORT_PREVIEW_NOTICE = (
    "以下报告为 AI 辅助决策报告，不构成最终救援结论。"
    "所有候选目标、路径建议、热红外证据和优先级排序均需人工复核。"
)

MISSION_CONTROL_TRUTHFULNESS_BOUNDARIES = [
    "Demo data is for workflow demonstration only and is not operational disaster evidence.",
    "Mock/imported detections are not real model inference results.",
    "Simulated Thermal is not real temperature measurement.",
    "Fast Preview is not a real ODM georeferenced orthomosaic.",
    "Uploaded/Demo Mask is not automatic model segmentation.",
    "Image-plane path is not GPS navigation.",
    "AI candidates are not confirmed civilians.",
    "Final Report is an AI-assisted decision-support report and not a final rescue conclusion.",
]

STAGE_ORDER = [
    "global_mapping",
    "macro_analysis",
    "area_tasking",
    "local_recon",
    "target_verification",
    "thermal_check",
    "decision_fusion",
    "rescue_recommendation",
    "evidence_report",
]

STAGE_LABELS = {
    "global_mapping": "S1 高空建图 / global_mapping",
    "macro_analysis": "S2 宏观灾情分析 / macro_analysis",
    "area_tasking": "S3 重点区域划分 / area_tasking",
    "local_recon": "S4 中高空局部精查 / local_recon",
    "target_verification": "S5 低空目标复核 / target_verification",
    "thermal_check": "S6 热红外辅助复查 / thermal_check",
    "decision_fusion": "S7 EC-TERP 决策融合 / decision_fusion",
    "rescue_recommendation": "S8 路径与任务建议 / rescue_recommendation",
    "evidence_report": "S9 证据链与报告 / evidence_report",
}

STAGE_TABLE_HEADERS = [
    "阶段",
    "状态",
    "核心输出",
    "候选/记录数量",
    "是否需要人工复核",
    "真实性边界摘要",
]


def _get_count(result):
    if not isinstance(result, dict):
        return ""
    count_keys = [
        "candidate_count",
        "area_count",
        "verification_count",
        "thermal_target_count",
    ]
    for key in count_keys:
        if key in result:
            return result.get(key)
    nested_counts = [
        ("verification_summary", "verification_count"),
        ("thermal_summary", "thermal_check_count"),
        ("decision_summary", "decision_candidate_count"),
        ("recommendation_summary", "recommendation_count"),
        ("evidence_summary", "total_evidence_count"),
    ]
    for parent, key in nested_counts:
        value = result.get(parent, {}) or {}
        if isinstance(value, dict) and key in value:
            return value.get(key)
    return ""


def _core_output(stage_key, result):
    if not isinstance(result, dict) or not result:
        return "未生成 / unavailable"
    if stage_key == "global_mapping":
        return result.get("base_map_type") or result.get("base_map_path") or "mapping result"
    if stage_key == "macro_analysis":
        return f"macro_zones={len(result.get('macro_zones', []) or [])}"
    if stage_key == "area_tasking":
        return f"area_tasks={len(result.get('area_tasks', []) or [])}"
    if stage_key == "local_recon":
        return f"candidates={result.get('candidate_count', 0)}"
    if stage_key == "target_verification":
        summary = result.get("verification_summary", {}) or {}
        return f"verification_records={summary.get('verification_count', 0)}"
    if stage_key == "thermal_check":
        summary = result.get("thermal_summary", {}) or {}
        return f"thermal_records={summary.get('thermal_check_count', 0)}"
    if stage_key == "decision_fusion":
        summary = result.get("decision_summary", {}) or {}
        return f"decision_candidates={summary.get('decision_candidate_count', 0)}"
    if stage_key == "rescue_recommendation":
        summary = result.get("recommendation_summary", {}) or {}
        return f"recommendations={summary.get('recommendation_count', 0)}"
    if stage_key == "evidence_report":
        return result.get("report_markdown_path") or result.get("report_json_path") or "Final Report 2.0"
    return result.get("result_type") or "available"


def _status_counts(stage_results):
    counts = {"completed": 0, "failed": 0, "degraded": 0}
    for result in (stage_results or {}).values():
        status = (result or {}).get("status")
        if status in counts:
            counts[status] += 1
    return counts


def format_truthfulness_markdown():
    """Return Mission Control truthfulness boundary notes."""
    items = "\n".join(f"- {item}" for item in MISSION_CONTROL_TRUTHFULNESS_BOUNDARIES)
    return f"### 真实性边界说明\n\n{items}"


def format_demo_result_summary(demo_result):
    """Return Markdown summary for a one-click demo result."""
    if not demo_result:
        return "尚未运行一键演示任务。"

    mission = demo_result.get("mission", {}) or {}
    stage_results = demo_result.get("stage_results", {}) or {}
    workflow_summary = demo_result.get("workflow_summary", {}) or {}
    counts = _status_counts(stage_results)
    truthfulness_note = demo_result.get("truthfulness_note", "")
    return f"""
### Demo Mission 结果摘要

- Mission ID: `{mission.get('mission_id', '')}`
- Mission Name: {mission.get('mission_name', '')}
- Mission Dir: `{demo_result.get('mission_dir', '')}`
- Demo Dataset Dir: `{demo_result.get('demo_dataset_dir', '')}`
- Final Report Markdown: `{demo_result.get('final_report_markdown_path', '')}`
- Evidence Ledger: `{demo_result.get('evidence_ledger_path', '')}`
- Workflow Completed Stages: `{workflow_summary.get('completed_stage_count', counts.get('completed', 0))}`
- Workflow Failed Stages: `{workflow_summary.get('failed_stage_count', counts.get('failed', 0))}`
- Stage Result Status: completed={counts.get('completed', 0)}, degraded={counts.get('degraded', 0)}, failed={counts.get('failed', 0)}

**Truthfulness Warning:** {truthfulness_note}
""".strip()


def format_stage_result_table(stage_results):
    """Return rows for a Gradio Dataframe summarizing S1-S9 stage results."""
    rows = []
    stage_results = stage_results or {}
    for stage_key in STAGE_ORDER:
        result = stage_results.get(stage_key, {}) or {}
        rows.append(
            [
                STAGE_LABELS[stage_key],
                result.get("status", "missing"),
                _core_output(stage_key, result),
                _get_count(result),
                "是" if result.get("human_review_required", True) else "否",
                result.get("truthfulness_note", "") or "未生成 / unavailable",
            ]
        )
    return rows


def format_candidate_summary(stage_results):
    """Return Markdown for candidate, verification, thermal, EC-TERP, and route summaries."""
    stage_results = stage_results or {}
    local_recon = stage_results.get("local_recon") or {}
    target_verification = stage_results.get("target_verification") or {}
    thermal_check = stage_results.get("thermal_check") or {}
    decision_fusion = stage_results.get("decision_fusion") or {}
    rescue_recommendation = stage_results.get("rescue_recommendation") or {}

    return f"""
### Candidate / Thermal / EC-TERP / Route 摘要

- S4 local_recon candidate_count: `{local_recon.get('candidate_count', '未生成 / unavailable')}`
- S5 verification_summary: `{target_verification.get('verification_summary', '未生成 / unavailable')}`
- S6 thermal_summary: `{thermal_check.get('thermal_summary', '未生成 / unavailable')}`
- S7 decision_summary: `{decision_fusion.get('decision_summary', '未生成 / unavailable')}`
- S8 recommendation_summary: `{rescue_recommendation.get('recommendation_summary', '未生成 / unavailable')}`
""".strip()


def load_final_report_preview(report_markdown_path, max_chars=6000):
    """Load a bounded Final Report Markdown preview."""
    if not report_markdown_path:
        return "Final Report 尚未生成。"
    path = Path(report_markdown_path)
    if not path.exists():
        return "Final Report 尚未生成。"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Final Report 读取失败：{exc}"
    max_chars = int(max_chars or 6000)
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...（报告内容较长，已截断预览。）"
    return text


def run_one_click_demo_from_ui(missions_root, mission_name):
    """Run the one-click demo mission and return Gradio output values."""
    try:
        try:
            from ..demo.one_click_mission_orchestrator import run_one_click_demo_mission
        except ImportError:  # pragma: no cover - app.py runs with app/ on sys.path.
            from demo.one_click_mission_orchestrator import run_one_click_demo_mission

        missions_root = missions_root or "outputs/demo_missions"
        mission_name = mission_name or "AeroRescue-AI One-Click Demo Mission"
        demo_result = run_one_click_demo_mission(
            missions_root=missions_root,
            mission_name=mission_name,
            demo_output_root=str(Path(missions_root) / "_demo_dataset"),
        )
        preview = load_final_report_preview(demo_result.get("final_report_markdown_path", ""))
        final_report_preview = f"### Final Report Preview\n\n**{FINAL_REPORT_PREVIEW_NOTICE}**\n\n{preview}"
        return (
            format_demo_result_summary(demo_result),
            format_stage_result_table(demo_result.get("stage_results", {})),
            format_candidate_summary(demo_result.get("stage_results", {})),
            final_report_preview,
            format_truthfulness_markdown(),
        )
    except Exception as exc:
        error_summary = f"### Mission Control 运行失败\n\n错误：`{exc}`\n\nDemo 编排失败不会代表真实救援流程失败，请检查本地依赖和输出目录权限。"
        return (
            error_summary,
            [],
            "未生成 / unavailable",
            "Final Report 尚未生成。",
            format_truthfulness_markdown(),
        )


def attach_mission_control_panel():
    """Attach Mission Control components to the active Gradio context."""
    import gradio as gr

    gr.Markdown(MISSION_CONTROL_INTRO)
    with gr.Row():
        with gr.Column(scale=2):
            mission_name = gr.Textbox(
                label="Mission Name",
                value="AeroRescue-AI One-Click Demo Mission",
            )
        with gr.Column(scale=2):
            missions_root = gr.Textbox(
                label="Missions Root",
                value="outputs/demo_missions",
            )
        with gr.Column(scale=1):
            run_button = gr.Button("Run One-Click Demo Mission", variant="primary")

    summary_markdown = gr.Markdown("尚未运行一键演示任务。")
    stage_table = gr.Dataframe(
        headers=STAGE_TABLE_HEADERS,
        value=format_stage_result_table({}),
        label="S1-S9 Stage Results",
        interactive=False,
        wrap=True,
    )
    candidate_summary = gr.Markdown("未生成 / unavailable")
    final_report_preview = gr.Markdown(f"### Final Report Preview\n\n**{FINAL_REPORT_PREVIEW_NOTICE}**\n\nFinal Report 尚未生成。")
    truthfulness_markdown = gr.Markdown(format_truthfulness_markdown())

    run_button.click(
        fn=run_one_click_demo_from_ui,
        inputs=[missions_root, mission_name],
        outputs=[
            summary_markdown,
            stage_table,
            candidate_summary,
            final_report_preview,
            truthfulness_markdown,
        ],
    )

    return {
        "mission_name": mission_name,
        "missions_root": missions_root,
        "run_button": run_button,
        "summary": summary_markdown,
        "stage_table": stage_table,
        "candidate_summary": candidate_summary,
        "final_report_preview": final_report_preview,
        "truthfulness": truthfulness_markdown,
    }
