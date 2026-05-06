"""Mission Dashboard panel for the 灾情感知及影响评估 rescue workflow.

This module only formats and displays mission/workflow state. It does not run
YOLO, ODM, thermal analysis, segmentation, EC-TERP, path planning, or reports.
"""

from __future__ import annotations

from html import escape

try:
    from ..workflow.stage_definitions import RESCUE_WORKFLOW_STAGES, get_stage_definition
    from ..workflow.workflow_state import create_initial_workflow_state
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from workflow.stage_definitions import RESCUE_WORKFLOW_STAGES, get_stage_definition
    from workflow.workflow_state import create_initial_workflow_state


WORKFLOW_DASHBOARD_INTRO = """
## 任务总览

灾情感知及影响评估 按真实救援任务组织：高空建图 → 灾情感知 → 重点区域 → 局部精查 → 目标复核 → 热红外复查 → 决策融合 → 路径建议 → 证据报告。默认页面只展示主线状态，详细模块结果在各阶段页或折叠区查看。
"""

DASHBOARD_TRUTHFULNESS_BOUNDARIES = [
    "Fast Preview 不等于真实 ODM 正射影像。",
    "Simulated Thermal 不等于真实热红外测温。",
    "RGB/JPG/PNG 不能生成真实 temperature_matrix。",
    "Uploaded/Demo Mask 不等于自动语义分割模型结果。",
    "Image-plane path 不等于 GPS 导航路线。",
    "AI 检测到的人只是 human_candidate，不是 confirmed civilian。",
    "Final Report 是辅助决策报告，不是最终救援结论。",
]

WORKFLOW_STAGE_TABLE_HEADERS = [
    "阶段编号",
    "阶段名称",
    "真实现场动作",
    "无人机/数据层级",
    "系统模块",
    "输出",
    "状态",
    "是否需要人工复核",
    "真实性边界",
]

STATUS_ICONS = {
    "completed": "✅",
    "ready": "🟡",
    "running": "🔵",
    "pending": "⚪",
    "skipped": "⏭️",
    "failed": "❌",
    "completed_external": "📥",
    "manual_required": "📝",
}


def _workflow_state_or_default(workflow_state):
    """Return a workflow state dictionary, creating the default when absent."""
    if workflow_state:
        return workflow_state
    return create_initial_workflow_state()


def _join_values(value):
    """Format lists and dictionaries for compact Markdown/table display."""
    if value is None or value == "":
        return ""
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if isinstance(item, dict):
                detail = item.get("reason") or item.get("status") or item.get("message") or item
                parts.append(f"{key}: {detail}")
            else:
                parts.append(f"{key}: {item}")
        return ", ".join(str(part) for part in parts)
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            if isinstance(item, dict):
                module = item.get("module") or item.get("name") or item.get("key") or "unknown"
                reason = item.get("reason") or item.get("message") or item.get("status") or ""
                parts.append(f"{module}: {reason}" if reason else str(module))
            else:
                parts.append(str(item))
        return ", ".join(parts)
    return str(value)


def format_workflow_stage_table(workflow_state):
    """Return rows for a gr.Dataframe showing all nine rescue workflow stages."""
    state = _workflow_state_or_default(workflow_state)
    stage_states = state.get("stages", {})
    rows = []
    for definition in RESCUE_WORKFLOW_STAGES:
        stage_key = definition["stage_key"]
        runtime_state = stage_states.get(stage_key, {})
        status = runtime_state.get("status", "pending")
        human_review_required = runtime_state.get(
            "human_review_required",
            definition.get("required_human_review", True),
        )
        rows.append(
            [
                definition["stage_id"],
                f"{definition['stage_name_zh']} / {stage_key}",
                definition["real_action"],
                definition["uav_layer"],
                _join_values(definition["system_modules"]),
                _join_values(definition["outputs"]),
                status,
                "是" if human_review_required else "否",
                runtime_state.get("truthfulness_boundary") or definition["truthfulness_boundary"],
            ]
        )
    return rows


def format_workflow_stage_cards_html(workflow_state):
    """Return a compact HTML card view for the nine-stage workflow."""
    state = _workflow_state_or_default(workflow_state)
    stage_states = state.get("stages", {})
    cards = []
    for definition in RESCUE_WORKFLOW_STAGES:
        stage_key = definition["stage_key"]
        status = stage_states.get(stage_key, {}).get("status", "pending")
        icon = STATUS_ICONS.get(status, "⚪")
        status_class = escape(str(status).replace("_", "-"))
        cards.append(
            f"""
            <div class="mission-stage-card mission-stage-{status_class}">
                <div class="mission-stage-head">
                    <span class="mission-stage-id">{escape(definition["stage_id"])}</span>
                    <span class="mission-stage-status">{icon} {escape(status)}</span>
                </div>
                <div class="mission-stage-name">{escape(definition["stage_name_zh"])}</div>
                <div class="mission-stage-action">{escape(definition["real_action"])}</div>
            </div>
            """
        )
    return f"<div class='mission-stage-grid'>{''.join(cards)}</div>"


def format_mission_summary_markdown(mission):
    """Return a Markdown mission summary for the dashboard."""
    if not mission:
        return "当前尚未创建任务。请先创建或导入一次救援任务。"

    workflow_state = _workflow_state_or_default(mission.get("workflow_state"))
    current_stage_key = workflow_state.get("current_stage_key", "global_mapping")
    try:
        current_stage = get_stage_definition(current_stage_key)
        current_stage_label = f"{current_stage['stage_id']} {current_stage['stage_name_zh']} / {current_stage_key}"
    except KeyError:
        current_stage_label = current_stage_key

    available_modules = _join_values(mission.get("available_modules", [])) or "暂无"
    disabled_modules = _join_values(mission.get("disabled_modules", [])) or "暂无"
    truthfulness_boundaries = mission.get("truthfulness_boundaries", [])
    truthfulness_text = "\n".join(f"- {item}" for item in truthfulness_boundaries) if truthfulness_boundaries else "- 暂无"

    return f"""
### 任务信息

| 项目 | 内容 |
| --- | --- |
| 任务编号 | `{mission.get('mission_id', '')}` |
| 任务名称 | {mission.get('mission_name', '')} |
| 状态 | `{mission.get('status', '')}` |
| 当前阶段 | {current_stage_label} |
| 可用模块 | {available_modules} |
| 暂不可用模块 | {disabled_modules} |
| 证据链路径 | `{mission.get('evidence_ledger_path', '')}` |

### 任务真实性边界

{truthfulness_text}
""".strip()


def format_workflow_progress_markdown(workflow_state):
    """Return a Markdown progress list for the nine rescue workflow stages."""
    state = _workflow_state_or_default(workflow_state)
    stage_states = state.get("stages", {})
    lines = ["### 流程进度"]
    for definition in RESCUE_WORKFLOW_STAGES:
        stage_key = definition["stage_key"]
        status = stage_states.get(stage_key, {}).get("status", "pending")
        icon = STATUS_ICONS.get(status, "⚪")
        lines.append(f"{icon} **{definition['stage_id']} {definition['stage_name_zh']}** `{status}`")
    return "\n".join(lines)


def format_dashboard_truthfulness_markdown():
    """Return the fixed dashboard truthfulness boundary notes."""
    items = "\n".join(f"- {item}" for item in DASHBOARD_TRUTHFULNESS_BOUNDARIES)
    return f"""
### 真实性边界说明

{items}

所有关键阶段输出均应保留人工复核要求；Dashboard 展示的是辅助决策流程状态，不是最终救援结论。
""".strip()


def attach_mission_dashboard_panel(mission=None, workflow_state=None):
    """Attach the Mission Dashboard components to the current Gradio context."""
    import gradio as gr

    state = _workflow_state_or_default(workflow_state or (mission or {}).get("workflow_state"))
    components = {}
    components["intro"] = gr.Markdown(WORKFLOW_DASHBOARD_INTRO)
    components["summary"] = gr.Markdown(format_mission_summary_markdown(mission), elem_classes=["mission-summary-card"])
    components["stage_cards"] = gr.HTML(format_workflow_stage_cards_html(state))
    with gr.Accordion("查看 9 阶段详细表格", open=False):
        components["stage_table"] = gr.Dataframe(
            headers=WORKFLOW_STAGE_TABLE_HEADERS,
            value=format_workflow_stage_table(state),
            label="真实救援任务 9 阶段工作流",
            interactive=False,
            wrap=True,
        )
    with gr.Accordion("真实性边界", open=False):
        components["truthfulness"] = gr.Markdown(format_dashboard_truthfulness_markdown())
    return components
