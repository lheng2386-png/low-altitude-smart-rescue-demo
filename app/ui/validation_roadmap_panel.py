"""Validation Roadmap UI panel for 灾情感知及影响评估.

This panel is a static planning surface. It does not train models, download
datasets, run YOLO, run ODM, parse real thermal files, or call LLM APIs.
"""

from __future__ import annotations

try:
    from ..roadmap.validation_roadmap import (
        CAPABILITY_LAYERS,
        LIGHTWEIGHT_CAPABILITY_NOTES,
        NEXT_PHASE_ORDER,
        ROADMAP_TRUTHFULNESS_REMINDERS,
        VALIDATION_TASKS,
    )
except ImportError:  # pragma: no cover - supports app.py running with app/ on sys.path.
    from roadmap.validation_roadmap import (
        CAPABILITY_LAYERS,
        LIGHTWEIGHT_CAPABILITY_NOTES,
        NEXT_PHASE_ORDER,
        ROADMAP_TRUTHFULNESS_REMINDERS,
        VALIDATION_TASKS,
    )


ROADMAP_INTRO = """
## 真实能力验证路线图

本面板用于记录 灾情感知及影响评估 后续真实能力验证计划。当前系统已经具备 S1-S9 工作流闭环，但部分 AI 能力仍处于演示、模拟、导入或轻量验证状态。后续需要通过数据集、模型权重、指标和消融实验逐步验证。
""".strip()

VALIDATION_TASK_HEADERS = [
    "任务编号",
    "任务",
    "类别",
    "优先级",
    "当前状态",
    "当前说明",
    "成功标准摘要",
    "真实性边界",
]

STATUS_LABELS = {
    "completed": "已完成",
    "in_progress": "进行中",
    "pending": "待验证",
    "blocked": "受阻",
}

PRIORITY_LABELS = {
    "Must": "必须补齐",
    "High": "高优先级",
    "Medium": "中优先级",
    "Optional": "可选增强",
}

CATEGORY_LABELS = {
    "dataset": "数据集",
    "model_validation": "模型验证",
    "algorithm_validation": "算法验证",
    "mapping_validation": "建图验证",
    "thermal_validation": "热红外验证",
    "integration": "工程接入",
    "product_ui": "产品界面",
    "infrastructure": "基础设施",
}


def _join_items(items):
    if not items:
        return ""
    if isinstance(items, str):
        return items
    return "; ".join(str(item) for item in items)


def _status_label(value):
    return STATUS_LABELS.get(str(value or ""), str(value or ""))


def _priority_label(value):
    return PRIORITY_LABELS.get(str(value or ""), str(value or ""))


def _category_label(value):
    return CATEGORY_LABELS.get(str(value or ""), str(value or ""))


def format_capability_layers_markdown():
    """Return Markdown describing the three capability layers."""
    lines = ["### 当前能力三层状态", ""]
    for layer in CAPABILITY_LAYERS:
        lines.extend(
            [
                f"#### {layer.get('title_zh', '')}",
                f"- 状态：{_status_label(layer.get('status', ''))}",
                f"- 已包含能力：{_join_items(layer.get('items', []))}",
                f"- 说明：{layer.get('note', '')}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def format_validation_tasks_table(tasks=None):
    """Return rows for the roadmap validation tasks table."""
    rows = []
    for task in tasks or VALIDATION_TASKS:
        rows.append(
            [
                task.get("task_id", ""),
                task.get("title_zh", ""),
                _category_label(task.get("category", "")),
                _priority_label(task.get("priority", "")),
                _status_label(task.get("current_status", "")),
                task.get("current_state_note", ""),
                _join_items(task.get("success_criteria", [])),
                task.get("truthfulness_boundary", ""),
            ]
        )
    return rows


def format_priority_tasks_markdown(priority="Must"):
    """Return Markdown for tasks at one priority level."""
    priority = str(priority or "Must")
    selected = [task for task in VALIDATION_TASKS if task.get("priority") == priority]
    lines = [f"### {_priority_label(priority)}任务", ""]
    if not selected:
        lines.append("暂无。")
        return "\n".join(lines)
    for task in selected:
        lines.extend(
            [
                f"- **{task.get('title_zh', '')}** (`{task.get('task_id', '')}`)",
                f"  - 类别：{_category_label(task.get('category', ''))}",
                f"  - 目标产物：{_join_items(task.get('target_deliverables', []))}",
                f"  - 成功标准：{_join_items(task.get('success_criteria', []))}",
                f"  - 边界：{task.get('truthfulness_boundary', '')}",
            ]
        )
    return "\n".join(lines)


def format_lightweight_notes_markdown():
    """Return Markdown documenting lightweight/demo capability boundaries."""
    lines = ["### 轻量能力可以继续使用，但必须保留边界", ""]
    for item in LIGHTWEIGHT_CAPABILITY_NOTES:
        lines.extend(
            [
                f"- **{item.get('capability', '')}**：{item.get('note', '')}",
                f"  - 边界：{item.get('truthfulness_boundary', '')}",
            ]
        )
    return "\n".join(lines)


def format_next_phase_order_markdown():
    """Return Markdown with the recommended next phase order."""
    lines = ["### 推荐后续推进顺序", ""]
    for index, item in enumerate(NEXT_PHASE_ORDER, start=1):
        lines.append(f"{index}. {item}")
    return "\n".join(lines)


def format_truthfulness_reminder_markdown():
    """Return fixed roadmap truthfulness reminders."""
    lines = ["### 真实性提醒", ""]
    lines.extend(f"- {item}" for item in ROADMAP_TRUTHFULNESS_REMINDERS)
    return "\n".join(lines)


def attach_validation_roadmap_panel():
    """Attach the Validation Roadmap components to the current Gradio context."""
    import gradio as gr

    gr.Markdown(ROADMAP_INTRO)
    with gr.Accordion("1. 当前能力层级", open=False):
        gr.Markdown(format_capability_layers_markdown())
    with gr.Accordion("2. 后续验证任务清单", open=False):
        gr.Dataframe(
            headers=VALIDATION_TASK_HEADERS,
            value=format_validation_tasks_table(),
            label="真实能力验证任务",
            interactive=False,
            wrap=True,
        )
    with gr.Accordion("3. 必须补齐任务", open=False):
        gr.Markdown(format_priority_tasks_markdown("Must"))
    with gr.Accordion("4. 高优先级任务", open=False):
        gr.Markdown(format_priority_tasks_markdown("High"))
    with gr.Accordion("5. 轻量能力边界", open=False):
        gr.Markdown(format_lightweight_notes_markdown())
    with gr.Accordion("6. 推荐推进顺序", open=False):
        gr.Markdown(format_next_phase_order_markdown())
    with gr.Accordion("7. 真实性提醒", open=False):
        gr.Markdown(format_truthfulness_reminder_markdown())
