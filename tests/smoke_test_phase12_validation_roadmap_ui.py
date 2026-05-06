import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.roadmap.validation_roadmap import (  # noqa: E402
    CAPABILITY_LAYERS,
    LIGHTWEIGHT_CAPABILITY_NOTES,
    NEXT_PHASE_ORDER,
    VALIDATION_TASKS,
)
from app.ui.validation_roadmap_panel import (  # noqa: E402
    format_capability_layers_markdown,
    format_lightweight_notes_markdown,
    format_next_phase_order_markdown,
    format_priority_tasks_markdown,
    format_validation_tasks_table,
)


def main():
    assert CAPABILITY_LAYERS
    assert LIGHTWEIGHT_CAPABILITY_NOTES
    assert NEXT_PHASE_ORDER

    task_ids = {task["task_id"] for task in VALIDATION_TASKS}
    expected = {
        "mini_benchmark_dataset",
        "detection_weight_validation",
        "segmentation_validation",
        "ec_terp_ablation",
        "path_planning_comparison",
        "real_odm_validation",
        "thermal_reality_check",
        "human_review_center",
        "evidence_drilldown",
    }
    assert expected.issubset(task_ids)

    must_titles = {task["title_zh"] for task in VALIDATION_TASKS if task["priority"] == "Must"}
    assert "小型验证数据集骨架" in must_titles
    assert "目标检测权重验证" in must_titles
    assert "语义分割模型或 Mask 验证" in must_titles
    assert "EC-TERP 消融实验" in must_titles
    assert "路径规划对比实验" in must_titles

    table = format_validation_tasks_table()
    assert isinstance(table, list)
    assert len(table) == len(VALIDATION_TASKS)
    assert any("图像平面路径不是 GPS 导航路线。" in row[-1] for row in table)

    capability_md = format_capability_layers_markdown()
    priority_md = format_priority_tasks_markdown("Must")
    lightweight_md = format_lightweight_notes_markdown()
    next_phase_md = format_next_phase_order_markdown()
    joined = "\n".join([capability_md, priority_md, lightweight_md, next_phase_md])

    assert "工程工作流闭环" in capability_md
    assert "小型验证数据集" in priority_md
    assert "模拟热红外" in lightweight_md
    assert "目标检测" in next_phase_md
    assert capability_md and priority_md and lightweight_md and next_phase_md

    assert "模拟或导入的检测结果不是真实模型推理结果。" in joined
    assert "模拟热红外不是真实测温。" in joined
    assert "图像平面路径不是 GPS 导航路线。" in joined
    assert "AI 检测到的人只能称为候选目标" in joined

    print("灾情感知及影响评估 phase 12 validation roadmap UI smoke test passed.")


if __name__ == "__main__":
    main()
