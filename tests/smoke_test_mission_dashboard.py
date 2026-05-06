import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ui import mission_dashboard_panel  # noqa: E402
from app.workflow.stage_definitions import RESCUE_WORKFLOW_STAGES  # noqa: E402
from app.workflow.workflow_state import create_initial_workflow_state  # noqa: E402


def main():
    assert len(RESCUE_WORKFLOW_STAGES) == 9

    workflow_state = create_initial_workflow_state()
    rows = mission_dashboard_panel.format_workflow_stage_table(workflow_state)
    assert len(rows) == 9
    assert rows[0][0] == "S1"
    assert "global_mapping" in rows[0][1]
    assert "高空建图" in rows[0][1]
    assert any("Fast Preview" in str(cell) and "real ODM" in str(cell) for row in rows for cell in row)

    progress_markdown = mission_dashboard_panel.format_workflow_progress_markdown(workflow_state)
    assert isinstance(progress_markdown, str)
    assert "高空建图" in progress_markdown
    assert "宏观灾情分析" in progress_markdown
    assert "重点区域划分" in progress_markdown
    assert "ready" in progress_markdown
    assert "pending" in progress_markdown

    mission = {
        "mission_id": "M_TEST",
        "mission_name": "测试救援任务",
        "status": "running",
        "workflow_state": workflow_state,
        "available_modules": ["object_detection", "image_plane_path_planning"],
        "disabled_modules": [{"module": "real_temperature_analysis", "reason": "No radiometric thermal input."}],
        "truthfulness_boundaries": ["Simulated Thermal is not real temperature measurement."],
        "evidence_ledger_path": "outputs/missions/M_TEST/evidence/ledger.json",
    }
    summary_markdown = mission_dashboard_panel.format_mission_summary_markdown(mission)
    assert "M_TEST" in summary_markdown
    assert "测试救援任务" in summary_markdown
    assert "object_detection" in summary_markdown
    assert "real_temperature_analysis" in summary_markdown
    assert "Simulated Thermal is not real temperature measurement" in summary_markdown

    empty_summary = mission_dashboard_panel.format_mission_summary_markdown(None)
    assert "当前尚未创建任务" in empty_summary

    truthfulness_markdown = mission_dashboard_panel.format_dashboard_truthfulness_markdown()
    assert "Fast Preview 不等于真实 ODM 正射影像" in truthfulness_markdown
    assert "Simulated Thermal 不等于真实热红外测温" in truthfulness_markdown
    assert "Image-plane path 不等于 GPS 导航路线" in truthfulness_markdown
    assert "human_candidate" in truthfulness_markdown
    assert "confirmed civilian" in truthfulness_markdown

    assert callable(mission_dashboard_panel.attach_mission_dashboard_panel)

    print("灾情感知及影响评估 mission dashboard smoke test passed.")


if __name__ == "__main__":
    main()
