import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from evidence_ledger import load_ledger  # noqa: E402
from mission_schema import create_mission, load_mission, save_mission  # noqa: E402
from workflow.stage_definitions import RESCUE_WORKFLOW_STAGES  # noqa: E402
from workflow.workflow_orchestrator import fail_stage, initialize_rescue_workflow, record_stage_output  # noqa: E402
from workflow.workflow_state import create_initial_workflow_state, summarize_workflow_state, update_stage_status  # noqa: E402


REQUIRED_STAGE_FIELDS = {
    "stage_id",
    "stage_key",
    "stage_name_zh",
    "real_action",
    "uav_layer",
    "input_data",
    "system_modules",
    "outputs",
    "truthfulness_boundary",
    "required_human_review",
}


def main():
    assert len(RESCUE_WORKFLOW_STAGES) == 9
    for stage in RESCUE_WORKFLOW_STAGES:
        missing_fields = REQUIRED_STAGE_FIELDS - set(stage)
        assert not missing_fields, f"{stage.get('stage_key', '<unknown>')} missing {missing_fields}"
        assert stage["truthfulness_boundary"]
        assert stage["required_human_review"] is True

    workflow_state = create_initial_workflow_state()
    assert workflow_state["current_stage_key"] == "global_mapping"
    assert workflow_state["stages"]["global_mapping"]["status"] == "ready"
    for stage_key, stage in workflow_state["stages"].items():
        if stage_key != "global_mapping":
            assert stage["status"] == "pending"

    workflow_state = update_stage_status(
        workflow_state,
        "global_mapping",
        "completed",
        output_ref="outputs/orthomosaic/preview.png",
        result_type="orthomosaic_preview",
        evidence_id="E0001",
    )
    assert workflow_state["stages"]["global_mapping"]["status"] == "completed"
    assert workflow_state["stages"]["macro_analysis"]["status"] == "ready"
    assert workflow_state["stages"]["global_mapping"]["evidence_ids"] == ["E0001"]

    summary = summarize_workflow_state(workflow_state)
    assert summary["total_stage_count"] == 9
    assert summary["completed_stage_count"] == 1
    assert summary["ready_stage_key"] == "macro_analysis"
    assert summary["failed_stage_count"] == 0

    mission = create_mission(mission_name="Workflow Smoke Mission")
    mission = initialize_rescue_workflow(mission)
    assert mission["workflow_state"]
    assert mission["workflow_state"]["stages"]["global_mapping"]["status"] == "ready"

    with tempfile.TemporaryDirectory() as tmp:
        mission_dir = Path(tmp) / mission["mission_id"]
        mission["evidence_ledger_path"] = str(mission_dir / "evidence" / "ledger.json")
        save_mission(mission, mission_dir)

        entry = record_stage_output(
            mission,
            mission_dir,
            "global_mapping",
            output_ref="outputs/orthomosaic/fast_preview.png",
            result_type="fast_preview",
            source_type="orthomosaic_preview",
        )
        assert entry["evidence_id"]
        assert entry["module"] == "workflow:global_mapping"
        assert "not a real ODM" in entry["truthfulness_note"]
        assert mission["workflow_state"]["stages"]["global_mapping"]["status"] == "completed"
        assert entry["evidence_id"] in mission["workflow_state"]["stages"]["global_mapping"]["evidence_ids"]

        ledger_path = Path(mission["evidence_ledger_path"])
        assert ledger_path.exists()
        ledger = load_ledger(ledger_path)
        assert ledger["entries"]
        assert ledger["entries"][0]["human_review_required"] is True

        record_stage_output(
            mission,
            mission_dir,
            "macro_analysis",
            output_ref="outputs/segmentation/macro_mask.png",
            result_type="uploaded_mask",
            source_type="uploaded_mask",
        )
        saved_mission = load_mission(mission_dir / "mission.json")
        assert saved_mission["workflow_state"]["stages"]["global_mapping"]["status"] == "completed"
        assert saved_mission["workflow_state"]["stages"]["macro_analysis"]["status"] == "completed"

        mission = fail_stage(mission, mission_dir, "macro_analysis", "Smoke test failure")
        assert mission["workflow_state"]["stages"]["macro_analysis"]["status"] == "failed"
        assert mission["workflow_state"]["stages"]["macro_analysis"]["error"] == "Smoke test failure"

    print("AeroRescue-AI rescue workflow smoke test passed.")


if __name__ == "__main__":
    main()
