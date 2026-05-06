import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.evidence_ledger import load_ledger  # noqa: E402
from app.mission_schema import create_mission, save_mission  # noqa: E402
from app.services.area_tasking_service import build_area_tasks_from_macro_zones  # noqa: E402
from app.stages import run_area_tasking_stage, run_global_mapping_stage, run_macro_analysis_stage  # noqa: E402
from app.workflow.workflow_orchestrator import initialize_rescue_workflow  # noqa: E402


def _create_test_mission(mission_dir):
    mission = create_mission(mission_name="Phase 3 Smoke Mission")
    mission = initialize_rescue_workflow(mission)
    mission["evidence_ledger_path"] = str(mission_dir / "evidence" / "ledger.json")
    save_mission(mission, mission_dir)
    return mission


def _write_test_image_and_mask(root):
    from PIL import Image

    image_path = root / "map_rgb.png"
    mask_path = root / "macro_mask.png"
    image = Image.new("RGB", (64, 64), (120, 130, 140))
    mask = Image.new("L", (64, 64), 0)
    for y in range(0, 24):
        for x in range(0, 24):
            mask.putpixel((x, y), 1)
    for y in range(24, 44):
        for x in range(0, 64):
            mask.putpixel((x, y), 7)
    for y in range(44, 64):
        for x in range(32, 64):
            mask.putpixel((x, y), 8)
    image.save(image_path)
    mask.save(mask_path)
    return image_path, mask_path


def main():
    macro_zones = [
        {"zone_id": "Z001", "zone_type": "flood_or_water_zone", "risk_level": "High", "area_percent": 22.5},
        {"zone_id": "Z002", "zone_type": "accessible_road_zone", "risk_level": "Low", "area_percent": 30.0},
        {"zone_id": "Z003", "zone_type": "blocked_road_zone", "risk_level": "High", "area_percent": 8.0},
    ]
    area_tasks = build_area_tasks_from_macro_zones(macro_zones)
    assert len(area_tasks) == 3
    assert area_tasks[0]["area_id"] == "A"
    assert area_tasks[0]["source_zone_id"] == "Z001"
    assert all(task["human_review_required"] is True for task in area_tasks)

    fallback_tasks = build_area_tasks_from_macro_zones([])
    assert fallback_tasks[0]["recommended_next_action"] == "manual_area_selection_required"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mission_dir = root / "mission"
        mission = _create_test_mission(mission_dir)

        macro_analysis_result = {
            "stage_key": "macro_analysis",
            "status": "completed",
            "macro_zones": macro_zones,
        }
        mission, area_result = run_area_tasking_stage(mission, mission_dir, macro_analysis_result)
        assert area_result["stage_key"] == "area_tasking"
        assert area_result["area_count"] == 3
        assert mission["workflow_state"]["stages"]["area_tasking"]["status"] == "completed"
        ledger_path = Path(mission["evidence_ledger_path"])
        assert ledger_path.exists()
        assert len(load_ledger(ledger_path)["entries"]) >= 1

        image_path, mask_path = _write_test_image_and_mask(root)
        mission, macro_result = run_macro_analysis_stage(
            mission,
            mission_dir,
            map_image_path=str(image_path),
            segmentation_mask_path=str(mask_path),
            segmentation_source="uploaded_mask",
        )
        assert macro_result["status"] in {"completed", "degraded"}
        zone_types = {zone["zone_type"] for zone in macro_result["macro_zones"]}
        assert "flood_or_water_zone" in zone_types
        assert "blocked_road_zone" in zone_types
        assert "Uploaded/Demo Mask is not automatic model segmentation." in macro_result["truthfulness_note"]

        fresh_mission = _create_test_mission(root / "mapping_mission")
        fresh_mission, mapping_result = run_global_mapping_stage(fresh_mission, root / "mapping_mission", image_files=[])
        assert mapping_result["status"] in {"failed", "skipped"}
        assert mapping_result["base_map_type"] == "none"
        assert "No high-altitude overlapping RGB images were provided." in mapping_result["truthfulness_note"]

    print("灾情感知及影响评估 phase 3 early workflow smoke test passed.")


if __name__ == "__main__":
    main()
