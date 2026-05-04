import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from path_planner import (  # noqa: E402
    build_cost_map,
    compare_path_plans,
    plan_baseline_path,
    plan_rescue_path,
    plan_risk_aware_path,
)
from segmentation_engine import (  # noqa: E402
    create_segmentation_overlay,
    summarize_segmentation,
    validate_segmentation_mask,
)
from segmentation_model import get_segmentation_model_status  # noqa: E402
from scene_applicability_gate import evaluate_scene_applicability  # noqa: E402
from terp_engine import calculate_terp, rank_targets_by_terp  # noqa: E402
from evidence_ledger import (  # noqa: E402
    add_evidence_entry,
    create_ledger,
    load_ledger,
    save_ledger,
    summarize_ledger,
)
from input_validator import validate_mission_inputs  # noqa: E402
from mission_orchestrator import (  # noqa: E402
    finalize_mission,
    initialize_mission_from_inputs,
    record_module_result,
)
from mission_schema import create_mission, load_mission, save_mission  # noqa: E402
from scripts.generate_demo_cases import _manual_demo_mask  # noqa: E402
from model_comparison.evaluate_detection_models import load_registry  # noqa: E402


def main():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :] = (90, 110, 130)

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    mask[24:44, 8:44] = 7
    mask[44:60, 32:60] = 8

    validation = validate_segmentation_mask(mask)
    assert validation["valid"] is True
    assert validation["unique_class_ids"] == [0, 1, 7, 8]
    assert validation["unknown_class_ids"] == []

    invalid_mask = mask.copy()
    invalid_mask[0, 0] = 99
    invalid_validation = validate_segmentation_mask(invalid_mask)
    assert invalid_validation["valid"] is False
    assert invalid_validation["unknown_class_ids"] == [99]

    summary = summarize_segmentation(mask)
    assert "water" in summary
    assert "road_clear" in summary
    assert "road_blocked" in summary

    overlay = create_segmentation_overlay(image, mask)
    assert overlay is not None
    assert overlay.shape == image.shape

    cost_map = build_cost_map(mask, 64, 64)
    assert cost_map.shape == (64, 64)
    assert float(cost_map[16, 16]) == 100.0
    assert float(cost_map[30, 20]) == 1.0
    assert float(cost_map[50, 40]) == 80.0

    ranked_targets = [
        {
            "id": "T001",
            "target_id": "T001",
            "class_name": "civilian",
            "center": [56, 8],
            "bbox": [52, 4, 60, 12],
            "confidence": 0.92,
            "area": 64.0,
            "risk_score": 95.0,
        }
    ]
    path_result = plan_rescue_path(ranked_targets, mask, 64, 64, start_point=(4, 60))
    assert path_result["found"] is True
    assert path_result["path_length"] > 0

    baseline = plan_baseline_path(ranked_targets, 64, 64, start_point=(4, 60))
    risk_aware = plan_risk_aware_path(ranked_targets, mask, 64, 64, start_point=(4, 60))
    comparison = compare_path_plans(baseline, risk_aware, mask)
    assert comparison["baseline_length"] > 0
    assert comparison["risk_aware_length"] > 0
    assert "risk_reduction" in comparison

    no_mask_baseline = plan_baseline_path(ranked_targets, 64, 64, start_point=(4, 60))
    no_mask_risk_aware = plan_risk_aware_path(ranked_targets, None, 64, 64, start_point=(4, 60))
    no_mask_comparison = compare_path_plans(no_mask_baseline, no_mask_risk_aware, None)
    assert "语义分割掩码" in no_mask_comparison["message"]

    terp = calculate_terp(
        ranked_targets[0],
        64,
        64,
        environment_context={"environment_risk_score": 18.0, "environment_reason": "目标附近存在水域。"},
        path_result=risk_aware,
    )
    assert terp["terp_score"] > 0
    assert terp["terp_level"] in {"Low", "Medium", "High", "Critical"}

    terp_ranking = rank_targets_by_terp(
        ranked_targets,
        64,
        64,
        environment_contexts={"T001": {"environment_risk_score": 18.0, "environment_reason": "目标附近存在水域。"}},
        path_results={"T001": risk_aware},
    )
    assert terp_ranking[0]["target_id"] == "T001"

    multi_targets = [
        ranked_targets[0],
        {
            "id": "T002",
            "target_id": "T002",
            "class_name": "rescuer",
            "center": [12, 12],
            "bbox": [8, 8, 16, 16],
            "confidence": 0.88,
            "area": 64.0,
            "risk_score": 20.0,
        },
    ]
    multi_ranking = rank_targets_by_terp(
        multi_targets,
        64,
        64,
        environment_contexts={
            "T001": {"environment_risk_score": 18.0, "environment_reason": "目标附近存在水域。"},
            "T002": {"environment_risk_score": 0.0, "environment_reason": "环境风险较低。"},
        },
        path_results={"T001": risk_aware, "T002": baseline},
    )
    assert multi_ranking[0]["terp_score"] >= multi_ranking[1]["terp_score"]

    gate_full = evaluate_scene_applicability(ranked_targets, segmentation_mask=mask, segmentation_source="uploaded")
    assert gate_full["allow_environment_fusion"] is True
    gate_no_target = evaluate_scene_applicability([], segmentation_mask=mask, segmentation_source="uploaded")
    assert gate_no_target["allow_path_planning"] is False

    generated_mask = _manual_demo_mask(64, 64, "road_blocked")
    generated_validation = validate_segmentation_mask(generated_mask)
    assert generated_validation["valid"] is True
    assert 8 in generated_validation["unique_class_ids"]

    registry = load_registry(ROOT_DIR / "model_comparison" / "model_registry.json")
    assert any(item["name"] == "yolov11n" for item in registry)

    missing_status = get_segmentation_model_status(ROOT_DIR / "checkpoints" / "missing_smoke_test.pth")
    assert missing_status["available"] is False

    tmp_path = Path(tempfile.mkdtemp(prefix="aerorescue_mission_smoke_"))
    rgb_path = tmp_path / "input_rgb.png"
    Image.fromarray(image).save(rgb_path)

    input_summary = validate_mission_inputs(rgb_images=[str(rgb_path)])
    assert "object_detection" in input_summary["available_modules"]
    assert "simulated_thermal_risk" in input_summary["available_modules"]
    assert "image_plane_path_planning" in input_summary["available_modules"]
    assert not input_summary["real_temperature_available"]
    disabled_modules = {item["module"] for item in input_summary["disabled_modules"]}
    assert "real_temperature_analysis" in disabled_modules
    assert "gps_navigation" in disabled_modules
    assert "RGB images can support visual detection but cannot provide real temperature_matrix." in input_summary[
        "truthfulness_boundaries"
    ]

    mission = create_mission(
        mission_name="Smoke Mission",
        input_summary=input_summary,
        available_modules=input_summary["available_modules"],
        disabled_modules=input_summary["disabled_modules"],
        truthfulness_boundaries=input_summary["truthfulness_boundaries"],
    )
    mission_dir = tmp_path / "missions" / mission["mission_id"]
    mission["evidence_ledger_path"] = str(mission_dir / "evidence" / "ledger.json")
    mission_json_path = save_mission(mission, mission_dir)
    loaded_mission = load_mission(mission_json_path)
    assert loaded_mission["mission_id"] == mission["mission_id"]
    assert mission_json_path.exists()

    ledger = create_ledger(mission["mission_id"])
    add_evidence_entry(
        ledger,
        module="thermal",
        input_ref=str(rgb_path),
        output_ref="outputs/thermal/simulated_heatmap.png",
        result_type="simulated_heatmap",
        source_type="simulated_thermal",
    )
    add_evidence_entry(
        ledger,
        module="path_planning",
        input_ref=str(rgb_path),
        output_ref="outputs/path/path_overlay.png",
        result_type="image_plane_path",
        source_type="image_plane_path",
    )
    add_evidence_entry(
        ledger,
        module="segmentation",
        input_ref="input/masks/uploaded.png",
        output_ref="outputs/segmentation/uploaded_mask_overlay.png",
        result_type="uploaded_mask",
        source_type="uploaded_mask",
    )
    assert all("human_review_required" in entry for entry in ledger["entries"])
    ledger_summary = summarize_ledger(ledger)
    assert ledger_summary["total_evidence_count"] == 3
    assert ledger_summary["human_review_required_count"] == 3
    assert ledger_summary["module_counts"]["thermal"] == 1
    assert ledger_summary["module_counts"]["path_planning"] == 1
    assert ledger_summary["module_counts"]["segmentation"] == 1
    ledger_path = save_ledger(ledger, mission["evidence_ledger_path"])
    assert ledger_path.exists()
    assert load_ledger(ledger_path)["entries"][0]["source_type"] == "simulated_thermal"

    orchestrated_mission, orchestrated_dir = initialize_mission_from_inputs(
        tmp_path / "orchestrated_missions",
        mission_name="Orchestrated Smoke Mission",
        rgb_images=[str(rgb_path)],
    )
    record_module_result(
        orchestrated_mission,
        orchestrated_dir,
        module="thermal",
        input_ref=str(rgb_path),
        output_ref="outputs/thermal/simulated_heatmap.png",
        result_type="simulated_heatmap",
        source_type="simulated_thermal",
    )
    summary = finalize_mission(orchestrated_mission, orchestrated_dir)
    assert (orchestrated_dir / "mission.json").exists()
    assert (orchestrated_dir / "evidence" / "ledger.json").exists()
    assert (orchestrated_dir / "mission_summary.json").exists()
    assert summary["evidence_summary"]["total_evidence_count"] == 1

    help_result = subprocess.run(
        [sys.executable, str(ROOT_DIR / "scripts" / "generate_demo_cases.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "Generate AeroRescue-AI offline showcase demo cases" in help_result.stdout

    print("AeroRescue-AI demo case and model comparison smoke test passed.")


if __name__ == "__main__":
    main()
