import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.evidence_ledger import load_ledger  # noqa: E402
from app.mission_schema import create_mission, save_mission  # noqa: E402
from app.services.thermal_support_service import (  # noqa: E402
    build_thermal_check_record,
    build_thermal_check_records,
    estimate_thermal_support_level,
    extract_hotspot_summary,
    normalize_thermal_inputs,
    summarize_thermal_check_records,
)
from app.stages.thermal_check_stage import run_thermal_check_stage  # noqa: E402
from app.workflow.workflow_orchestrator import initialize_rescue_workflow  # noqa: E402


def _verification(verification_id, class_name, required=True):
    return {
        "verification_id": verification_id,
        "candidate_id": verification_id.replace("V", "C"),
        "target_id": verification_id.replace("V", "T"),
        "area_id": "A",
        "class_name": class_name,
        "thermal_check_required": required,
        "human_review_required": True,
        "is_confirmed_civilian": False,
    }


def main():
    simulated = {
        "thermal_mode": "simulated",
        "temperature_matrix": None,
        "hotspot_count": 2,
        "hotspot_area_ratio": 0.04,
        "risk_level": "Medium",
        "is_real_temperature_measurement": False,
    }
    summary = extract_hotspot_summary(simulated)
    assert summary["thermal_mode"] == "simulated"
    assert summary["temperature_matrix_available"] is False
    assert summary["is_real_temperature_measurement"] is False
    assert "Simulated Thermal is not real temperature measurement." in summary["truthfulness_note"]
    assert "RGB/JPG/PNG images cannot provide real temperature_matrix." in summary["truthfulness_note"]

    assert estimate_thermal_support_level(summary)["thermal_support_level"] == "weak"
    strong = dict(simulated, hotspot_count=3, risk_level="High")
    assert estimate_thermal_support_level(extract_hotspot_summary(strong))["thermal_support_level"] == "strong"
    none = dict(simulated, hotspot_count=0)
    assert estimate_thermal_support_level(extract_hotspot_summary(none))["thermal_support_level"] == "none"
    assert estimate_thermal_support_level(extract_hotspot_summary({"thermal_mode": "unknown"}))["thermal_support_level"] == "unavailable"

    verification_record = _verification("V001", "human_candidate", True)
    thermal_record = build_thermal_check_record(
        verification_record,
        thermal_image_path="thermal.jpg",
        thermal_result=simulated,
        check_index=1,
        source_type="simulated_thermal",
    )
    assert thermal_record["thermal_check_id"] == "TH001"
    assert thermal_record["candidate_id"] == "C001"
    assert thermal_record["thermal_support_level"] in {"weak", "strong", "none", "unavailable"}
    assert thermal_record["is_confirmed_civilian"] is False
    assert thermal_record["human_review_required"] is True
    assert "not confirmation of a civilian" in thermal_record["truthfulness_note"]
    assert "Simulated Thermal is not real temperature measurement" in thermal_record["truthfulness_note"]
    assert "RGB-thermal matching may be approximate unless calibrated registration is provided." in thermal_record["truthfulness_note"]

    records = build_thermal_check_records(
        [
            verification_record,
            _verification("V002", "vehicle", False),
        ],
        thermal_results=simulated,
        source_type="simulated_thermal",
    )
    assert len(records) == 1

    mixed_summary = summarize_thermal_check_records(
        [
            dict(thermal_record, thermal_support_level="strong"),
            dict(thermal_record, thermal_support_level="weak"),
            dict(thermal_record, thermal_support_level="none"),
            dict(thermal_record, thermal_support_level="unavailable"),
        ]
    )
    assert mixed_summary["strong_support_count"] == 1
    assert mixed_summary["weak_support_count"] == 1
    assert mixed_summary["none_support_count"] == 1
    assert mixed_summary["unavailable_count"] == 1

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        thermal_path = root / "thermal.jpg"
        thermal_path.write_bytes(b"placeholder")
        assert normalize_thermal_inputs(str(thermal_path)) == [str(thermal_path)]

        mission_dir = root / "mission"
        mission = create_mission(mission_name="Phase 6 Thermal Check")
        mission = initialize_rescue_workflow(mission)
        mission["evidence_ledger_path"] = str(mission_dir / "evidence" / "ledger.json")
        save_mission(mission, mission_dir)
        target_verification_result = {
            "verification_records": [
                verification_record,
                _verification("V002", "vehicle", False),
            ]
        }
        mission, result = run_thermal_check_stage(
            mission,
            mission_dir,
            target_verification_result=target_verification_result,
            thermal_images=[str(thermal_path)],
            thermal_results=simulated,
            run_thermal_analysis=False,
        )
        assert result["stage_key"] == "thermal_check"
        assert result["thermal_target_count"] == 1
        assert result["thermal_summary"]["thermal_check_count"] == 1
        assert mission["workflow_state"]["stages"]["thermal_check"]["status"] == "completed"
        assert (mission_dir / "outputs" / "thermal_check" / "thermal_check_result.json").exists()
        ledger = load_ledger(mission["evidence_ledger_path"])
        assert len(ledger["entries"]) >= 1
        assert "auxiliary support" in ledger["entries"][-1]["truthfulness_note"]

        mission, empty_result = run_thermal_check_stage(
            mission,
            mission_dir,
            verification_records=[],
        )
        assert empty_result["status"] == "degraded"
        assert "must not invent thermal hotspots" in empty_result["truthfulness_note"]
        assert "The system must not invent thermal hotspots or temperature values." in empty_result["truthfulness_note"]

    print("AeroRescue-AI phase 6 thermal check smoke test passed.")


if __name__ == "__main__":
    main()
