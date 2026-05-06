import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.evidence_ledger import load_ledger  # noqa: E402
from app.mission_schema import create_mission, save_mission  # noqa: E402
from app.services.target_verification_service import (  # noqa: E402
    apply_review_action,
    build_verification_record,
    build_verification_records,
    clamp_bbox,
    crop_candidate_evidence,
    expand_bbox,
    summarize_verification_records,
)
from app.stages.target_verification_stage import run_target_verification_stage  # noqa: E402
from app.workflow.workflow_orchestrator import initialize_rescue_workflow  # noqa: E402


def _write_rgb_image(path, size=(100, 100)):
    from PIL import Image

    Image.new("RGB", size, (120, 130, 140)).save(path)
    return path


def _candidate(candidate_id, class_name, image_path):
    return {
        "candidate_id": candidate_id,
        "target_id": candidate_id.replace("C", "T"),
        "area_id": "A",
        "class_name": class_name,
        "confidence": 0.88 if class_name == "human_candidate" else 0.72,
        "source_image": str(image_path),
        "bbox": [20, 20, 50, 60],
        "center": [35, 40],
        "human_review_required": True,
        "is_confirmed_civilian": False,
    }


def main():
    clamped = clamp_bbox([-10, 5, 80, 100], 64, 64)
    assert clamped is not None
    assert all(0 <= value <= 63 for value in clamped)
    expanded = expand_bbox([20, 20, 50, 60], 64, 64)
    assert expanded is not None
    assert expanded[0] <= 20 and expanded[2] >= 50

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image_path = _write_rgb_image(root / "candidate_source.jpg")
        candidate = _candidate("C001", "human_candidate", image_path)

        crop_result = crop_candidate_evidence(
            str(image_path),
            candidate,
            root / "verification",
            verification_id="V001",
        )
        assert crop_result["crop_success"] is True
        assert Path(crop_result["target_crop_path"]).exists()
        assert Path(crop_result["context_crop_path"]).exists()

        record = build_verification_record(candidate, verification_index=1, crop_result=crop_result)
        assert record["verification_id"] == "V001"
        assert record["candidate_id"] == "C001"
        assert record["human_review_required"] is True
        assert record["is_confirmed_civilian"] is False
        assert "not a final rescue conclusion" in record["truthfulness_note"]
        assert "A human-reviewed candidate is still not equivalent to a confirmed rescued civilian." in record["truthfulness_note"]
        assert "Cropped evidence is derived from image pixels and may miss context outside the crop." in record["truthfulness_note"]

        reviewed = apply_review_action(
            record,
            "confirmed_candidate",
            review_note="目标清晰，建议热红外复查。",
            reviewer="operator",
        )
        assert reviewed["review_status"] == "confirmed_candidate"
        assert reviewed["thermal_check_required"] is True
        assert reviewed["is_confirmed_civilian"] is False

        candidates = [
            candidate,
            _candidate("C002", "vehicle", image_path),
        ]
        review_actions = {
            "C001": {
                "review_status": "confirmed_candidate",
                "review_note": "目标清晰，建议热红外复查。",
                "reviewer": "operator",
            }
        }
        records = build_verification_records(candidates, root / "records", review_actions=review_actions)
        summary = summarize_verification_records(records)
        assert len(records) == 2
        assert summary["verification_count"] == 2
        assert summary["confirmed_candidate_count"] == 1
        assert all(item["human_review_required"] is True for item in records)
        assert all(item["is_confirmed_civilian"] is False for item in records)

        mission_dir = root / "mission"
        mission = create_mission(mission_name="Phase 5 Target Verification")
        mission = initialize_rescue_workflow(mission)
        mission["evidence_ledger_path"] = str(mission_dir / "evidence" / "ledger.json")
        save_mission(mission, mission_dir)
        local_recon_result = {
            "stage_key": "local_recon",
            "candidates": candidates,
        }
        mission, result = run_target_verification_stage(
            mission,
            mission_dir,
            local_recon_result=local_recon_result,
            review_actions=review_actions,
        )
        assert result["stage_key"] == "target_verification"
        assert result["candidate_count"] == 2
        assert result["verification_summary"]["verification_count"] == 2
        assert mission["workflow_state"]["stages"]["target_verification"]["status"] == "completed"
        result_path = mission_dir / "outputs" / "target_verification" / "target_verification_result.json"
        assert result_path.exists()
        ledger = load_ledger(mission["evidence_ledger_path"])
        assert len(ledger["entries"]) >= 1
        assert "not a final rescue conclusion" in ledger["entries"][-1]["truthfulness_note"]

        mission, empty_result = run_target_verification_stage(
            mission,
            mission_dir,
            candidates=[],
        )
        assert empty_result["status"] == "degraded"
        assert empty_result["candidate_count"] == 0
        assert "must not invent candidates" in empty_result["truthfulness_note"]
        assert "The system must not invent candidates or review decisions." in empty_result["truthfulness_note"]

    print("灾情感知及影响评估 phase 5 target verification smoke test passed.")


if __name__ == "__main__":
    main()
