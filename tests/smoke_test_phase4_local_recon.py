import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.evidence_ledger import load_ledger  # noqa: E402
from app.mission_schema import create_mission, save_mission  # noqa: E402
from app.services.local_recon_service import (  # noqa: E402
    build_candidate_from_detection,
    normalize_imported_detections,
    normalize_local_rgb_images,
    summarize_candidates,
)
from app.stages.local_recon_stage import run_local_recon_stage  # noqa: E402
from app.workflow.workflow_orchestrator import initialize_direct_local_recon_workflow  # noqa: E402


def _write_rgb_image(path):
    from PIL import Image

    Image.new("RGB", (64, 64), (120, 130, 140)).save(path)
    return path


def main():
    candidate = build_candidate_from_detection(
        {
            "class_name": "person",
            "confidence": 0.9,
            "bbox": [10, 20, 30, 60],
            "center": [20, 40],
            "area": 800,
        },
        image_path="local_rgb.png",
        area_id="A",
    )
    assert candidate["class_name"] == "human_candidate"
    assert candidate["original_class_name"] == "person"
    assert candidate["is_confirmed_civilian"] is False
    assert candidate["human_review_required"] is True
    assert candidate["is_georeferenced"] is False
    assert "AI detections are candidates and not confirmed civilians." in candidate["truthfulness_note"]
    assert "Local RGB detections are image-level evidence and require human review." in candidate["truthfulness_note"]
    assert "not confirmed civilians" in candidate["truthfulness_note"]

    for class_name in ["car", "truck", "vehicle"]:
        vehicle_candidate = build_candidate_from_detection({"class_name": class_name}, candidate_index=1)
        assert vehicle_candidate["class_name"] == "vehicle"

    detections = [
        {"class_name": "person", "confidence": 0.91, "bbox": [1, 2, 20, 40], "center": [10, 20], "area": 760},
        {"class_name": "vehicle", "confidence": 0.73, "bbox": [30, 20, 50, 45], "center": [40, 32], "area": 500},
    ]
    candidates = normalize_imported_detections(detections, image_path="local_rgb.png", area_id="A")
    assert len(candidates) == 2
    assert candidates[0]["candidate_id"] == "C001"
    assert candidates[1]["candidate_id"] == "C002"
    summary = summarize_candidates(candidates)
    assert summary["candidate_count"] == 2
    assert summary["human_candidate_count"] == 1
    assert summary["vehicle_count"] == 1

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image_path = _write_rgb_image(root / "local_rgb.png")
        assert normalize_local_rgb_images(str(image_path)) == [str(image_path)]
        assert normalize_local_rgb_images([str(image_path), str(root / "missing.png")]) == [str(image_path)]

        mission_dir = root / "mission"
        mission = create_mission(mission_name="Phase 4 Local Recon")
        mission = initialize_direct_local_recon_workflow(
            mission,
            local_rgb_images=[str(image_path)],
            manual_area_id="A",
        )
        mission["evidence_ledger_path"] = str(mission_dir / "evidence" / "ledger.json")
        save_mission(mission, mission_dir)

        mission, result = run_local_recon_stage(
            mission,
            mission_dir,
            local_rgb_images=[str(image_path)],
            area_id="A",
            detections=detections,
            detection_backend="mock",
        )
        assert result["stage_key"] == "local_recon"
        assert result["candidate_count"] == 2
        assert result["global_context_available"] is False
        assert result["map_registration_available"] is False
        assert mission["workflow_state"]["stages"]["local_recon"]["status"] == "completed"
        ledger_path = Path(mission["evidence_ledger_path"])
        assert ledger_path.exists()
        ledger = load_ledger(ledger_path)
        assert len(ledger["entries"]) >= 1
        assert "AI detections are candidates and not confirmed civilians." in ledger["entries"][-1]["truthfulness_note"]
        assert "Local RGB detections are image-level evidence and require human review." in ledger["entries"][-1]["truthfulness_note"]
        assert "not confirmed civilians" in ledger["entries"][-1]["truthfulness_note"]
        assert "Local detection results cannot be treated as georeferenced rescue targets unless map registration is provided." in ledger["entries"][-1]["truthfulness_note"]

        mission, empty_result = run_local_recon_stage(
            mission,
            mission_dir,
            local_rgb_images=[str(image_path)],
            area_id="A",
            detections=None,
            detection_backend="none",
        )
        assert empty_result["candidate_count"] == 0
        assert empty_result["status"] in {"degraded", "completed"}
        assert "The system must not invent detections." in empty_result["truthfulness_note"]

    print("灾情感知及影响评估 phase 4 local recon smoke test passed.")


if __name__ == "__main__":
    main()
