import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.demo.demo_dataset_builder import (  # noqa: E402
    build_demo_detections,
    build_demo_route_result,
    build_demo_thermal_result,
    ensure_demo_dataset,
)
from app.demo.one_click_mission_orchestrator import run_one_click_demo_mission  # noqa: E402
from app.evidence_ledger import load_ledger  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        demo_dir = Path(tmp) / "demo_dataset"
        manifest = ensure_demo_dataset(demo_dir)
        assert Path(manifest["manifest_path"]).exists()
        assert len(manifest["mapping_images"]) >= 1
        assert Path(manifest["local_recon_image"]).exists()
        assert Path(manifest["macro_mask"]).exists()
        assert Path(manifest["thermal_like_image"]).exists()
        assert "workflow demonstration" in manifest["truthfulness_note"]
        assert "Mock/imported detections are not real model inference results." in manifest["mock_detection_note"]

    detections = build_demo_detections()
    assert len(detections) >= 2
    assert any(item["class_name"] == "person" for item in detections)
    assert any("mock detections" in item.get("truthfulness_note", "") for item in detections)

    thermal_result = build_demo_thermal_result()
    assert thermal_result["thermal_mode"] == "simulated"
    assert thermal_result["is_real_temperature_measurement"] is False
    assert thermal_result["temperature_matrix"] is None
    assert "Simulated Thermal is not real temperature measurement" in thermal_result["truthfulness_note"]

    route_result = build_demo_route_result()
    assert route_result["path_type"] == "image_plane_path"
    assert route_result["is_gps_navigation"] is False
    assert "Image-plane path is not GPS navigation" in route_result["message"]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        result = run_one_click_demo_mission(
            missions_root=root / "missions",
            demo_output_root=root / "demo_dataset",
            mission_name="Phase 10 One Click Demo",
        )
        mission_dir = Path(result["mission_dir"])
        assert mission_dir.exists()
        assert Path(result["evidence_ledger_path"]).exists()
        assert Path(result["final_report_markdown_path"]).exists()
        assert Path(result["final_report_json_path"]).exists()

        required_stage_keys = {
            "global_mapping",
            "macro_analysis",
            "area_tasking",
            "local_recon",
            "target_verification",
            "thermal_check",
            "decision_fusion",
            "rescue_recommendation",
            "evidence_report",
        }
        assert required_stage_keys.issubset(set(result["stage_results"].keys()))
        assert result["stage_results"]["local_recon"]["candidate_count"] >= 1
        assert result["stage_results"]["decision_fusion"]["decision_summary"]["decision_candidate_count"] >= 1
        assert result["stage_results"]["rescue_recommendation"]["recommendation_summary"]["recommendation_count"] >= 1

        markdown = Path(result["final_report_markdown_path"]).read_text(encoding="utf-8")
        assert "灾情感知及影响评估 Final Report 2.0" in markdown
        assert "AI 辅助决策报告" in markdown
        assert "不构成最终救援结论" in markdown

        ledger = load_ledger(result["evidence_ledger_path"])
        assert len(ledger["entries"]) >= 6
        joined_notes = "\n".join(str(entry.get("truthfulness_note", "")) for entry in ledger["entries"])
        assert "Demo data is for workflow demonstration only" in result["truthfulness_note"]
        assert "Mock/imported detections are not real model inference results." in joined_notes
        assert "Simulated Thermal is not real temperature measurement." in joined_notes
        assert "Fast Preview / OpenCV Stitch / ORB Homography is not a real ODM georeferenced orthomosaic." in joined_notes
        assert "Uploaded/Demo Mask is not automatic model segmentation." in joined_notes
        assert "Image-plane path is not GPS navigation." in joined_notes
        assert "AI candidates are not confirmed civilians." in joined_notes
        assert "Final Report is an AI-assisted decision-support report and not a final rescue conclusion." in joined_notes

    print("灾情感知及影响评估 phase 10 one-click demo smoke test passed.")


if __name__ == "__main__":
    main()
