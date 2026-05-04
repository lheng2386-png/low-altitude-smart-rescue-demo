import json
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.reconstruction_3d import reconstruction_workflow  # noqa: E402


def _write_image(path: Path) -> None:
    path.write_bytes(b"placeholder workflow image")


def test_report_only_workflow_writes_status_and_report():
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "workflow"
        result = reconstruction_workflow.run_reconstruction_workflow("report_only", str(output))

        assert result["status"] == "report_only"
        assert result["success"] is True
        assert (output / "workflow_status.json").exists()
        assert Path(result["report"]["paths"]["json"]).exists()
        assert Path(result["report"]["paths"]["markdown"]).exists()


def test_standard_workflow_dependency_missing_is_transparent():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image_dir = tmp_path / "images"
        output = tmp_path / "workflow"
        image_dir.mkdir()
        _write_image(image_dir / "frame_001.jpg")

        original = reconstruction_workflow.colmap_standard_pipeline.run_colmap_standard_pipeline

        def fake_standard(input_dir, output_dir, **kwargs):
            status_path = Path(output_dir) / "reconstruction_status.json"
            status_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "success": False,
                "status": "dependency_missing",
                "method": "COLMAP standard SfM",
                "input_type": "uav_frames",
                "frame_count": 1,
                "outputs": {},
                "limitations": ["No fake reconstruction success."],
            }
            status_path.write_text(json.dumps(payload), encoding="utf-8")
            payload["status_path"] = str(status_path)
            return payload

        try:
            reconstruction_workflow.colmap_standard_pipeline.run_colmap_standard_pipeline = fake_standard
            result = reconstruction_workflow.run_reconstruction_workflow(
                "standard_uav",
                str(output),
                image_dir=str(image_dir),
                run_quality_filter=False,
            )
        finally:
            reconstruction_workflow.colmap_standard_pipeline.run_colmap_standard_pipeline = original

        assert result["status"] == "dependency_missing"
        assert result["success"] is False
        assert result["steps"][1]["name"] == "frame_quality_filter"
        assert result["steps"][2]["name"] == "colmap_standard"
        assert (output / "workflow_status.json").exists()


def test_odm_workflow_success_propagates_only_from_pipeline_status():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image_dir = tmp_path / "images"
        output = tmp_path / "workflow"
        image_dir.mkdir()
        _write_image(image_dir / "uav_001.jpg")

        original = reconstruction_workflow.odm_pipeline.run_odm_pipeline

        def fake_odm(input_dir, output_dir, **kwargs):
            status_path = Path(output_dir) / "reconstruction_status.json"
            status_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "success": True,
                "status": "success",
                "method": "opendronemap_odm",
                "input_type": "uav_images",
                "image_count": 1,
                "docker_image": "opendronemap/odm",
                "camera_lens": "auto",
                "outputs": {
                    "orthophoto": {"exists": True, "path": str(Path(output_dir) / "aerorescue_odm/odm_orthophoto/odm_orthophoto.tif")},
                    "available_output_count": 1,
                },
                "geo_reference": "unknown",
                "scale_type": "depends_on_exif_gps_gcp",
                "limitations": ["All outputs are auxiliary spatial evidence and require human review."],
            }
            status_path.write_text(json.dumps(payload), encoding="utf-8")
            payload["status_path"] = str(status_path)
            return payload

        try:
            reconstruction_workflow.odm_pipeline.run_odm_pipeline = fake_odm
            result = reconstruction_workflow.run_reconstruction_workflow(
                "odm",
                str(output),
                image_dir=str(image_dir),
                run_quality_filter=False,
            )
        finally:
            reconstruction_workflow.odm_pipeline.run_odm_pipeline = original

        assert result["status"] == "success"
        assert result["success"] is True
        assert result["steps"][2]["name"] == "odm"
        assert result["report"]["report"]["method"] == "opendronemap_odm"


def test_dependency_panel_reports_script_missing_without_crash():
    original_colmap = reconstruction_workflow.check_colmap_available
    original_docker = reconstruction_workflow.check_docker_available
    try:
        reconstruction_workflow.check_colmap_available = lambda: {
            "available": False,
            "colmap_path": None,
            "version": "unknown",
            "status": "dependency_missing",
        }
        reconstruction_workflow.check_docker_available = lambda: {
            "available": False,
            "docker_path": None,
            "version": "unknown",
            "status": "dependency_missing",
        }
        status = reconstruction_workflow.check_reconstruction_dependencies(panorama_sfm_script="/missing/panorama_sfm.py")
    finally:
        reconstruction_workflow.check_colmap_available = original_colmap
        reconstruction_workflow.check_docker_available = original_docker

    assert status["colmap"]["status"] == "dependency_missing"
    assert status["docker"]["status"] == "dependency_missing"
    assert status["panorama_sfm_script"]["status"] == "script_missing"
    assert "limitations" in status


def main():
    test_report_only_workflow_writes_status_and_report()
    test_standard_workflow_dependency_missing_is_transparent()
    test_odm_workflow_success_propagates_only_from_pipeline_status()
    test_dependency_panel_reports_script_missing_without_crash()
    print("AeroRescue-AI reconstruction workflow tests passed.")


if __name__ == "__main__":
    main()
