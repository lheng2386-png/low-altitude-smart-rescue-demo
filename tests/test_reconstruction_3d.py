import json
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.reconstruction_3d import colmap_360_pipeline, odm_pipeline, reconstruction_report, video_to_frames  # noqa: E402


def _write_image(path: Path, value: int = 128) -> None:
    path.write_bytes(f"placeholder image file {value}".encode("utf-8"))


def test_colmap_missing_dependency_status():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        frames = tmp_path / "frames"
        output = tmp_path / "colmap"
        frames.mkdir()
        _write_image(frames / "frame_000001.jpg")

        result = colmap_360_pipeline.run_colmap_360(
            frames,
            output,
            colmap_binary=str(tmp_path / "missing_colmap"),
            panorama_script=str(tmp_path / "missing_panorama_sfm.py"),
            execute=False,
        )

        assert result["status"] == "dependency_missing"
        assert result["success"] is False
        assert result["frame_count"] == 1
        status_path = output / "reconstruction_status.json"
        assert status_path.exists()
        saved = json.loads(status_path.read_text(encoding="utf-8"))
        assert saved["missing_dependency"] == "COLMAP executable"


def test_video_to_frames_missing_input_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        output = tmp_path / "frames"

        result = video_to_frames.extract_frames(tmp_path / "missing.mp4", output, fps=1)

        assert result["status"] == "input_missing"
        assert result["success"] is False
        assert result["original_video"].endswith("missing.mp4")
        assert result["fps"] == 1.0
        assert result["frame_count"] == 0
        assert result["extracted_count"] == 0
        metadata_path = output / "frames_metadata.json"
        assert metadata_path.exists()
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert saved["output_dir"] == str(output)


def test_odm_command_prepared_and_outputs_only_existing():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        images = tmp_path / "images"
        output = tmp_path / "odm"
        images.mkdir()
        _write_image(images / "uav_001.jpg")

        result = odm_pipeline.run_odm(images, output, camera_lens="fisheye", execute=False)

        assert result["image_count"] == 1
        assert result["success"] is False
        assert result["outputs"] == {}
        assert "--camera-lens" in result["command_template"]
        assert "fisheye" in result["command_template"]

        project_dir = output / "project"
        orthophoto = project_dir / "odm_orthophoto" / "odm_orthophoto.tif"
        orthophoto.parent.mkdir(parents=True)
        orthophoto.write_text("placeholder path existence only", encoding="utf-8")
        outputs = odm_pipeline.collect_existing_outputs(project_dir)
        assert outputs == {"orthophoto_tif": str(orthophoto)}


def test_reconstruction_report_generation():
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        status = {
            "method": "COLMAP panorama_sfm.py",
            "status": "dependency_missing",
            "success": False,
            "outputs": {},
        }
        status_path = output / "status.json"
        status_path.write_text(json.dumps(status), encoding="utf-8")

        generated = reconstruction_report.generate_report(
            output,
            source_video="data/demo/input.mp4",
            reconstruction_status_path=status_path,
            gps_available=False,
            gcp_available=False,
        )

        report = generated["report"]
        assert report["scale_status"] == "relative_scale_only"
        assert "360 panorama preview is not 3D reconstruction" in "\n".join(report["truthfulness_boundaries"])
        assert Path(generated["paths"]["json"]).exists()
        md_text = Path(generated["paths"]["markdown"]).read_text(encoding="utf-8")
        assert "No verified reconstruction output files" in md_text
        assert "auxiliary decision support" in md_text


def main():
    test_colmap_missing_dependency_status()
    test_video_to_frames_missing_input_metadata()
    test_odm_command_prepared_and_outputs_only_existing()
    test_reconstruction_report_generation()
    print("AeroRescue-AI 3D reconstruction unit tests passed.")


if __name__ == "__main__":
    main()
