import json
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.reconstruction_3d import colmap_360_pipeline, colmap_standard_pipeline, colmap_utils, reconstruction_report  # noqa: E402


def _write_placeholder_image(path: Path) -> None:
    path.write_bytes(b"placeholder image path for COLMAP pipeline tests")


def _write_sparse_model(output_dir: Path) -> None:
    (output_dir / "database.db").write_bytes(b"verified database file existence only")
    sparse_0 = output_dir / "sparse" / "0"
    sparse_0.mkdir(parents=True, exist_ok=True)
    for name in ["cameras.bin", "images.bin", "points3D.bin"]:
        (sparse_0 / name).write_bytes(b"verified file existence only")


def test_missing_colmap_returns_dependency_missing():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        images = tmp_path / "images"
        output = tmp_path / "standard"
        images.mkdir()
        _write_placeholder_image(images / "frame_001.jpg")

        original_check = colmap_standard_pipeline.check_colmap_available
        try:
            colmap_standard_pipeline.check_colmap_available = lambda: {
                "available": False,
                "colmap_path": None,
                "version": "unknown",
                "status": "dependency_missing",
            }
            result = colmap_standard_pipeline.run_colmap_standard_pipeline(str(images), str(output))
        finally:
            colmap_standard_pipeline.check_colmap_available = original_check

        assert result["status"] == "dependency_missing"
        assert result["success"] is False
        assert result["outputs"] == {}
        assert (output / "reconstruction_status.json").exists()


def test_missing_input_folder_returns_invalid_input_when_colmap_available():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        output = tmp_path / "standard"

        original_check = colmap_standard_pipeline.check_colmap_available
        try:
            colmap_standard_pipeline.check_colmap_available = lambda: {
                "available": True,
                "colmap_path": "/bin/echo",
                "version": "test",
                "status": "ok",
            }
            result = colmap_standard_pipeline.run_colmap_standard_pipeline(str(tmp_path / "missing"), str(output))
        finally:
            colmap_standard_pipeline.check_colmap_available = original_check

        assert result["status"] == "invalid_input"
        assert result["success"] is False


def test_single_360_panorama_frame_returns_insufficient_input():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        images = tmp_path / "panorama"
        output = tmp_path / "colmap_360"
        script = tmp_path / "panorama_sfm.py"
        images.mkdir()
        _write_placeholder_image(images / "pano_001.jpg")
        script.write_text("print('not executed for insufficient input')\n", encoding="utf-8")

        original_check = colmap_360_pipeline.check_colmap_available
        try:
            colmap_360_pipeline.check_colmap_available = lambda: {
                "available": True,
                "colmap_path": "/bin/echo",
                "version": "test",
                "status": "ok",
            }
            result = colmap_360_pipeline.run_colmap_360_pipeline(str(images), str(output), panorama_sfm_script=str(script))
        finally:
            colmap_360_pipeline.check_colmap_available = original_check

        assert result["status"] == "insufficient_input"
        assert result["success"] is False
        assert "single panorama image is not 3D reconstruction" in result["message"]


def test_failed_subprocess_command_returns_success_false():
    result = colmap_utils.run_command([sys.executable, "-c", "import sys; sys.exit(7)"], cwd=None, timeout=30)
    assert result["success"] is False
    assert result["returncode"] == 7
    assert result["duration_seconds"] >= 0


def test_standard_pipeline_writes_status_and_requires_sparse_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        images = tmp_path / "images"
        output = tmp_path / "standard"
        images.mkdir()
        _write_placeholder_image(images / "frame_001.jpg")
        _write_placeholder_image(images / "frame_002.jpg")

        original_check = colmap_standard_pipeline.check_colmap_available
        original_run = colmap_standard_pipeline.run_command
        try:
            colmap_standard_pipeline.check_colmap_available = lambda: {
                "available": True,
                "colmap_path": "/bin/echo",
                "version": "test",
                "status": "ok",
            }
            colmap_standard_pipeline.run_command = lambda cmd, cwd=None, timeout=None: {
                "cmd": cmd,
                "cwd": cwd,
                "timeout": timeout,
                "returncode": 0,
                "stdout": "ok",
                "stderr": "",
                "duration_seconds": 0.001,
                "success": True,
                "error": None,
                "timed_out": False,
            }
            result = colmap_standard_pipeline.run_colmap_standard_pipeline(str(images), str(output))
        finally:
            colmap_standard_pipeline.check_colmap_available = original_check
            colmap_standard_pipeline.run_command = original_run

        assert result["status"] == "failed"
        assert result["success"] is False
        assert result["sparse_success"] is False
        assert (output / "reconstruction_status.json").exists()


def test_success_only_true_when_expected_sparse_files_exist():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        images = tmp_path / "images"
        output = tmp_path / "standard"
        images.mkdir()
        _write_placeholder_image(images / "frame_001.jpg")
        _write_placeholder_image(images / "frame_002.jpg")

        original_check = colmap_standard_pipeline.check_colmap_available
        original_run = colmap_standard_pipeline.run_command

        def fake_run(cmd, cwd=None, timeout=None):
            if len(cmd) > 1 and cmd[1] == "mapper":
                _write_sparse_model(output)
            return {
                "cmd": cmd,
                "cwd": cwd,
                "timeout": timeout,
                "returncode": 0,
                "stdout": "ok",
                "stderr": "",
                "duration_seconds": 0.001,
                "success": True,
                "error": None,
                "timed_out": False,
            }

        try:
            colmap_standard_pipeline.check_colmap_available = lambda: {
                "available": True,
                "colmap_path": "/bin/echo",
                "version": "test",
                "status": "ok",
            }
            colmap_standard_pipeline.run_command = fake_run
            result = colmap_standard_pipeline.run_colmap_standard_pipeline(str(images), str(output), run_dense=False)
        finally:
            colmap_standard_pipeline.check_colmap_available = original_check
            colmap_standard_pipeline.run_command = original_run

        assert result["success"] is True
        assert result["sparse_success"] is True
        assert result["dense_success"] is False
        assert result["mesh_success"] is False
        assert "sparse_model" in result["outputs"]


def test_report_generation_includes_truthfulness_boundaries_for_colmap_status():
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        status = {
            "method": "COLMAP standard SfM",
            "input_type": "uav_frames",
            "frame_count": 12,
            "status": "completed",
            "success": True,
            "sparse_success": True,
            "dense_success": False,
            "mesh_success": False,
            "geo_reference": False,
            "scale_type": "relative",
            "outputs": {"sparse_model": "/tmp/sparse/0"},
        }
        status_path = output / "status.json"
        status_path.write_text(json.dumps(status), encoding="utf-8")

        generated = reconstruction_report.generate_report(output, reconstruction_status_path=status_path)
        report = generated["report"]
        text = Path(generated["paths"]["markdown"]).read_text(encoding="utf-8")

        assert report["input_type"] == "uav_frames"
        assert report["method"] == "COLMAP standard SfM"
        assert report["frame_count"] == 12
        assert report["sparse_success"] is True
        assert report["scale_type"] == "relative"
        assert "No GPS/GCP means no absolute georeferenced rescue route" in "\n".join(report["truthfulness_boundaries"])
        assert "Human review required" in text


def main():
    test_missing_colmap_returns_dependency_missing()
    test_missing_input_folder_returns_invalid_input_when_colmap_available()
    test_single_360_panorama_frame_returns_insufficient_input()
    test_failed_subprocess_command_returns_success_false()
    test_standard_pipeline_writes_status_and_requires_sparse_outputs()
    test_success_only_true_when_expected_sparse_files_exist()
    test_report_generation_includes_truthfulness_boundaries_for_colmap_status()
    print("AeroRescue-AI COLMAP pipeline tests passed.")


if __name__ == "__main__":
    main()
