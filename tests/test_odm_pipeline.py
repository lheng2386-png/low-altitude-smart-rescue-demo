import json
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.reconstruction_3d import odm_pipeline, reconstruction_report  # noqa: E402
from modules.reconstruction_3d.odm_outputs import detect_odm_outputs  # noqa: E402


def _write_image(path: Path) -> None:
    path.write_bytes(b"placeholder test image")


def _docker_ok():
    return {"available": True, "docker_path": "/usr/bin/docker", "version": "Docker test", "status": "ok"}


def _odm_image_ok(image_name="opendronemap/odm"):
    return {"available": True, "image_name": image_name, "status": "ok", "message": "available"}


def _odm_image_missing(image_name="opendronemap/odm"):
    return {"available": False, "image_name": image_name, "status": "image_missing", "message": "missing"}


def _command_result(returncode=0):
    return {
        "cmd": ["docker", "run"],
        "cwd": None,
        "timeout": None,
        "returncode": returncode,
        "stdout": "stdout",
        "stderr": "stderr" if returncode else "",
        "duration_seconds": 0.01,
        "success": returncode == 0,
        "error": None,
        "timed_out": False,
    }


def test_missing_image_dir_returns_invalid_input():
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "odm"
        result = odm_pipeline.run_odm_pipeline(str(Path(tmp) / "missing"), str(output))
        assert result["status"] == "invalid_input"
        assert result["success"] is False
        assert (output / "reconstruction_status.json").exists()


def test_empty_image_dir_returns_invalid_input():
    with tempfile.TemporaryDirectory() as tmp:
        image_dir = Path(tmp) / "images"
        output = Path(tmp) / "odm"
        image_dir.mkdir()
        result = odm_pipeline.run_odm_pipeline(str(image_dir), str(output))
        assert result["status"] == "invalid_input"
        assert result["success"] is False


def test_missing_docker_returns_dependency_missing():
    with tempfile.TemporaryDirectory() as tmp:
        image_dir = Path(tmp) / "images"
        output = Path(tmp) / "odm"
        image_dir.mkdir()
        _write_image(image_dir / "uav_001.jpg")

        original = odm_pipeline.check_docker_available
        try:
            odm_pipeline.check_docker_available = lambda: {
                "available": False,
                "docker_path": None,
                "version": "unknown",
                "status": "dependency_missing",
            }
            result = odm_pipeline.run_odm_pipeline(str(image_dir), str(output))
        finally:
            odm_pipeline.check_docker_available = original

        assert result["status"] == "dependency_missing"
        assert result["dependency"] == "docker"
        assert result["success"] is False


def test_missing_odm_image_without_auto_pull_returns_dependency_missing():
    with tempfile.TemporaryDirectory() as tmp:
        image_dir = Path(tmp) / "images"
        output = Path(tmp) / "odm"
        image_dir.mkdir()
        _write_image(image_dir / "uav_001.jpg")

        original_docker = odm_pipeline.check_docker_available
        original_image = odm_pipeline.check_odm_image_available
        try:
            odm_pipeline.check_docker_available = _docker_ok
            odm_pipeline.check_odm_image_available = _odm_image_missing
            result = odm_pipeline.run_odm_pipeline(str(image_dir), str(output), auto_pull=False)
        finally:
            odm_pipeline.check_docker_available = original_docker
            odm_pipeline.check_odm_image_available = original_image

        assert result["status"] == "dependency_missing"
        assert result["dependency"] == "odm_docker_image"
        assert result["success"] is False


def test_command_failure_returns_command_failed():
    with tempfile.TemporaryDirectory() as tmp:
        image_dir = Path(tmp) / "images"
        output = Path(tmp) / "odm"
        image_dir.mkdir()
        _write_image(image_dir / "uav_001.jpg")

        original_docker = odm_pipeline.check_docker_available
        original_image = odm_pipeline.check_odm_image_available
        original_run = odm_pipeline.run_command
        try:
            odm_pipeline.check_docker_available = _docker_ok
            odm_pipeline.check_odm_image_available = _odm_image_ok
            odm_pipeline.run_command = lambda cmd, cwd=None, timeout=None: _command_result(returncode=9)
            result = odm_pipeline.run_odm_pipeline(str(image_dir), str(output))
        finally:
            odm_pipeline.check_docker_available = original_docker
            odm_pipeline.check_odm_image_available = original_image
            odm_pipeline.run_command = original_run

        assert result["status"] == "command_failed"
        assert result["success"] is False
        assert result["logs"]["returncode"] == 9


def test_returncode_zero_missing_outputs_returns_output_missing():
    with tempfile.TemporaryDirectory() as tmp:
        image_dir = Path(tmp) / "images"
        output = Path(tmp) / "odm"
        image_dir.mkdir()
        _write_image(image_dir / "uav_001.jpg")

        original_docker = odm_pipeline.check_docker_available
        original_image = odm_pipeline.check_odm_image_available
        original_run = odm_pipeline.run_command
        try:
            odm_pipeline.check_docker_available = _docker_ok
            odm_pipeline.check_odm_image_available = _odm_image_ok
            odm_pipeline.run_command = lambda cmd, cwd=None, timeout=None: _command_result(returncode=0)
            result = odm_pipeline.run_odm_pipeline(str(image_dir), str(output))
        finally:
            odm_pipeline.check_docker_available = original_docker
            odm_pipeline.check_odm_image_available = original_image
            odm_pipeline.run_command = original_run

        assert result["status"] == "output_missing"
        assert result["success"] is False
        assert result["outputs"]["available_output_count"] == 0


def test_detect_odm_outputs_finds_existing_test_files():
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "aerorescue_odm"
        orthophoto = project / "odm_orthophoto" / "odm_orthophoto.tif"
        report = project / "odm_report" / "report.pdf"
        orthophoto.parent.mkdir(parents=True)
        report.parent.mkdir(parents=True)
        orthophoto.write_bytes(b"test orthophoto presence")
        report.write_bytes(b"test report presence")

        outputs = detect_odm_outputs(str(project))
        assert outputs["orthophoto"]["exists"] is True
        assert outputs["report_pdf"]["exists"] is True
        assert outputs["dsm"]["exists"] is False
        assert outputs["available_output_count"] == 2


def test_success_true_only_when_returncode_zero_and_expected_output_exists():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image_dir = tmp_path / "images"
        output = tmp_path / "odm"
        image_dir.mkdir()
        _write_image(image_dir / "uav_001.jpg")

        original_docker = odm_pipeline.check_docker_available
        original_image = odm_pipeline.check_odm_image_available
        original_run = odm_pipeline.run_command

        def fake_run(cmd, cwd=None, timeout=None):
            orthophoto = output / "aerorescue_odm" / "odm_orthophoto" / "odm_orthophoto.tif"
            orthophoto.parent.mkdir(parents=True, exist_ok=True)
            orthophoto.write_bytes(b"test verified output file")
            return _command_result(returncode=0)

        try:
            odm_pipeline.check_docker_available = _docker_ok
            odm_pipeline.check_odm_image_available = _odm_image_ok
            odm_pipeline.run_command = fake_run
            result = odm_pipeline.run_odm_pipeline(str(image_dir), str(output))
        finally:
            odm_pipeline.check_docker_available = original_docker
            odm_pipeline.check_odm_image_available = original_image
            odm_pipeline.run_command = original_run

        assert result["status"] == "success"
        assert result["success"] is True
        assert result["outputs"]["orthophoto"]["exists"] is True
        assert (output / "reconstruction_status.json").exists()


def test_odm_report_includes_truthfulness_boundaries():
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        status = {
            "success": True,
            "status": "success",
            "method": "opendronemap_odm",
            "input_type": "uav_images",
            "image_count": 3,
            "docker_image": "opendronemap/odm",
            "camera_lens": "auto",
            "outputs": {
                "orthophoto": {"exists": True, "path": "/tmp/odm/odm_orthophoto/odm_orthophoto.tif"},
                "available_output_count": 1,
            },
            "geo_reference": "unknown",
            "scale_type": "depends_on_exif_gps_gcp",
        }
        status_path = output / "odm_status.json"
        status_path.write_text(json.dumps(status), encoding="utf-8")

        generated = reconstruction_report.generate_report(output, reconstruction_status_path=status_path)
        report = generated["report"]
        md = Path(generated["paths"]["markdown"]).read_text(encoding="utf-8")

        assert report["method"] == "opendronemap_odm"
        assert report["input_type"] == "uav_images"
        assert report["image_count"] == 3
        assert report["scale_type"] == "depends_on_exif_gps_gcp"
        assert report["output_availability"]["orthophoto"]["exists"] is True
        assert "No GPS/GCP means no absolute georeferenced rescue route" in "\n".join(report["truthfulness_boundaries"])
        assert "Human review required" in md


def main():
    test_missing_image_dir_returns_invalid_input()
    test_empty_image_dir_returns_invalid_input()
    test_missing_docker_returns_dependency_missing()
    test_missing_odm_image_without_auto_pull_returns_dependency_missing()
    test_command_failure_returns_command_failed()
    test_returncode_zero_missing_outputs_returns_output_missing()
    test_detect_odm_outputs_finds_existing_test_files()
    test_success_true_only_when_returncode_zero_and_expected_output_exists()
    test_odm_report_includes_truthfulness_boundaries()
    print("AeroRescue-AI ODM pipeline tests passed.")


if __name__ == "__main__":
    main()
