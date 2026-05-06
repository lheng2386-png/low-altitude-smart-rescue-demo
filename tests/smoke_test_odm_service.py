import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import odm_service  # noqa: E402


def _write_synthetic_jpg(path):
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 80
    image[:, :, 1] = np.linspace(40, 220, image.shape[1], dtype=np.uint8)
    image[12:36, 20:44, :] = (240, 240, 240)
    Image.fromarray(image).save(path)


def _write_synthetic_tif(path):
    image = np.zeros((32, 40), dtype=np.uint8)
    image[8:24, 10:30] = 180
    Image.fromarray(image).save(path)


def main():
    env_status, env_json, env_log = odm_service.check_odm_environment()
    env_result = json.loads(env_json)
    assert isinstance(env_status, str) and env_status
    assert "docker_available" in env_result
    assert "odm_image_available" in env_result
    assert "truthfulness_note" in env_result
    assert isinstance(env_log, str)

    safe_task_id = odm_service.make_safe_task_id("smoke odm task", timestamp="20260503_153012")
    assert safe_task_id == "smoke_odm_task_20260503_153012"

    preview, status, result_json, log_text = odm_service.run_odm_task([])
    result = json.loads(result_json)
    assert preview is None
    assert "未上传有效图像" in status
    assert result["image_count"] == 0
    assert "未找到有效航测图像" in log_text

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        jpg_path = tmp_dir / "synthetic_odm_input.jpg"
        tif_path = tmp_dir / "synthetic_orthophoto.tif"
        preview_path = tmp_dir / "preview.jpg"
        _write_synthetic_jpg(jpg_path)
        _write_synthetic_tif(tif_path)

        assert odm_service.normalize_uploaded_file_path(str(jpg_path)) == str(jpg_path)

        created_preview = odm_service.make_orthophoto_preview(tif_path, preview_path)
        assert created_preview is not None
        assert Path(created_preview).exists()

        original_check = odm_service.check_docker_available
        try:
            odm_service.check_docker_available = lambda: (False, "docker not installed for smoke test")
            preview, status, result_json, log_text = odm_service.run_odm_task(
                [str(jpg_path)],
                task_name="smoke odm task",
                max_images=1,
                fast_orthophoto=False,
            )
        finally:
            odm_service.check_docker_available = original_check

        result = json.loads(result_json)
        task_dir = Path(result["task_dir"])
        assert preview is None
        assert "Docker 不可用" in status
        assert result["image_count"] == 1
        assert result["docker_available"] is False
        assert task_dir.exists()
        assert (task_dir / "images").exists()
        assert len(list((task_dir / "images").glob("*.jpg"))) == 1
        assert Path(result["log_path"]).exists()
        assert "docker not installed" in log_text

        command = odm_service.build_odm_command(task_dir, fast_orthophoto=True)
        assert command[:3] == ["docker", "run", "--rm"]
        assert any(str(item).startswith("opendronemap/odm") for item in command)
        assert "--fast-orthophoto" in command

        outputs = odm_service.collect_odm_outputs(task_dir)
        assert "orthophoto_tif" in outputs
        assert "point_cloud_ply" in outputs

    print("灾情感知及影响评估 ODM service smoke test passed.")


if __name__ == "__main__":
    main()
