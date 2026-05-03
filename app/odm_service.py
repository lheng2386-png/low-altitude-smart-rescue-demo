import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
ODM_OUTPUT_ROOT = ROOT_DIR / "outputs" / "odm"
ODM_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ODM_IMAGE = "opendronemap/odm:latest"


def as_upload_path(file_obj):
    """Return a local path from a Gradio file value."""
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)


def normalize_uploaded_file_path(file_obj):
    """Compatibility wrapper used by tests and UI helpers."""
    return as_upload_path(file_obj)


def safe_name(value):
    """Create a filesystem-safe task name segment."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "aerorescue_odm_task")).strip("._-")
    return cleaned or "aerorescue_odm_task"


def make_safe_task_id(task_name, timestamp=None):
    """Create a unique, filesystem-safe ODM task id."""
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_name(task_name)}_{timestamp}"


def create_task_dir(task_name):
    """Create an ODM task directory with an images subdirectory."""
    task_id = make_safe_task_id(task_name)
    task_dir = ODM_OUTPUT_ROOT / task_id
    images_dir = task_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return task_id, task_dir, images_dir


def collect_valid_images(image_files, max_images=None):
    """Normalize upload values and keep supported image paths."""
    paths = []
    for item in image_files or []:
        raw_path = as_upload_path(item)
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        paths.append(path)
    if max_images is not None:
        max_images = int(max_images)
        if max_images > 0:
            paths = paths[:max_images]
    return paths


def copy_images_to_task(image_paths, images_dir):
    """Copy images into the ODM task images directory without overwriting duplicates."""
    copied = []
    used_names = set()
    for index, source in enumerate(image_paths, start=1):
        stem = safe_name(source.stem)
        suffix = source.suffix.lower()
        candidate = f"{stem}{suffix}"
        if candidate in used_names or (images_dir / candidate).exists():
            candidate = f"{stem}_{index:03d}{suffix}"
        used_names.add(candidate)
        destination = images_dir / candidate
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def copy_input_images(image_files, images_dir, max_images=None):
    """Normalize Gradio uploads, filter supported images, and copy them into an ODM images dir."""
    image_paths = collect_valid_images(image_files, max_images=max_images)
    return copy_images_to_task(image_paths, images_dir)


def check_docker_available():
    """Check whether Docker is available for local ODM execution."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.returncode == 0, (result.stdout or result.stderr or "").strip()
    except Exception as exc:
        return False, str(exc)


def check_odm_image_available():
    """Check whether the OpenDroneMap Docker image is available locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", ODM_IMAGE],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        message = "\n".join(part for part in [result.stdout, result.stderr] if part)
        return result.returncode == 0, message.strip()
    except Exception as exc:
        return False, str(exc)


def check_odm_environment():
    """Return Gradio-friendly Docker and ODM image environment status."""
    docker_available, docker_message = check_docker_available()
    image_available = False
    image_message = ""
    if docker_available:
        image_available, image_message = check_odm_image_available()

    image_permission_denied = "permission denied" in (image_message or "").lower()
    if docker_available and image_available:
        status = "ODM 环境可用：Docker 正常，已找到 opendronemap/odm 镜像。"
    elif not docker_available:
        status = "Docker 不可用，无法运行真实 ODM 正射处理。"
    elif image_permission_denied:
        status = "Docker CLI 可用，但当前进程无法访问 Docker Engine API，无法确认或运行 ODM 镜像。"
    else:
        status = "Docker 可用，但未找到 opendronemap/odm 镜像，无法运行真实 ODM 正射处理。"

    result = {
        "docker_available": docker_available,
        "docker_message": docker_message,
        "odm_image": ODM_IMAGE,
        "odm_image_available": image_available,
        "odm_image_message": image_message,
        "truthfulness_note": "只有 Docker 可用、ODM 镜像可用，并且运行后生成 odm_orthophoto.tif，才能说明真实 ODM 正射处理完成。",
    }
    log_text = "\n".join(
        [
            f"[docker]\n{docker_message or 'no docker output'}",
            f"\n[odm image: {ODM_IMAGE}]\n{image_message or 'not checked or no output'}",
        ]
    )
    return status, json.dumps(result, ensure_ascii=False, indent=2), log_text


def build_odm_command(task_dir, fast_orthophoto=False):
    """Build the Docker command for OpenDroneMap."""
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{Path(task_dir).resolve()}:/datasets/project",
        ODM_IMAGE,
        "--project-path",
        "/datasets",
    ]
    if fast_orthophoto:
        command.append("--fast-orthophoto")
    return command


def run_subprocess_command(command, log_path):
    """Run a command and persist stdout/stderr to a log file."""
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        log_text = "\n".join(
            [
                "$ " + " ".join(command),
                "\n[stdout]\n" + (result.stdout or ""),
                "\n[stderr]\n" + (result.stderr or ""),
            ]
        )
        Path(log_path).write_text(log_text, encoding="utf-8", errors="ignore")
        return result.returncode, log_text
    except Exception as exc:
        log_text = f"$ {' '.join(command)}\n\n[subprocess exception]\n{exc}"
        Path(log_path).write_text(log_text, encoding="utf-8", errors="ignore")
        return -1, log_text


def _normalize_preview_array(image):
    if image is None:
        return None
    array = np.asarray(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    if array.dtype != np.uint8:
        arr_min = float(np.nanmin(array))
        arr_max = float(np.nanmax(array))
        if arr_max > arr_min:
            array = ((array - arr_min) / (arr_max - arr_min) * 255).clip(0, 255).astype(np.uint8)
        else:
            array = np.zeros(array.shape, dtype=np.uint8)
    max_side = 1800
    h, w = array.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        array = cv2.resize(array, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return array


def make_orthophoto_preview(orthophoto_tif_path, output_path):
    """Create a JPG preview from an ODM GeoTIFF if possible."""
    if not orthophoto_tif_path:
        return None
    path = Path(orthophoto_tif_path)
    if not path.exists():
        return None

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    preview = _normalize_preview_array(image)
    if preview is not None:
        cv2.imwrite(str(output_path), preview)
        return str(output_path)

    try:
        pil_image = Image.open(path)
        pil_image.thumbnail((1800, 1800))
        pil_image.convert("RGB").save(output_path, quality=92)
        return str(output_path)
    except Exception:
        return None


def _existing_path(path):
    path = Path(path)
    return str(path) if path.exists() else ""


def _empty_result(task_id="", task_dir="", image_count=0, docker_available=False, return_code=None, log_path=""):
    return {
        "task_id": task_id,
        "task_dir": str(task_dir) if task_dir else "",
        "image_count": image_count,
        "docker_available": docker_available,
        "odm_return_code": return_code,
        "orthophoto_tif": "",
        "orthophoto_preview": "",
        "point_cloud_laz": "",
        "point_cloud_ply": "",
        "textured_model_obj": "",
        "dsm_tif": "",
        "dtm_tif": "",
        "log_path": str(log_path) if log_path else "",
    }


def collect_odm_outputs(task_dir, orthophoto_preview=""):
    """Collect known ODM output paths without requiring every artifact to exist."""
    task_dir = Path(task_dir)
    return {
        "orthophoto_tif": _existing_path(task_dir / "odm_orthophoto" / "odm_orthophoto.tif"),
        "orthophoto_preview": orthophoto_preview or "",
        "point_cloud_laz": _existing_path(task_dir / "odm_georeferencing" / "odm_georeferenced_model.laz"),
        "point_cloud_ply": _existing_path(task_dir / "odm_georeferencing" / "odm_georeferenced_model.ply"),
        "textured_model_obj": _existing_path(task_dir / "odm_texturing" / "odm_textured_model.obj"),
        "dsm_tif": _existing_path(task_dir / "odm_dem" / "dsm.tif"),
        "dtm_tif": _existing_path(task_dir / "odm_dem" / "dtm.tif"),
    }


def run_odm_task(
    image_files,
    task_name="aerorescue_odm_task",
    max_images=None,
    fast_orthophoto=False,
):
    """Run a local OpenDroneMap Docker task and return Gradio-friendly outputs."""
    image_paths = collect_valid_images(image_files, max_images=max_images)
    if not image_paths:
        result = _empty_result()
        log_text = "未找到有效航测图像。支持 jpg、jpeg、png、tif、tiff。"
        return (
            None,
            "未上传有效图像，无法运行真实 ODM 正射处理。建议上传具有 70%-80% 重叠度的无人机航测照片。",
            json.dumps(result, ensure_ascii=False, indent=2),
            log_text,
        )

    task_id, task_dir, images_dir = create_task_dir(task_name)
    copied_images = copy_images_to_task(image_paths, images_dir)
    log_path = task_dir / "odm_run.log"

    docker_available, docker_message = check_docker_available()
    if not docker_available:
        log_text = f"Docker 不可用，无法运行真实 ODM 正射处理。\n\n{docker_message}"
        log_path.write_text(log_text, encoding="utf-8", errors="ignore")
        result = _empty_result(
            task_id=task_id,
            task_dir=task_dir,
            image_count=len(copied_images),
            docker_available=False,
            return_code=None,
            log_path=log_path,
        )
        return (
            None,
            "Docker 不可用，无法运行真实 ODM 正射处理。已创建任务目录并复制图像，但未伪造 ODM 输出。",
            json.dumps(result, ensure_ascii=False, indent=2),
            log_text,
        )

    command = build_odm_command(task_dir, fast_orthophoto=fast_orthophoto)
    return_code, log_text = run_subprocess_command(command, log_path)

    orthophoto_tif = task_dir / "odm_orthophoto" / "odm_orthophoto.tif"
    preview_path = task_dir / "orthophoto_preview.jpg"
    orthophoto_preview = make_orthophoto_preview(orthophoto_tif, preview_path)
    if orthophoto_tif.exists() and not orthophoto_preview:
        log_text += "\n\n[preview]\n无法从 odm_orthophoto.tif 生成 JPG 预览。"
        log_path.write_text(log_text, encoding="utf-8", errors="ignore")

    result = _empty_result(
        task_id=task_id,
        task_dir=task_dir,
        image_count=len(copied_images),
        docker_available=True,
        return_code=return_code,
        log_path=log_path,
    )
    result.update(collect_odm_outputs(task_dir, orthophoto_preview=orthophoto_preview))

    if return_code != 0:
        status = "ODM 运行失败，请查看日志。未伪造正射影像输出。"
    elif result["orthophoto_tif"]:
        status = "ODM 正射处理完成，已生成 GeoTIFF 正射影像。"
    else:
        status = "ODM 运行完成，但未找到 odm_orthophoto.tif。请检查照片重叠度、GPS/纹理质量和 ODM 日志。"

    if len(copied_images) < 3:
        status += " 当前输入图像较少，建议上传具有 70%-80% 重叠度的无人机航测照片。"

    return orthophoto_preview, status, json.dumps(result, ensure_ascii=False, indent=2), log_text
