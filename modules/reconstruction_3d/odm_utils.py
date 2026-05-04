"""Shared Docker/OpenDroneMap helpers for reconstruction pipelines."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from shutil import which
from typing import Any, Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_executable(name: str) -> Optional[str]:
    return which(name)


def run_command(cmd: list[str], cwd: Optional[str] = None, timeout: Optional[int] = None) -> dict[str, Any]:
    started = time.monotonic()
    payload: dict[str, Any] = {
        "cmd": cmd,
        "cwd": cwd,
        "timeout": timeout,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "duration_seconds": 0.0,
        "success": False,
        "error": None,
        "timed_out": False,
    }
    try:
        result = subprocess.run(cmd, cwd=cwd, timeout=timeout, check=False, capture_output=True, text=True)
        payload.update(
            {
                "returncode": result.returncode,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                "success": result.returncode == 0,
            }
        )
    except subprocess.TimeoutExpired as exc:
        payload.update(
            {
                "returncode": -1,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "error": f"Command timed out after {timeout} seconds.",
                "timed_out": True,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive subprocess boundary
        payload.update({"returncode": -1, "stderr": str(exc), "error": str(exc)})
    finally:
        payload["duration_seconds"] = round(time.monotonic() - started, 3)
    return payload


def check_docker_available() -> dict[str, Any]:
    docker_path = find_executable("docker")
    if not docker_path:
        return {
            "available": False,
            "docker_path": None,
            "version": "unknown",
            "status": "dependency_missing",
        }
    result = run_command([docker_path, "--version"], cwd=None, timeout=30)
    version = "unknown"
    if result["returncode"] == 0:
        text = (result.get("stdout") or result.get("stderr") or "").strip()
        version = text.splitlines()[0] if text else "unknown"
    return {
        "available": True,
        "docker_path": docker_path,
        "version": version,
        "status": "ok",
    }


def check_odm_image_available(image_name: str = "opendronemap/odm") -> dict[str, Any]:
    docker = check_docker_available()
    if not docker["available"]:
        return {
            "available": False,
            "image_name": image_name,
            "status": "dependency_missing",
            "message": "Docker CLI is not available.",
        }
    result = run_command([docker["docker_path"], "image", "inspect", image_name], cwd=None, timeout=60)
    if result["returncode"] == 0:
        return {
            "available": True,
            "image_name": image_name,
            "status": "ok",
            "message": "ODM Docker image is available locally.",
        }
    return {
        "available": False,
        "image_name": image_name,
        "status": "image_missing",
        "message": "ODM Docker image is not available locally.",
        "stderr_tail": (result.get("stderr") or "")[-1000:],
    }


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _image_files(path: str | Path) -> list[Path]:
    directory = Path(path)
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(item for item in directory.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def contains_images(path: str | Path) -> bool:
    return bool(_image_files(path))


def count_images(path: str | Path) -> int:
    return len(_image_files(path))


def safe_relpath(path: str | Path, base: str | Path) -> str:
    resolved_path = Path(path).resolve()
    resolved_base = Path(base).resolve()
    try:
        return str(resolved_path.relative_to(resolved_base))
    except ValueError:
        return str(resolved_path)


def image_files(path: str | Path) -> list[Path]:
    return _image_files(path)
