"""Shared COLMAP helpers for AeroRescue-AI reconstruction pipelines."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from shutil import which
from typing import Any, Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_executable(name: str) -> Optional[str]:
    """Return an executable path from PATH, or None when missing."""
    candidate = Path(name).expanduser()
    if candidate.is_absolute() or len(candidate.parts) > 1:
        return str(candidate) if candidate.exists() and candidate.is_file() else None
    return which(name)


def check_colmap_available(executable: str = "colmap") -> dict[str, Any]:
    """Check whether COLMAP is callable and return structured status."""
    colmap_path = find_executable(executable)
    if not colmap_path:
        return {
            "available": False,
            "colmap_path": None,
            "version": "unknown",
            "status": "dependency_missing",
        }

    result = run_command([colmap_path, "--version"], cwd=None, timeout=30)
    if result["returncode"] != 0:
        result = run_command([colmap_path, "help"], cwd=None, timeout=30)
    version = "unknown"
    if result["returncode"] == 0:
        text = (result.get("stdout") or result.get("stderr") or "").strip()
        version = text.splitlines()[0] if text else "unknown"
    return {
        "available": True,
        "colmap_path": colmap_path,
        "version": version,
        "status": "ok",
    }


def run_command(cmd: list[str], cwd: Optional[str] = None, timeout: Optional[int] = None) -> dict[str, Any]:
    """Run a subprocess and capture JSON-serializable execution metadata."""
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
        result = subprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            check=False,
            capture_output=True,
            text=True,
        )
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
        payload.update({"returncode": -1, "error": str(exc), "stderr": str(exc)})
    finally:
        payload["duration_seconds"] = round(time.monotonic() - started, 3)
    return payload


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def file_exists(path: str | Path) -> bool:
    return Path(path).exists()


def image_files(path: str | Path) -> list[Path]:
    directory = Path(path)
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(item for item in directory.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def sparse_model_exists(path: str | Path) -> bool:
    sparse = Path(path)
    cameras = (sparse / "cameras.bin").exists() or (sparse / "cameras.txt").exists()
    images = (sparse / "images.bin").exists() or (sparse / "images.txt").exists()
    points = (sparse / "points3D.bin").exists() or (sparse / "points3D.txt").exists()
    return cameras and images and points


def list_expected_outputs(path: str | Path) -> dict[str, Any]:
    """List verified COLMAP output paths and success booleans."""
    root = Path(path)
    sparse_dir = root / "sparse" / "0"
    dense_dir = root / "dense"
    outputs: dict[str, str] = {}
    if (root / "database.db").exists():
        outputs["database"] = str(root / "database.db")
    if sparse_model_exists(sparse_dir):
        outputs["sparse_model"] = str(sparse_dir)
    if (dense_dir / "fused.ply").exists():
        outputs["dense_fused_ply"] = str(dense_dir / "fused.ply")
    for mesh_name in ["meshed-poisson.ply", "meshed-delaunay.ply", "mesh.ply"]:
        mesh_path = dense_dir / mesh_name
        if mesh_path.exists():
            outputs["mesh"] = str(mesh_path)
            break
    return {
        "outputs": outputs,
        "sparse_success": "sparse_model" in outputs,
        "dense_success": "dense_fused_ply" in outputs,
        "mesh_success": "mesh" in outputs,
    }
