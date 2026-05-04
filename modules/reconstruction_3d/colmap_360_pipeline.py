"""COLMAP 360 equirectangular panorama reconstruction pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from shutil import which
from typing import Any, Optional

from .colmap_utils import check_colmap_available, ensure_dir, image_files, list_expected_outputs, run_command

LIMITATIONS = [
    "This pipeline is for true 360 equirectangular image sequences, not ordinary perspective frames.",
    "It requires actual camera motion and parallax between panorama frames.",
    "A 360 panorama preview is not 3D reconstruction.",
    "Reconstruction depends on image overlap, parallax, texture, lighting, and blur.",
    "No GPS/GCP means no absolute georeferenced rescue route.",
    "3D reconstruction without GPS/GCP is relative-scale only.",
    "Output is auxiliary spatial evidence only.",
    "Human review required.",
]


def _status_base(panorama_image_dir: str, output_dir: str, frame_count: int) -> dict[str, Any]:
    return {
        "success": False,
        "status": "prepared",
        "method": "COLMAP 360 panorama SfM",
        "method_id": "colmap_360_panorama_sfm",
        "input_type": "360_panorama_frames",
        "panorama_image_dir": str(panorama_image_dir),
        "output_dir": str(output_dir),
        "frame_count": frame_count,
        "sparse_success": False,
        "dense_success": False,
        "mesh_success": False,
        "geo_reference": False,
        "scale_type": "relative",
        "outputs": {},
        "commands": [],
        "limitations": LIMITATIONS,
    }


def _write_status(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    status_path = output_dir / "reconstruction_status.json"
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["status_path"] = str(status_path)
    return payload


def _summarize_command(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "cmd": result.get("cmd"),
        "returncode": result.get("returncode"),
        "duration_seconds": result.get("duration_seconds"),
        "success": result.get("success"),
        "timed_out": result.get("timed_out"),
        "stdout_tail": (result.get("stdout") or "")[-2000:],
        "stderr_tail": (result.get("stderr") or "")[-2000:],
        "error": result.get("error"),
    }


def run_colmap_360_pipeline(
    panorama_image_dir: str,
    output_dir: str,
    panorama_sfm_script: Optional[str] = None,
    timeout: Optional[int] = None,
) -> dict[str, Any]:
    output = ensure_dir(output_dir)
    logs_dir = ensure_dir(output / "logs")
    images = image_files(panorama_image_dir)
    payload = _status_base(panorama_image_dir, output_dir, len(images))

    colmap = check_colmap_available()
    payload["colmap"] = colmap
    if not colmap["available"]:
        payload.update(
            {
                "status": "dependency_missing",
                "message": "COLMAP executable not found. 360 panorama reconstruction was not run.",
            }
        )
        return _write_status(output, payload)

    if not panorama_sfm_script:
        payload.update(
            {
                "status": "script_missing",
                "message": "panorama_sfm.py was not provided. 360 panorama reconstruction was not run.",
                "missing_dependency": "COLMAP panorama_sfm.py script",
            }
        )
        return _write_status(output, payload)

    script_path = Path(panorama_sfm_script)
    if not script_path.exists():
        payload.update(
            {
                "status": "script_missing",
                "message": f"panorama_sfm.py was not found: {panorama_sfm_script}",
                "missing_dependency": "COLMAP panorama_sfm.py script",
            }
        )
        return _write_status(output, payload)

    input_path = Path(panorama_image_dir)
    if not input_path.exists() or not input_path.is_dir():
        payload.update({"status": "invalid_input", "message": f"Input panorama directory does not exist: {panorama_image_dir}"})
        return _write_status(output, payload)
    if not images:
        payload.update({"status": "invalid_input", "message": f"No supported panorama frames found in: {panorama_image_dir}"})
        return _write_status(output, payload)
    if len(images) < 2:
        payload.update(
            {
                "status": "insufficient_input",
                "message": "At least two 360 panorama frames with camera motion/parallax are required. A single panorama image is not 3D reconstruction.",
            }
        )
        return _write_status(output, payload)

    command = [
        sys.executable,
        str(script_path),
        "--input_image_path",
        str(input_path),
        "--output_path",
        str(output),
    ]
    result = run_command(command, cwd=None, timeout=timeout)
    log_path = logs_dir / "panorama_sfm.json"
    log_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = _summarize_command(result)
    summary["name"] = "panorama_sfm"
    summary["log_path"] = str(log_path)
    payload["commands"].append(summary)

    payload.update(list_expected_outputs(output))
    payload["panorama_sfm_script"] = str(script_path)
    if not result["success"]:
        payload.update({"status": "failed", "message": "COLMAP panorama_sfm.py command failed."})
        payload["success"] = False
        return _write_status(output, payload)

    payload["success"] = bool(payload["sparse_success"])
    payload["status"] = "completed" if payload["success"] else "failed"
    if payload["success"]:
        payload["message"] = "COLMAP 360 panorama SfM completed and sparse output files were verified."
    else:
        payload["message"] = "panorama_sfm.py finished, but verified reconstruction output files were not found."
    return _write_status(output, payload)


def run_colmap_360(
    frames_dir: str | Path,
    output_dir: str | Path,
    colmap_binary: str = "colmap",
    panorama_script: str | Path | None = None,
    python_executable: str = "python",
    execute: bool = False,
) -> dict[str, Any]:
    """Backward-compatible Phase 1 wrapper.

    The Phase 2 runner always attempts the real panorama script when all
    dependencies and inputs are present. The old execute/python/colmap_binary
    arguments are retained so older tests and callers keep working.
    """
    del python_executable, execute
    custom_colmap_missing = colmap_binary != "colmap" and not (Path(colmap_binary).exists() or which(colmap_binary))
    if custom_colmap_missing:
        output = ensure_dir(output_dir)
        images = image_files(frames_dir)
        payload = _status_base(str(frames_dir), str(output_dir), len(images))
        payload.update(
            {
                "status": "dependency_missing",
                "message": "COLMAP executable not found. 360 panorama reconstruction was not run.",
                "missing_dependency": "COLMAP executable",
            }
        )
        return _write_status(output, payload)
    return run_colmap_360_pipeline(str(frames_dir), str(output_dir), panorama_sfm_script=str(panorama_script) if panorama_script else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run COLMAP 360 panorama SfM.")
    parser.add_argument("--panorama-image-dir", "--frames", dest="panorama_image_dir", required=True)
    parser.add_argument("--output-dir", "--output", dest="output_dir", required=True)
    parser.add_argument("--panorama-sfm-script", "--panorama-script", dest="panorama_sfm_script")
    parser.add_argument("--timeout", type=int)
    args = parser.parse_args(argv)
    payload = run_colmap_360_pipeline(
        args.panorama_image_dir,
        args.output_dir,
        panorama_sfm_script=args.panorama_sfm_script,
        timeout=args.timeout,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("success") or payload.get("status") in {"dependency_missing", "script_missing"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
