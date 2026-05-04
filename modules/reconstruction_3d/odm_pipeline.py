"""OpenDroneMap Docker pipeline for real UAV photogrammetry outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Optional

from .odm_outputs import detect_odm_outputs, has_major_output
from .odm_utils import check_docker_available, check_odm_image_available, count_images, ensure_dir, image_files, run_command

LENS_OPTIONS = {"auto", "perspective", "fisheye", "spherical", "equirectangular"}
QUALITY_OPTIONS = {"ultra", "high", "medium", "low", "lowest"}

LIMITATIONS = [
    "ODM output quality depends on image overlap, parallax, texture, lighting, blur, and camera metadata.",
    "Without GPS/GCP/RTK, outputs should not be treated as survey-grade georeferenced products.",
    "Fast orthophoto mode is not equivalent to full validated photogrammetric orthophoto.",
    "All outputs are auxiliary spatial evidence and require human review.",
]


def _status_base(image_dir: str, output_dir: str, image_count: int) -> dict[str, Any]:
    return {
        "success": False,
        "status": "prepared",
        "method": "opendronemap_odm",
        "input_type": "uav_images",
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "image_count": image_count,
        "outputs": {},
        "geo_reference": "unknown",
        "scale_type": "depends_on_exif_gps_gcp",
        "limitations": LIMITATIONS,
    }


def _write_status(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    status_path = output_dir / "reconstruction_status.json"
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["status_path"] = str(status_path)
    return payload


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _copy_images_to_workspace(source_dir: str | Path, images_dir: Path) -> list[str]:
    copied: list[str] = []
    used_names: set[str] = set()
    for index, source in enumerate(image_files(source_dir), start=1):
        candidate = source.name
        if candidate in used_names or (images_dir / candidate).exists():
            candidate = f"{source.stem}_{index:04d}{source.suffix.lower()}"
        used_names.add(candidate)
        destination = images_dir / candidate
        shutil.copy2(source, destination)
        copied.append(str(destination))
    return copied


def build_odm_command(
    output_dir: str | Path,
    project_name: str = "aerorescue_odm",
    docker_image: str = "opendronemap/odm",
    camera_lens: str = "auto",
    feature_quality: str = "medium",
    pc_quality: str = "medium",
    dsm: bool = True,
    dtm: bool = False,
    use_fast_orthophoto: bool = False,
    orthophoto_resolution: Optional[float] = None,
    additional_args: Optional[list[str]] = None,
) -> list[str]:
    if camera_lens not in LENS_OPTIONS:
        raise ValueError(f"camera_lens must be one of {sorted(LENS_OPTIONS)}")
    if feature_quality not in QUALITY_OPTIONS:
        raise ValueError(f"feature_quality must be one of {sorted(QUALITY_OPTIONS)}")
    if pc_quality not in QUALITY_OPTIONS:
        raise ValueError(f"pc_quality must be one of {sorted(QUALITY_OPTIONS)}")

    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{Path(output_dir).resolve()}:/datasets",
        docker_image,
        "--project-path",
        "/datasets",
        project_name,
        "--camera-lens",
        camera_lens,
        "--feature-quality",
        feature_quality,
        "--pc-quality",
        pc_quality,
    ]
    if dsm:
        command.append("--dsm")
    if dtm:
        command.append("--dtm")
    if use_fast_orthophoto:
        command.append("--fast-orthophoto")
    if orthophoto_resolution is not None:
        command.extend(["--orthophoto-resolution", str(orthophoto_resolution)])
    command.extend(additional_args or [])
    return command


def _logs_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "returncode": result.get("returncode"),
        "duration_seconds": result.get("duration_seconds"),
        "timed_out": result.get("timed_out"),
        "stdout_tail": (result.get("stdout") or "")[-3000:],
        "stderr_tail": (result.get("stderr") or "")[-3000:],
        "error": result.get("error"),
    }


def run_odm_pipeline(
    image_dir: str,
    output_dir: str,
    project_name: str = "aerorescue_odm",
    docker_image: str = "opendronemap/odm",
    camera_lens: str = "auto",
    use_fast_orthophoto: bool = False,
    feature_quality: str = "medium",
    pc_quality: str = "medium",
    dsm: bool = True,
    dtm: bool = False,
    orthophoto_resolution: Optional[float] = None,
    auto_pull: bool = False,
    timeout: Optional[int] = None,
) -> dict[str, Any]:
    output = ensure_dir(output_dir)
    workspace_images = ensure_dir(output / "images")
    logs_dir = ensure_dir(output / "logs")
    project_dir = output / project_name
    project_images = ensure_dir(project_dir / "images")
    image_count = count_images(image_dir)
    payload = _status_base(image_dir, output_dir, image_count)
    payload.update(
        {
            "project_name": project_name,
            "docker_image": docker_image,
            "camera_lens": camera_lens,
            "feature_quality": feature_quality,
            "pc_quality": pc_quality,
            "dsm_requested": bool(dsm),
            "dtm_requested": bool(dtm),
            "use_fast_orthophoto": bool(use_fast_orthophoto),
            "orthophoto_resolution": orthophoto_resolution,
            "auto_pull": bool(auto_pull),
        }
    )

    source = Path(image_dir)
    if not source.exists() or not source.is_dir():
        payload.update({"status": "invalid_input", "message": f"Input image directory does not exist: {image_dir}"})
        return _write_status(output, payload)
    if image_count == 0:
        payload.update({"status": "invalid_input", "message": f"No supported image files found in: {image_dir}"})
        return _write_status(output, payload)

    docker_status = check_docker_available()
    payload["docker"] = docker_status
    if not docker_status["available"]:
        payload.update(
            {
                "status": "dependency_missing",
                "dependency": "docker",
                "message": "Docker executable not found. ODM reconstruction was not run.",
            }
        )
        return _write_status(output, payload)

    image_status = check_odm_image_available(docker_image)
    payload["odm_image"] = image_status
    if not image_status["available"]:
        if not auto_pull:
            payload.update(
                {
                    "status": "dependency_missing",
                    "dependency": "odm_docker_image",
                    "message": "ODM Docker image not found. Reconstruction was not run.",
                }
            )
            return _write_status(output, payload)
        pull_result = run_command(["docker", "pull", docker_image], cwd=None, timeout=timeout)
        pull_log_path = logs_dir / "docker_pull.json"
        pull_log_path.write_text(json.dumps(pull_result, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["pull"] = _logs_from_result(pull_result)
        payload["pull"]["log_path"] = str(pull_log_path)
        if not pull_result["success"]:
            payload.update(
                {
                    "status": "dependency_missing",
                    "dependency": "odm_docker_image",
                    "message": "ODM Docker image pull failed. Reconstruction was not run.",
                }
            )
            return _write_status(output, payload)

    copied = _copy_images_to_workspace(image_dir, project_images)
    if not any(workspace_images.iterdir()):
        _copy_images_to_workspace(image_dir, workspace_images)
    payload["workspace"] = {
        "images_dir": str(workspace_images),
        "project_images_dir": str(project_images),
        "project_dir": str(project_dir),
        "copied_images": len(copied),
    }
    try:
        command = build_odm_command(
            output,
            project_name=project_name,
            docker_image=docker_image,
            camera_lens=camera_lens,
            feature_quality=feature_quality,
            pc_quality=pc_quality,
            dsm=dsm,
            dtm=dtm,
            use_fast_orthophoto=use_fast_orthophoto,
            orthophoto_resolution=orthophoto_resolution,
        )
    except ValueError as exc:
        payload.update({"status": "invalid_input", "message": str(exc)})
        return _write_status(output, payload)

    payload["command"] = command
    result = run_command(command, cwd=None, timeout=timeout)
    command_log_path = logs_dir / "odm_command.json"
    command_log_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["logs"] = _logs_from_result(result)
    payload["logs"]["command_log_path"] = str(command_log_path)
    outputs = detect_odm_outputs(str(project_dir))
    payload["outputs"] = outputs

    if not result["success"]:
        payload.update({"success": False, "status": "command_failed", "message": "ODM Docker command failed."})
    elif not has_major_output(outputs):
        payload.update(
            {
                "success": False,
                "status": "output_missing",
                "message": "ODM command returned 0, but no major expected ODM output files were found.",
            }
        )
    else:
        payload.update({"success": True, "status": "success", "message": "ODM command completed and expected output files were verified."})
    return _write_status(output, payload)


def collect_existing_outputs(project_dir: str | Path) -> dict[str, str]:
    outputs = detect_odm_outputs(str(project_dir))
    legacy_names = {
        "orthophoto": "orthophoto_tif",
        "dsm": "dsm_tif",
        "dtm": "dtm_tif",
        "point_cloud_laz": "georeferenced_model_laz",
        "textured_model_geo_obj": "textured_model_obj",
        "report_pdf": "report_pdf",
    }
    result: dict[str, str] = {}
    for name, legacy_name in legacy_names.items():
        entry = outputs.get(name) or {}
        if entry.get("exists"):
            result[legacy_name] = entry["path"]
    return result


def run_odm(
    images_dir: str | Path,
    output_dir: str | Path,
    docker_image: str = "opendronemap/odm:latest",
    camera_lens: str = "auto",
    execute: bool = False,
    additional_args: list[str] | None = None,
) -> dict[str, Any]:
    """Backward-compatible Phase 1 wrapper.

    With execute=False this only prepares and returns the command template.
    With execute=True callers should use run_odm_pipeline for full status.
    """
    output = ensure_dir(output_dir)
    project_dir = output / "project"
    project_images = ensure_dir(project_dir / "images")
    status_path = output / "reconstruction_status.json"
    image_count = count_images(images_dir)
    payload: dict[str, Any] = {
        "method": "OpenDroneMap Docker",
        "input_type": "uav_image_folder",
        "image_count": image_count,
        "camera_lens": camera_lens,
        "success": False,
        "status": "prepared",
        "outputs": {},
        "limitations": LIMITATIONS,
    }
    if image_count == 0:
        payload.update({"status": "input_missing", "error": f"No supported image files found in {images_dir}."})
    else:
        _copy_images_to_workspace(images_dir, project_images)
        try:
            command = build_odm_command(
                project_dir,
                project_name="project",
                docker_image=docker_image,
                camera_lens=camera_lens,
                additional_args=additional_args,
            )
        except ValueError as exc:
            payload.update({"status": "invalid_config", "error": str(exc)})
        else:
            payload["command_template"] = command
            if not execute:
                payload["status"] = "command_prepared"
            else:
                payload.update(run_odm_pipeline(str(images_dir), str(output), docker_image=docker_image, camera_lens=camera_lens))
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["status_path"] = str(status_path)
    return payload


def _exit_code(status: str, success: bool) -> int:
    if success:
        return 0
    return {
        "dependency_missing": 2,
        "invalid_input": 3,
        "command_failed": 4,
        "output_missing": 5,
    }.get(status, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OpenDroneMap Docker reconstruction.")
    parser.add_argument("--image-dir", "--images", dest="image_dir", required=True)
    parser.add_argument("--output-dir", "--output", dest="output_dir", required=True)
    parser.add_argument("--project-name", default="aerorescue_odm")
    parser.add_argument("--docker-image", default="opendronemap/odm")
    parser.add_argument("--camera-lens", default="auto", choices=sorted(LENS_OPTIONS))
    parser.add_argument("--feature-quality", default="medium", choices=sorted(QUALITY_OPTIONS))
    parser.add_argument("--pc-quality", default="medium", choices=sorted(QUALITY_OPTIONS))
    parser.add_argument("--dsm", default="true")
    parser.add_argument("--dtm", default="false")
    parser.add_argument("--use-fast-orthophoto", default="false")
    parser.add_argument("--orthophoto-resolution", type=float)
    parser.add_argument("--auto-pull", default="false")
    parser.add_argument("--timeout", type=int)
    args = parser.parse_args(argv)
    payload = run_odm_pipeline(
        args.image_dir,
        args.output_dir,
        project_name=args.project_name,
        docker_image=args.docker_image,
        camera_lens=args.camera_lens,
        use_fast_orthophoto=_parse_bool(args.use_fast_orthophoto),
        feature_quality=args.feature_quality,
        pc_quality=args.pc_quality,
        dsm=_parse_bool(args.dsm),
        dtm=_parse_bool(args.dtm),
        orthophoto_resolution=args.orthophoto_resolution,
        auto_pull=_parse_bool(args.auto_pull),
        timeout=args.timeout,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return _exit_code(payload.get("status", ""), bool(payload.get("success")))


if __name__ == "__main__":
    raise SystemExit(main())
