"""End-to-end reconstruction workflow orchestration.

The orchestrator coordinates frame extraction, quality filtering, COLMAP, ODM,
and reporting without weakening the truthfulness boundaries of the individual
pipeline modules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from . import colmap_360_pipeline, colmap_standard_pipeline, frame_quality_filter, odm_pipeline, reconstruction_report, video_to_frames
from .colmap_utils import check_colmap_available
from .odm_utils import check_docker_available, check_odm_image_available

MODES = {"standard_uav", "360_panorama", "odm", "report_only"}

LIMITATIONS = [
    "Fast Preview is not a real ODM orthophoto.",
    "A 360 panorama preview is not 3D reconstruction.",
    "3D reconstruction without GPS/GCP/EXIF geotags is relative-scale or non-georeferenced.",
    "Image-plane or reconstructed-space outputs are not GPS navigation routes.",
    "All outputs are auxiliary spatial evidence and require human review.",
]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _step(name: str, status: str, success: bool = False, payload: Optional[dict[str, Any]] = None, message: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "success": bool(success),
        "message": message,
        "payload": payload or {},
    }


def _image_count(path: str | Path | None) -> int:
    if not path:
        return 0
    directory = Path(path)
    if not directory.exists() or not directory.is_dir():
        return 0
    return sum(1 for item in directory.iterdir() if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})


def check_reconstruction_dependencies(
    include_colmap: bool = True,
    include_odm: bool = True,
    odm_image: str = "opendronemap/odm",
    panorama_sfm_script: Optional[str] = None,
) -> dict[str, Any]:
    """Return environment status for app/UI dependency panels."""
    status: dict[str, Any] = {
        "opencv": {
            "available": video_to_frames.cv2 is not None and frame_quality_filter.cv2 is not None,
            "status": "ok" if video_to_frames.cv2 is not None and frame_quality_filter.cv2 is not None else "dependency_missing",
            "message": "OpenCV is available." if video_to_frames.cv2 is not None and frame_quality_filter.cv2 is not None else "OpenCV/cv2 is not available.",
        },
        "limitations": LIMITATIONS,
    }
    if include_colmap:
        status["colmap"] = check_colmap_available()
        if panorama_sfm_script:
            script = Path(panorama_sfm_script)
            status["panorama_sfm_script"] = {
                "available": script.exists(),
                "path": str(script),
                "status": "ok" if script.exists() else "script_missing",
            }
        else:
            status["panorama_sfm_script"] = {"available": False, "path": None, "status": "script_missing"}
    if include_odm:
        status["docker"] = check_docker_available()
        status["odm_image"] = check_odm_image_available(odm_image) if status["docker"]["available"] else {
            "available": False,
            "image_name": odm_image,
            "status": "dependency_missing",
            "message": "Docker is missing, so the ODM image was not checked.",
        }
    return status


def run_reconstruction_workflow(
    mode: str,
    output_dir: str,
    video_path: Optional[str] = None,
    image_dir: Optional[str] = None,
    fps: float = 1.0,
    run_quality_filter: bool = True,
    blur_threshold: float = 100.0,
    brightness_threshold: float = 30.0,
    reduce_duplicates: bool = True,
    colmap_matcher: str = "sequential",
    colmap_run_dense: bool = False,
    colmap_run_mesher: bool = False,
    panorama_sfm_script: Optional[str] = None,
    odm_project_name: str = "aerorescue_odm",
    odm_docker_image: str = "opendronemap/odm",
    odm_camera_lens: str = "auto",
    odm_feature_quality: str = "medium",
    odm_pc_quality: str = "medium",
    odm_dsm: bool = True,
    odm_dtm: bool = False,
    odm_use_fast_orthophoto: bool = False,
    odm_auto_pull: bool = False,
    timeout: Optional[int] = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    workflow_status_path = root / "workflow_status.json"
    report_dir = root / "report"
    steps: list[dict[str, Any]] = []
    selected_input_dir: Optional[str] = None
    frames_metadata_path: Optional[str] = None
    quality_metadata_path: Optional[str] = None
    reconstruction_status_path: Optional[str] = None
    reconstruction_status: dict[str, Any] = {}

    payload: dict[str, Any] = {
        "success": False,
        "status": "prepared",
        "mode": mode,
        "input": {"video_path": video_path, "image_dir": image_dir},
        "output_dir": str(root),
        "steps": steps,
        "limitations": LIMITATIONS,
    }

    if mode == "report_only":
        steps.append(_step("input", "skipped", True, message="report_only mode does not require imagery input."))
    elif video_path:
        frames_dir = root / "frames"
        frames = video_to_frames.extract_frames(video_path, frames_dir, fps=fps)
        frames_metadata_path = frames.get("metadata_path") or str(frames_dir / "frames_metadata.json")
        steps.append(_step("video_to_frames", frames.get("status", "unknown"), bool(frames.get("success")), frames))
        if frames.get("success"):
            selected_input_dir = str(frames_dir)
    elif image_dir:
        image_count = _image_count(image_dir)
        if image_count > 0:
            selected_input_dir = image_dir
            steps.append(_step("input_images", "completed", True, {"image_dir": image_dir, "image_count": image_count}))
        else:
            steps.append(_step("input_images", "invalid_input", False, {"image_dir": image_dir, "image_count": image_count}, "No supported image files found."))
    else:
        steps.append(_step("input", "invalid_input", False, message="Provide either video_path or image_dir."))

    if mode != "report_only" and not selected_input_dir:
        payload.update({"status": "invalid_input", "message": "No usable frames or image folder were available for reconstruction."})
    elif mode == "report_only":
        payload["status"] = "report_only"
    else:
        if run_quality_filter:
            selected_dir = root / "selected_frames"
            quality = frame_quality_filter.filter_frames(
                selected_input_dir,
                selected_dir,
                blur_threshold=blur_threshold,
                brightness_threshold=brightness_threshold,
                reduce_duplicates=reduce_duplicates,
            )
            quality_metadata_path = quality.get("metadata_path") or str(selected_dir / "frame_quality_metadata.json")
            steps.append(_step("frame_quality_filter", quality.get("status", "unknown"), bool(quality.get("success")), quality))
            if quality.get("success") and int(quality.get("selected_frames", 0)) > 0:
                selected_input_dir = str(selected_dir)
            elif quality.get("status") == "dependency_missing":
                steps.append(_step("frame_quality_filter_fallback", "skipped", True, {"input_dir": selected_input_dir}, "OpenCV missing; using unfiltered input frames."))
            else:
                payload.update({"status": "invalid_input", "message": "Frame quality filter produced no selected frames."})
                selected_input_dir = None
        else:
            steps.append(_step("frame_quality_filter", "skipped", True, {"input_dir": selected_input_dir}))

        if selected_input_dir:
            if mode == "standard_uav":
                reconstruction_status = colmap_standard_pipeline.run_colmap_standard_pipeline(
                    selected_input_dir,
                    str(root / "colmap_standard"),
                    matcher=colmap_matcher,
                    run_dense=colmap_run_dense,
                    run_mesher=colmap_run_mesher,
                    timeout=timeout,
                )
                reconstruction_status_path = reconstruction_status.get("status_path")
                steps.append(_step("colmap_standard", reconstruction_status.get("status", "unknown"), bool(reconstruction_status.get("success")), reconstruction_status))
            elif mode == "360_panorama":
                reconstruction_status = colmap_360_pipeline.run_colmap_360_pipeline(
                    selected_input_dir,
                    str(root / "colmap_360"),
                    panorama_sfm_script=panorama_sfm_script,
                    timeout=timeout,
                )
                reconstruction_status_path = reconstruction_status.get("status_path")
                steps.append(_step("colmap_360", reconstruction_status.get("status", "unknown"), bool(reconstruction_status.get("success")), reconstruction_status))
            elif mode == "odm":
                reconstruction_status = odm_pipeline.run_odm_pipeline(
                    selected_input_dir,
                    str(root / "odm"),
                    project_name=odm_project_name,
                    docker_image=odm_docker_image,
                    camera_lens=odm_camera_lens,
                    use_fast_orthophoto=odm_use_fast_orthophoto,
                    feature_quality=odm_feature_quality,
                    pc_quality=odm_pc_quality,
                    dsm=odm_dsm,
                    dtm=odm_dtm,
                    auto_pull=odm_auto_pull,
                    timeout=timeout,
                )
                reconstruction_status_path = reconstruction_status.get("status_path")
                steps.append(_step("odm", reconstruction_status.get("status", "unknown"), bool(reconstruction_status.get("success")), reconstruction_status))

    report = reconstruction_report.generate_report(
        report_dir,
        source_video=video_path,
        frames_metadata_path=frames_metadata_path,
        quality_metadata_path=quality_metadata_path,
        reconstruction_status_path=reconstruction_status_path,
    )
    steps.append(_step("reconstruction_report", "completed", True, report))

    failed_or_blocked = [step for step in steps if not step["success"] and step["status"] not in {"dependency_missing", "script_missing", "skipped"}]
    dependency_blocked = [step for step in steps if step["status"] in {"dependency_missing", "script_missing"}]
    if reconstruction_status.get("success"):
        payload["status"] = "success"
        payload["success"] = True
    elif dependency_blocked:
        payload["status"] = "dependency_missing"
        payload["success"] = False
    elif failed_or_blocked:
        payload["status"] = failed_or_blocked[-1]["status"]
        payload["success"] = False
    elif mode == "report_only":
        payload["status"] = "report_only"
        payload["success"] = True
    else:
        payload["status"] = reconstruction_status.get("status", "skipped")
        payload["success"] = False

    payload["report"] = report
    payload["reconstruction_status_path"] = reconstruction_status_path
    _write_json(workflow_status_path, payload)
    payload["workflow_status_path"] = str(workflow_status_path)
    return payload


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run 灾情感知及影响评估 reconstruction workflow.")
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video")
    parser.add_argument("--image-dir")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--run-quality-filter", default="true")
    parser.add_argument("--blur-threshold", type=float, default=100.0)
    parser.add_argument("--brightness-threshold", type=float, default=30.0)
    parser.add_argument("--matcher", default="sequential", choices=["sequential", "exhaustive"])
    parser.add_argument("--run-dense", default="false")
    parser.add_argument("--run-mesher", default="false")
    parser.add_argument("--panorama-sfm-script")
    parser.add_argument("--odm-project-name", default="aerorescue_odm")
    parser.add_argument("--odm-docker-image", default="opendronemap/odm")
    parser.add_argument("--odm-camera-lens", default="auto")
    parser.add_argument("--timeout", type=int)
    args = parser.parse_args(argv)
    result = run_reconstruction_workflow(
        args.mode,
        args.output_dir,
        video_path=args.video,
        image_dir=args.image_dir,
        fps=args.fps,
        run_quality_filter=_parse_bool(args.run_quality_filter),
        blur_threshold=args.blur_threshold,
        brightness_threshold=args.brightness_threshold,
        colmap_matcher=args.matcher,
        colmap_run_dense=_parse_bool(args.run_dense),
        colmap_run_mesher=_parse_bool(args.run_mesher),
        panorama_sfm_script=args.panorama_sfm_script,
        odm_project_name=args.odm_project_name,
        odm_docker_image=args.odm_docker_image,
        odm_camera_lens=args.odm_camera_lens,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
