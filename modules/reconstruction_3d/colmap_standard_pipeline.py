"""Standard COLMAP SfM pipeline for UAV image folders and extracted frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from .colmap_utils import check_colmap_available, ensure_dir, image_files, list_expected_outputs, run_command

LIMITATIONS = [
    "Reconstruction depends on image overlap, parallax, texture, lighting, and blur.",
    "No GPS/GCP means no absolute georeferenced rescue route.",
    "3D reconstruction without GPS/GCP is relative-scale only.",
    "Output is auxiliary spatial evidence only.",
    "Human review required.",
]


def _status_base(image_dir: str, output_dir: str, frame_count: int = 0) -> dict[str, Any]:
    return {
        "success": False,
        "status": "prepared",
        "method": "COLMAP standard SfM",
        "method_id": "colmap_standard_sfm",
        "input_type": "uav_frames",
        "image_dir": str(image_dir),
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


def _run_step(
    name: str,
    cmd: list[str],
    logs_dir: Path,
    payload: dict[str, Any],
    timeout: Optional[int],
) -> bool:
    result = run_command(cmd, cwd=None, timeout=timeout)
    log_path = logs_dir / f"{name}.json"
    log_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = _summarize_command(result)
    summary["name"] = name
    summary["log_path"] = str(log_path)
    payload["commands"].append(summary)
    return bool(result.get("success"))


def run_colmap_standard_pipeline(
    image_dir: str,
    output_dir: str,
    use_gpu: bool = True,
    max_image_size: int = 2000,
    matcher: str = "sequential",
    run_dense: bool = False,
    run_mesher: bool = False,
    timeout: Optional[int] = None,
) -> dict[str, Any]:
    output = ensure_dir(output_dir)
    logs_dir = ensure_dir(output / "logs")
    sparse_dir = ensure_dir(output / "sparse")
    dense_dir = ensure_dir(output / "dense")
    database_path = output / "database.db"

    images = image_files(image_dir)
    payload = _status_base(image_dir, output_dir, frame_count=len(images))
    payload.update(
        {
            "use_gpu": bool(use_gpu),
            "max_image_size": int(max_image_size),
            "matcher": matcher,
            "run_dense_requested": bool(run_dense),
            "run_mesher_requested": bool(run_mesher),
        }
    )

    colmap = check_colmap_available()
    payload["colmap"] = colmap
    if not colmap["available"]:
        payload.update(
            {
                "status": "dependency_missing",
                "message": "COLMAP executable not found. Reconstruction was not run.",
            }
        )
        return _write_status(output, payload)

    if not Path(image_dir).exists() or not Path(image_dir).is_dir():
        payload.update({"status": "invalid_input", "message": f"Input image directory does not exist: {image_dir}"})
        return _write_status(output, payload)
    if not images:
        payload.update({"status": "invalid_input", "message": f"No supported images found in: {image_dir}"})
        return _write_status(output, payload)
    if matcher not in {"sequential", "exhaustive"}:
        payload.update({"status": "invalid_config", "message": "matcher must be 'sequential' or 'exhaustive'."})
        return _write_status(output, payload)

    colmap_path = colmap["colmap_path"]
    gpu_flag = "1" if use_gpu else "0"
    feature_cmd = [
        colmap_path,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(Path(image_dir)),
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.use_gpu",
        gpu_flag,
        "--SiftExtraction.max_image_size",
        str(max_image_size),
    ]
    if not _run_step("feature_extractor", feature_cmd, logs_dir, payload, timeout):
        payload.update({"status": "failed", "message": "COLMAP feature_extractor failed."})
        payload.update(list_expected_outputs(output))
        payload["success"] = False
        return _write_status(output, payload)

    matcher_cmd = [colmap_path, f"{matcher}_matcher", "--database_path", str(database_path)]
    if matcher == "sequential":
        matcher_cmd.extend(["--SequentialMatching.overlap", "10"])
    if not _run_step(f"{matcher}_matcher", matcher_cmd, logs_dir, payload, timeout):
        payload.update({"status": "failed", "message": f"COLMAP {matcher}_matcher failed."})
        payload.update(list_expected_outputs(output))
        payload["success"] = False
        return _write_status(output, payload)

    mapper_cmd = [
        colmap_path,
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(Path(image_dir)),
        "--output_path",
        str(sparse_dir),
    ]
    if not _run_step("mapper", mapper_cmd, logs_dir, payload, timeout):
        payload.update({"status": "failed", "message": "COLMAP mapper failed."})
        payload.update(list_expected_outputs(output))
        payload["success"] = False
        return _write_status(output, payload)

    payload.update(list_expected_outputs(output))
    if "database" not in payload["outputs"] or not payload["sparse_success"]:
        payload.update(
            {
                "status": "failed",
                "message": "COLMAP mapper finished, but database.db or sparse/0 cameras/images/points3D files were not found.",
            }
        )
        payload["success"] = False
        return _write_status(output, payload)

    if run_dense:
        undistorter_cmd = [
            colmap_path,
            "image_undistorter",
            "--image_path",
            str(Path(image_dir)),
            "--input_path",
            str(sparse_dir / "0"),
            "--output_path",
            str(dense_dir),
            "--output_type",
            "COLMAP",
            "--max_image_size",
            str(max_image_size),
        ]
        if not _run_step("image_undistorter", undistorter_cmd, logs_dir, payload, timeout):
            payload.update({"status": "failed", "message": "COLMAP image_undistorter failed."})
            payload.update(list_expected_outputs(output))
            payload["success"] = False
            return _write_status(output, payload)

        patch_match_cmd = [
            colmap_path,
            "patch_match_stereo",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true",
        ]
        if not _run_step("patch_match_stereo", patch_match_cmd, logs_dir, payload, timeout):
            payload.update({"status": "failed", "message": "COLMAP patch_match_stereo failed."})
            payload.update(list_expected_outputs(output))
            payload["success"] = False
            return _write_status(output, payload)

        fusion_cmd = [
            colmap_path,
            "stereo_fusion",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            str(dense_dir / "fused.ply"),
        ]
        if not _run_step("stereo_fusion", fusion_cmd, logs_dir, payload, timeout):
            payload.update({"status": "failed", "message": "COLMAP stereo_fusion failed."})
            payload.update(list_expected_outputs(output))
            payload["success"] = False
            return _write_status(output, payload)
    else:
        payload["dense_status"] = "skipped"

    payload.update(list_expected_outputs(output))
    if run_dense and not payload["dense_success"]:
        payload.update({"status": "failed", "message": "Dense reconstruction was requested, but dense/fused.ply was not found."})
        payload["success"] = False
        return _write_status(output, payload)

    if run_mesher:
        if not payload["dense_success"]:
            payload["mesh_status"] = "skipped_no_dense_fused_ply"
        else:
            mesh_path = dense_dir / "meshed-poisson.ply"
            mesher_cmd = [colmap_path, "poisson_mesher", "--input_path", str(dense_dir / "fused.ply"), "--output_path", str(mesh_path)]
            if not _run_step("poisson_mesher", mesher_cmd, logs_dir, payload, timeout):
                payload.update({"status": "failed", "message": "COLMAP poisson_mesher failed."})
                payload.update(list_expected_outputs(output))
                payload["success"] = False
                return _write_status(output, payload)
    else:
        payload["mesh_status"] = "skipped"

    payload.update(list_expected_outputs(output))
    payload["success"] = bool(
        "database" in payload["outputs"]
        and payload["sparse_success"]
        and (not run_dense or payload["dense_success"])
        and (not run_mesher or payload["mesh_success"])
    )
    payload["status"] = "completed" if payload["success"] else "failed"
    if payload["success"]:
        payload["message"] = "COLMAP reconstruction completed and expected output files were verified."
    else:
        payload["message"] = "COLMAP commands finished, but expected output files were missing."
    return _write_status(output, payload)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run standard COLMAP SfM for UAV frames.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-gpu", default="true")
    parser.add_argument("--max-image-size", type=int, default=2000)
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    parser.add_argument("--run-dense", default="false")
    parser.add_argument("--run-mesher", default="false")
    parser.add_argument("--timeout", type=int)
    args = parser.parse_args(argv)
    result = run_colmap_standard_pipeline(
        args.image_dir,
        args.output_dir,
        use_gpu=_parse_bool(args.use_gpu),
        max_image_size=args.max_image_size,
        matcher=args.matcher,
        run_dense=_parse_bool(args.run_dense),
        run_mesher=_parse_bool(args.run_mesher),
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") or result.get("status") == "dependency_missing" else 1


if __name__ == "__main__":
    raise SystemExit(main())
