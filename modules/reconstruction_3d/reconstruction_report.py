"""Generate truthfulness-first 3D reconstruction reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRUTHFULNESS_BOUNDARIES = [
    "Reconstruction is relative-scale unless reliable GPS/GCP constraints are available and used.",
    "A 360 panorama preview is not 3D reconstruction.",
    "Reconstruction depends on image overlap, parallax, texture, lighting, and blur.",
    "No GPS/GCP means no absolute georeferenced rescue route.",
    "Output is auxiliary spatial evidence only.",
    "Human review required.",
]


def _unique_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def build_report(
    output_dir: str | Path,
    source_video: str | None = None,
    frames_metadata: dict[str, Any] | None = None,
    quality_metadata: dict[str, Any] | None = None,
    reconstruction_status: dict[str, Any] | None = None,
    gps_available: bool = False,
    gcp_available: bool = False,
) -> dict[str, Any]:
    reconstruction_status = reconstruction_status or {}
    scale_status = "gps_or_gcp_constrained" if gps_available or gcp_available else "relative_scale_only"
    raw_geo_reference = reconstruction_status.get("geo_reference")
    geo_reference = bool(gps_available or gcp_available or raw_geo_reference in {True, "available", "true", "georeferenced"})
    scale_type = reconstruction_status.get("scale_type") or ("georeferenced" if geo_reference else "relative")
    frame_count = reconstruction_status.get("frame_count")
    if frame_count is None:
        frame_count = (frames_metadata or {}).get("extracted_count") or (frames_metadata or {}).get("frame_count") or 0
    image_count = reconstruction_status.get("image_count")
    output_availability = _output_availability(reconstruction_status.get("outputs") or {})
    limitations = _unique_texts(TRUTHFULNESS_BOUNDARIES + list(reconstruction_status.get("limitations") or []))
    return {
        "report_type": "AeroRescue-AI 3D Reconstruction Report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_video": source_video,
        "input_type": reconstruction_status.get("input_type", "unknown"),
        "method": reconstruction_status.get("method", "not run"),
        "frame_count": frame_count,
        "image_count": image_count,
        "docker_image": reconstruction_status.get("docker_image"),
        "camera_lens": reconstruction_status.get("camera_lens"),
        "sparse_success": bool(reconstruction_status.get("sparse_success", False)),
        "dense_success": bool(reconstruction_status.get("dense_success", False)),
        "mesh_success": bool(reconstruction_status.get("mesh_success", False)),
        "geo_reference": "available" if geo_reference else (raw_geo_reference if raw_geo_reference is not None else False),
        "scale_type": scale_type,
        "output_availability": output_availability,
        "scale_status": scale_status,
        "gps_available": bool(gps_available),
        "gcp_available": bool(gcp_available),
        "frames": frames_metadata or {},
        "quality_filter": quality_metadata or {},
        "reconstruction": reconstruction_status,
        "truthfulness_boundaries": limitations,
        "limitations": limitations,
        "decision_support_note": "This report is auxiliary decision support for human review. It is not an autonomous rescue command, GPS route, or flight-control instruction.",
        "output_dir": str(Path(output_dir)),
    }


def _output_availability(outputs: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(outputs, dict):
        return {}
    keys = [
        "orthophoto",
        "dsm",
        "dtm",
        "point_cloud_laz",
        "point_cloud_las",
        "textured_model_obj",
        "textured_model_geo_obj",
        "report_pdf",
    ]
    availability: dict[str, Any] = {}
    for key in keys:
        value = outputs.get(key)
        if isinstance(value, dict):
            availability[key] = {"exists": bool(value.get("exists")), "path": value.get("path")}
        elif value:
            availability[key] = {"exists": True, "path": value}
    return availability


def write_report(
    output_dir: str | Path,
    report: dict[str, Any],
    json_name: str = "reconstruction_report.json",
    md_name: str = "reconstruction_report.md",
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / json_name
    md_path = output / md_name
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def render_markdown(report: dict[str, Any]) -> str:
    reconstruction = report.get("reconstruction") or {}
    outputs = reconstruction.get("outputs") or {}
    lines = [
        "# AeroRescue-AI 3D Reconstruction Report",
        "",
        f"- Generated UTC: {report.get('generated_at_utc')}",
        f"- Source video: {report.get('source_video') or 'not provided'}",
        f"- Input type: {report.get('input_type')}",
        f"- Reconstruction method: {report.get('method')}",
        f"- Reconstruction status: {reconstruction.get('status', 'not run')}",
        f"- Success: {bool(reconstruction.get('success'))}",
        f"- Frame count: {report.get('frame_count')}",
        f"- Image count: {report.get('image_count')}",
        f"- Docker image: {report.get('docker_image') or 'not applicable'}",
        f"- Camera lens: {report.get('camera_lens') or 'not applicable'}",
        f"- Sparse success: {report.get('sparse_success')}",
        f"- Dense success: {report.get('dense_success')}",
        f"- Mesh success: {report.get('mesh_success')}",
        f"- Geo reference: {report.get('geo_reference')}",
        f"- Scale type: {report.get('scale_type')}",
        "",
        "## Outputs",
        "",
    ]
    if outputs:
        for name, path in outputs.items():
            if isinstance(path, dict):
                lines.append(f"- {name}: {'available' if path.get('exists') else 'missing'}" + (f" (`{path.get('path')}`)" if path.get("path") else ""))
            else:
                lines.append(f"- {name}: `{path}`")
    else:
        lines.append("- No verified reconstruction output files were reported.")
    availability = report.get("output_availability") or {}
    if availability:
        lines.extend(["", "## ODM Output Availability", ""])
        for name, entry in availability.items():
            lines.append(f"- {name}: {'available' if entry.get('exists') else 'missing'}" + (f" (`{entry.get('path')}`)" if entry.get("path") else ""))
    lines.extend(["", "## Truthfulness Boundaries", ""])
    for boundary in report.get("truthfulness_boundaries", TRUTHFULNESS_BOUNDARIES):
        lines.append(f"- {boundary}")
    lines.extend(["", "## Decision Support", "", report.get("decision_support_note", "")])
    return "\n".join(lines).rstrip() + "\n"


def generate_report(
    output_dir: str | Path,
    source_video: str | None = None,
    frames_metadata_path: str | Path | None = None,
    quality_metadata_path: str | Path | None = None,
    reconstruction_status_path: str | Path | None = None,
    gps_available: bool = False,
    gcp_available: bool = False,
) -> dict[str, Any]:
    report = build_report(
        output_dir,
        source_video=source_video,
        frames_metadata=_load_json(frames_metadata_path),
        quality_metadata=_load_json(quality_metadata_path),
        reconstruction_status=_load_json(reconstruction_status_path),
        gps_available=gps_available,
        gcp_available=gcp_available,
    )
    paths = write_report(output_dir, report)
    return {"report": report, "paths": paths}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a 3D reconstruction truthfulness report.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-video")
    parser.add_argument("--frames-metadata")
    parser.add_argument("--quality-metadata")
    parser.add_argument("--reconstruction-status")
    parser.add_argument("--gps-available", action="store_true")
    parser.add_argument("--gcp-available", action="store_true")
    args = parser.parse_args(argv)
    payload = generate_report(
        args.output,
        source_video=args.source_video,
        frames_metadata_path=args.frames_metadata,
        quality_metadata_path=args.quality_metadata,
        reconstruction_status_path=args.reconstruction_status,
        gps_available=args.gps_available,
        gcp_available=args.gcp_available,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
