"""Detect real OpenDroneMap output files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

EXPECTED_OUTPUTS = {
    "orthophoto": "odm_orthophoto/odm_orthophoto.tif",
    "dsm": "odm_dem/dsm.tif",
    "dtm": "odm_dem/dtm.tif",
    "point_cloud_laz": "odm_georeferencing/odm_georeferenced_model.laz",
    "point_cloud_las": "odm_georeferencing/odm_georeferenced_model.las",
    "textured_model_obj": "odm_texturing/odm_textured_model.obj",
    "textured_model_geo_obj": "odm_texturing/odm_textured_model_geo.obj",
    "report_pdf": "odm_report/report.pdf",
}


def detect_odm_outputs(project_dir: str) -> dict[str, Any]:
    root = Path(project_dir)
    result: dict[str, Any] = {}
    count = 0
    for name, rel_path in EXPECTED_OUTPUTS.items():
        path = root / rel_path
        exists = path.exists() and path.is_file()
        if exists:
            count += 1
        result[name] = {"exists": exists, "path": str(path) if exists else None}
    result["available_output_count"] = count
    return result


def has_major_output(outputs: dict[str, Any]) -> bool:
    major = [
        "orthophoto",
        "dsm",
        "dtm",
        "point_cloud_laz",
        "point_cloud_las",
        "textured_model_obj",
        "textured_model_geo_obj",
    ]
    return any(bool((outputs.get(name) or {}).get("exists")) for name in major)
