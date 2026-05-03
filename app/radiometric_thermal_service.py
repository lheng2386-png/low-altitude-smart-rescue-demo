"""Radiometric thermal image analysis service for AeroRescue-AI.

This module is intentionally separate from ``thermal_service.py``.

- ``thermal_service.py`` keeps the simulated RGB/gray hotspot fallback.
- This module only reports real temperature measurement when a real
  radiometric temperature matrix is extracted from the uploaded file.

Truthfulness rule:
    A result is real temperature measurement only if ``temperature_matrix`` is
    successfully parsed from a FLIR radiometric JPG or another supported
    radiometric thermal source. Ordinary JPG/PNG files must fail in this mode.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "thermal" / "radiometric"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_FLIR_HINTS = (
    "FLIR",
    "Flir",
    "Thermal",
)
SUPPORTED_DJI_HINTS = (
    "DJI",
    "MAVIC2-ENTERPRISE-ADVANCED",
    "M3T",
    "M30T",
    "H20N",
    "ZH20T",
)


class RadiometricThermalError(RuntimeError):
    """Raised when a file cannot be parsed as real radiometric thermal data."""


def _as_path(file_obj: Any) -> str | None:
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)


def check_radiometric_environment() -> dict[str, Any]:
    """Return dependency status without installing anything automatically."""
    exiftool_path = shutil.which("exiftool")
    return {
        "exiftool_available": exiftool_path is not None,
        "exiftool_path": exiftool_path,
        "dji_sdk_configured": False,
        "dji_sdk_note": (
            "DJI R-JPEG parsing requires DJI Thermal SDK / DIRP. "
            "This repository does not bundle or auto-install DJI SDK binaries."
        ),
    }


def _run_exiftool_bytes(file_path: str | Path, args: list[str] | None = None) -> bytes:
    env = check_radiometric_environment()
    if not env["exiftool_available"]:
        raise RadiometricThermalError(
            "exiftool is not available. Install exiftool before parsing radiometric thermal metadata."
        )
    cmd = [env["exiftool_path"]]
    if args:
        cmd.extend(args)
    cmd.append(str(file_path))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RadiometricThermalError(f"exiftool failed: {stderr or 'unknown error'}")
    return proc.stdout


def run_exiftool(file_path: str | Path, args: list[str] | None = None) -> str:
    """Run exiftool and return UTF-8 text output.

    This public wrapper is useful for smoke checks and debugging. It never
    installs dependencies automatically.
    """
    output = _run_exiftool_bytes(file_path, args=args)
    return output.decode("utf-8", errors="ignore")


def _read_metadata_json(file_path: str | Path) -> dict[str, Any]:
    text = run_exiftool(file_path, ["-j", "-n", "-s", "-a", "-G1"])
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RadiometricThermalError(f"Failed to parse exiftool JSON output: {exc}") from exc
    if not payload:
        return {}
    raw = payload[0]
    metadata: dict[str, Any] = {}
    # exiftool with -G1 prefixes keys like EXIF:Model. Keep both full and short names.
    for key, value in raw.items():
        metadata[key] = value
        if ":" in key:
            metadata[key.split(":", 1)[1]] = value
    return metadata


def extract_flir_metadata(file_path: str | Path) -> dict[str, Any]:
    """Extract radiometric metadata from a possible FLIR radiometric JPG."""
    metadata = _read_metadata_json(file_path)
    flir_keys = [
        "PlanckR1",
        "PlanckB",
        "PlanckF",
        "PlanckO",
        "PlanckR2",
        "Emissivity",
        "ObjectDistance",
        "RawThermalImageWidth",
        "RawThermalImageHeight",
    ]
    metadata["_radiometric_key_hits"] = [key for key in flir_keys if key in metadata]
    return metadata


def detect_radiometric_file_type(file_path: str | Path) -> dict[str, Any]:
    """Detect whether a file appears to contain real radiometric thermal data."""
    file_path = Path(file_path)
    if not file_path.exists():
        return {
            "file_type": "missing",
            "supported": False,
            "reason": f"File does not exist: {file_path}",
        }

    env = check_radiometric_environment()
    binary_hint = b""
    try:
        binary_hint = file_path.read_bytes()[:1024 * 1024]
    except Exception:
        binary_hint = b""

    metadata: dict[str, Any] = {}
    metadata_error = None
    if env["exiftool_available"]:
        try:
            metadata = extract_flir_metadata(file_path)
        except Exception as exc:  # detection should not crash the UI
            metadata_error = str(exc)

    model_text = " ".join(str(metadata.get(key, "")) for key in ["Model", "CameraModelName", "Make"])
    keys_text = " ".join(metadata.keys())
    values_text = " ".join(str(value) for value in metadata.values() if not isinstance(value, (list, dict)))[:4000]
    combined_text = f"{model_text} {keys_text} {values_text}"

    has_flir_binary = b"FLIR" in binary_hint
    has_flir_metadata = all(key in metadata for key in ["PlanckR1", "PlanckB", "PlanckF", "PlanckO", "PlanckR2"])
    has_raw_thermal = "RawThermalImage" in combined_text or "RawThermalImageType" in combined_text
    has_dji_hint = any(hint in combined_text for hint in SUPPORTED_DJI_HINTS) or b"DJI" in binary_hint

    if has_flir_binary or has_flir_metadata or has_raw_thermal:
        return {
            "file_type": "flir_radiometric_jpg",
            "supported": True,
            "parser": "exiftool_flir_planck",
            "camera_model": metadata.get("Model") or metadata.get("CameraModelName") or "Unknown FLIR-compatible camera",
            "metadata_error": metadata_error,
            "radiometric_key_hits": metadata.get("_radiometric_key_hits", []),
        }

    if has_dji_hint:
        return {
            "file_type": "dji_rjpeg",
            "supported": False,
            "parser": "dji_dirp_sdk_not_configured",
            "camera_model": metadata.get("Model") or metadata.get("CameraModelName") or "DJI thermal camera",
            "metadata_error": metadata_error,
            "reason": "DJI R-JPEG appears to be detected, but DJI Thermal SDK / DIRP is not configured in this repository.",
        }

    return {
        "file_type": "ordinary_image_or_unsupported",
        "supported": False,
        "parser": None,
        "camera_model": metadata.get("Model") or metadata.get("CameraModelName") if metadata else None,
        "metadata_error": metadata_error,
        "reason": (
            "This file does not expose FLIR/DJI radiometric metadata. "
            "It may be an ordinary JPG/PNG, a screenshot, a compressed preview, or an unsupported thermal image."
        ),
    }


def _metadata_float(metadata: dict[str, Any], key: str, default: float | None = None) -> float:
    if key not in metadata:
        if default is None:
            raise RadiometricThermalError(f"Missing required FLIR metadata field: {key}")
        return float(default)
    value = metadata[key]
    if isinstance(value, str):
        value = value.replace(" deg C", "").replace(" C", "").replace("m", "").replace("%", "").strip()
    return float(value)


def _extract_flir_raw_thermal_image(file_path: str | Path) -> np.ndarray:
    """Extract FLIR RawThermalImage as a 2D numeric array using exiftool."""
    raw_bytes = _run_exiftool_bytes(file_path, ["-RawThermalImage", "-b"])
    if not raw_bytes or len(raw_bytes) < 32:
        raise RadiometricThermalError(
            "RawThermalImage is missing or empty. This is not a usable FLIR radiometric JPG."
        )
    try:
        image = Image.open(BytesIO(raw_bytes))
        raw = np.array(image)
    except Exception as exc:
        raise RadiometricThermalError(f"Failed to decode RawThermalImage: {exc}") from exc
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    raw = raw.astype(np.float32)
    if raw.size == 0:
        raise RadiometricThermalError("RawThermalImage decoded to an empty matrix.")
    return raw


def parse_flir_radiometric_jpg(file_path: str | Path) -> np.ndarray:
    """Parse a FLIR radiometric JPG into an estimated Celsius temperature matrix.

    The calculation uses FLIR Planck metadata and RawThermalImage extracted by
    exiftool. It refuses to run when those fields are missing.
    """
    metadata = extract_flir_metadata(file_path)
    required = ["PlanckR1", "PlanckB", "PlanckF", "PlanckO", "PlanckR2"]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise RadiometricThermalError(f"Missing FLIR Planck metadata fields: {missing}")

    raw = _extract_flir_raw_thermal_image(file_path)
    planck_r1 = _metadata_float(metadata, "PlanckR1")
    planck_b = _metadata_float(metadata, "PlanckB")
    planck_f = _metadata_float(metadata, "PlanckF")
    planck_o = _metadata_float(metadata, "PlanckO")
    planck_r2 = _metadata_float(metadata, "PlanckR2")

    denominator = planck_r2 * (raw + planck_o)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        temp_kelvin = planck_b / np.log((planck_r1 / denominator) + planck_f)
        temp_celsius = temp_kelvin - 273.15

    temp_celsius = temp_celsius.astype(np.float32)
    finite = np.isfinite(temp_celsius)
    if not np.any(finite):
        raise RadiometricThermalError("FLIR temperature conversion produced no finite values.")
    temp_celsius[~finite] = np.nanmedian(temp_celsius[finite])
    return temp_celsius


def parse_dji_rjpeg(file_path: str | Path) -> np.ndarray:
    """Placeholder for DJI R-JPEG parsing.

    DJI R-JPEG real measurement requires DJI Thermal SDK / DIRP binaries. This
    repository does not bundle them, so this function fails honestly instead of
    fabricating a temperature matrix.
    """
    raise RadiometricThermalError(
        "DJI R-JPEG detected, but DJI Thermal SDK / DIRP integration is not configured. "
        "Do not treat this file as real temperature measurement until DIRP returns a temperature matrix."
    )


def compute_temperature_statistics(
    temperature_matrix: np.ndarray,
    threshold_celsius: float | None = None,
) -> dict[str, Any]:
    """Compute truthful statistics from a real temperature matrix."""
    matrix = np.asarray(temperature_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("temperature_matrix must be a non-empty 2D array")
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError("temperature_matrix contains no finite values")
    threshold = float(threshold_celsius) if threshold_celsius is not None else float(np.percentile(finite, 95))
    hotspot = matrix >= threshold
    return {
        "min_temperature": round(float(np.min(finite)), 2),
        "max_temperature": round(float(np.max(finite)), 2),
        "mean_temperature": round(float(np.mean(finite)), 2),
        "median_temperature": round(float(np.median(finite)), 2),
        "hotspot_threshold": round(threshold, 2),
        "hotspot_area_ratio": round(float(np.count_nonzero(hotspot)) / float(matrix.size), 4),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
    }


def render_temperature_heatmap(temperature_matrix: np.ndarray, output_path: str | Path) -> str:
    """Render a visual heatmap from a parsed temperature matrix."""
    matrix = np.asarray(temperature_matrix, dtype=np.float32)
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError("temperature_matrix contains no finite values")
    clipped = np.nan_to_num(matrix, nan=float(np.median(finite)))
    normalized = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), heatmap)
    return str(output_path)


def create_hotspot_overlay(
    base_image_path_or_array: str | Path | np.ndarray,
    temperature_matrix: np.ndarray,
    output_path: str | Path,
    threshold_celsius: float | None = None,
) -> str:
    """Create a visual overlay for hotspots using a real temperature matrix."""
    if isinstance(base_image_path_or_array, np.ndarray):
        base = base_image_path_or_array
    else:
        base = cv2.imread(str(base_image_path_or_array), cv2.IMREAD_COLOR)
    if base is None:
        raise ValueError("Base image cannot be read for hotspot overlay")
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    base = base[:, :, :3].astype(np.uint8)

    matrix = np.asarray(temperature_matrix, dtype=np.float32)
    stats = compute_temperature_statistics(matrix, threshold_celsius)
    threshold = stats["hotspot_threshold"]
    resized_matrix = cv2.resize(matrix, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    finite = resized_matrix[np.isfinite(resized_matrix)]
    resized_matrix = np.nan_to_num(resized_matrix, nan=float(np.median(finite)) if finite.size else 0.0)
    normalized = cv2.normalize(resized_matrix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(base, 0.45, heatmap, 0.55, 0)

    hotspot = resized_matrix >= threshold
    hotspot_u8 = (hotspot.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(hotspot_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 12:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    return str(output_path)


def analyze_radiometric_thermal(
    file_path: str | Path,
    threshold_celsius: float | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze a real radiometric thermal image.

    Ordinary images fail truthfully in this mode. They should be processed by
    ``thermal_service.analyze_thermal`` only as simulated thermal fallback.
    """
    path = _as_path(file_path)
    if not path:
        return _failure_result("No file uploaded.", file_type="missing")
    path = str(path)
    detection = detect_radiometric_file_type(path)
    if not detection.get("supported"):
        return _failure_result(
            detection.get("reason", "Unsupported radiometric thermal file."),
            file_type=detection.get("file_type"),
            parser=detection.get("parser"),
            camera_model=detection.get("camera_model"),
            detection=detection,
        )

    try:
        if detection["file_type"] == "flir_radiometric_jpg":
            temperature_matrix = parse_flir_radiometric_jpg(path)
        elif detection["file_type"] == "dji_rjpeg":
            temperature_matrix = parse_dji_rjpeg(path)
        else:
            raise RadiometricThermalError(f"Unsupported radiometric type: {detection['file_type']}")

        output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem
        matrix_path = output_dir / f"{stem}_temperature_matrix.npy"
        heatmap_path = output_dir / f"{stem}_radiometric_heatmap.jpg"
        overlay_path = output_dir / f"{stem}_radiometric_overlay.jpg"
        result_path = output_dir / f"{stem}_radiometric_result.json"

        np.save(matrix_path, temperature_matrix)
        render_temperature_heatmap(temperature_matrix, heatmap_path)
        create_hotspot_overlay(path, temperature_matrix, overlay_path, threshold_celsius)
        stats = compute_temperature_statistics(temperature_matrix, threshold_celsius)

        result = {
            "success": True,
            "thermal_mode": "Radiometric Thermal",
            "is_real_temperature_measurement": True,
            "file_type": detection.get("file_type"),
            "parser": detection.get("parser"),
            "camera_model": detection.get("camera_model"),
            "temperature_matrix_path": str(matrix_path),
            "heatmap_path": str(heatmap_path),
            "overlay_path": str(overlay_path),
            "result_path": str(result_path),
            **stats,
            "truthfulness_note": (
                "Real temperature measurement: a temperature_matrix was extracted from radiometric thermal data. "
                "Statistics are computed from the parsed matrix, not from RGB grayscale simulation."
            ),
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        return _failure_result(
            str(exc),
            file_type=detection.get("file_type"),
            parser=detection.get("parser"),
            camera_model=detection.get("camera_model"),
            detection=detection,
        )


def analyze_radiometric_thermal_for_ui(
    file_obj: Any,
    threshold_celsius: float | None = None,
) -> tuple[str | None, str | None, str, str]:
    """Gradio-friendly wrapper returning heatmap, overlay, status, JSON text."""
    result = analyze_radiometric_thermal(file_obj, threshold_celsius=threshold_celsius)
    status = (
        "真实 Radiometric Thermal 解析完成，已生成 temperature_matrix。"
        if result.get("success")
        else "Radiometric Thermal 解析失败：该文件不能证明包含可解析的真实测温数据。"
    )
    if not result.get("success"):
        status += " 可切换到 Simulated Thermal 做热点模拟，但不能称为真实测温。"
    return (
        result.get("heatmap_path"),
        result.get("overlay_path"),
        status,
        json.dumps(result, ensure_ascii=False, indent=2),
    )


def _failure_result(
    message: str,
    file_type: str | None = None,
    parser: str | None = None,
    camera_model: str | None = None,
    detection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "thermal_mode": "Radiometric Thermal",
        "is_real_temperature_measurement": False,
        "file_type": file_type,
        "parser": parser,
        "camera_model": camera_model,
        "error": message,
        "detection": detection or {},
        "truthfulness_note": (
            "No temperature_matrix was extracted. This result must not be described as real temperature measurement. "
            "Use Simulated Thermal only as a visual hotspot fallback for ordinary images."
        ),
    }
