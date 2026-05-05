"""Thermal support evidence helpers for AeroRescue-AI S6.

Thermal records are auxiliary support for human review. Simulated thermal is
never treated as real temperature measurement, and missing thermal inputs never
produce invented hotspots or temperature values.
"""

from __future__ import annotations

import json
from pathlib import Path


THERMAL_TRUTHFULNESS_NOTE = (
    "Thermal hotspot evidence is only auxiliary support and not confirmation of a civilian. "
    "RGB-thermal matching may be approximate unless calibrated registration is provided. "
    "Thermal evidence requires human review."
)
SIMULATED_THERMAL_NOTE = (
    "Simulated Thermal is not real temperature measurement. "
    "RGB/JPG/PNG images cannot provide real temperature_matrix."
)
RADIOMETRIC_THERMAL_NOTE = (
    "Radiometric thermal results are valid only when a real temperature matrix is successfully parsed."
)
NO_THERMAL_INVENTION_NOTE = "The system must not invent thermal hotspots or temperature values."


def _as_path(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)


def normalize_thermal_inputs(thermal_images):
    """Return existing thermal image paths from strings or Gradio file values."""
    files = thermal_images or []
    if not isinstance(files, (list, tuple)):
        files = [files]
    paths = []
    for item in files:
        raw_path = _as_path(item)
        if raw_path and Path(raw_path).exists():
            paths.append(str(Path(raw_path)))
    return paths


def _load_thermal_result(thermal_result):
    if thermal_result is None:
        return {}
    if isinstance(thermal_result, str):
        try:
            return json.loads(thermal_result)
        except Exception:
            return {"thermal_mode": "unknown", "parse_error": "Unable to parse thermal_result JSON string."}
    if isinstance(thermal_result, dict):
        return dict(thermal_result)
    return {"thermal_mode": "unknown", "parse_error": "Unsupported thermal_result type."}


def _normalize_mode(value):
    text = str(value or "unknown").strip().lower()
    if "radiometric" in text:
        return "radiometric"
    if "simulated" in text:
        return "simulated"
    if "infrared" in text:
        return "infrared_detection"
    return text if text in {"simulated", "radiometric", "infrared_detection"} else "unknown"


def extract_hotspot_summary(thermal_result):
    """Extract a uniform hotspot summary from existing thermal analysis output."""
    data = _load_thermal_result(thermal_result)
    thermal_mode = _normalize_mode(data.get("thermal_mode") or data.get("mode"))
    real_measurement = bool(data.get("is_real_temperature_measurement"))
    temperature_matrix_available = False
    if thermal_mode == "radiometric" and real_measurement:
        temperature_matrix_available = bool(
            data.get("temperature_matrix_available")
            or data.get("temperature_matrix_path")
            or data.get("temperature_matrix") is not None
        )
    if thermal_mode == "simulated":
        temperature_matrix_available = False
        real_measurement = False

    max_temperature = data.get("max_temperature") if temperature_matrix_available else None
    mean_temperature = data.get("mean_temperature") if temperature_matrix_available else None
    unit = data.get("unit") or ("Celsius" if temperature_matrix_available else "none")
    if not temperature_matrix_available:
        unit = "none"

    truthfulness_note = THERMAL_TRUTHFULNESS_NOTE
    if thermal_mode == "simulated":
        truthfulness_note = f"{truthfulness_note} {SIMULATED_THERMAL_NOTE}"
    elif thermal_mode == "radiometric":
        truthfulness_note = f"{truthfulness_note} {RADIOMETRIC_THERMAL_NOTE}"
    elif thermal_mode == "unknown":
        truthfulness_note = f"{truthfulness_note} {NO_THERMAL_INVENTION_NOTE}"

    return {
        "thermal_mode": thermal_mode,
        "hotspot_count": int(data.get("hotspot_count") or 0),
        "hotspot_area_ratio": float(data.get("hotspot_area_ratio") or 0.0),
        "risk_level": str(data.get("risk_level") or "Unknown"),
        "temperature_matrix_available": temperature_matrix_available,
        "is_real_temperature_measurement": real_measurement and temperature_matrix_available,
        "max_temperature": max_temperature,
        "mean_temperature": mean_temperature,
        "unit": unit,
        "truthfulness_note": truthfulness_note,
    }


def estimate_thermal_support_level(hotspot_summary):
    """Estimate a bounded support level from normalized hotspot summary."""
    summary = hotspot_summary or {}
    thermal_mode = summary.get("thermal_mode", "unknown")
    hotspot_count = int(summary.get("hotspot_count") or 0)
    risk_level = str(summary.get("risk_level") or "Unknown")
    area_ratio = float(summary.get("hotspot_area_ratio") or 0.0)

    if thermal_mode == "unknown":
        return {
            "thermal_support_level": "unavailable",
            "thermal_support_score": 0,
            "reason": "No usable thermal result is available; the system must not invent thermal hotspots or temperature values.",
        }
    if hotspot_count <= 0:
        return {
            "thermal_support_level": "none",
            "thermal_support_score": 0,
            "reason": "No thermal hotspot was reported near the candidate.",
        }
    if risk_level == "High" or area_ratio >= 0.05:
        return {
            "thermal_support_level": "strong",
            "thermal_support_score": 20,
            "reason": "Thermal result reports high hotspot risk or substantial hotspot area.",
        }
    if risk_level == "Medium" or area_ratio >= 0.015:
        return {
            "thermal_support_level": "weak",
            "thermal_support_score": 10,
            "reason": "Thermal result reports limited hotspot support and requires human review.",
        }
    return {
        "thermal_support_level": "none",
        "thermal_support_score": 0,
        "reason": "Thermal result does not provide meaningful hotspot support.",
    }


def build_thermal_check_record(
    verification_record,
    thermal_image_path="",
    thermal_result=None,
    check_index=1,
    rgb_thermal_alignment="unregistered_or_approximate",
    source_type="simulated_thermal",
):
    """Build one thermal support record for a verification target."""
    verification_record = dict(verification_record or {})
    hotspot_summary = extract_hotspot_summary(thermal_result)
    support = estimate_thermal_support_level(hotspot_summary)
    truthfulness_note = THERMAL_TRUTHFULNESS_NOTE
    if source_type == "simulated_thermal" or hotspot_summary.get("thermal_mode") == "simulated":
        truthfulness_note = f"{truthfulness_note} {SIMULATED_THERMAL_NOTE}"
    if hotspot_summary.get("thermal_mode") == "radiometric":
        truthfulness_note = f"{truthfulness_note} {RADIOMETRIC_THERMAL_NOTE}"
    if support["thermal_support_level"] == "unavailable":
        truthfulness_note = f"{truthfulness_note} {NO_THERMAL_INVENTION_NOTE}"
    if rgb_thermal_alignment == "unregistered_or_approximate":
        truthfulness_note = f"{truthfulness_note} RGB-thermal matching may be approximate unless calibrated registration is provided."

    return {
        "thermal_check_id": f"TH{int(check_index):03d}",
        "verification_id": verification_record.get("verification_id", ""),
        "candidate_id": verification_record.get("candidate_id", ""),
        "target_id": verification_record.get("target_id", ""),
        "area_id": verification_record.get("area_id", ""),
        "class_name": verification_record.get("class_name", ""),
        "thermal_image_path": str(thermal_image_path or ""),
        "thermal_mode": hotspot_summary["thermal_mode"],
        "hotspot_count": hotspot_summary["hotspot_count"],
        "hotspot_area_ratio": hotspot_summary["hotspot_area_ratio"],
        "risk_level": hotspot_summary["risk_level"],
        "thermal_support_level": support["thermal_support_level"],
        "thermal_support_score": support["thermal_support_score"],
        "temperature_matrix_available": hotspot_summary["temperature_matrix_available"],
        "is_real_temperature_measurement": hotspot_summary["is_real_temperature_measurement"],
        "max_temperature": hotspot_summary["max_temperature"],
        "mean_temperature": hotspot_summary["mean_temperature"],
        "unit": hotspot_summary["unit"],
        "rgb_thermal_alignment": str(rgb_thermal_alignment or "unregistered_or_approximate"),
        "source_type": str(source_type or ""),
        "human_review_required": True,
        "is_confirmed_civilian": False,
        "truthfulness_note": truthfulness_note,
        "support_reason": support["reason"],
    }


def _normalize_results(thermal_results):
    if thermal_results is None:
        return []
    if isinstance(thermal_results, (dict, str)):
        return [thermal_results]
    if isinstance(thermal_results, (list, tuple)):
        return list(thermal_results)
    return []


def build_thermal_check_records(
    verification_records,
    thermal_images=None,
    thermal_results=None,
    rgb_thermal_alignment="unregistered_or_approximate",
    source_type="simulated_thermal",
):
    """Build thermal check records for targets requiring thermal review."""
    thermal_paths = normalize_thermal_inputs(thermal_images)
    results = _normalize_results(thermal_results)
    records = []
    thermal_targets = [item for item in (verification_records or []) if item.get("thermal_check_required")]
    for index, verification_record in enumerate(thermal_targets, start=1):
        thermal_image_path = thermal_paths[min(index - 1, len(thermal_paths) - 1)] if thermal_paths else ""
        if results:
            thermal_result = results[index - 1] if index - 1 < len(results) else results[0]
        else:
            thermal_result = {"thermal_mode": "unknown"}
        records.append(
            build_thermal_check_record(
                verification_record,
                thermal_image_path=thermal_image_path,
                thermal_result=thermal_result,
                check_index=index,
                rgb_thermal_alignment=rgb_thermal_alignment,
                source_type=source_type,
            )
        )
    return records


def summarize_thermal_check_records(records):
    """Summarize thermal support records for S7 fusion."""
    records = list(records or [])
    return {
        "thermal_check_count": len(records),
        "strong_support_count": sum(1 for item in records if item.get("thermal_support_level") == "strong"),
        "weak_support_count": sum(1 for item in records if item.get("thermal_support_level") == "weak"),
        "none_support_count": sum(1 for item in records if item.get("thermal_support_level") == "none"),
        "unavailable_count": sum(1 for item in records if item.get("thermal_support_level") == "unavailable"),
        "real_temperature_count": sum(1 for item in records if item.get("is_real_temperature_measurement")),
        "simulated_thermal_count": sum(1 for item in records if item.get("thermal_mode") == "simulated"),
        "human_review_required_count": sum(1 for item in records if item.get("human_review_required")),
    }
