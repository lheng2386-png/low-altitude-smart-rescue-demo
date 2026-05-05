"""S2 macro_analysis stage wrapper for map-level disaster analysis."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is available in normal test/runtime envs.
    Image = None

try:
    from ..workflow.workflow_orchestrator import initialize_rescue_workflow
    from ._stage_recording import record_stage_evidence
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from workflow.workflow_orchestrator import initialize_rescue_workflow
    from stages._stage_recording import record_stage_evidence


NO_SEGMENTATION_NOTE = (
    "No segmentation mask/model output was provided; macro analysis is limited. "
    "System outputs are decision-support results and not final rescue conclusions."
)
MASK_BOUNDARY_NOTE = (
    "Uploaded/Demo Mask is not automatic model segmentation. "
    "System outputs are decision-support results and not final rescue conclusions."
)
AUTO_SEGMENTATION_NOTE = (
    "Automatic segmentation model output is a macro-analysis aid and requires human review. "
    "System outputs are decision-support results and not final rescue conclusions."
)

ZONE_CLASS_MAP = {
    "water": ("flood_or_water_zone", "High", "水域占比较高，可能影响地面救援通行。"),
    "pool": ("flood_or_water_zone", "High", "积水或水面区域可能影响地面救援通行。"),
    "road_blocked": ("blocked_road_zone", "High", "道路阻断区域可能影响救援车辆通行。"),
    "major_damage": ("damaged_building_zone", "High", "严重损毁建筑区域需要局部精查和人工复核。"),
    "destroyed_building": ("damaged_building_zone", "High", "完全毁坏建筑区域需要局部精查和人工复核。"),
    "road_clear": ("accessible_road_zone", "Low", "可通行道路可作为潜在接近路线，但仍需现场复核。"),
    "vehicle": ("vehicle_zone", "Medium", "车辆区域可能提示道路占用或人员活动线索。"),
    "tree": ("vegetation_or_obstacle_zone", "Medium", "植被或障碍物区域可能影响视线和通行。"),
}


def _stage_output_dir(mission_dir):
    output_dir = Path(mission_dir) / "outputs" / "workflow"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_stage_result(mission_dir, result):
    result_path = _stage_output_dir(mission_dir) / "macro_analysis_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(result_path)


def _normalize_segmentation_source(segmentation_source, has_mask):
    source = str(segmentation_source or "none").strip().lower()
    if not has_mask:
        return "none"
    if source in {"uploaded_mask", "uploaded", "mask", "user_mask"}:
        return "uploaded_mask"
    if source in {"auto_model", "automatic", "auto_segmentation_model"}:
        return "auto_model"
    if source in {"demo", "demo_fallback", "fallback"}:
        return "demo_fallback"
    return "uploaded_mask"


def _source_type(segmentation_source):
    if segmentation_source == "uploaded_mask":
        return "uploaded_mask"
    if segmentation_source == "auto_model":
        return "auto_segmentation_model"
    if segmentation_source == "demo_fallback":
        return "demo_fallback"
    return "rule_based"


def _truthfulness_note(segmentation_source):
    if segmentation_source in {"uploaded_mask", "demo_fallback"}:
        return MASK_BOUNDARY_NOTE
    if segmentation_source == "auto_model":
        return AUTO_SEGMENTATION_NOTE
    return NO_SEGMENTATION_NOTE


def _build_macro_zones(segmentation_summary):
    zones = []
    for class_name, ratio in (segmentation_summary or {}).items():
        if class_name not in ZONE_CLASS_MAP:
            continue
        zone_type, risk_level, reason = ZONE_CLASS_MAP[class_name]
        zones.append(
            {
                "zone_id": f"Z{len(zones) + 1:03d}",
                "zone_type": zone_type,
                "source_class": class_name,
                "risk_level": risk_level,
                "area_percent": round(float(ratio or 0.0) * 100.0, 2),
                "reason": reason,
            }
        )
    return zones


def _save_overlay(map_image_path, mask, mission_dir):
    if not map_image_path or Image is None:
        return ""
    path = Path(map_image_path)
    if not path.exists():
        return ""
    try:
        from ..segmentation_engine import create_segmentation_overlay
    except Exception:
        try:
            from segmentation_engine import create_segmentation_overlay
        except Exception:
            return ""

    try:
        image = Image.open(path).convert("RGB")
        overlay = create_segmentation_overlay(image, mask)
        if overlay is None:
            return ""
        overlay_path = _stage_output_dir(mission_dir) / "macro_analysis_overlay.png"
        Image.fromarray(overlay).save(overlay_path)
        return str(overlay_path)
    except Exception:
        return ""


def run_macro_analysis_stage(
    mission,
    mission_dir,
    map_image_path=None,
    segmentation_mask_path=None,
    segmentation_source="none",
):
    """Run S2 macro disaster analysis from a map image and optional mask."""
    mission = initialize_rescue_workflow(mission)
    mission_dir = Path(mission_dir)
    mask_path = Path(segmentation_mask_path) if segmentation_mask_path else None
    has_mask = bool(mask_path and mask_path.exists())
    normalized_source = _normalize_segmentation_source(segmentation_source, has_mask)
    truthfulness_note = _truthfulness_note(normalized_source)
    segmentation_valid = False
    segmentation_summary = {}
    macro_zones = []
    overlay_path = ""
    status = "degraded"

    if not has_mask:
        result = {
            "stage_key": "macro_analysis",
            "status": status,
            "map_image_path": str(map_image_path or ""),
            "segmentation_source": "none",
            "segmentation_valid": False,
            "segmentation_summary": {},
            "macro_zones": [],
            "overlay_path": "",
            "truthfulness_note": truthfulness_note,
            "human_review_required": True,
        }
        result_path = _save_stage_result(mission_dir, result)
        record_stage_evidence(
            mission,
            mission_dir,
            "macro_analysis",
            "completed",
            output_ref=result_path,
            result_type="macro_segmentation_risk_summary",
            source_type="rule_based",
            truthfulness_note=truthfulness_note,
            limitation=truthfulness_note,
            human_review_required=True,
        )
        return mission, result

    try:
        from ..segmentation_engine import load_segmentation_mask, summarize_segmentation, validate_segmentation_mask
    except Exception:
        try:
            from segmentation_engine import load_segmentation_mask, summarize_segmentation, validate_segmentation_mask
        except Exception as exc:
            result = {
                "stage_key": "macro_analysis",
                "status": "failed",
                "map_image_path": str(map_image_path or ""),
                "segmentation_source": normalized_source,
                "segmentation_valid": False,
                "segmentation_summary": {},
                "macro_zones": [],
                "overlay_path": "",
                "truthfulness_note": f"{truthfulness_note} Segmentation engine unavailable: {exc}",
                "human_review_required": True,
            }
            result_path = _save_stage_result(mission_dir, result)
            record_stage_evidence(
                mission,
                mission_dir,
                "macro_analysis",
                "failed",
                output_ref=result_path,
                result_type="macro_segmentation_risk_summary",
                source_type=_source_type(normalized_source),
                truthfulness_note=result["truthfulness_note"],
                limitation=result["truthfulness_note"],
                human_review_required=True,
            )
            return mission, result

    try:
        mask = load_segmentation_mask(mask_path)
        validation = validate_segmentation_mask(mask)
        segmentation_valid = bool(validation.get("valid"))
        if segmentation_valid:
            segmentation_summary = summarize_segmentation(mask)
            macro_zones = _build_macro_zones(segmentation_summary)
            overlay_path = _save_overlay(map_image_path, mask, mission_dir)
            status = "completed" if macro_zones else "degraded"
        else:
            truthfulness_note = f"{truthfulness_note} Segmentation mask validation failed: {validation.get('message')}"
    except Exception as exc:
        status = "failed"
        truthfulness_note = f"{truthfulness_note} Macro analysis failed while reading segmentation mask: {exc}"

    result = {
        "stage_key": "macro_analysis",
        "status": status,
        "map_image_path": str(map_image_path or ""),
        "segmentation_source": normalized_source,
        "segmentation_valid": segmentation_valid,
        "segmentation_summary": segmentation_summary,
        "macro_zones": macro_zones,
        "overlay_path": overlay_path,
        "truthfulness_note": truthfulness_note,
        "human_review_required": True,
    }
    result_path = _save_stage_result(mission_dir, result)
    record_stage_evidence(
        mission,
        mission_dir,
        "macro_analysis",
        "completed" if status in {"completed", "degraded"} else "failed",
        output_ref=result_path,
        result_type="macro_segmentation_risk_summary",
        source_type=_source_type(normalized_source),
        truthfulness_note=truthfulness_note,
        limitation=truthfulness_note,
        human_review_required=True,
    )
    return mission, result
