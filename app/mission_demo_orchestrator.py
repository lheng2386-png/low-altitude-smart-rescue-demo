"""One-click mission demo orchestrator for 灾情感知及影响评估.

This module chains current runnable stages without fabricating missing results.
"""

import json
import shutil
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


class MissionDemoOrchestratorError(Exception):
    pass


DEMO_STAGE_STATUS = {
    "not_requested": "not_requested",
    "skipped": "skipped",
    "running": "running",
    "success": "success",
    "failed": "failed",
    "partial_success": "partial_success",
}


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_BASE = ROOT_DIR / "outputs" / "mission_demo"


def _lazy_import(name):
    try:
        return __import__(name, fromlist=["*"])
    except Exception:
        return None


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _save_json(path, payload):
    path = Path(path)
    _ensure_dir(path.parent)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    _ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows or []:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return str(path)


def _copy_if_exists(source, destination_dir):
    if not source:
        return None
    source = Path(source)
    if not source.exists():
        return None
    destination_dir = _ensure_dir(destination_dir)
    destination = destination_dir / source.name
    shutil.copy2(source, destination)
    return str(destination)


def _copy_paths(paths, destination_dir):
    copied = []
    for path in paths or []:
        copied_path = _copy_if_exists(path, destination_dir)
        if copied_path:
            copied.append(copied_path)
    return copied


def _load_mask_fallback(mask):
    if isinstance(mask, (str, Path)):
        path = Path(mask)
        if path.suffix.lower() == ".npy":
            return np.load(path, allow_pickle=False)
        return np.asarray(Image.open(path))
    if isinstance(mask, Image.Image):
        return np.asarray(mask)
    return np.asarray(mask)


def _resize_mask_fallback(mask, width, height):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[:2] == (height, width):
        return mask.astype(np.uint8)
    resized = Image.fromarray(mask.astype(np.uint8)).resize((width, height), Image.NEAREST)
    return np.asarray(resized).astype(np.uint8)


def _validate_mask_fallback(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2:
        return {"valid": False, "message": "Segmentation mask must be a 2D class-id mask."}
    unique_values = np.unique(mask)
    invalid_values = [int(value) for value in unique_values if value < 0 or value > 10]
    if invalid_values:
        return {
            "valid": False,
            "message": f"Segmentation mask contains class ids outside 0-10: {invalid_values[:10]}",
        }
    return {
        "valid": True,
        "message": "Segmentation mask is a valid 0-10 class-id mask.",
        "unique_values": [int(value) for value in unique_values],
    }


def _summarize_mask_fallback(mask):
    class_names = {
        0: "background",
        1: "water",
        2: "building_no_damage",
        3: "building_medium_damage",
        4: "building_major_damage",
        5: "building_total_destruction",
        6: "vehicle",
        7: "road_clear",
        8: "road_blocked",
        9: "tree",
        10: "pool",
    }
    mask = np.asarray(mask)
    total = max(1, int(mask.size))
    summary = {}
    for class_id, name in class_names.items():
        pixels = int(np.sum(mask == class_id))
        summary[name] = {"pixels": pixels, "ratio": round(pixels / total, 6)}
    return summary


def _create_overlay_fallback(image_rgb, color_mask, alpha=0.45):
    image_rgb = np.asarray(image_rgb).astype(np.uint8)
    color_mask = np.asarray(color_mask).astype(np.uint8)
    if color_mask.shape[:2] != image_rgb.shape[:2]:
        color_mask = np.asarray(Image.fromarray(color_mask).resize((image_rgb.shape[1], image_rgb.shape[0]), Image.NEAREST))
    return np.clip((1 - alpha) * image_rgb + alpha * color_mask, 0, 255).astype(np.uint8)


def normalize_demo_image_input(image):
    """Normalize input image, preferring detection runtime adapter if available."""
    try:
        from detection_runtime_service import normalize_image_input

        return normalize_image_input(image)
    except Exception:
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
            np_image = np.asarray(pil_image).copy()
            width, height = pil_image.size
            return {
                "pil_image": pil_image,
                "np_image": np_image,
                "width": width,
                "height": height,
                "source_path": None,
            }
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
            np_image = np.asarray(pil_image).copy()
            width, height = pil_image.size
            return {
                "pil_image": pil_image,
                "np_image": np_image,
                "width": width,
                "height": height,
                "source_path": str(image),
            }
        array = np.asarray(image)
        if array.ndim == 2:
            pil_image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
        elif array.ndim == 3:
            if array.shape[-1] == 4:
                array = array[:, :, :3]
            pil_image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
        else:
            raise MissionDemoOrchestratorError("无效的输入图像格式。")
        np_image = np.asarray(pil_image).copy()
        width, height = pil_image.size
        return {
            "pil_image": pil_image,
            "np_image": np_image,
            "width": width,
            "height": height,
            "source_path": None,
        }


def build_stage_result(stage_name, status, result=None, error_code=None, message="", artifacts=None, truthfulness_note=""):
    """Build a standardized stage result."""
    success = status in {DEMO_STAGE_STATUS["success"], DEMO_STAGE_STATUS["partial_success"]}
    return {
        "stage_name": stage_name,
        "status": status,
        "success": success,
        "result": result,
        "error_code": error_code,
        "message": message,
        "artifacts": [str(item) for item in (artifacts or []) if item],
        "truthfulness_note": truthfulness_note or "",
    }


def run_detection_stage(
    image,
    detection_mode="yolo_rescue_targets",
    model_variant="yolov11m",
    transformer_model_key="rescuedet_deformable_detr",
    confidence_threshold=0.3,
    transformer_confidence_threshold=0.4,
    output_dir=None,
):
    """Run detection through the unified runtime adapter."""
    artifacts = []
    try:
        runtime = _lazy_import("detection_runtime_service")
        if runtime is None:
            return build_stage_result(
                "detection",
                DEMO_STAGE_STATUS["failed"],
                result=None,
                error_code="DETECTION_RUNTIME_MISSING",
                message="Detection runtime service is unavailable.",
                artifacts=[],
                truthfulness_note="Detection stage could not run because the runtime adapter is missing.",
            )
        result = runtime.run_detection(
            image,
            detection_mode=detection_mode,
            model_variant=model_variant,
            transformer_model_key=transformer_model_key,
            confidence_threshold=confidence_threshold,
            transformer_confidence_threshold=transformer_confidence_threshold,
            output_dir=output_dir,
        )
        if output_dir is not None and not result.get("detection_result_path"):
            result = runtime.save_detection_artifacts(result, output_dir)
        artifacts = [
            result.get("annotated_image_path"),
            result.get("detection_result_path"),
            result.get("metadata_path"),
        ]
        if result.get("detection_mode") == "dual_backend_compare" and result.get("detection_result_path"):
            consensus = result.get("consensus")
            if consensus is not None:
                artifacts.append(_save_json(Path(output_dir) / "dual_detection_consensus.json", consensus))
        status = DEMO_STAGE_STATUS["success"] if result.get("success") else DEMO_STAGE_STATUS["failed"]
        if result.get("partial_success"):
            status = DEMO_STAGE_STATUS["partial_success"] if not result.get("success") else status
        return build_stage_result(
            "detection",
            status,
            result=result,
            error_code=result.get("error_code"),
            message=result.get("message", ""),
            artifacts=[item for item in artifacts if item],
            truthfulness_note=result.get("truthfulness_note", ""),
        )
    except Exception as exc:
        return build_stage_result(
            "detection",
            DEMO_STAGE_STATUS["failed"],
            result=None,
            error_code="DETECTION_STAGE_FAILED",
            message=str(exc),
            artifacts=[],
            truthfulness_note="Detection stage failed structurally. No detections were fabricated.",
        )


def _build_segmentation_overlay_artifacts(mask, image, output_dir):
    artifacts = []
    try:
        from damage_segmentation_visualizer import create_legend_image, create_segmentation_panel, render_segmentation_mask

        color_mask = render_segmentation_mask(mask)
        panel = create_segmentation_panel(image, color_mask)
        overlay_path = Path(output_dir) / "segmentation_overlay.png"
        panel_path = Path(output_dir) / "segmentation_panel.png"
        color_path = Path(output_dir) / "segmentation_color_mask.png"
        Image.fromarray(color_mask).save(color_path)
        Image.fromarray(panel).save(panel_path)
        artifacts.extend([str(color_path), str(panel_path)])
        legend = create_legend_image()
        if legend is not None:
            legend_path = Path(output_dir) / "segmentation_legend.png"
            Image.fromarray(legend).save(legend_path)
            artifacts.append(str(legend_path))
        return artifacts, str(overlay_path), str(panel_path), str(color_path)
    except Exception:
        return artifacts, None, None, None


def run_segmentation_stage(image, segmentation_source="none", segmentation_mask=None, output_dir=None):
    """Run segmentation according to declared source type."""
    output_dir = Path(output_dir) if output_dir else None
    artifacts = []
    try:
        seg_engine = _lazy_import("segmentation_engine")
        seg_meta = _lazy_import("segmentation_source_metadata")
        seg_model_service = _lazy_import("segmentation_model_service")
        if seg_meta is None:
            return build_stage_result(
                "segmentation",
                DEMO_STAGE_STATUS["failed"],
                result=None,
                error_code="SEGMENTATION_SERVICE_MISSING",
                message="Segmentation service is unavailable.",
                artifacts=[],
                truthfulness_note="Segmentation stage could not run because supporting modules are missing.",
            )

        normalized = normalize_demo_image_input(image)
        image_rgb = normalized["np_image"]
        width, height = normalized["width"], normalized["height"]
        output_dir = _ensure_dir(output_dir or ROOT_DIR / "outputs" / "mission_demo" / "segmentation")
        source_type = str(segmentation_source or "none").strip().lower()

        if source_type == "none":
            metadata = seg_meta.build_segmentation_source_metadata("none")
            result = {
                "segmentation_source": metadata,
                "segmentation_summary": {},
                "mask_available": False,
                "is_model_prediction": False,
                "mask_path": None,
                "overlay_path": None,
                "truthfulness_note": metadata.get("truthfulness_note", ""),
                "damage_statistics": {},
                "damage_level": "Unknown",
            }
            return build_stage_result(
                "segmentation",
                DEMO_STAGE_STATUS["skipped"],
                result=result,
                message="No segmentation was requested.",
                artifacts=[],
                truthfulness_note=metadata.get("truthfulness_note", ""),
            )

        if source_type == "uploaded_mask":
            mask = segmentation_mask
            mask_path = None
            if isinstance(mask, (str, Path)):
                mask_path = str(mask)
                if seg_engine is not None:
                    mask = seg_engine.load_segmentation_mask(mask)
                else:
                    mask = _load_mask_fallback(mask)
            elif mask is not None:
                mask = np.asarray(mask)
            if mask is None:
                metadata = seg_meta.build_segmentation_source_metadata(
                    "uploaded_mask",
                    model_available=False,
                    prediction_success=False,
                    mask_path=mask_path,
                    fallback_reason="Uploaded mask was not provided or could not be read.",
                )
                result = {
                    "segmentation_source": metadata,
                    "segmentation_summary": {},
                    "mask_available": False,
                    "is_model_prediction": False,
                    "mask_path": mask_path,
                    "overlay_path": None,
                    "truthfulness_note": metadata.get("truthfulness_note", ""),
                    "damage_statistics": {},
                    "damage_level": "Unknown",
                }
                return build_stage_result(
                    "segmentation",
                    DEMO_STAGE_STATUS["failed"],
                    result=result,
                    error_code="UPLOADED_MASK_MISSING",
                    message="Uploaded segmentation mask was missing or unreadable.",
                    artifacts=[],
                    truthfulness_note=metadata.get("truthfulness_note", ""),
                )
            if seg_engine is not None:
                mask = seg_engine.resize_segmentation_mask(mask, width, height)
                validation = seg_engine.validate_segmentation_mask(mask)
            else:
                mask = _resize_mask_fallback(mask, width, height)
                validation = _validate_mask_fallback(mask)
            if not validation.get("valid"):
                metadata = seg_meta.build_segmentation_source_metadata(
                    "uploaded_mask",
                    model_available=False,
                    prediction_success=False,
                    mask_path=mask_path,
                    fallback_reason=validation.get("message", ""),
                )
                result = {
                    "segmentation_source": metadata,
                    "segmentation_summary": {},
                    "mask_available": False,
                    "is_model_prediction": False,
                    "mask_path": mask_path,
                    "overlay_path": None,
                    "truthfulness_note": metadata.get("truthfulness_note", ""),
                    "damage_statistics": {},
                    "damage_level": "Unknown",
                }
                return build_stage_result(
                    "segmentation",
                    DEMO_STAGE_STATUS["failed"],
                    result=result,
                    error_code="UPLOADED_MASK_INVALID",
                    message=validation.get("message", "Invalid uploaded mask."),
                    artifacts=[],
                    truthfulness_note=metadata.get("truthfulness_note", ""),
                )

            summary = seg_engine.summarize_segmentation(mask) if seg_engine is not None else _summarize_mask_fallback(mask)
            damage_statistics = {}
            damage_level = "Unknown"
            try:
                from damage_segmentation_visualizer import classify_damage_level, compute_damage_statistics

                damage_statistics = compute_damage_statistics(mask)
                damage_level = classify_damage_level(damage_statistics)
            except Exception:
                damage_statistics = {}
            overlay = None
            color_mask = seg_engine.render_segmentation_mask(mask) if seg_engine is not None and hasattr(seg_engine, "render_segmentation_mask") else None
            if color_mask is None:
                try:
                    from damage_segmentation_visualizer import render_segmentation_mask

                    color_mask = render_segmentation_mask(mask)
                except Exception:
                    color_mask = None
            if seg_engine is not None:
                overlay = seg_engine.create_segmentation_overlay(image_rgb, mask)
            elif color_mask is not None:
                overlay = _create_overlay_fallback(image_rgb, color_mask)
            overlay_path = None
            panel_path = None
            color_path = None
            if overlay is not None:
                overlay_path = str(output_dir / "segmentation_overlay.png")
                Image.fromarray(np.asarray(overlay).astype(np.uint8)).save(overlay_path)
                artifacts.append(overlay_path)
            if color_mask is not None:
                color_path = str(output_dir / "segmentation_color_mask.png")
                Image.fromarray(color_mask).save(color_path)
                artifacts.append(color_path)
                try:
                    from damage_segmentation_visualizer import create_segmentation_panel

                    panel = create_segmentation_panel(image_rgb, color_mask)
                    panel_path = str(output_dir / "segmentation_panel.png")
                    Image.fromarray(panel).save(panel_path)
                    artifacts.append(panel_path)
                except Exception:
                    panel_path = None

            metadata = seg_meta.build_segmentation_source_metadata(
                "uploaded_mask",
                model_available=False,
                prediction_success=False,
                mask_path=mask_path or str(output_dir / "uploaded_mask.npy"),
            )
            if mask_path is None:
                np.save(output_dir / "uploaded_mask.npy", mask.astype(np.uint8))
                artifacts.append(str(output_dir / "uploaded_mask.npy"))
            result = {
                "segmentation_source": metadata,
                "segmentation_summary": summary,
                "mask_available": True,
                "is_model_prediction": False,
                "mask_path": mask_path or str(output_dir / "uploaded_mask.npy"),
                "overlay_path": overlay_path,
                "panel_path": panel_path,
                "color_mask_path": color_path,
                "truthfulness_note": metadata.get("truthfulness_note", ""),
                "damage_statistics": damage_statistics,
                "damage_level": damage_level,
                "validation": validation,
            }
            _save_json(output_dir / "segmentation_source.json", metadata)
            _save_json(output_dir / "segmentation_result.json", result)
            return build_stage_result(
                "segmentation",
                DEMO_STAGE_STATUS["success"],
                result=result,
                message="Uploaded segmentation mask processed successfully.",
                artifacts=artifacts + [str(output_dir / "segmentation_source.json"), str(output_dir / "segmentation_result.json")],
                truthfulness_note=metadata.get("truthfulness_note", ""),
            )

        if source_type == "auto_model":
            if seg_model_service is None:
                metadata = seg_meta.build_segmentation_source_metadata(
                    "auto_model",
                    model_available=False,
                    prediction_success=False,
                    fallback_reason="segmentation_model_service module is missing.",
                )
                result = {
                    "segmentation_source": metadata,
                    "segmentation_summary": {},
                    "mask_available": False,
                    "is_model_prediction": False,
                    "mask_path": None,
                    "overlay_path": None,
                    "truthfulness_note": metadata.get("truthfulness_note", ""),
                    "damage_statistics": {},
                    "damage_level": "Unknown",
                }
                return build_stage_result(
                    "segmentation",
                    DEMO_STAGE_STATUS["failed"],
                    result=result,
                    error_code="SEGMENTATION_MODEL_SERVICE_MISSING",
                    message=metadata.get("truthfulness_note", ""),
                    artifacts=[],
                    truthfulness_note=metadata.get("truthfulness_note", ""),
                )

            prediction, message, status = seg_model_service.predict_segmentation(image_rgb, img_size=max(width, height))
            model_available = bool(status.get("ok"))
            metadata = seg_meta.build_segmentation_source_metadata(
                "auto_model",
                checkpoint_path=status.get("checkpoint_path"),
                model_available=model_available,
                prediction_success=prediction is not None,
                fallback_reason=None if prediction is not None else message,
            )
            if prediction is None:
                result = {
                    "segmentation_source": metadata,
                    "segmentation_summary": {},
                    "mask_available": False,
                    "is_model_prediction": False,
                    "mask_path": None,
                    "overlay_path": None,
                    "truthfulness_note": metadata.get("truthfulness_note", ""),
                    "damage_statistics": {},
                    "damage_level": "Unknown",
                }
                return build_stage_result(
                    "segmentation",
                    DEMO_STAGE_STATUS["failed"],
                    result=result,
                    error_code="AUTO_SEGMENTATION_FAILED",
                    message=message,
                    artifacts=[],
                    truthfulness_note=metadata.get("truthfulness_note", ""),
                )

            mask = seg_engine.resize_segmentation_mask(prediction, width, height)
            summary = seg_engine.summarize_segmentation(mask)
            damage_statistics = {}
            damage_level = "Unknown"
            try:
                from damage_segmentation_visualizer import classify_damage_level, compute_damage_statistics, create_segmentation_panel, render_segmentation_mask

                damage_statistics = compute_damage_statistics(mask)
                damage_level = classify_damage_level(damage_statistics)
                color_mask = render_segmentation_mask(mask)
                panel = create_segmentation_panel(image_rgb, color_mask)
                color_path = str(output_dir / "segmentation_color_mask.png")
                panel_path = str(output_dir / "segmentation_panel.png")
                overlay_path = str(output_dir / "segmentation_overlay.png")
                Image.fromarray(color_mask).save(color_path)
                Image.fromarray(panel).save(panel_path)
                overlay = seg_engine.create_segmentation_overlay(image_rgb, mask)
                if overlay is not None:
                    Image.fromarray(np.asarray(overlay).astype(np.uint8)).save(overlay_path)
                    artifacts.append(overlay_path)
                artifacts.extend([color_path, panel_path])
            except Exception:
                color_path = None
                panel_path = None
                overlay_path = None

            mask_path = str(output_dir / "auto_segmentation_mask.npy")
            np.save(mask_path, mask.astype(np.uint8))
            artifacts.append(mask_path)
            _save_json(output_dir / "segmentation_source.json", metadata)
            result = {
                "segmentation_source": metadata,
                "segmentation_summary": summary,
                "mask_available": True,
                "is_model_prediction": True,
                "mask_path": mask_path,
                "overlay_path": overlay_path,
                "panel_path": panel_path,
                "color_mask_path": color_path,
                "truthfulness_note": metadata.get("truthfulness_note", ""),
                "damage_statistics": damage_statistics,
                "damage_level": damage_level,
                "prediction_message": message,
                "prediction_status": status,
            }
            _save_json(output_dir / "segmentation_result.json", result)
            return build_stage_result(
                "segmentation",
                DEMO_STAGE_STATUS["success"],
                result=result,
                message=message,
                artifacts=artifacts + [str(output_dir / "segmentation_source.json"), str(output_dir / "segmentation_result.json")],
                truthfulness_note=metadata.get("truthfulness_note", ""),
            )

        metadata = seg_meta.build_segmentation_source_metadata(
            "demo_fallback",
            fallback_reason=f"Unsupported segmentation source: {segmentation_source}",
        )
        result = {
            "segmentation_source": metadata,
            "segmentation_summary": {},
            "mask_available": False,
            "is_model_prediction": False,
            "mask_path": None,
            "overlay_path": None,
            "truthfulness_note": metadata.get("truthfulness_note", ""),
            "damage_statistics": {},
            "damage_level": "Unknown",
        }
        return build_stage_result(
            "segmentation",
            DEMO_STAGE_STATUS["skipped"],
            result=result,
            message=metadata.get("truthfulness_note", ""),
            artifacts=[],
            truthfulness_note=metadata.get("truthfulness_note", ""),
        )
    except Exception as exc:
        return build_stage_result(
            "segmentation",
            DEMO_STAGE_STATUS["failed"],
            result=None,
            error_code="SEGMENTATION_STAGE_FAILED",
            message=str(exc),
            artifacts=[],
            truthfulness_note="Segmentation stage failed structurally. No mask was fabricated.",
        )


def _filter_decision_targets(detection_stage_result, segmentation_mask):
    result = detection_stage_result.get("result") if detection_stage_result else {}
    targets = result.get("targets", []) if isinstance(result, dict) else []
    filtered = []
    for target in targets:
        class_name = str(target.get("class_name", "")).lower()
        if class_name in {"civilian", "rescuer", "dog", "cat", "horse", "cow"}:
            filtered.append(target)
    return filtered


def _ec_component_score(item, component_name):
    component = (item.get("components") or {}).get(component_name) or {}
    try:
        return float(component.get("score", 0.0))
    except Exception:
        return 0.0


def _source_modules_for_ec(detection_result, segmentation_result, path_result, decision_fusion_result):
    modules = []
    if detection_result and detection_result.get("success"):
        modules.append("detection")
    if segmentation_result and segmentation_result.get("mask_available"):
        modules.append("segmentation")
    if path_result:
        modules.append("path_planning")
    if decision_fusion_result:
        modules.append("decision_fusion")
    return modules


def _normalize_ec_rankings_for_output(ec_rankings, raw_targets, evidence_level, source_modules, path_result):
    target_lookup = {str(target.get("id") or target.get("target_id")): target for target in raw_targets or []}
    normalized = []
    for item in ec_rankings or []:
        target_id = str(item.get("target_id") or "unknown")
        target = target_lookup.get(target_id, {})
        target_type = item.get("class_name") or target.get("class_name", "unknown")
        target_type_lower = str(target_type).lower()
        limitations = [
            "EC-TERP is an assistive priority ranking algorithm.",
            "It does not replace human rescue command decisions.",
            "Image-plane route accessibility is not GPS navigation.",
        ]
        if target_type_lower == "human_candidate" or str(target.get("source_backend", "")).startswith("transformer"):
            limitations.append("Transformer human_candidate is not a confirmed civilian and requires manual review.")
        if not path_result or not path_result.get("found"):
            limitations.append("Route accessibility is degraded or unavailable for this target.")
        normalized.append(
            {
                "target_id": target_id,
                "target_type": target_type,
                "rank": int(item.get("rank", len(normalized) + 1)),
                "ec_terp_score": float(item.get("ec_terp_score", 0.0)),
                "ec_terp_level": item.get("ec_terp_level", "low"),
                "score_components": {
                    "target_urgency": _ec_component_score(item, "target_urgency"),
                    "environment_risk": _ec_component_score(item, "environment_risk"),
                    "route_accessibility": _ec_component_score(item, "route_accessibility"),
                    "coverage_gap": _ec_component_score(item, "coverage_gap"),
                    "evidence_quality": _ec_component_score(item, "evidence_quality"),
                    "uncertainty_penalty": _ec_component_score(item, "uncertainty_penalty"),
                },
                "evidence_level": evidence_level,
                "source_modules": list(source_modules),
                "is_confirmed_rescue_target": False,
                "human_review_required": True,
                "recommendation_type": "assistive_priority_ranking",
                "explanation": item.get("explanation", ""),
                "limitations": limitations,
                "truthfulness_note": "EC-TERP provides assistive image-plane priority ranking only and does not replace human rescue decisions.",
            }
        )
    return normalized


def _write_ec_terp_outputs(
    output_dir,
    raw_rankings,
    normalized_rankings,
    comparison,
    metadata,
    limitations,
):
    output_dir = _ensure_dir(output_dir)
    artifacts = []
    artifacts.append(_save_json(output_dir / "ec_terp_rankings.json", {
        "success": bool(normalized_rankings),
        "status": "executed_success" if normalized_rankings else "executed_failed",
        "module": "ec_terp_ranking",
        "rankings": normalized_rankings,
        "raw_ec_terp_rankings": raw_rankings,
        "comparison": comparison,
        "truthfulness_note": "EC-TERP provides assistive image-plane priority ranking only and does not replace human rescue decisions.",
    }))
    artifacts.append(
        _write_csv(
            output_dir / "ec_terp_rankings.csv",
            normalized_rankings,
            [
                "rank",
                "target_id",
                "target_type",
                "ec_terp_score",
                "ec_terp_level",
                "evidence_level",
                "is_confirmed_rescue_target",
                "human_review_required",
                "recommendation_type",
            ],
        )
    )
    artifacts.append(_save_json(output_dir / "ec_terp_metadata.json", metadata))
    artifacts.append(_save_json(output_dir / "ec_terp_limitations.json", {
        "limitations": limitations,
        "truthfulness_note": "EC-TERP is assistive decision support. It is not an automatic rescue decision system.",
    }))
    return artifacts


def _write_ec_terp_ui_summary(output_dir, rankings, status, limitations, visuals_metadata=None):
    output_dir = _ensure_dir(output_dir)
    visuals = {}
    if visuals_metadata:
        try:
            from ec_terp_visualization_service import build_visuals_map

            visuals = build_visuals_map(visuals_metadata)
        except Exception:
            visuals = {}
    summary = {
        "module": "ec_terp",
        "title": "EC-TERP Assistive Priority Ranking",
        "status": status,
        "top_rankings": rankings[:5],
        "score_component_labels": {
            "target_urgency": "Target Urgency",
            "environment_risk": "Environment Risk",
            "route_accessibility": "Route Accessibility",
            "coverage_gap": "Coverage Gap",
            "evidence_quality": "Evidence Quality",
            "uncertainty_penalty": "Uncertainty Penalty",
        },
        "visuals": {
            "topk_ranking_chart": visuals.get("topk_ranking_chart"),
            "component_breakdown_chart": visuals.get("component_breakdown_chart"),
            "evidence_quality_distribution_chart": visuals.get("evidence_quality_distribution_chart"),
            "sensitivity_summary_chart": visuals.get("sensitivity_summary_chart"),
        },
        "explainability": {
            "formula": "EC-TERP = αT + βE + γR + δC + λQ - μU",
            "component_meanings": {
                "T": "Target urgency",
                "E": "Environment risk",
                "R": "Route accessibility",
                "C": "Coverage gap",
                "Q": "Evidence quality",
                "U": "Uncertainty penalty",
            },
            "key_message": "Higher priority means the target should be reviewed earlier by human operators, not automatically rescued.",
        },
        "truthfulness_badges": [
            "Assistive ranking",
            "Human review required",
            "Image-plane path only",
            "Not GPS navigation",
            "Not automatic rescue decision",
        ],
        "limitations": limitations,
        "human_review_required": True,
    }
    return _save_json(output_dir / "ec_terp_summary.json", summary)


def run_decision_stage(
    image,
    detection_stage_result,
    segmentation_stage_result,
    start_point=None,
    output_dir=None,
):
    """Run risk ranking, TERP, path planning, and optional decision fusion."""
    output_dir = _ensure_dir(output_dir or ROOT_DIR / "outputs" / "mission_demo" / "decision")
    artifacts = []
    try:
        detection_result = detection_stage_result.get("result") if detection_stage_result else {}
        segmentation_result = segmentation_stage_result.get("result") if segmentation_stage_result else {}
        segmentation_mask = None
        if isinstance(segmentation_result, dict) and segmentation_result.get("mask_path"):
            mask_path = Path(segmentation_result["mask_path"])
            if mask_path.suffix.lower() == ".npy" and mask_path.exists():
                segmentation_mask = np.load(mask_path, allow_pickle=False)
            elif mask_path.exists():
                from segmentation_engine import load_segmentation_mask

                segmentation_mask = load_segmentation_mask(mask_path)
        elif isinstance(segmentation_result, dict) and segmentation_result.get("mask_available") and segmentation_result.get("segmentation_summary"):
            # No raw mask path available, keep None and rely on summaries.
            segmentation_mask = None

        normalized = normalize_demo_image_input(image)
        image_width, image_height = normalized["width"], normalized["height"]

        raw_targets = detection_result.get("targets", []) if isinstance(detection_result, dict) else []
        if not raw_targets:
            result = {
                "ranked_targets": [],
                "terp_rankings": [],
                "path_result": None,
                "path_comparison": None,
                "decision_fusion_result": None,
                "search_priority_overlay_path": None,
                "path_overlay_path": None,
                "can_support_decision": False,
                "truthfulness_note": "No executable detection targets were available for decision-stage reasoning.",
                "scene_mode_result": None,
                "entry_result": None,
                "gate_result": None,
                "reliability_status": None,
            }
            _save_json(output_dir / "decision_result.json", result)
            return build_stage_result(
                "decision",
                DEMO_STAGE_STATUS["failed"],
                result=result,
                error_code="NO_DETECTION_TARGETS",
                message="No detection targets are available for TERP/path planning.",
                artifacts=[str(output_dir / "decision_result.json")],
                truthfulness_note=result["truthfulness_note"],
            )

        try:
            from detection_decision_bridge import filter_targets_for_terp

            terp_candidates = filter_targets_for_terp(raw_targets)
        except Exception:
            terp_candidates = _filter_decision_targets(detection_stage_result, segmentation_mask)

        if not terp_candidates:
            terp_candidates = [target for target in raw_targets if str(target.get("class_name", "")).lower() in {"civilian", "rescuer", "dog", "cat", "horse", "cow"}]

        try:
            from risk_engine import calculate_risk
        except Exception:
            calculate_risk = None

        risk_ranked = []
        if terp_candidates and calculate_risk is not None:
            try:
                from priority_ranker import rank_targets

                risk_ranked = rank_targets(terp_candidates, image_width, image_height, segmentation_mask=segmentation_mask, language="zh")
            except Exception:
                risk_ranked = terp_candidates
        else:
            risk_ranked = terp_candidates

        target_map = {target.get("id") or target.get("target_id"): target for target in raw_targets}
        top_target_id = None
        if risk_ranked:
            top_target_id = risk_ranked[0].get("target_id") or risk_ranked[0].get("id")
        primary_target = target_map.get(top_target_id) or (raw_targets[0] if raw_targets else None)

        scene_mode_result = None
        entry_result = None
        gate_result = None
        reliability_status = None
        path_result = None
        baseline_path_result = None
        path_comparison = None
        path_overlay_path = None
        search_priority_overlay_path = None
        decision_fusion_result = None
        coverage_result = None
        ec_terp_rankings = []
        ec_terp_comparison = None

        try:
            from scene_mode_and_entry_service import (
                analyze_scene_mode,
                build_path_planning_gate_result,
                build_path_planning_reliability_status,
                find_rescue_entry_point,
            )

            scene_mode_result = analyze_scene_mode(image, segmentation_mask=segmentation_mask, detections=raw_targets)
            entry_result = find_rescue_entry_point(segmentation_mask, target_point=(primary_target or {}).get("center"))
            gate_result = build_path_planning_gate_result(
                scene_mode_result,
                entry_result,
                use_manual_start=start_point is not None,
                manual_start_point=start_point,
                force_path_planning=False,
            )
            reliability_status = build_path_planning_reliability_status(
                scene_mode_result,
                entry_result,
                gate_result,
                segmentation_source_metadata=(segmentation_result.get("segmentation_source") if isinstance(segmentation_result, dict) else None),
                force_path_planning=False,
            )
            _save_json(output_dir / "scene_mode_result.json", scene_mode_result)
            _save_json(output_dir / "entry_result.json", entry_result)
            _save_json(output_dir / "path_planning_gate_result.json", gate_result)
            _save_json(output_dir / "path_planning_reliability_status.json", reliability_status)
            artifacts.extend(
                [
                    str(output_dir / "scene_mode_result.json"),
                    str(output_dir / "entry_result.json"),
                    str(output_dir / "path_planning_gate_result.json"),
                    str(output_dir / "path_planning_reliability_status.json"),
                ]
            )
        except Exception as exc:
            scene_mode_result = {"scene_mode": "unknown", "scene_mode_label": "Unknown / 信息不足", "path_planning_allowed": False, "reason": str(exc), "evidence": {}}
            entry_result = {"entry_found": False, "entry_point": None, "entry_reason": str(exc), "candidate_count": 0}
            gate_result = {"path_enabled": False, "start_point": None, "start_source": "disabled", "gate_reason": str(exc), "display_message": str(exc)}
            reliability_status = {"reliability_level": "not_applicable", "is_real_gps_navigation": False, "path_type": "image_plane_reference_path", "scene_mode_method": "rule_based", "mask_dependency": segmentation_mask is not None, "mask_source": "unknown", "mask_risk_note": str(exc), "reliability_note": str(exc), "human_review_required": True}

        can_support_decision = False
        if risk_ranked:
            can_support_decision = True
            try:
                from terp_engine import rank_targets_by_terp

                environment_contexts = {}
                path_results = {}
                for target in risk_ranked:
                    target_id = target.get("id") or target.get("target_id")
                    if segmentation_mask is not None:
                        try:
                            from segmentation_engine import get_environment_context_for_target

                            environment_contexts[target_id] = get_environment_context_for_target(target, segmentation_mask, language="zh")
                        except Exception:
                            environment_contexts[target_id] = None
                if gate_result and gate_result.get("path_enabled") and primary_target is not None:
                    path_target_list = [primary_target]
                else:
                    path_target_list = []
                    path_comparison = {
                        "success": False,
                        "comparison_available": False,
                        "reason": (gate_result or {}).get(
                            "display_message",
                            "路径规划门控未启用，因此不生成图像平面参考路径。",
                        ),
                        "truthfulness_note": "Path planning was not run because the Scene Mode / Rescue Entry gate did not allow a reliable image-plane path.",
                    }
                    _save_json(output_dir / "path_planning_result.json", None)
                    _save_json(output_dir / "baseline_path_result.json", None)
                    _save_json(output_dir / "path_comparison.json", path_comparison)
                    artifacts.extend(
                        [
                            str(output_dir / "path_planning_result.json"),
                            str(output_dir / "baseline_path_result.json"),
                            str(output_dir / "path_comparison.json"),
                        ]
                    )
                if path_target_list:
                    from path_planner import plan_baseline_path, plan_risk_aware_path, compare_path_plans, create_path_overlay, create_dual_path_overlay

                    path_targets_for_planning = [target_map.get(path_target_list[0].get("target_id") or path_target_list[0].get("id"), path_target_list[0])]
                    use_start = gate_result.get("start_point") if gate_result else start_point
                    baseline_path_result = plan_baseline_path(path_targets_for_planning, image_width, image_height, start_point=use_start)
                    path_result = plan_risk_aware_path(path_targets_for_planning, segmentation_mask, image_width, image_height, start_point=use_start)
                    path_comparison = compare_path_plans(baseline_path_result, path_result, segmentation_mask)
                    if path_result and path_result.get("found"):
                        path_overlay = create_path_overlay(normalized["pil_image"], path_result)
                        if path_overlay is not None:
                            path_overlay_path = str(output_dir / "path_overlay.png")
                            Image.fromarray(path_overlay).save(path_overlay_path)
                            artifacts.append(path_overlay_path)
                    if baseline_path_result and path_result:
                        try:
                            dual_overlay = create_dual_path_overlay(normalized["pil_image"], baseline_path_result, path_result)
                            if dual_overlay is not None:
                                dual_path = str(output_dir / "dual_path_overlay.png")
                                Image.fromarray(dual_overlay).save(dual_path)
                                artifacts.append(dual_path)
                        except Exception:
                            pass
                    _save_json(output_dir / "path_planning_result.json", path_result)
                    _save_json(output_dir / "baseline_path_result.json", baseline_path_result)
                    _save_json(output_dir / "path_comparison.json", path_comparison)
                    artifacts.extend(
                        [
                            str(output_dir / "path_planning_result.json"),
                            str(output_dir / "baseline_path_result.json"),
                            str(output_dir / "path_comparison.json"),
                        ]
                    )
            except Exception as exc:
                path_result = None
                baseline_path_result = None
                path_comparison = {"success": False, "message": str(exc)}

        try:
            from decision_fusion_adapter import (
                build_decision_fusion_summary,
                compute_coverage_planning_score,
                compute_image_plane_search_priority_map,
                compute_segmentation_damage_impact_score,
                render_priority_map_overlay,
            )

            segmentation_summary = segmentation_result.get("segmentation_summary") if isinstance(segmentation_result, dict) else None
            search_priority_result = compute_image_plane_search_priority_map(
                normalized["np_image"].shape,
                targets=raw_targets,
                segmentation_summary=segmentation_summary,
                segmentation_mask=segmentation_mask,
                detection_bridge_result=detection_stage_result.get("result") if detection_stage_result else None,
            )
            damage_impact_result = compute_segmentation_damage_impact_score(
                segmentation_summary=segmentation_summary,
                segmentation_mask=segmentation_mask,
            )
            coverage_result = compute_coverage_planning_score(
                normalized["np_image"].shape,
                path_result=path_result,
                segmentation_mask=segmentation_mask,
                priority_map=search_priority_result.get("priority_map"),
            )
            decision_fusion_result = build_decision_fusion_summary(
                search_priority_result=search_priority_result,
                damage_impact_result=damage_impact_result,
                coverage_result=coverage_result,
                detection_bridge_result=detection_stage_result.get("result") if detection_stage_result else None,
            )
            _save_json(output_dir / "search_priority_map.npy.json", search_priority_result.get("priority_statistics", {}))
            np.save(output_dir / "search_priority_map.npy", search_priority_result.get("priority_map"))
            search_priority_overlay_path = render_priority_map_overlay(
                normalized["np_image"],
                search_priority_result.get("priority_map"),
                output_path=output_dir / "search_priority_overlay.png",
            )
            _save_json(output_dir / "damage_impact_result.json", damage_impact_result)
            _save_json(output_dir / "coverage_score_result.json", coverage_result)
            _save_json(output_dir / "decision_fusion_summary.json", decision_fusion_result)
            artifacts.extend(
                [
                    str(output_dir / "search_priority_map.npy"),
                    search_priority_overlay_path,
                    str(output_dir / "damage_impact_result.json"),
                    str(output_dir / "coverage_score_result.json"),
                    str(output_dir / "decision_fusion_summary.json"),
                ]
            )
        except Exception as exc:
            decision_fusion_result = None

        terp_rankings = []
        try:
            from terp_engine import rank_targets_by_terp

            environment_contexts = {}
            path_results = {}
            if segmentation_mask is not None:
                from segmentation_engine import get_environment_context_for_target

                for target in raw_targets:
                    target_id = target.get("id") or target.get("target_id")
                    environment_contexts[target_id] = get_environment_context_for_target(target, segmentation_mask, language="zh")
            if path_result and risk_ranked:
                target_id = path_result.get("target_id") or (risk_ranked[0].get("target_id") if risk_ranked else None)
                if target_id:
                    path_results[target_id] = path_result
            terp_rankings = rank_targets_by_terp(
                raw_targets,
                image_width,
                image_height,
                environment_contexts=environment_contexts,
                path_results=path_results,
                language="zh",
            )
            _save_json(output_dir / "terp_ranking.json", terp_rankings)
            artifacts.append(str(output_dir / "terp_ranking.json"))
        except Exception:
            terp_rankings = []

        try:
            from ec_terp_engine import (
                compare_terp_and_ec_terp,
                format_ec_terp_result_markdown,
                rank_targets_by_ec_terp,
            )

            target_evidence_level = "strong" if detection_result.get("is_model_output") else "none"
            if not detection_result.get("is_model_output") and segmentation_mask is not None:
                target_evidence_level = "medium"
            transformer_only = str(detection_result.get("detection_mode", "")).startswith("transformer")
            if transformer_only:
                target_evidence_level = "medium" if segmentation_mask is not None else "weak"
            safe_coverage_result = coverage_result if isinstance(coverage_result, dict) else None
            safe_decision_fusion_result = decision_fusion_result if isinstance(decision_fusion_result, dict) else None
            ec_terp_rankings = rank_targets_by_ec_terp(
                raw_targets,
                segmentation_summary=(segmentation_result.get("segmentation_summary") if isinstance(segmentation_result, dict) else None),
                path_result=path_result,
                path_comparison=path_comparison,
                coverage_result=safe_coverage_result,
                decision_fusion_result=safe_decision_fusion_result,
                target_evidence_level=target_evidence_level,
                segmentation_available=segmentation_mask is not None,
                transformer_only=transformer_only,
            )
            ec_terp_comparison = compare_terp_and_ec_terp(terp_rankings, ec_terp_rankings)
            ec_markdown = format_ec_terp_result_markdown(ec_terp_rankings)
            source_modules = _source_modules_for_ec(detection_result, segmentation_result, path_result, decision_fusion_result)
            normalized_ec_rankings = _normalize_ec_rankings_for_output(
                ec_terp_rankings,
                raw_targets,
                target_evidence_level,
                source_modules,
                path_result,
            )
            ec_limitations = [
                "EC-TERP is an assistive priority ranking algorithm.",
                "It does not replace human rescue command decisions.",
                "Image-plane route accessibility is not GPS navigation.",
                "Synthetic demo cases are not real rescue data.",
            ]
            if segmentation_mask is None:
                ec_limitations.append("Segmentation/environment evidence is missing or unavailable.")
            if not path_result or not path_result.get("found"):
                ec_limitations.append("Route accessibility evidence is missing, degraded, or unavailable.")
            ec_metadata = {
                "success": bool(normalized_ec_rankings),
                "module": "ec_terp_ranking",
                "formula": "EC-TERP = αT + βE + γR + δC + λQ - μU",
                "evidence_level": target_evidence_level,
                "source_modules": source_modules,
                "ranking_count": len(normalized_ec_rankings),
                "is_real_measurement": False,
                "is_automatic_rescue_decision": False,
                "is_gps_navigation": False,
                "is_gis_route": False,
                "human_review_required": True,
                "truthfulness_note": "EC-TERP provides assistive image-plane priority ranking only and does not replace human rescue decisions.",
            }
            ec_output_dir = _ensure_dir(output_dir.parent / "ec_terp")
            ec_artifacts = _write_ec_terp_outputs(
                ec_output_dir,
                ec_terp_rankings,
                normalized_ec_rankings,
                ec_terp_comparison,
                ec_metadata,
                ec_limitations,
            )
            artifacts.extend(ec_artifacts)
            visuals_metadata = None
            try:
                from ec_terp_visualization_service import generate_ec_terp_visuals

                visuals_metadata = generate_ec_terp_visuals(
                    ranking_path=ec_output_dir / "ec_terp_rankings.json",
                    eval_dir=output_dir.parent / "ec_terp_evaluation",
                    output_dir=output_dir.parent / "ec_terp_visuals",
                )
                if visuals_metadata.get("metadata_path"):
                    artifacts.append(visuals_metadata["metadata_path"])
                for figure in visuals_metadata.get("generated_figures", []):
                    if isinstance(figure, dict) and figure.get("path"):
                        artifacts.append(figure["path"])
                if visuals_metadata.get("limitations"):
                    ec_limitations.extend(
                        limitation
                        for limitation in visuals_metadata.get("limitations", [])
                        if limitation not in ec_limitations
                    )
            except Exception as exc:
                visuals_metadata = {
                    "status": "degraded",
                    "generated_figures": [],
                    "limitations": [f"EC-TERP visualization was not generated: {exc}"],
                    "truthfulness_notes": [
                        "EC-TERP provides assistive image-plane priority ranking only.",
                        "It does not replace human rescue command decisions.",
                        "Image-plane path planning is not GPS navigation.",
                        "Synthetic demo cases are not real rescue data.",
                    ],
                }
                ec_limitations.extend(visuals_metadata["limitations"])
            ui_artifact = _write_ec_terp_ui_summary(
                output_dir.parent / "ui",
                normalized_ec_rankings,
                "executed_success" if normalized_ec_rankings else "failed",
                ec_limitations,
                visuals_metadata=visuals_metadata,
            )
            artifacts.append(ui_artifact)
            _save_json(output_dir / "ec_terp_ranking.json", ec_terp_rankings)
            _save_json(output_dir / "ec_terp_comparison.json", ec_terp_comparison)
            ec_markdown_path = output_dir / "ec_terp_summary.md"
            ec_markdown_path.write_text(ec_markdown, encoding="utf-8")
            artifacts.extend(
                [
                    str(output_dir / "ec_terp_ranking.json"),
                    str(output_dir / "ec_terp_comparison.json"),
                    str(ec_markdown_path),
                ]
            )
        except Exception as exc:
            ec_terp_rankings = []
            ec_terp_comparison = {
                "success": False,
                "changed_rankings": [],
                "summary": f"EC-TERP 计算未完成：{exc}",
                "truthfulness_note": "EC-TERP failure is reported without fabricating priority scores.",
            }

        result = {
            "ranked_targets": risk_ranked,
            "terp_rankings": terp_rankings,
            "ec_terp_rankings": ec_terp_rankings,
            "ec_terp_comparison": ec_terp_comparison,
            "path_result": path_result,
            "baseline_path_result": baseline_path_result,
            "path_comparison": path_comparison,
            "decision_fusion_result": decision_fusion_result,
            "search_priority_overlay_path": search_priority_overlay_path,
            "path_overlay_path": path_overlay_path,
            "can_support_decision": bool(can_support_decision),
            "truthfulness_note": "Decision stage uses YOLO targets, TERP, image-plane path planning and lightweight decision fusion. Paths are reference paths, not GPS navigation.",
            "scene_mode_result": scene_mode_result,
            "entry_result": entry_result,
            "gate_result": gate_result,
            "reliability_status": reliability_status,
            "segmentation_mask_available": segmentation_mask is not None,
        }
        _save_json(output_dir / "decision_result.json", result)
        artifacts.append(str(output_dir / "decision_result.json"))
        status = DEMO_STAGE_STATUS["success"] if (path_result and path_result.get("found")) or can_support_decision else DEMO_STAGE_STATUS["partial_success"]
        if not risk_ranked:
            status = DEMO_STAGE_STATUS["failed"]
        return build_stage_result(
            "decision",
            status,
            result=result,
            message="Decision stage completed.",
            artifacts=[item for item in artifacts if item],
            truthfulness_note=result["truthfulness_note"],
        )
    except Exception as exc:
        return build_stage_result(
            "decision",
            DEMO_STAGE_STATUS["failed"],
            result=None,
            error_code="DECISION_STAGE_FAILED",
            message=str(exc),
            artifacts=[],
            truthfulness_note="Decision stage failed structurally. No path or fusion result was fabricated.",
        )


def run_thermal_stage(thermal_image=None, thermal_mode="skip", output_dir=None):
    """Run simulated or radiometric thermal analysis."""
    output_dir = _ensure_dir(output_dir or ROOT_DIR / "outputs" / "mission_demo" / "thermal")
    mode = str(thermal_mode or "skip").lower()
    if mode == "skip":
        result = {
            "thermal_mode": "skip",
            "is_real_temperature_measurement": False,
            "truthfulness_note": "Thermal stage was skipped on purpose.",
        }
        return build_stage_result(
            "thermal",
            DEMO_STAGE_STATUS["skipped"],
            result=result,
            message="Thermal stage skipped.",
            artifacts=[],
            truthfulness_note=result["truthfulness_note"],
        )

    thermal_service = _lazy_import("thermal_service")
    if thermal_service is None:
        return build_stage_result(
            "thermal",
            DEMO_STAGE_STATUS["failed"],
            result=None,
            error_code="THERMAL_SERVICE_MISSING",
            message="Thermal service is unavailable.",
            artifacts=[],
            truthfulness_note="Thermal stage could not run because the service module is missing.",
        )

    try:
        if mode.startswith("radiometric"):
            heatmap, overlay, status_text, result_text = thermal_service.analyze_thermal(
                thermal_image,
                mode="Radiometric Thermal / 真实热红外测温",
            )
        else:
            heatmap, overlay, status_text, result_text = thermal_service.analyze_thermal(
                thermal_image,
                mode="Simulated Thermal / 模拟热红外",
            )
        result = json.loads(result_text) if result_text else {}
        artifacts = []
        thermal_dir = output_dir
        if result_text:
            _save_json(thermal_dir / "thermal_result.json", result)
            artifacts.append(str(thermal_dir / "thermal_result.json"))
        for source in [heatmap, overlay]:
            copied = _copy_if_exists(source, thermal_dir)
            if copied:
                artifacts.append(copied)
        if isinstance(result, dict) and result.get("temperature_matrix_path"):
            copied = _copy_if_exists(result.get("temperature_matrix_path"), thermal_dir)
            if copied:
                artifacts.append(copied)
        status = DEMO_STAGE_STATUS["success"] if result.get("success", True) or result.get("thermal_mode") == "simulated" else DEMO_STAGE_STATUS["failed"]
        return build_stage_result(
            "thermal",
            status,
            result=result,
            message=status_text,
            artifacts=[item for item in artifacts if item],
            truthfulness_note=result.get("truthfulness_note", ""),
        )
    except Exception as exc:
        return build_stage_result(
            "thermal",
            DEMO_STAGE_STATUS["failed"],
            result=None,
            error_code="THERMAL_STAGE_FAILED",
            message=str(exc),
            artifacts=[],
            truthfulness_note="Thermal stage failed structurally. No temperature result was fabricated.",
        )


def run_report_stage(root_dir=None, output_dir=None):
    """Run scanner, evidence ledger, and Final Report 2.0."""
    try:
        root_dir = Path(root_dir) if root_dir else ROOT_DIR
        output_dir = _ensure_dir(output_dir or root_dir / "outputs" / "reports")
        artifacts = []

        module_status_scanner = _lazy_import("module_status_scanner")
        mission_evidence_ledger = _lazy_import("mission_evidence_ledger")
        final_report_v2_service = _lazy_import("final_report_v2_service")
        if module_status_scanner is None or mission_evidence_ledger is None or final_report_v2_service is None:
            return build_stage_result(
                "report",
                DEMO_STAGE_STATUS["failed"],
                result=None,
                error_code="REPORT_SERVICE_MISSING",
                message="Report stack is unavailable.",
                artifacts=[],
                truthfulness_note="Report stage could not run because one or more reporting modules are missing.",
            )

        scan_result = module_status_scanner.scan_all_modules(root_dir=root_dir)
        scan_report = module_status_scanner.save_module_status_report(scan_result, output_dir=output_dir)
        ledger = mission_evidence_ledger.build_mission_evidence_ledger(scan_result=scan_result, root_dir=root_dir)
        ledger_report = mission_evidence_ledger.save_mission_evidence_ledger(ledger=ledger, root_dir=root_dir, output_dir=output_dir)
        final_report = final_report_v2_service.save_final_report_v2(
            report=final_report_v2_service.build_final_report_v2(ledger=ledger, root_dir=root_dir),
            root_dir=root_dir,
            output_dir=output_dir,
        )
        artifacts.extend(
            [
                scan_report.get("markdown_path"),
                scan_report.get("json_path"),
                ledger_report.get("markdown_path"),
                ledger_report.get("json_path"),
                final_report.get("markdown_path"),
                final_report.get("html_path"),
                final_report.get("json_path"),
            ]
        )
        result = {
            "scan_result": scan_result,
            "ledger": ledger,
            "module_status_report": scan_report,
            "mission_evidence_ledger": ledger_report,
            "final_report_v2": final_report,
            "truthfulness_note": "Final Report 2.0 is derived from scanner and mission evidence ledger outputs, not from code existence alone.",
        }
        _save_json(output_dir / "mission_report_bundle.json", result)
        artifacts.append(str(output_dir / "mission_report_bundle.json"))
        return build_stage_result(
            "report",
            DEMO_STAGE_STATUS["success"],
            result=result,
            message=final_report.get("message", "Report stage completed."),
            artifacts=[item for item in artifacts if item],
            truthfulness_note=result["truthfulness_note"],
        )
    except Exception as exc:
        return build_stage_result(
            "report",
            DEMO_STAGE_STATUS["failed"],
            result=None,
            error_code="REPORT_STAGE_FAILED",
            message=str(exc),
            artifacts=[],
            truthfulness_note="Report stage failed structurally. No report was fabricated.",
        )

