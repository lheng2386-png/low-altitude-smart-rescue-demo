"""Optional Transformer RescueDet detection backend.

This module never loads large models at import time. Transformer detections are
auxiliary candidates; human-like labels are not confirmed civilians.
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image


TRANSFORMER_DETECTION_MODELS = {
    "rescuedet_deformable_detr": {
        "model_id": "RoblabWhGe/rescuedet-deformable-detr",
        "description": "Transformer-based rescue-scene detector. Expected labels may include human, vehicle, fire.",
        "backend": "transformer_rescuedet",
    },
    "rescuedet_yolos_small": {
        "model_id": "RoblabWhGe/rescuedet-yolos-small",
        "description": "YOLOS-small rescue-scene detector. Optional backup model.",
        "backend": "transformer_rescuedet",
    },
}

DEFAULT_TRANSFORMER_MODEL = "rescuedet_deformable_detr"
_TRANSFORMER_MODEL_CACHE = {}


class TransformerDetectionError(Exception):
    """Raised when the optional Transformer detector cannot run truthfully."""


def check_transformer_detection_environment():
    """Check optional Transformer dependencies without loading or downloading models."""
    warnings = []
    try:
        import transformers  # noqa: F401

        transformers_available = True
    except Exception as exc:
        transformers_available = False
        warnings.append(f"transformers unavailable: {exc}")

    try:
        import torch  # noqa: F401

        torch_available = True
    except Exception as exc:
        torch_available = False
        warnings.append(f"torch unavailable: {exc}")

    try:
        import huggingface_hub  # noqa: F401

        huggingface_hub_available = True
    except Exception as exc:
        huggingface_hub_available = False
        warnings.append(f"huggingface_hub unavailable: {exc}")

    success = bool(transformers_available and torch_available and huggingface_hub_available)
    return {
        "success": success,
        "transformers_available": transformers_available,
        "torch_available": torch_available,
        "huggingface_hub_available": huggingface_hub_available,
        "available_models": list(TRANSFORMER_DETECTION_MODELS.keys()),
        "warnings": warnings,
        "truthfulness_note": "This check only verifies optional dependencies. It does not load, download, or validate any Transformer model.",
    }


def get_transformer_model_config(model_key=None):
    """Return a configured Transformer detector entry."""
    key = model_key or DEFAULT_TRANSFORMER_MODEL
    if key not in TRANSFORMER_DETECTION_MODELS:
        raise TransformerDetectionError(f"Unknown Transformer detection model key: {key}")
    config = dict(TRANSFORMER_DETECTION_MODELS[key])
    config["model_key"] = key
    return config


def _select_device(device=None):
    if device is not None:
        return device
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def load_transformer_detector(model_key=None, device=None, local_model_path=None, allow_download=False):
    """Load a Transformer object-detection pipeline lazily and cache it.

    By default this function uses local cache/local paths only. Set
    allow_download=True explicitly when the user wants Hugging Face downloads.
    """
    env = check_transformer_detection_environment()
    if not env["success"]:
        raise TransformerDetectionError("Transformer detection dependencies are missing: " + "; ".join(env["warnings"]))
    config = get_transformer_model_config(model_key)
    selected_device = _select_device(device)
    model_source = str(local_model_path).strip() if local_model_path else config["model_id"]
    cache_key = (config["model_key"], selected_device, model_source, bool(allow_download))
    if cache_key in _TRANSFORMER_MODEL_CACHE:
        return _TRANSFORMER_MODEL_CACHE[cache_key]
    try:
        from transformers import pipeline

        detector = pipeline(
            task="object-detection",
            model=model_source,
            device=selected_device,
            model_kwargs={"local_files_only": not bool(allow_download)},
        )
    except Exception as exc:
        download_note = (
            "Downloads are disabled by default; provide a local model path/cache or call with allow_download=True."
            if not allow_download
            else "Downloading was allowed, but model loading still failed."
        )
        raise TransformerDetectionError(
            f"Unable to load Transformer detector '{model_source}'. {download_note} The model may be unavailable locally, network access may be blocked, or dependencies may be incomplete: {exc}"
        ) from exc
    _TRANSFORMER_MODEL_CACHE[cache_key] = detector
    return detector


def normalize_transformer_label(label):
    """Normalize model labels without promoting human candidates to confirmed civilians."""
    normalized = str(label or "unknown").strip().lower().replace(" ", "_")
    aliases = {
        "human": "human_candidate",
        "person": "human_candidate",
        "people": "human_candidate",
        "pedestrian": "human_candidate",
        "vehicle": "vehicle",
        "car": "vehicle",
        "truck": "vehicle",
        "bus": "vehicle",
        "fire": "fire",
        "flame": "fire",
        "smoke": "fire",
    }
    return aliases.get(normalized, normalized)


def map_transformer_label_to_rescue_semantics(label):
    """Map a raw Transformer label into rescue semantics with review boundaries."""
    normalized = normalize_transformer_label(label)
    if normalized == "human_candidate":
        return {
            "raw_label": str(label),
            "normalized_label": normalized,
            "rescue_role": "human_candidate",
            "can_enter_rescue_priority": True,
            "human_review_required": True,
            "mapping_note": "Transformer detected a human-like target. It is treated as a human candidate, not confirmed civilian.",
        }
    if normalized in {"vehicle", "fire"}:
        return {
            "raw_label": str(label),
            "normalized_label": normalized,
            "rescue_role": "environment_risk",
            "can_enter_rescue_priority": False,
            "human_review_required": False,
            "mapping_note": f"{normalized} is treated as disaster-scene context, not a confirmed rescue target.",
        }
    return {
        "raw_label": str(label),
        "normalized_label": normalized,
        "rescue_role": "unknown",
        "can_enter_rescue_priority": False,
        "human_review_required": True,
        "mapping_note": "Unknown Transformer label. Manual review is required before rescue use.",
    }


def _clip_bbox(box, image_width, image_height):
    x1 = float(box.get("xmin", box.get("x1", 0.0)))
    y1 = float(box.get("ymin", box.get("y1", 0.0)))
    x2 = float(box.get("xmax", box.get("x2", 0.0)))
    y2 = float(box.get("ymax", box.get("y2", 0.0)))
    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def normalize_transformer_detections(raw_results, image_width, image_height, model_key, confidence_threshold):
    """Normalize Hugging Face object-detection outputs into AeroRescue-AI target records."""
    config = get_transformer_model_config(model_key)
    targets = []
    for item in raw_results or []:
        confidence = float(item.get("score", 0.0) or 0.0)
        if confidence < float(confidence_threshold):
            continue
        raw_label = str(item.get("label", "unknown"))
        semantics = map_transformer_label_to_rescue_semantics(raw_label)
        bbox = _clip_bbox(item.get("box", {}), image_width, image_height)
        x1, y1, x2, y2 = bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        target_id = f"TR{len(targets) + 1:03d}"
        targets.append(
            {
                "id": target_id,
                "class_name": semantics["normalized_label"],
                "raw_label": raw_label,
                "confidence": round(confidence, 4),
                "bbox": bbox,
                "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                "area": area,
                "source_backend": config["backend"],
                "model_key": config["model_key"],
                "model_name": config["model_id"],
                "rescue_role": semantics["rescue_role"],
                "can_enter_rescue_priority": semantics["can_enter_rescue_priority"],
                "human_review_required": semantics["human_review_required"],
                "mapping_note": semantics["mapping_note"],
                "truthfulness_note": "This detection was produced by a Transformer object detector. Human-like detections are candidates and require manual review.",
            }
        )
    return targets


def _as_pil_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8)).convert("RGB")
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return Image.fromarray(array.astype(np.uint8)).convert("RGB")


def run_transformer_detection(
    image,
    model_key=None,
    confidence_threshold=0.4,
    local_model_path=None,
    allow_download=False,
):
    """Run optional Transformer detection and return a structured result."""
    config = get_transformer_model_config(model_key)
    env = check_transformer_detection_environment()
    if not env["success"]:
        return {
            "success": False,
            "detection_backend": config["backend"],
            "model_key": config["model_key"],
            "model_name": config["model_id"],
            "confidence_threshold": float(confidence_threshold),
            "image_size": None,
            "target_count": 0,
            "targets": [],
            "raw_result_count": 0,
            "inference_time_ms": None,
            "is_model_output": False,
            "allow_download": bool(allow_download),
            "local_model_path": str(local_model_path) if local_model_path else None,
            "human_review_required": True,
            "truthfulness_note": "Transformer detector did not run because optional dependencies are missing. No detections are fabricated.",
            "error_code": "DEPENDENCY_MISSING",
            "message": "; ".join(env["warnings"]) or "Transformer dependencies are missing.",
        }
    try:
        pil_image = _as_pil_image(image)
    except Exception as exc:
        return {
            "success": False,
            "detection_backend": config["backend"],
            "model_key": config["model_key"],
            "model_name": config["model_id"],
            "confidence_threshold": float(confidence_threshold),
            "image_size": None,
            "target_count": 0,
            "targets": [],
            "raw_result_count": 0,
            "inference_time_ms": None,
            "is_model_output": False,
            "allow_download": bool(allow_download),
            "local_model_path": str(local_model_path) if local_model_path else None,
            "human_review_required": True,
            "truthfulness_note": "Input image could not be converted for Transformer detection. No detections are fabricated.",
            "error_code": "INVALID_INPUT",
            "message": str(exc),
        }
    width, height = pil_image.size
    try:
        detector = load_transformer_detector(
            config["model_key"],
            local_model_path=local_model_path,
            allow_download=allow_download,
        )
    except TransformerDetectionError as exc:
        return {
            "success": False,
            "detection_backend": config["backend"],
            "model_key": config["model_key"],
            "model_name": config["model_id"],
            "confidence_threshold": float(confidence_threshold),
            "image_size": [width, height],
            "target_count": 0,
            "targets": [],
            "raw_result_count": 0,
            "inference_time_ms": None,
            "is_model_output": False,
            "allow_download": bool(allow_download),
            "local_model_path": str(local_model_path) if local_model_path else None,
            "human_review_required": True,
            "truthfulness_note": "Transformer model is unavailable. Downloads are disabled unless allow_download=True is explicitly requested. No detections are fabricated.",
            "error_code": "MODEL_UNAVAILABLE",
            "message": str(exc),
        }
    try:
        start = time.perf_counter()
        raw_results = detector(pil_image)
        inference_time_ms = (time.perf_counter() - start) * 1000.0
        targets = normalize_transformer_detections(raw_results, width, height, config["model_key"], confidence_threshold)
        human_review_required = any(target.get("human_review_required") for target in targets)
        return {
            "success": True,
            "detection_backend": config["backend"],
            "model_key": config["model_key"],
            "model_name": config["model_id"],
            "confidence_threshold": float(confidence_threshold),
            "image_size": [width, height],
            "target_count": len(targets),
            "targets": targets,
            "raw_result_count": len(raw_results or []),
            "inference_time_ms": round(float(inference_time_ms), 2),
            "is_model_output": True,
            "allow_download": bool(allow_download),
            "local_model_path": str(local_model_path) if local_model_path else None,
            "human_review_required": human_review_required,
            "truthfulness_note": "Transformer detections are auxiliary model outputs. Human-like detections are candidates and require manual review; they do not replace rescue decisions.",
            "error_code": None,
            "message": "Transformer detection completed.",
        }
    except Exception as exc:
        return {
            "success": False,
            "detection_backend": config["backend"],
            "model_key": config["model_key"],
            "model_name": config["model_id"],
            "confidence_threshold": float(confidence_threshold),
            "image_size": [width, height],
            "target_count": 0,
            "targets": [],
            "raw_result_count": 0,
            "inference_time_ms": None,
            "is_model_output": False,
            "allow_download": bool(allow_download),
            "local_model_path": str(local_model_path) if local_model_path else None,
            "human_review_required": True,
            "truthfulness_note": "Transformer inference failed. No detections are fabricated.",
            "error_code": "INFERENCE_FAILED",
            "message": str(exc),
        }


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    return 0.0 if denom <= 0 else inter_area / denom


def compare_yolo_and_transformer_detections(yolo_targets, transformer_targets, iou_threshold=0.3):
    """Compare YOLO rescue targets with Transformer auxiliary candidates."""
    matched_pairs = []
    matched_yolo = set()
    matched_transformer = set()
    yolo_human_classes = {"civilian", "rescuer", "person", "people"}
    for yolo in yolo_targets or []:
        for transformer in transformer_targets or []:
            if yolo.get("class_name") not in yolo_human_classes:
                continue
            if transformer.get("class_name") != "human_candidate":
                continue
            iou = _bbox_iou(yolo.get("bbox", [0, 0, 0, 0]), transformer.get("bbox", [0, 0, 0, 0]))
            if iou >= float(iou_threshold):
                matched_yolo.add(yolo.get("id"))
                matched_transformer.add(transformer.get("id"))
                matched_pairs.append(
                    {
                        "yolo_target_id": yolo.get("id"),
                        "transformer_target_id": transformer.get("id"),
                        "iou": round(float(iou), 4),
                        "consensus_type": "human_target_overlap",
                        "note": "YOLO human-class target overlaps with Transformer human_candidate. This is auxiliary consistency only and still requires manual review.",
                    }
                )
    yolo_only = [target for target in yolo_targets or [] if target.get("id") not in matched_yolo]
    transformer_only = [target for target in transformer_targets or [] if target.get("id") not in matched_transformer]
    if matched_pairs:
        summary = f"发现 {len(matched_pairs)} 组 YOLO 人员类目标与 Transformer human_candidate 重叠，提示人员目标一致性增强，但仍需人工复核。"
    else:
        summary = "未发现 YOLO 人员类目标与 Transformer human_candidate 的显著重叠。Transformer-only human_candidate 只作为人工复核线索。"
    return {
        "success": True,
        "matched_pairs": matched_pairs,
        "yolo_only": yolo_only,
        "transformer_only": transformer_only,
        "consensus_summary": summary,
        "truthfulness_note": "This comparison is auxiliary bbox consistency analysis. It is not final rescue judgment and does not create confirmed civilians from Transformer-only detections.",
    }
