"""Unified detection runtime adapter for 灾情感知及影响评估.

This module executes only current runnable detection modes. Planned/reference
backends remain registry entries and are not executed here.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_CACHE = {}


EXECUTABLE_DETECTION_MODES = {
    "yolo_rescue_targets": {
        "display_name": "YOLO Rescue Targets",
        "runtime_role": "primary",
    },
    "transformer_rescuedet_argus": {
        "display_name": "Transformer RescueDet from ARGUS",
        "runtime_role": "auxiliary",
    },
    "dual_backend_compare": {
        "display_name": "YOLO + Transformer Compare",
        "runtime_role": "consensus",
    },
}

YOLO_CLASSES = ["civilian", "rescuer", "dog", "cat", "horse", "cow"]
REFERENCE_OR_PLANNED_MODES = {
    "qazi_disaster_management_reference",
    "air_retinanet_sar_reference",
    "bahmanyar_merkle_person_detection_reference",
    "vtsar_dataset_reference",
    "sardet_or_vtsar_reference",
    "post_disaster_survivor_yolo",
}


class DetectionRuntimeError(Exception):
    """Raised when detection runtime input cannot be processed."""


def normalize_image_input(image):
    """Normalize image input into PIL RGB, numpy RGB, width, height, and path."""
    source_path = None
    try:
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif isinstance(image, (str, Path)):
            source_path = str(image)
            pil_image = Image.open(image).convert("RGB")
        else:
            array = np.asarray(image)
            if array.ndim == 2:
                pil_image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
            elif array.ndim == 3:
                if array.shape[-1] == 4:
                    array = array[:, :, :3]
                pil_image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
            else:
                raise ValueError("Unsupported image array shape.")
    except Exception as exc:
        raise DetectionRuntimeError(f"Unable to normalize image input: {exc}") from exc
    np_image = np.asarray(pil_image).copy()
    width, height = pil_image.size
    return {
        "pil_image": pil_image,
        "np_image": np_image,
        "width": width,
        "height": height,
        "source_path": source_path,
    }


def build_empty_detection_result(
    detection_mode,
    backend_key=None,
    model_name=None,
    model_variant=None,
    confidence_threshold=None,
    image_size=None,
    truthfulness_note="No detection result was produced.",
    error_code="DETECTION_FAILED",
    message="Detection did not run.",
):
    """Build a structured failure result."""
    result = {
        "success": False,
        "detection_mode": detection_mode,
        "backend_key": backend_key,
        "model_name": model_name,
        "model_variant": model_variant,
        "confidence_threshold": confidence_threshold,
        "image_size": image_size,
        "target_count": 0,
        "targets": [],
        "annotated_image_path": None,
        "detection_result_path": None,
        "metadata_path": None,
        "inference_time_ms": None,
        "is_model_output": False,
        "human_review_required": True,
        "can_enter_terp": False,
        "can_enter_path_planning": False,
        "truthfulness_note": truthfulness_note,
        "error_code": error_code,
        "message": message,
    }
    try:
        from s4_reference_fusion import enrich_detection_result_with_reference_fusion

        return enrich_detection_result_with_reference_fusion(result, root_dir=ROOT_DIR)
    except Exception:
        return result


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _weights_path(model_variant):
    return ROOT_DIR / "models" / str(model_variant) / "best.pt"


def _get_yolo_model(model_variant):
    weights = _weights_path(model_variant)
    if not weights.exists():
        raise DetectionRuntimeError(f"YOLO weights not found: {weights}")
    if model_variant not in MODEL_CACHE:
        from ultralytics import YOLO

        MODEL_CACHE[model_variant] = YOLO(str(weights))
    return MODEL_CACHE[model_variant]


def _extract_yolo_targets(results):
    targets = []
    if not results:
        return targets
    result = results[0]
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return targets
    for idx, box in enumerate(boxes):
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
        confidence = float(box.conf[0].detach().cpu().item()) if getattr(box, "conf", None) is not None else 0.0
        cls_id = int(box.cls[0].detach().cpu().item()) if getattr(box, "cls", None) is not None else -1
        class_name = str(names.get(cls_id, cls_id)).strip().lower()
        x1, y1, x2, y2 = xyxy
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        targets.append(
            {
                "id": f"T{idx + 1:03d}",
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)],
                "area": float(area),
                "source_backend": "yolo_rescue_targets",
                "human_review_required": True,
                "truthfulness_note": "YOLO detection result from local model weights. Requires human review before real rescue use.",
            }
        )
    return targets


def _draw_targets(pil_image, targets):
    image = pil_image.copy()
    draw = ImageDraw.Draw(image)
    for target in targets:
        x1, y1, x2, y2 = [float(v) for v in target.get("bbox", [0, 0, 0, 0])]
        label = f"{target.get('class_name', 'target')} {target.get('confidence', 0):.2f}"
        color = (255, 230, 0) if target.get("source_backend") == "yolo_rescue_targets" else (0, 180, 255)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.rectangle([x1, max(0, y1 - 16), x1 + max(80, len(label) * 7), y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - 15)), label, fill=(0, 0, 0))
    return image


def save_detection_artifacts(result, output_dir):
    """Save JSON result/metadata when requested."""
    if output_dir is None:
        return result
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    mode = result.get("detection_mode", "detection")
    result_name = "dual_detection_result.json" if mode == "dual_backend_compare" else "detection_result.json"
    metadata_name = "detection_metadata.json"
    result["detection_result_path"] = _write_json(output_path / result_name, result)
    metadata = {
        "backend_key": result.get("backend_key"),
        "detection_mode": result.get("detection_mode"),
        "model_name": result.get("model_name"),
        "model_variant": result.get("model_variant"),
        "confidence_threshold": result.get("confidence_threshold"),
        "target_count": result.get("target_count"),
        "inference_time_ms": result.get("inference_time_ms"),
        "can_enter_terp": result.get("can_enter_terp"),
        "can_enter_path_planning": result.get("can_enter_path_planning"),
        "truthfulness_note": result.get("truthfulness_note"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    result["metadata_path"] = _write_json(output_path / metadata_name, metadata)
    return result


def _with_s4_reference_fusion(result):
    try:
        from s4_reference_fusion import enrich_detection_result_with_reference_fusion

        return enrich_detection_result_with_reference_fusion(result, root_dir=ROOT_DIR)
    except Exception:
        return result


def run_yolo_detection_runtime(image, model_variant="yolov11m", confidence_threshold=0.3, output_dir=None):
    """Run YOLO rescue-target detection with local weights only."""
    try:
        normalized = normalize_image_input(image)
    except DetectionRuntimeError as exc:
        return build_empty_detection_result(
            "yolo_rescue_targets",
            backend_key="yolo_rescue_targets",
            model_variant=model_variant,
            confidence_threshold=float(confidence_threshold),
            error_code="INVALID_INPUT",
            message=str(exc),
        )
    weights = _weights_path(model_variant)
    image_size = [normalized["width"], normalized["height"]]
    if not weights.exists():
        return build_empty_detection_result(
            "yolo_rescue_targets",
            backend_key="yolo_rescue_targets",
            model_variant=model_variant,
            confidence_threshold=float(confidence_threshold),
            image_size=image_size,
            truthfulness_note="YOLO cannot run because local best.pt weights are missing. No detections are fabricated.",
            error_code="YOLO_WEIGHTS_MISSING",
            message=f"Missing YOLO weights: {weights}",
        )
    try:
        model = _get_yolo_model(model_variant)
        start = time.perf_counter()
        results = model.predict(normalized["np_image"], conf=float(confidence_threshold), verbose=False)
        inference_time_ms = (time.perf_counter() - start) * 1000.0
        targets = _extract_yolo_targets(results)
        for target in targets:
            target["model_variant"] = model_variant
        annotated_image_path = None
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            overlay = _draw_targets(normalized["pil_image"], targets)
            annotated_image_path = str(output_path / "detection_overlay.png")
            overlay.save(annotated_image_path)
        result = {
            "success": True,
            "detection_mode": "yolo_rescue_targets",
            "backend_key": "yolo_rescue_targets",
            "model_name": "Ultralytics YOLO local weights",
            "model_variant": model_variant,
            "confidence_threshold": float(confidence_threshold),
            "image_size": image_size,
            "target_count": len(targets),
            "targets": targets,
            "annotated_image_path": annotated_image_path,
            "detection_result_path": None,
            "metadata_path": None,
            "inference_time_ms": round(float(inference_time_ms), 2),
            "is_model_output": True,
            "human_review_required": True,
            "can_enter_terp": True,
            "can_enter_path_planning": True,
            "truthfulness_note": "YOLO detection result from local model weights. Requires human review before real rescue use.",
            "error_code": None,
            "message": "YOLO detection completed.",
            "metadata": {
                "backend_key": "yolo_rescue_targets",
                "model_variant": model_variant,
                "weights_path": str(weights),
                "confidence_threshold": float(confidence_threshold),
                "class_schema": YOLO_CLASSES,
                "target_count": len(targets),
                "inference_time_ms": round(float(inference_time_ms), 2),
                "truthfulness_note": "YOLO detections are local model outputs if weights exist.",
            },
        }
        return save_detection_artifacts(_with_s4_reference_fusion(result), output_dir)
    except Exception as exc:
        return build_empty_detection_result(
            "yolo_rescue_targets",
            backend_key="yolo_rescue_targets",
            model_variant=model_variant,
            confidence_threshold=float(confidence_threshold),
            image_size=image_size,
            truthfulness_note="YOLO inference failed. No detections are fabricated.",
            error_code="YOLO_INFERENCE_FAILED",
            message=str(exc),
        )


def run_transformer_detection_runtime(
    image,
    model_key="rescuedet_deformable_detr",
    confidence_threshold=0.4,
    output_dir=None,
):
    """Run optional Transformer RescueDet through the existing service."""
    try:
        normalized = normalize_image_input(image)
    except DetectionRuntimeError as exc:
        return build_empty_detection_result(
            "transformer_rescuedet_argus",
            backend_key="transformer_rescuedet_argus",
            model_name=model_key,
            confidence_threshold=float(confidence_threshold),
            error_code="INVALID_INPUT",
            message=str(exc),
        )
    try:
        from transformer_detection_service import run_transformer_detection
    except Exception as exc:
        return build_empty_detection_result(
            "transformer_rescuedet_argus",
            backend_key="transformer_rescuedet_argus",
            model_name=model_key,
            confidence_threshold=float(confidence_threshold),
            image_size=[normalized["width"], normalized["height"]],
            truthfulness_note="Transformer service is missing. No detections are fabricated.",
            error_code="TRANSFORMER_SERVICE_MISSING",
            message=str(exc),
        )
    transformer_result = run_transformer_detection(
        normalized["pil_image"],
        model_key=model_key,
        confidence_threshold=float(confidence_threshold),
        allow_download=False,
    )
    result = {
        **transformer_result,
        "detection_mode": "transformer_rescuedet_argus",
        "backend_key": "transformer_rescuedet_argus",
        "model_variant": None,
        "annotated_image_path": None,
        "detection_result_path": None,
        "metadata_path": None,
        "can_enter_terp": "human_candidate_only_with_review" if transformer_result.get("success") else False,
        "can_enter_path_planning": False,
    }
    result = _with_s4_reference_fusion(result)
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result["detection_result_path"] = _write_json(output_path / "transformer_detection_result.json", result)
        metadata = {
            "backend_key": "transformer_rescuedet_argus",
            "model_key": model_key,
            "confidence_threshold": float(confidence_threshold),
            "target_count": result.get("target_count", 0),
            "is_model_output": result.get("is_model_output", False),
            "truthfulness_note": result.get("truthfulness_note", ""),
        }
        result["metadata_path"] = _write_json(output_path / "transformer_detection_metadata.json", metadata)
    return result


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def compare_detection_targets(yolo_targets, transformer_targets, iou_threshold=0.3):
    """Fallback YOLO/Transformer bbox consistency analysis."""
    matched_pairs = []
    matched_yolo = set()
    matched_transformer = set()
    yolo_human_classes = {"civilian", "rescuer", "person", "people"}
    for yolo in yolo_targets or []:
        for transformer in transformer_targets or []:
            if str(yolo.get("class_name", "")).lower() not in yolo_human_classes:
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
                        "note": "YOLO human-class target overlaps with Transformer human_candidate. This is auxiliary evidence and requires manual review.",
                    }
                )
    yolo_only = [target for target in yolo_targets or [] if target.get("id") not in matched_yolo]
    transformer_only = [target for target in transformer_targets or [] if target.get("id") not in matched_transformer]
    summary = (
        f"发现 {len(matched_pairs)} 组 YOLO 人员类目标与 Transformer human_candidate 重叠，建议优先人工复核。"
        if matched_pairs
        else "未发现 YOLO 人员类目标与 Transformer human_candidate 的显著重叠。Transformer-only human_candidate 不能直接升级为 confirmed civilian。"
    )
    return {
        "success": True,
        "matched_pairs": matched_pairs,
        "yolo_only": yolo_only,
        "transformer_only": transformer_only,
        "consensus_summary": summary,
        "truthfulness_note": "Dual-backend comparison is auxiliary consistency evidence, not a final rescue judgment.",
    }


def run_dual_backend_compare_runtime(
    image,
    yolo_model_variant="yolov11m",
    transformer_model_key="rescuedet_deformable_detr",
    confidence_threshold=0.3,
    transformer_confidence_threshold=0.4,
    output_dir=None,
):
    """Run YOLO as primary and Transformer as auxiliary consensus evidence."""
    yolo_result = run_yolo_detection_runtime(image, yolo_model_variant, confidence_threshold, output_dir=output_dir)
    transformer_result = run_transformer_detection_runtime(
        image,
        model_key=transformer_model_key,
        confidence_threshold=transformer_confidence_threshold,
        output_dir=output_dir,
    )
    try:
        from transformer_detection_service import compare_yolo_and_transformer_detections

        consensus = compare_yolo_and_transformer_detections(
            yolo_result.get("targets", []),
            transformer_result.get("targets", []),
        )
    except Exception:
        consensus = compare_detection_targets(yolo_result.get("targets", []), transformer_result.get("targets", []))
    success = bool(yolo_result.get("success"))
    partial_success = bool(yolo_result.get("success") or transformer_result.get("success"))
    error_code = None if success else yolo_result.get("error_code") or transformer_result.get("error_code")
    result = {
        "success": success,
        "partial_success": partial_success,
        "detection_mode": "dual_backend_compare",
        "backend_key": "dual_backend_compare",
        "primary_result": yolo_result,
        "auxiliary_result": transformer_result,
        "consensus": consensus,
        "targets": yolo_result.get("targets", []),
        "target_count": len(yolo_result.get("targets", [])),
        "can_enter_terp": bool(yolo_result.get("success")),
        "can_enter_path_planning": bool(yolo_result.get("success")),
        "human_review_required": True,
        "truthfulness_note": "YOLO remains the primary rescue-target detector. Transformer RescueDet is used only as auxiliary consensus evidence.",
        "error_code": error_code,
        "message": "Dual backend comparison completed." if partial_success else "Dual backend comparison failed because no executable backend succeeded.",
        "annotated_image_path": yolo_result.get("annotated_image_path"),
        "detection_result_path": None,
        "metadata_path": None,
    }
    result = _with_s4_reference_fusion(result)
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result["detection_result_path"] = _write_json(output_path / "dual_detection_result.json", result)
        _write_json(output_path / "dual_detection_consensus.json", consensus)
    return result


def run_detection(
    image,
    detection_mode="yolo_rescue_targets",
    model_variant="yolov11m",
    transformer_model_key="rescuedet_deformable_detr",
    confidence_threshold=0.3,
    transformer_confidence_threshold=0.4,
    output_dir=None,
):
    """Unified detection runtime entrypoint."""
    if detection_mode == "yolo_rescue_targets":
        return run_yolo_detection_runtime(image, model_variant, confidence_threshold, output_dir=output_dir)
    if detection_mode == "transformer_rescuedet_argus":
        return run_transformer_detection_runtime(
            image,
            model_key=transformer_model_key,
            confidence_threshold=transformer_confidence_threshold,
            output_dir=output_dir,
        )
    if detection_mode == "dual_backend_compare":
        return run_dual_backend_compare_runtime(
            image,
            yolo_model_variant=model_variant,
            transformer_model_key=transformer_model_key,
            confidence_threshold=confidence_threshold,
            transformer_confidence_threshold=transformer_confidence_threshold,
            output_dir=output_dir,
        )
    if detection_mode in REFERENCE_OR_PLANNED_MODES:
        return build_empty_detection_result(
            detection_mode,
            backend_key=detection_mode,
            confidence_threshold=float(confidence_threshold),
            truthfulness_note="This backend is planned or reference-only and is not executable in the current runtime.",
            error_code="REFERENCE_BACKEND_NOT_EXECUTABLE",
            message=f"{detection_mode} is not an executable 灾情感知及影响评估 detection runtime mode.",
        )
    return build_empty_detection_result(
        detection_mode,
        backend_key=detection_mode,
        confidence_threshold=float(confidence_threshold),
        truthfulness_note="Unsupported detection mode. No model was executed and no results are fabricated.",
        error_code="UNSUPPORTED_DETECTION_MODE",
        message=f"Unsupported detection mode: {detection_mode}",
    )


def format_detection_runtime_status(result):
    """Format a detection runtime result as Chinese Markdown."""
    if not isinstance(result, dict):
        return "## 检测运行状态\n\n检测结果格式无效。"
    lines = [
        "## 检测运行状态",
        f"- 检测模式：{result.get('detection_mode')}",
        f"- 是否成功：{'是' if result.get('success') else '否'}",
        f"- 主后端：{result.get('backend_key')}",
        f"- 模型变体：{result.get('model_variant') or result.get('model_key') or result.get('model_name')}",
        f"- 检测目标数：{result.get('target_count', 0)}",
        f"- 可进入 TERP：{result.get('can_enter_terp')}",
        f"- 可进入路径规划：{result.get('can_enter_path_planning')}",
        f"- 需要人工复核：{'是' if result.get('human_review_required', True) else '否'}",
        f"- 真实性说明：{result.get('truthfulness_note', '')}",
    ]
    if result.get("error_code"):
        lines.append(f"- 错误代码：{result.get('error_code')}")
    if result.get("message"):
        lines.append(f"- 状态说明：{result.get('message')}")
    if result.get("consensus"):
        lines.append(f"- 双后端一致性：{result['consensus'].get('consensus_summary', '')}")
    reference_fusion = result.get("s4_reference_fusion") or {}
    if reference_fusion:
        lines.append(f"- S4 源码级参考融合：{reference_fusion.get('reference_count', 0)} 个参考源")
    return "\n".join(lines)


def format_detection_result_for_report(result):
    """Return a compact Chinese detection summary for future report export."""
    if not result:
        return "目标检测结果尚未生成。"
    text = [
        f"检测后端：{result.get('backend_key')}",
        f"检测模式：{result.get('detection_mode')}",
        f"目标数量：{result.get('target_count', 0)}",
        f"是否可进入 TERP：{result.get('can_enter_terp')}",
        f"是否需要人工复核：{'是' if result.get('human_review_required', True) else '否'}",
        f"真实性说明：{result.get('truthfulness_note', '')}",
    ]
    if result.get("consensus"):
        text.append(f"双后端一致性摘要：{result['consensus'].get('consensus_summary', '')}")
    reference_fusion = result.get("s4_reference_fusion") or {}
    if reference_fusion:
        text.append(
            "S4 源码级参考融合："
            f"{reference_fusion.get('reference_count', 0)} 个参考源，"
            f"{reference_fusion.get('person_detection_reference_count', 0)} 个人员检测参考源。"
        )
    return "\n".join(text)
