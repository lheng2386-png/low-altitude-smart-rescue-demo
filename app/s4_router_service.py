"""S4 Router, execution plan, model execution, and evidence export service.

S4 produces rescue candidates and model evidence for human review. It must not
confirm civilians, survivors, or operational rescue conclusions.
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detection.router.model_router_service import ModelRouterService
from detection.router.route_labels import (
    AIR_BACKEND,
    BACKEND_DISPLAY,
    BACKEND_OUTPUT_ROLE,
    QAZI_BACKEND,
    ROUTE_LABELS,
    TRANSFORMER_BACKEND,
    YOLO_BACKEND,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT_DIR / "outputs" / "detection" / "s4_workbench"
TRUTHFULNESS_BOUNDARY = "模型输出为辅助研判结果，需人工复核，不代表确认人员或真实救援结论。"
MODEL_EVIDENCE_NOTE = "该结果为模型识别出的疑似目标，不代表已确认人员或真实救援结论。"
ROUTE_CONFIG = ROUTE_LABELS


def _utc_timestamp():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


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


def _output_dir(output_dir=None):
    if output_dir:
        path = Path(output_dir)
    else:
        path = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_pil_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8)).convert("RGB")
    if array.ndim == 3:
        if array.shape[-1] == 4:
            array = array[:, :, :3]
        return Image.fromarray(array.astype(np.uint8)).convert("RGB")
    raise ValueError("Unsupported S4 image input.")


def _image_features(image):
    pil = _to_pil_rgb(image)
    arr = np.asarray(pil).astype(np.float32)
    gray = arr.mean(axis=2)
    width, height = pil.size
    gx = np.abs(np.diff(gray, axis=1)).mean() if width > 1 else 0.0
    gy = np.abs(np.diff(gray, axis=0)).mean() if height > 1 else 0.0
    edge_strength = float((gx + gy) / 2.0)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    red_ratio = float(np.mean((arr[:, :, 0] > 150) & (arr[:, :, 1] < 120) & (arr[:, :, 2] < 120)))
    blue_ratio = float(np.mean((arr[:, :, 2] > 135) & (arr[:, :, 0] < 130)))
    dark_ratio = float(np.mean(gray < 55))
    small_input_score = 1.0 - min(width, height) / 512.0
    return {
        "width": width,
        "height": height,
        "edge_strength": round(edge_strength, 4),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "red_ratio": round(red_ratio, 4),
        "blue_ratio": round(blue_ratio, 4),
        "dark_ratio": round(dark_ratio, 4),
        "small_input_score": round(max(0.0, min(1.0, small_input_score)), 4),
    }


def classify_s4_route(image=None, video_frames=None, input_type="image", route_override=None):
    """Compatibility wrapper around the packaged S4 ModelRouterService."""
    decision = ModelRouterService().classify(
        image=image,
        video_frames=video_frames,
        input_type=input_type,
        route_override=route_override,
    )
    decision["truthfulness_boundary"] = TRUTHFULNESS_BOUNDARY
    return decision


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def check_s4_backend_availability(model_variant="yolov11m", root_dir=None, availability_overrides=None):
    root = Path(root_dir or ROOT_DIR)
    overrides = availability_overrides or {}
    availability = {}

    yolo_weights = root / "models" / str(model_variant) / "best.pt"
    availability[YOLO_BACKEND] = {
        "backend": YOLO_BACKEND,
        "display_name": BACKEND_DISPLAY[YOLO_BACKEND],
        "available": yolo_weights.exists(),
        "status": "available" if yolo_weights.exists() else "yolo_unavailable",
        "reason": "local YOLO weights found" if yolo_weights.exists() else f"missing_weights: {yolo_weights}",
        "output_role": BACKEND_OUTPUT_ROLE[YOLO_BACKEND],
    }

    try:
        from transformer_detection_service import check_transformer_detection_environment

        env = check_transformer_detection_environment()
        transformer_available = bool(env.get("success"))
        transformer_reason = "Transformer dependencies importable" if transformer_available else str(env)
    except Exception as exc:
        transformer_available = False
        transformer_reason = f"dependency_check_failed: {exc}"
    availability[TRANSFORMER_BACKEND] = {
        "backend": TRANSFORMER_BACKEND,
        "display_name": BACKEND_DISPLAY[TRANSFORMER_BACKEND],
        "available": transformer_available,
        "status": "available" if transformer_available else "adapter_unavailable",
        "reason": transformer_reason,
        "output_role": BACKEND_OUTPUT_ROLE[TRANSFORMER_BACKEND],
    }

    for backend, external_key in [
        (AIR_BACKEND, "air_sar_detection"),
        (QAZI_BACKEND, "qazi_disaster_management"),
    ]:
        status = _read_json(root / "external_integrations" / "detection" / external_key / "status.json")
        availability[backend] = {
            "backend": backend,
            "display_name": BACKEND_DISPLAY[backend],
            "available": False,
            "status": "adapter_unavailable",
            "reason": status.get("current_state", "planned") + ": executable adapter not reproduced locally",
            "output_role": BACKEND_OUTPUT_ROLE[backend],
        }

    for backend, override in overrides.items():
        availability.setdefault(backend, {"backend": backend, "display_name": backend, "output_role": BACKEND_OUTPUT_ROLE.get(backend, "")})
        availability[backend].update(override)
    return availability


def build_s4_execution_plan(router_decision, availability, low_conf_threshold=0.60):
    """Compatibility wrapper around the packaged S4 ModelRouterService."""
    plan = ModelRouterService(low_confidence_threshold=low_conf_threshold).build_execution_plan(
        router_decision,
        availability,
    )
    plan["truthfulness_boundary"] = TRUTHFULNESS_BOUNDARY
    return plan


def _font(size=16):
    for font_path in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"]:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _draw_targets(image, targets, output_path, title):
    pil = _to_pil_rgb(image).copy()
    draw = ImageDraw.Draw(pil)
    font = _font(15)
    draw.rectangle([0, 0, min(pil.width, 760), 30], fill=(255, 247, 214))
    draw.text((8, 6), title, fill=(24, 33, 47), font=font)
    colors = {
        YOLO_BACKEND: (31, 111, 235),
        TRANSFORMER_BACKEND: (184, 134, 11),
        AIR_BACKEND: (196, 61, 61),
        QAZI_BACKEND: (111, 78, 55),
    }
    for target in targets or []:
        bbox = target.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        color = colors.get(target.get("source_backend"), (24, 33, 47))
        label = f"{target.get('candidate_id') or target.get('id')} {target.get('class_name')} {float(target.get('confidence', 0.0)):.2f}"
        if target.get("human_review_required"):
            label += " | review"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        draw.rectangle([x1, max(0, y1 - 24), x1 + text_bbox[2] + 8, y1], fill=color)
        draw.text((x1 + 4, max(0, y1 - 21)), label, fill=(255, 255, 255), font=font)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path)
    return str(output_path)


def _placeholder_image(image, output_path, message):
    pil = _to_pil_rgb(image).copy()
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, min(pil.height, 90)], fill=(255, 241, 241))
    draw.text((12, 18), message[:120], fill=(138, 31, 31), font=_font(16))
    pil.save(output_path)
    return str(output_path)


def _runtime_target_to_detection(target, backend):
    class_name = str(target.get("class_name") or "unknown").lower()
    if class_name in {"person", "people"}:
        class_name = "human_candidate"
    bbox = [float(v) for v in target.get("bbox", [0, 0, 0, 0])[:4]]
    center = target.get("center") or [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
    return {
        "class_name": class_name,
        "bbox": [round(v, 2) for v in bbox],
        "center": [round(float(v), 2) for v in center[:2]],
        "confidence": round(float(target.get("confidence", 0.0)), 4),
        "source_backend": backend,
        "source_role": BACKEND_OUTPUT_ROLE.get(backend, ""),
        "human_review_required": True,
    }


def _run_yolo(image, model_variant, confidence_threshold, output_dir, mock_backend_results=None):
    if mock_backend_results and YOLO_BACKEND in mock_backend_results:
        detections = [_runtime_target_to_detection(item, YOLO_BACKEND) for item in mock_backend_results[YOLO_BACKEND]]
        overlay = _draw_targets(image, detections, Path(output_dir) / "s4_yolo_main_overlay.png", "YOLO Detector")
        raw_path = _write_json(Path(output_dir) / "yolo_detections.json", {"detections": detections})
        return {"success": True, "backend": YOLO_BACKEND, "detections": detections, "overlay_path": overlay, "result_path": raw_path}
    from detection_runtime_service import run_yolo_detection_runtime

    runtime = run_yolo_detection_runtime(image, model_variant=model_variant, confidence_threshold=confidence_threshold, output_dir=Path(output_dir) / "runtime_yolo")
    detections = [_runtime_target_to_detection(item, YOLO_BACKEND) for item in runtime.get("targets", [])]
    overlay = _draw_targets(image, detections, Path(output_dir) / "s4_yolo_main_overlay.png", "YOLO Detector")
    raw_path = _write_json(Path(output_dir) / "yolo_detections.json", {"runtime": runtime, "detections": detections})
    return {"success": bool(runtime.get("success")), "backend": YOLO_BACKEND, "detections": detections, "overlay_path": overlay, "result_path": raw_path, "runtime": runtime}


def _run_transformer(image, model_key, confidence_threshold, output_dir, mock_backend_results=None):
    if mock_backend_results and TRANSFORMER_BACKEND in mock_backend_results:
        detections = [_runtime_target_to_detection(item, TRANSFORMER_BACKEND) for item in mock_backend_results[TRANSFORMER_BACKEND]]
        overlay = _draw_targets(image, detections, Path(output_dir) / "s4_transformer_compare_overlay.png", "Transformer Detector")
        raw_path = _write_json(Path(output_dir) / "transformer_candidates.json", {"detections": detections})
        return {"success": True, "backend": TRANSFORMER_BACKEND, "detections": detections, "overlay_path": overlay, "result_path": raw_path}
    from detection_runtime_service import run_transformer_detection_runtime

    runtime = run_transformer_detection_runtime(image, model_key=model_key, confidence_threshold=confidence_threshold, output_dir=Path(output_dir) / "runtime_transformer")
    detections = [_runtime_target_to_detection(item, TRANSFORMER_BACKEND) for item in runtime.get("targets", [])]
    overlay = _draw_targets(image, detections, Path(output_dir) / "s4_transformer_compare_overlay.png", "Transformer Detector")
    raw_path = _write_json(Path(output_dir) / "transformer_candidates.json", {"runtime": runtime, "detections": detections})
    return {"success": bool(runtime.get("success")), "backend": TRANSFORMER_BACKEND, "detections": detections, "overlay_path": overlay, "result_path": raw_path, "runtime": runtime}


def _run_unimplemented_adapter(backend, output_dir, reason):
    filename = "air_adapter_status.json" if backend == AIR_BACKEND else "qazi_adapter_status.json"
    status = {
        "backend": backend,
        "status": "adapter_unavailable",
        "reason": reason,
        "truthfulness_boundary": TRUTHFULNESS_BOUNDARY,
        "detections": [],
    }
    return {"success": False, "backend": backend, "detections": [], "adapter_status_path": _write_json(Path(output_dir) / filename, status), "status": status}


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


def _class_can_match(a, b):
    human = {"human_candidate", "civilian", "rescuer", "person", "victim"}
    if a in human and b in human:
        return True
    return a == b


def fuse_rescue_candidates(detections_by_backend):
    flat = []
    for backend, detections in detections_by_backend.items():
        for item in detections or []:
            flat.append(dict(item))

    candidates = []
    used = set()
    for idx, detection in enumerate(flat):
        if idx in used:
            continue
        group = [detection]
        used.add(idx)
        for other_idx, other in enumerate(flat):
            if other_idx in used:
                continue
            if detection.get("source_backend") == other.get("source_backend"):
                continue
            if not _class_can_match(detection.get("class_name"), other.get("class_name")):
                continue
            if _bbox_iou(detection.get("bbox", []), other.get("bbox", [])) >= 0.3:
                group.append(other)
                used.add(other_idx)

        best = max(group, key=lambda item: float(item.get("confidence", 0.0)))
        matched = sorted({item.get("source_backend") for item in group if item.get("source_backend")})
        class_name = best.get("class_name", "unknown")
        human_candidate = class_name == "human_candidate"
        source_backend = best.get("source_backend")
        cross_backend = len(matched) > 1
        can_enter_terp = bool(source_backend == YOLO_BACKEND or cross_backend or human_candidate)
        can_enter_path = bool(source_backend == YOLO_BACKEND and not (human_candidate and not cross_backend))
        review_priority = "high" if human_candidate or cross_backend else "normal"
        if cross_backend:
            review_priority = "urgent_review"
        candidates.append(
            {
                "candidate_id": f"S4-CAND-{len(candidates) + 1:04d}",
                "class_name": class_name,
                "bbox": best.get("bbox"),
                "center": best.get("center"),
                "confidence": best.get("confidence"),
                "source_backend": source_backend,
                "source_role": best.get("source_role", BACKEND_OUTPUT_ROLE.get(source_backend, "")),
                "human_review_required": True,
                "can_enter_terp": can_enter_terp,
                "can_enter_path_planning": can_enter_path,
                "cross_backend_agreement": cross_backend,
                "matched_backends": matched,
                "review_priority": review_priority,
                "notes": "疑似人员候选，需人工复核。" if human_candidate else "模型候选目标，需人工复核。",
                "truthfulness_boundary": MODEL_EVIDENCE_NOTE,
            }
        )
    return candidates


def _crop_candidates(image, candidates, output_dir):
    pil = _to_pil_rgb(image)
    thumbs = []
    for candidate in candidates:
        x1, y1, x2, y2 = [int(round(float(v))) for v in candidate.get("bbox", [0, 0, 0, 0])[:4]]
        pad = 16
        box = (max(0, x1 - pad), max(0, y1 - pad), min(pil.width, x2 + pad), min(pil.height, y2 + pad))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        crop = pil.crop(box)
        crop_path = Path(output_dir) / f"{candidate['candidate_id']}_crop.png"
        crop.save(crop_path)
        candidate["crop_path"] = str(crop_path)
        thumb = crop.copy()
        thumb.thumbnail((190, 130))
        thumbs.append((candidate, thumb))
    sheet_path = Path(output_dir) / "s4_candidate_crops_sheet.png"
    if not thumbs:
        sheet = Image.new("RGB", (620, 150), (255, 247, 214))
        draw = ImageDraw.Draw(sheet)
        draw.text((14, 56), "No rescue candidates generated. No detections are fabricated.", fill=(24, 33, 47), font=_font(16))
        sheet.save(sheet_path)
        return str(sheet_path)
    cell_w, cell_h = 240, 190
    cols = min(3, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (246, 247, 249))
    draw = ImageDraw.Draw(sheet)
    font = _font(13)
    for idx, (candidate, thumb) in enumerate(thumbs):
        col = idx % cols
        row = idx // cols
        x = col * cell_w + 12
        y = row * cell_h + 48
        label = f"{candidate['candidate_id']} | {candidate['class_name']} | {candidate['confidence']}"
        draw.text((x, y - 38), label, fill=(24, 33, 47), font=font)
        draw.text((x, y - 20), "review_required=True", fill=(138, 31, 31), font=font)
        sheet.paste(thumb, (x, y))
    sheet.save(sheet_path)
    return str(sheet_path)


def _evidence_records(candidates, output_dir):
    records = []
    for index, candidate in enumerate(candidates, start=1):
        records.append(
            {
                "evidence_id": f"S4-EVID-{index:04d}",
                "candidate_id": candidate.get("candidate_id"),
                "class_name": candidate.get("class_name"),
                "source_backend": candidate.get("source_backend"),
                "confidence": candidate.get("confidence"),
                "bbox": candidate.get("bbox"),
                "visualization_path": str(Path(output_dir) / "s4_fused_rescue_candidates.png"),
                "crop_path": candidate.get("crop_path", ""),
                "human_review_required": True,
                "generated_at": _utc_timestamp(),
                "used_by": ["S5_target_verification", "S7_terp_ranking", "S8_path_planning", "Final_Report_V2"],
                "truthfulness_note": MODEL_EVIDENCE_NOTE,
            }
        )
    return records


def _candidate_table_rows(candidates):
    rows = []
    for candidate in candidates or []:
        class_label = {
            "human_candidate": "疑似人员候选",
            "civilian": "平民候选",
            "rescuer": "救援人员候选",
            "vehicle": "车辆",
            "fire": "火点/烟火迹象",
        }.get(candidate.get("class_name"), candidate.get("class_name", "unknown"))
        review = "需人工复核"
        usage = "优先复核" if candidate.get("review_priority") in {"high", "urgent_review"} else "可进入排序" if candidate.get("can_enter_terp") else "风险参考"
        rows.append([candidate.get("candidate_id"), class_label, candidate.get("confidence"), review, usage])
    return rows


def _candidate_detail(candidate):
    if not candidate:
        return "请选择候选目标。", None
    source = "多模型一致性" if candidate.get("cross_backend_agreement") else (
        "小目标人体检测流程" if candidate.get("source_backend") == AIR_BACKEND else "单模型检测"
    )
    recommendation = "优先人工复核" if candidate.get("review_priority") in {"high", "urgent_review"} else "进入后续排序前需复核"
    text = (
        f"### 候选目标详情\n"
        f"- 编号：{candidate.get('candidate_id')}\n"
        f"- 类型：{candidate.get('class_name')}\n"
        f"- 置信度：{candidate.get('confidence')}\n"
        f"- 复核状态：需人工复核\n"
        f"- 检测来源：{source}\n"
        f"- 建议：{recommendation}\n"
        f"- 边界说明：{MODEL_EVIDENCE_NOTE}"
    )
    return text, candidate.get("crop_path")


def _confidence_label(value):
    value = float(value or 0.0)
    if value >= 0.8:
        return "较高"
    if value >= 0.6:
        return "中等"
    return "较低"


def _input_summary(image, input_source, input_type):
    if image is None:
        return {
            "input_source": input_source,
            "input_type": input_type,
            "resolution": "",
            "processing_mode": "unsupported",
            "status": "Unsupported",
        }
    pil = _to_pil_rgb(image)
    return {
        "input_source": input_source,
        "input_type": "RGB 图片" if input_type == "image" else "视频关键帧",
        "resolution": f"{pil.width} x {pil.height}",
        "processing_mode": "单帧检测" if input_type == "image" else "视频抽帧检测",
        "status": "Valid",
    }


def run_s4_router_detection(
    image,
    input_source="本地上传",
    input_type="image",
    model_variant="yolov11m",
    transformer_model_key="rescuedet_deformable_detr",
    confidence_threshold=0.3,
    output_dir=None,
    route_override=None,
    availability_overrides=None,
    mock_backend_results=None,
):
    output_dir = _output_dir(output_dir)
    pil = _to_pil_rgb(image) if image is not None else Image.new("RGB", (640, 360), (246, 247, 249))
    router_decision = classify_s4_route(pil if image is not None else None, input_type=input_type, route_override=route_override)
    availability = check_s4_backend_availability(model_variant=model_variant, availability_overrides=availability_overrides)
    execution_plan = build_s4_execution_plan(router_decision, availability)

    selected = list(execution_plan.get("selected_backend_combo", []))
    raw_results = {}
    detections_by_backend = {}
    adapter_status = {}
    for backend in selected:
        if not availability.get(backend, {}).get("available"):
            adapter_status[backend] = {"status": "adapter_unavailable", "reason": availability.get(backend, {}).get("reason", "")}
            continue
        if backend == YOLO_BACKEND:
            raw_results[backend] = _run_yolo(pil, model_variant, confidence_threshold, output_dir, mock_backend_results=mock_backend_results)
        elif backend == TRANSFORMER_BACKEND:
            raw_results[backend] = _run_transformer(pil, transformer_model_key, confidence_threshold, output_dir, mock_backend_results=mock_backend_results)
        elif backend in {AIR_BACKEND, QAZI_BACKEND}:
            raw_results[backend] = _run_unimplemented_adapter(backend, output_dir, availability.get(backend, {}).get("reason", "adapter_unavailable"))
        if raw_results.get(backend, {}).get("success"):
            detections_by_backend[backend] = raw_results[backend].get("detections", [])
        elif backend in {AIR_BACKEND, QAZI_BACKEND}:
            adapter_status[backend] = raw_results[backend].get("status", {})

    for item in execution_plan.get("unavailable_backends", []):
        backend = item.get("backend")
        if backend in {AIR_BACKEND, QAZI_BACKEND} and backend not in adapter_status:
            raw_results[backend] = _run_unimplemented_adapter(backend, output_dir, item.get("reason", "adapter_unavailable"))
            adapter_status[backend] = raw_results[backend].get("status", {})

    candidates = fuse_rescue_candidates(detections_by_backend)
    crops_sheet = _crop_candidates(pil, candidates, output_dir)
    detection_overlay = None
    first_backend = selected[0] if selected else None
    if first_backend and raw_results.get(first_backend, {}).get("overlay_path"):
        detection_overlay = str(Path(output_dir) / "s4_detection_overlay.png")
        shutil.copyfile(raw_results[first_backend]["overlay_path"], detection_overlay)
    else:
        detection_overlay = _placeholder_image(pil, Path(output_dir) / "s4_detection_overlay.png", "No executable main detector output. No detections are fabricated.")
    fused_overlay = _draw_targets(pil, candidates, Path(output_dir) / "s4_fused_rescue_candidates.png", "S4 fused rescue candidates")
    agreement_map = _draw_targets(pil, candidates, Path(output_dir) / "s4_backend_agreement_map.png", "S4 backend agreement map")

    backend_agreement = {
        "candidate_count": len(candidates),
        "items": [
            {
                "candidate_id": item.get("candidate_id"),
                "cross_backend_agreement": item.get("cross_backend_agreement"),
                "matched_backends": item.get("matched_backends"),
                "review_priority": item.get("review_priority"),
            }
            for item in candidates
        ],
        "truthfulness_boundary": TRUTHFULNESS_BOUNDARY,
    }
    evidence_records = _evidence_records(candidates, output_dir)
    input_summary = _input_summary(pil if image is not None else None, input_source, input_type)
    report_summary = {
        "input_source": input_source,
        "display_mode_name": execution_plan.get("display_mode_name"),
        "candidate_count": len(candidates),
        "human_review_required": True,
        "truthfulness_boundary": TRUTHFULNESS_BOUNDARY,
    }

    paths = {
        "execution_plan": _write_json(Path(output_dir) / "execution_plan.json", execution_plan),
        "router_decision": _write_json(Path(output_dir) / "router_decision.json", router_decision),
        "selected_backend_raw_results": _write_json(Path(output_dir) / "selected_backend_raw_results.json", raw_results),
        "rescue_candidates": _write_json(Path(output_dir) / "rescue_candidates.json", {"rescue_candidates": candidates}),
        "backend_agreement": _write_json(Path(output_dir) / "backend_agreement.json", backend_agreement),
        "evidence_records": _write_json(Path(output_dir) / "evidence_records.json", {"evidence_records": evidence_records}),
        "adapter_status": _write_json(Path(output_dir) / "adapter_status.json", adapter_status),
        "final_report_v2_s4_summary": _write_json(Path(output_dir) / "final_report_v2_s4_summary.json", report_summary),
        "s4_detection_overlay": detection_overlay,
        "s4_fused_rescue_candidates": fused_overlay,
        "s4_candidate_crops_sheet": crops_sheet,
        "s4_backend_agreement_map": agreement_map,
    }
    execution_plan["executed_backends"] = [backend for backend, raw in raw_results.items() if raw.get("success")]
    _write_json(paths["execution_plan"], execution_plan)
    return {
        "success": True,
        "output_dir": str(output_dir),
        "input_summary": input_summary,
        "router_decision": router_decision,
        "execution_plan": execution_plan,
        "availability": availability,
        "adapter_status": adapter_status,
        "raw_results": raw_results,
        "rescue_candidates": candidates,
        "candidate_table_rows": _candidate_table_rows(candidates),
        "candidate_detail": _candidate_detail(candidates[0] if candidates else None),
        "backend_agreement": backend_agreement,
        "evidence_records": evidence_records,
        "paths": paths,
        "system_mode_summary": {
            "display_mode_name": execution_plan.get("display_mode_name"),
            "confidence_label": _confidence_label(execution_plan.get("router_confidence")),
            "reason": execution_plan.get("reason"),
            "fallback_applied": execution_plan.get("fallback_applied"),
        },
        "truthfulness_boundary": TRUTHFULNESS_BOUNDARY,
    }


def select_candidate_detail(candidates, index):
    try:
        return _candidate_detail((candidates or [])[int(index)])
    except Exception:
        return _candidate_detail(None)
