"""Scene mode detection, rescue-entry generation, and path-planning gate.

This module keeps path planning conservative: local reconnaissance images do
not receive a synthetic route unless the user explicitly enables force mode.
"""

import cv2
import numpy as np


ROAD_CLEAR_CLASS_ID = 7
ROAD_BLOCKED_CLASS_ID = 8
BUILDING_CLASS_IDS = {2, 3, 4, 5}


def _image_size(image):
    if image is None:
        return 0, 0
    if hasattr(image, "size") and not isinstance(image, np.ndarray):
        return int(image.size[0]), int(image.size[1])
    array = np.asarray(image)
    if array.ndim < 2:
        return 0, 0
    return int(array.shape[1]), int(array.shape[0])


def _target_area_ratios(detections, image_area):
    detections = detections or []
    if not detections:
        return 0.0, 0.0
    ratios = []
    for target in detections:
        area = float(target.get("area", 0.0) or 0.0)
        if area <= 0 and target.get("bbox"):
            x1, y1, x2, y2 = [float(v) for v in target.get("bbox", [0, 0, 0, 0])[:4]]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ratios.append(area / max(1.0, float(image_area)))
    return float(np.mean(ratios)), float(max(ratios))


def _aligned_mask(segmentation_mask, width, height):
    if segmentation_mask is None:
        return None
    mask = np.asarray(segmentation_mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if width > 0 and height > 0 and mask.shape[:2] != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.uint8)


def _road_boundary_evidence(mask, boundary_margin_ratio=0.08):
    if mask is None:
        return False, 0
    road = (mask == ROAD_CLEAR_CLASS_ID).astype(np.uint8)
    if road.sum() == 0:
        return False, 0
    height, width = road.shape[:2]
    margin = max(3, int(min(width, height) * boundary_margin_ratio))
    min_area = max(12, int(width * height * 0.001))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road, connectivity=8)
    boundary_components = 0
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        if x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin:
            boundary_components += 1
    return boundary_components > 0, boundary_components


def analyze_scene_mode(image, segmentation_mask=None, detections=None):
    """Classify whether the scene is suitable for image-plane route planning."""
    width, height = _image_size(image)
    image_area = max(1, width * height)
    avg_target_ratio, max_target_ratio = _target_area_ratios(detections, image_area)

    evidence = {
        "image_width": width,
        "image_height": height,
        "target_count": len(detections or []),
        "avg_target_area_ratio": round(avg_target_ratio, 4),
        "max_target_area_ratio": round(max_target_ratio, 4),
        "has_segmentation_mask": segmentation_mask is not None,
        "road_clear_ratio": 0.0,
        "road_blocked_ratio": 0.0,
        "building_ratio": 0.0,
        "boundary_connected_road": False,
        "boundary_road_component_count": 0,
    }

    if not detections:
        return {
            "scene_mode": "unknown",
            "scene_mode_label": "Unknown / 信息不足",
            "path_planning_allowed": False,
            "reason": "当前未检测到明确救援目标，无法判断救援终点，因此不启用路径规划。",
            "evidence": evidence,
        }

    if max_target_ratio > 0.12 or avg_target_ratio > 0.06:
        return {
            "scene_mode": "local_reconnaissance",
            "scene_mode_label": "Local Reconnaissance / 局部侦察模式",
            "path_planning_allowed": False,
            "reason": "目标框占画面比例较大，当前图像更像近距离局部侦察图，缺乏完整道路与可达性信息。",
            "evidence": evidence,
        }

    mask = _aligned_mask(segmentation_mask, width, height)
    if mask is None:
        return {
            "scene_mode": "unknown",
            "scene_mode_label": "Unknown / 信息不足",
            "path_planning_allowed": False,
            "reason": "当前缺少语义分割掩码，无法确认 Road-Clear 可通行道路和场景结构，因此不默认生成路径。",
            "evidence": evidence,
        }

    road_clear_ratio = float(np.count_nonzero(mask == ROAD_CLEAR_CLASS_ID)) / image_area
    road_blocked_ratio = float(np.count_nonzero(mask == ROAD_BLOCKED_CLASS_ID)) / image_area
    building_ratio = float(np.count_nonzero(np.isin(mask, list(BUILDING_CLASS_IDS)))) / image_area
    boundary_road, boundary_count = _road_boundary_evidence(mask)
    evidence.update(
        {
            "road_clear_ratio": round(road_clear_ratio, 4),
            "road_blocked_ratio": round(road_blocked_ratio, 4),
            "building_ratio": round(building_ratio, 4),
            "boundary_connected_road": boundary_road,
            "boundary_road_component_count": int(boundary_count),
        }
    )

    if road_clear_ratio <= 0:
        return {
            "scene_mode": "unknown",
            "scene_mode_label": "Unknown / 信息不足",
            "path_planning_allowed": False,
            "reason": "语义分割中没有 Road-Clear 可通行道路区域，系统不会伪造救援入口或路径。",
            "evidence": evidence,
        }

    if road_clear_ratio < 0.015:
        return {
            "scene_mode": "local_reconnaissance",
            "scene_mode_label": "Local Reconnaissance / 局部侦察模式",
            "path_planning_allowed": False,
            "reason": "Road-Clear 可通行道路比例过低，当前图像缺乏可靠路径规划基础。",
            "evidence": evidence,
        }

    if not boundary_road:
        return {
            "scene_mode": "local_reconnaissance",
            "scene_mode_label": "Local Reconnaissance / 局部侦察模式",
            "path_planning_allowed": False,
            "reason": "未发现与图像边界相连或接近边界的 Road-Clear 道路入口，因此不生成路径建议。",
            "evidence": evidence,
        }

    if road_clear_ratio >= 0.015 and (building_ratio >= 0.01 or road_clear_ratio >= 0.05):
        return {
            "scene_mode": "wide_area_assessment",
            "scene_mode_label": "Wide-area Assessment / 广域评估模式",
            "path_planning_allowed": True,
            "reason": "图像中存在可通行道路并连接边界，且具备一定道路/建筑场景结构，可进行图像平面路径评估。",
            "evidence": evidence,
        }

    return {
        "scene_mode": "unknown",
        "scene_mode_label": "Unknown / 信息不足",
        "path_planning_allowed": False,
        "reason": "道路信息存在但场景结构不足，暂不启用路径规划。",
        "evidence": evidence,
    }


def find_rescue_entry_point(segmentation_mask, target_point=None):
    """Find a boundary-connected Road-Clear point to use as a rescue entry."""
    mask = _aligned_mask(segmentation_mask, 0, 0)
    if mask is None:
        return {
            "entry_found": False,
            "entry_point": None,
            "entry_reason": "未接入语义分割掩码，无法寻找 Road-Clear 救援入口。",
            "candidate_count": 0,
        }

    road = (mask == ROAD_CLEAR_CLASS_ID).astype(np.uint8)
    if road.sum() == 0:
        return {
            "entry_found": False,
            "entry_point": None,
            "entry_reason": "未检测到 Road-Clear 可通行道路区域，不生成伪入口。",
            "candidate_count": 0,
        }

    height, width = road.shape[:2]
    margin = max(3, int(min(width, height) * 0.08))
    min_area = max(12, int(width * height * 0.001))
    if target_point is None:
        goal_x, goal_y = width / 2.0, height / 2.0
    else:
        goal_x, goal_y = float(target_point[0]), float(target_point[1])

    candidates = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road, connectivity=8)
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        component = labels == label
        ys, xs = np.where(component)
        boundary_mask = (xs <= margin) | (ys <= margin) | (xs >= width - 1 - margin) | (ys >= height - 1 - margin)
        if not np.any(boundary_mask):
            continue
        bx = xs[boundary_mask]
        by = ys[boundary_mask]
        distances = (bx - goal_x) ** 2 + (by - goal_y) ** 2
        idx = int(np.argmin(distances))
        candidates.append((float(distances[idx]), int(bx[idx]), int(by[idx]), int(area)))

    if not candidates:
        return {
            "entry_found": False,
            "entry_point": None,
            "entry_reason": "Road-Clear 存在，但没有找到靠近图像边界的可靠道路入口。",
            "candidate_count": 0,
        }

    _, x, y, area = sorted(candidates, key=lambda item: (item[0], -item[3]))[0]
    return {
        "entry_found": True,
        "entry_point": [x, y],
        "entry_reason": f"入口来自靠近图像边界的 Road-Clear 连通区域，并优先选择距离目标最近的道路边界点；连通区域面积约 {area} 像素。",
        "candidate_count": len(candidates),
    }


def build_path_planning_gate_result(scene_mode_result, entry_result, use_manual_start=False, manual_start_point=None, force_path_planning=False):
    """Decide whether route planning is allowed and which start point to use."""
    scene_mode_result = scene_mode_result or {}
    entry_result = entry_result or {}
    scene_mode = scene_mode_result.get("scene_mode", "unknown")

    if force_path_planning:
        start_point = manual_start_point or entry_result.get("entry_point")
        return {
            "path_enabled": start_point is not None,
            "start_point": list(start_point) if start_point is not None else None,
            "start_source": "manual_force" if use_manual_start or manual_start_point else "force",
            "gate_reason": "已启用强制路径规划调试模式，可能在局部近景图上产生不可靠路径。",
            "display_message": "路径规划：强制启用。该路径仅用于调试，可靠性有限，不代表真实 GPS 导航。",
            "force_path_planning": True,
        }

    if scene_mode != "wide_area_assessment":
        return {
            "path_enabled": False,
            "start_point": None,
            "start_source": "disabled",
            "gate_reason": scene_mode_result.get("reason", "当前场景不适合路径规划。"),
            "display_message": f"路径规划未启用：{scene_mode_result.get('reason', '当前场景不适合路径规划。')}",
            "force_path_planning": False,
        }

    if not entry_result.get("entry_found"):
        return {
            "path_enabled": False,
            "start_point": None,
            "start_source": "disabled",
            "gate_reason": entry_result.get("entry_reason", "未找到可靠救援入口。"),
            "display_message": f"路径规划未启用：{entry_result.get('entry_reason', '未找到可靠救援入口。')}",
            "force_path_planning": False,
        }

    start_point = manual_start_point if use_manual_start and manual_start_point is not None else entry_result.get("entry_point")
    start_source = "manual_start" if use_manual_start and manual_start_point is not None else "auto_road_clear_entry"
    return {
        "path_enabled": True,
        "start_point": list(start_point),
        "start_source": start_source,
        "gate_reason": (
            "用户选择手动起点；场景已通过广域评估门控。"
            if start_source == "manual_start"
            else "已找到 Road-Clear 道路入口，可启用图像平面路径规划。"
        ),
        "display_message": (
            f"路径规划已启用，起点 {list(start_point)}，来源："
            + ("用户手动起点。" if start_source == "manual_start" else "Road-Clear 自动救援入口。")
        ),
        "force_path_planning": False,
    }


def build_path_planning_reliability_status(
    scene_mode_result,
    entry_result,
    gate_result,
    segmentation_source_metadata=None,
    force_path_planning=False,
):
    """Build a conservative reliability statement for image-plane path planning."""
    scene_mode_result = scene_mode_result or {}
    entry_result = entry_result or {}
    gate_result = gate_result or {}
    metadata = segmentation_source_metadata or {}

    path_enabled = bool(gate_result.get("path_enabled"))
    source_type = metadata.get("source_type", "none")
    source_label = metadata.get("source_label") or source_type or "unknown"
    prediction_success = bool(metadata.get("prediction_success"))
    is_model_prediction = bool(metadata.get("is_model_prediction"))
    force_enabled = bool(force_path_planning or gate_result.get("force_path_planning"))
    mask_dependency = source_type not in {"none", None, ""}

    if not path_enabled:
        reliability_level = "not_applicable"
        human_review_required = True
        reliability_note = (
            "当前路径规划未启用，因此没有可评价的路径可靠性。系统保守关闭路径，避免在局部近景图、无道路信息或无可靠入口时生成误导性路线。"
        )
    elif force_enabled:
        reliability_level = "low"
        human_review_required = True
        reliability_note = (
            "当前路径由强制调试模式生成，可能绕过场景门控，不适合作为真实救援路径依据，必须人工复核。"
        )
    elif source_type == "demo_fallback":
        reliability_level = "low"
        human_review_required = True
        reliability_note = (
            "当前路径依赖演示/兜底 mask，不代表真实模型输出或可靠人工标注，路径仅适合流程演示。"
        )
    elif source_type == "uploaded_mask":
        reliability_level = "medium"
        human_review_required = True
        reliability_note = (
            "当前路径基于用户上传 mask 和 Road-Clear 入口生成，可用于决策层验证；由于 mask 不是自动模型预测，仍需人工确认标注质量。"
        )
    elif source_type == "auto_model" and is_model_prediction and prediction_success:
        reliability_level = "medium"
        human_review_required = True
        reliability_note = (
            "当前路径基于本地语义分割模型预测结果生成。由于系统尚未接入明确的模型评估指标和真实地图/GPS，可靠性保守评为 medium，仍需人工复核。"
        )
    else:
        reliability_level = "low"
        human_review_required = True
        reliability_note = (
            "当前路径缺少可靠分割来源或模型预测未成功，路径结果可信度较低，不建议作为真实救援路线依据。"
        )

    if not entry_result.get("entry_found") and path_enabled:
        reliability_level = "low"
        human_review_required = True
        reliability_note += " 另外，当前未确认可靠 Road-Clear 入口，起点来源需要重点复核。"

    mask_risk_note = (
        "入口生成和路径门控依赖 segmentation mask；如果 mask 不准确，Road-Clear 入口、道路阻断判断和路径结果都会受影响。"
        if mask_dependency
        else "当前没有可靠 segmentation mask，系统不应默认生成环境感知路径。"
    )

    return {
        "reliability_level": reliability_level,
        "is_real_gps_navigation": False,
        "path_type": "image_plane_reference_path",
        "scene_mode_method": "rule_based",
        "mask_dependency": mask_dependency,
        "mask_source": source_label,
        "mask_risk_note": mask_risk_note,
        "reliability_note": reliability_note,
        "human_review_required": human_review_required,
    }


def format_path_planning_reliability_status(status):
    """Format path reliability metadata for the Chinese Gradio UI."""
    status = status or {}
    return "\n".join(
        [
            "路径类型：图像平面参考路径，不是真实 GPS 导航。",
            "Scene Mode 方法：基于规则判断，不是训练出的场景分类模型。",
            f"当前可靠性等级：{status.get('reliability_level', 'unknown')}",
            f"是否真实 GPS 导航：{'是' if status.get('is_real_gps_navigation') else '否'}",
            f"Mask 来源：{status.get('mask_source', 'unknown')}",
            f"Mask 依赖：{'是' if status.get('mask_dependency') else '否'}",
            f"Mask 风险说明：{status.get('mask_risk_note', '')}",
            f"是否需要人工复核：{'是' if status.get('human_review_required') else '否'}",
            f"可靠性说明：{status.get('reliability_note', '')}",
        ]
    )
