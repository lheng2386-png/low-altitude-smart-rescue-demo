"""Post-disaster damage assessment and rescue-entry generation."""

import cv2
import numpy as np


CLASS_NAMES = {
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

BUILDING_CLASS_IDS = {2, 3, 4, 5}


def _empty_assessment(image_width=0, image_height=0, reason="未接入有效语义分割掩码。"):
    return {
        "class_stats": {},
        "building_damage": {
            "no_damage_area": 0,
            "medium_damage_area": 0,
            "major_damage_area": 0,
            "total_destruction_area": 0,
        },
        "road_stats": {
            "road_clear_area": 0,
            "road_blocked_area": 0,
            "road_clear_ratio": 0.0,
            "road_blocked_ratio": 0.0,
        },
        "water_area": 0,
        "tree_area": 0,
        "vehicle_area": 0,
        "overall_damage_level": "Unknown",
        "scene_mode": "Local Reconnaissance",
        "scene_mode_reason": reason,
        "entry": {
            "entry_found": False,
            "entry_point_x": None,
            "entry_point_y": None,
            "entry_reason": "未找到可用道路入口。",
        },
        "path_planning_enabled": False,
        "path_planning_reason": "当前场景缺乏完整道路与可达性信息，因此不生成路径建议。",
        "image_width": int(image_width or 0),
        "image_height": int(image_height or 0),
    }


def _align_mask(mask, image_width, image_height):
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[:2] != (int(image_height), int(image_width)):
        mask = cv2.resize(mask.astype(np.uint8), (int(image_width), int(image_height)), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.uint8)


def summarize_damage(mask, image_width=None, image_height=None):
    """Summarize class areas, building damage, road state, and global damage level."""
    if mask is None:
        return _empty_assessment(image_width or 0, image_height or 0)

    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    height, width = mask.shape[:2]
    total_pixels = max(1, int(mask.size))
    class_stats = {}
    for class_id, class_name in CLASS_NAMES.items():
        pixels = int(np.count_nonzero(mask == class_id))
        class_stats[class_name] = {
            "class_id": class_id,
            "pixels": pixels,
            "ratio": round(pixels / total_pixels, 4),
        }

    building_damage = {
        "no_damage_area": class_stats["building_no_damage"]["pixels"],
        "medium_damage_area": class_stats["building_medium_damage"]["pixels"],
        "major_damage_area": class_stats["building_major_damage"]["pixels"],
        "total_destruction_area": class_stats["building_total_destruction"]["pixels"],
    }
    road_stats = {
        "road_clear_area": class_stats["road_clear"]["pixels"],
        "road_blocked_area": class_stats["road_blocked"]["pixels"],
        "road_clear_ratio": class_stats["road_clear"]["ratio"],
        "road_blocked_ratio": class_stats["road_blocked"]["ratio"],
    }

    weighted_damage = (
        building_damage["medium_damage_area"]
        + building_damage["major_damage_area"] * 2
        + building_damage["total_destruction_area"] * 3
    )
    building_area = max(1, sum(building_damage.values()))
    damage_index = weighted_damage / building_area
    if building_area <= 1:
        overall = "Unknown"
    elif damage_index < 0.65:
        overall = "Superficial Damage"
    elif damage_index < 1.65:
        overall = "Medium Damage"
    else:
        overall = "Major Damage"

    return {
        "class_stats": class_stats,
        "building_damage": building_damage,
        "road_stats": road_stats,
        "water_area": class_stats["water"]["pixels"] + class_stats["pool"]["pixels"],
        "tree_area": class_stats["tree"]["pixels"],
        "vehicle_area": class_stats["vehicle"]["pixels"],
        "overall_damage_level": overall,
        "image_width": int(width),
        "image_height": int(height),
    }


def _target_area_stats(targets, image_area):
    targets = targets or []
    if not targets:
        return 0.0, 0.0
    ratios = []
    for target in targets:
        area = float(target.get("area", 0.0) or 0.0)
        if area <= 0 and target.get("bbox"):
            x1, y1, x2, y2 = [float(v) for v in target["bbox"][:4]]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ratios.append(area / max(1.0, image_area))
    return float(np.mean(ratios)), float(max(ratios))


def has_boundary_connected_clear_road(mask, boundary_margin_ratio=0.08):
    """Return whether a Road-Clear component touches or approaches the image boundary."""
    if mask is None:
        return False
    road = (mask == 7).astype(np.uint8)
    if road.sum() == 0:
        return False
    height, width = road.shape[:2]
    margin = max(3, int(min(width, height) * boundary_margin_ratio))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road, connectivity=8)
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < max(12, int(width * height * 0.001)):
            continue
        if x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin:
            return True
    return False


def determine_scene_mode(mask, targets=None, image_width=None, image_height=None):
    """Classify the image as local reconnaissance or wide-area assessment."""
    if mask is None:
        return "Local Reconnaissance", "未接入语义分割掩码，无法确认完整道路与建筑分布。"

    mask = _align_mask(mask, image_width or mask.shape[1], image_height or mask.shape[0])
    height, width = mask.shape[:2]
    image_area = max(1, width * height)
    road_clear_ratio = float(np.count_nonzero(mask == 7)) / image_area
    road_blocked_ratio = float(np.count_nonzero(mask == 8)) / image_area
    building_ratio = float(np.count_nonzero(np.isin(mask, list(BUILDING_CLASS_IDS)))) / image_area
    avg_target_ratio, max_target_ratio = _target_area_stats(targets, image_area)
    boundary_road = has_boundary_connected_clear_road(mask)

    if max_target_ratio > 0.12 or avg_target_ratio > 0.06:
        return "Local Reconnaissance", "目标框占画面比例较大，图像更像近距离局部侦察，不适合生成路径。"
    if road_clear_ratio < 0.015:
        return "Local Reconnaissance", "语义分割中可通行道路比例过低，缺乏可靠路径规划基础。"
    if not boundary_road:
        return "Local Reconnaissance", "未发现与图像边界相连或接近边界的可通行道路入口。"
    if (road_clear_ratio + road_blocked_ratio) >= 0.02 and building_ratio >= 0.01:
        return "Wide-area Assessment", "画面具备道路与建筑分布，并存在边界连通道路，适合进行广域路径评估。"
    if road_clear_ratio >= 0.06 and boundary_road:
        return "Wide-area Assessment", "画面中可通行道路较完整，适合进行广域可达性评估。"
    return "Local Reconnaissance", "当前图像缺乏足够广域道路、建筑或环境上下文。"


def _target_point(target):
    center = target.get("center") if target else None
    if center and len(center) >= 2:
        return float(center[0]), float(center[1])
    bbox = target.get("bbox", [0, 0, 0, 0]) if target else [0, 0, 0, 0]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return (x1 + x2) / 2, (y1 + y2) / 2


def generate_rescue_entry(mask, target=None):
    """Find a rescue entry from boundary-connected Road-Clear components."""
    if mask is None:
        return {
            "entry_found": False,
            "entry_point_x": None,
            "entry_point_y": None,
            "entry_reason": "未接入语义分割掩码，无法生成道路入口。",
        }
    road = (mask == 7).astype(np.uint8)
    if road.sum() == 0:
        return {
            "entry_found": False,
            "entry_point_x": None,
            "entry_point_y": None,
            "entry_reason": "未检测到 Road-Clear 可通行道路区域，不生成伪入口。",
        }

    height, width = road.shape[:2]
    margin = max(3, int(min(width, height) * 0.08))
    goal_x, goal_y = _target_point(target)
    candidates = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road, connectivity=8)
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < max(12, int(width * height * 0.001)):
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
            "entry_point_x": None,
            "entry_point_y": None,
            "entry_reason": "未找到与图像边界相连或接近边界的 Road-Clear 连通区域。",
        }

    _, x, y, area = sorted(candidates, key=lambda item: (item[0], -item[3]))[0]
    return {
        "entry_found": True,
        "entry_point_x": x,
        "entry_point_y": y,
        "entry_reason": f"入口来自与图像边界相连的 Road-Clear 连通区域，优先选择距离目标最近的边界道路点；连通区域面积约 {area} 像素。",
    }


def assess_damage_and_entry(mask, targets=None, image_width=None, image_height=None, top_target=None):
    """Run damage assessment, scene mode classification, and rescue-entry generation."""
    if mask is None:
        return _empty_assessment(image_width or 0, image_height or 0)

    mask = _align_mask(mask, image_width or mask.shape[1], image_height or mask.shape[0])
    assessment = summarize_damage(mask, image_width=image_width, image_height=image_height)
    scene_mode, scene_reason = determine_scene_mode(mask, targets=targets, image_width=image_width, image_height=image_height)
    assessment["scene_mode"] = scene_mode
    assessment["scene_mode_reason"] = scene_reason

    if not targets:
        assessment["entry"] = {
            "entry_found": False,
            "entry_point_x": None,
            "entry_point_y": None,
            "entry_reason": "当前未检测到明确救援目标，因此不生成救援入口和路径建议。",
        }
        assessment["path_planning_enabled"] = False
        assessment["path_planning_reason"] = assessment["entry"]["entry_reason"]
        return assessment

    if scene_mode != "Wide-area Assessment":
        assessment["entry"] = {
            "entry_found": False,
            "entry_point_x": None,
            "entry_point_y": None,
            "entry_reason": "当前场景属于局部侦察图像，缺乏完整道路与可达性信息，因此不生成路径建议。",
        }
        assessment["path_planning_enabled"] = False
        assessment["path_planning_reason"] = assessment["entry"]["entry_reason"]
        return assessment

    entry = generate_rescue_entry(mask, target=top_target)
    assessment["entry"] = entry
    assessment["path_planning_enabled"] = bool(entry.get("entry_found"))
    assessment["path_planning_reason"] = (
        "已找到 Road-Clear 道路入口，可启用图像平面路径规划。"
        if entry.get("entry_found")
        else "当前场景具备广域特征，但未找到可靠的道路接入入口，因此不生成路径建议。"
    )
    return assessment


def format_damage_summary(assessment):
    """Format a compact Chinese damage summary for the UI."""
    if not assessment:
        return "暂无灾损评估结果。"
    bd = assessment.get("building_damage", {})
    road = assessment.get("road_stats", {})
    return "\n".join(
        [
            f"整体灾损等级：{assessment.get('overall_damage_level', 'Unknown')}",
            f"无损建筑面积：{bd.get('no_damage_area', 0)} 像素",
            f"中度损毁建筑面积：{bd.get('medium_damage_area', 0)} 像素",
            f"严重损毁建筑面积：{bd.get('major_damage_area', 0)} 像素",
            f"完全毁坏建筑面积：{bd.get('total_destruction_area', 0)} 像素",
            f"可通行道路比例：{road.get('road_clear_ratio', 0.0) * 100:.2f}%",
            f"阻断道路比例：{road.get('road_blocked_ratio', 0.0) * 100:.2f}%",
            f"水域/积水面积：{assessment.get('water_area', 0)} 像素",
            f"树木区域面积：{assessment.get('tree_area', 0)} 像素",
            f"车辆区域面积：{assessment.get('vehicle_area', 0)} 像素",
        ]
    )


def format_scene_mode(assessment):
    if not assessment:
        return "场景模式：Unknown"
    mode = assessment.get("scene_mode", "Unknown")
    mode_display = {
        "Local Reconnaissance": "Local Reconnaissance / 局部侦察模式",
        "Wide-area Assessment": "Wide-area Assessment / 广域评估模式",
    }.get(mode, mode)
    return f"场景模式：{mode_display}\n说明：{assessment.get('scene_mode_reason', '')}"


def format_entry_suggestion(assessment):
    if not assessment:
        return "暂无救援入口建议。"
    entry = assessment.get("entry", {})
    if entry.get("entry_found"):
        return (
            f"已找到救援入口：({entry.get('entry_point_x')}, {entry.get('entry_point_y')})\n"
            f"说明：{entry.get('entry_reason')}"
        )
    return f"未启用救援入口。\n说明：{entry.get('entry_reason') or assessment.get('path_planning_reason', '')}"
