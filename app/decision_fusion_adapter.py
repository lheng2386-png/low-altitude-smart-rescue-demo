"""Lightweight decision fusion adapter inspired by external reference projects."""

import math
from pathlib import Path

import cv2
import numpy as np


class DecisionFusionAdapterError(Exception):
    pass


def _normalize_shape(image_shape):
    if isinstance(image_shape, np.ndarray):
        height, width = image_shape.shape[:2]
    else:
        height, width = int(image_shape[0]), int(image_shape[1])
    return int(height), int(width)


def _ensure_priority_map_shape(priority_map, image_shape):
    height, width = _normalize_shape(image_shape)
    if priority_map.shape[:2] != (height, width):
        priority_map = cv2.resize(priority_map.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return priority_map.astype(np.float32)


def compute_image_plane_search_priority_map(
    image_shape,
    targets=None,
    segmentation_summary=None,
    segmentation_mask=None,
    detection_bridge_result=None,
):
    """Create a lightweight image-plane search priority map inspired by SAREnv."""
    height, width = _normalize_shape(image_shape)
    priority_map = np.zeros((height, width), dtype=np.float32)

    y_grid, x_grid = np.mgrid[0:height, 0:width]
    targets = targets or []
    for target in targets:
        center = target.get("center")
        if not center:
            bbox = target.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        cx, cy = float(center[0]), float(center[1])
        sigma = max(10.0, min(height, width) * 0.08)
        dist2 = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        bump = np.exp(-dist2 / (2.0 * sigma**2))
        class_name = str(target.get("class_name", "")).lower()
        if class_name in {"civilian", "rescuer", "human_candidate"}:
            weight = 1.0
        elif class_name in {"dog", "cat"}:
            weight = 0.55
        else:
            weight = 0.35
        weight *= float(target.get("confidence", 0.5) or 0.5)
        priority_map += weight * bump

    if segmentation_mask is not None:
        mask = np.asarray(segmentation_mask)
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        risky_classes = {1, 4, 5, 8, 10}
        supportive_classes = {7}
        risky_mask = np.isin(mask, list(risky_classes)).astype(np.float32)
        supportive_mask = np.isin(mask, list(supportive_classes)).astype(np.float32)
        priority_map += 0.8 * cv2.GaussianBlur(risky_mask, (0, 0), sigmaX=max(3.0, min(height, width) * 0.02))
        priority_map += 0.2 * supportive_mask

    if segmentation_summary:
        major_damage = float(segmentation_summary.get("major_damage", 0.0) or 0.0)
        destroyed = float(segmentation_summary.get("destroyed_building", 0.0) or 0.0)
        water = float(segmentation_summary.get("water", 0.0) or 0.0)
        road_blocked = float(segmentation_summary.get("road_blocked", 0.0) or 0.0)
        road_clear = float(segmentation_summary.get("road_clear", 0.0) or 0.0)
        multiplier = 1.0 + 0.8 * major_damage + 1.0 * destroyed + 0.6 * water + 0.7 * road_blocked - 0.2 * road_clear
        priority_map *= max(0.2, float(multiplier))

    if detection_bridge_result and detection_bridge_result.get("consensus"):
        consensus = detection_bridge_result["consensus"]
        matched_pairs = consensus.get("matched_pairs", [])
        for pair in matched_pairs:
            if pair.get("consensus_type") == "human_target_overlap":
                priority_map += 0.15 * np.exp(-((x_grid - width / 2.0) ** 2 + (y_grid - height / 2.0) ** 2) / max(1.0, (0.3 * min(height, width)) ** 2))

    priority_map = np.clip(priority_map, 0.0, None)
    max_priority = float(np.max(priority_map)) if priority_map.size else 0.0
    mean_priority = float(np.mean(priority_map)) if priority_map.size else 0.0
    high_priority_area_ratio = float(np.mean(priority_map > (mean_priority + 0.5 * (max_priority - mean_priority)))) if priority_map.size else 0.0
    return {
        "success": True,
        "priority_map": priority_map.astype(np.float32),
        "priority_map_shape": [height, width],
        "priority_statistics": {
            "max_priority": round(max_priority, 4),
            "mean_priority": round(mean_priority, 4),
            "high_priority_area_ratio": round(high_priority_area_ratio, 4),
        },
        "source": "image_plane_lightweight_sarenv_adaptation",
        "truthfulness_note": "This is a lightweight image-plane search priority map inspired by SAR probability heatmaps. It is not a full SAREnv geospatial probability model.",
    }


def compute_segmentation_damage_impact_score(segmentation_summary=None, segmentation_mask=None):
    """Compute a lightweight damage impact score inspired by SKAI and InaSAFE."""
    if not segmentation_summary and segmentation_mask is None:
        return {
            "success": False,
            "impact_score": 0.0,
            "impact_level": "low",
            "components": {
                "major_damage_ratio": 0.0,
                "destroyed_building_ratio": 0.0,
                "water_ratio": 0.0,
                "road_blocked_ratio": 0.0,
                "road_clear_ratio": 0.0,
            },
            "dominant_factors": [],
            "source": "lightweight_skai_inasafe_adaptation",
            "truthfulness_note": "This score is derived from the available segmentation mask/summary. It is not a SKAI model output and not a full GIS impact assessment.",
        }

    summary = segmentation_summary or {}
    if not summary and segmentation_mask is not None:
        mask = np.asarray(segmentation_mask)
        total = float(mask.size) if mask.size else 1.0
        summary = {
            "major_damage": float(np.mean(mask == 4)) if mask.size else 0.0,
            "destroyed_building": float(np.mean(mask == 5)) if mask.size else 0.0,
            "water": float(np.mean(mask == 1)) if mask.size else 0.0,
            "road_blocked": float(np.mean(mask == 8)) if mask.size else 0.0,
            "road_clear": float(np.mean(mask == 7)) if mask.size else 0.0,
        }

    major_damage = float(summary.get("major_damage", 0.0) or 0.0)
    destroyed = float(summary.get("destroyed_building", 0.0) or 0.0)
    water = float(summary.get("water", 0.0) or 0.0)
    road_blocked = float(summary.get("road_blocked", 0.0) or 0.0)
    road_clear = float(summary.get("road_clear", 0.0) or 0.0)
    score = 100.0 * (
        0.35 * major_damage
        + 0.45 * destroyed
        + 0.12 * water
        + 0.12 * road_blocked
        - 0.08 * road_clear
    )
    score = float(np.clip(score, 0.0, 100.0))
    if score < 25:
        level = "low"
    elif score < 50:
        level = "medium"
    elif score < 75:
        level = "high"
    else:
        level = "critical"
    dominant = []
    for name, value in [
        ("destroyed_building", destroyed),
        ("major_damage", major_damage),
        ("road_blocked", road_blocked),
        ("water", water),
    ]:
        if value > 0.05:
            dominant.append(name)
    return {
        "success": True,
        "impact_score": round(score, 4),
        "impact_level": level,
        "components": {
            "major_damage_ratio": round(major_damage, 4),
            "destroyed_building_ratio": round(destroyed, 4),
            "water_ratio": round(water, 4),
            "road_blocked_ratio": round(road_blocked, 4),
            "road_clear_ratio": round(road_clear, 4),
        },
        "dominant_factors": dominant,
        "source": "lightweight_skai_inasafe_adaptation",
        "truthfulness_note": "This score is derived from the available segmentation mask/summary. It is not a SKAI model output and not a full GIS impact assessment.",
    }


def compute_coverage_planning_score(image_shape, path_result=None, segmentation_mask=None, priority_map=None):
    """Compute a lightweight coverage score inspired by Fields2Cover and PythonRobotics."""
    height, width = _normalize_shape(image_shape)
    if path_result is None or not path_result.get("path"):
        return {
            "success": False,
            "coverage_score": 0.0,
            "covered_area_ratio": 0.0,
            "high_priority_covered_ratio": 0.0,
            "unsearched_high_priority_ratio": 0.0,
            "path_length": 0,
            "source": "lightweight_coverage_planning_adaptation",
            "truthfulness_note": "This is an image-plane coverage score inspired by coverage path planning. It is not a real UAV flight route or full Fields2Cover output.",
        }

    path = path_result.get("path", [])
    path_mask = np.zeros((height, width), dtype=np.uint8)
    for point in path:
        if len(point) < 2:
            continue
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(path_mask, (x, y), 4, 1, -1)
    covered_area_ratio = float(np.mean(path_mask > 0))
    if priority_map is None:
        priority_map = np.zeros((height, width), dtype=np.float32)
    else:
        priority_map = _ensure_priority_map_shape(np.asarray(priority_map, dtype=np.float32), (height, width))
    if float(np.max(priority_map)) > 0:
        high_threshold = float(np.percentile(priority_map, 80))
        high_priority_mask = priority_map >= high_threshold
    else:
        high_priority_mask = np.zeros((height, width), dtype=bool)
    high_priority_total = float(np.mean(high_priority_mask)) if high_priority_mask.size else 0.0
    high_priority_covered = float(np.mean(high_priority_mask & (path_mask > 0))) if high_priority_mask.size else 0.0
    high_priority_covered_ratio = 0.0 if high_priority_total <= 0 else high_priority_covered / high_priority_total
    unsearched_high_priority_ratio = max(0.0, 1.0 - high_priority_covered_ratio)
    coverage_score = float(np.clip(100.0 * (0.4 * covered_area_ratio + 0.6 * high_priority_covered_ratio), 0.0, 100.0))
    return {
        "success": True,
        "coverage_score": round(coverage_score, 4),
        "covered_area_ratio": round(covered_area_ratio, 4),
        "high_priority_covered_ratio": round(high_priority_covered_ratio, 4),
        "unsearched_high_priority_ratio": round(unsearched_high_priority_ratio, 4),
        "path_length": len(path),
        "source": "lightweight_coverage_planning_adaptation",
        "truthfulness_note": "This is an image-plane coverage score inspired by coverage path planning. It is not a real UAV flight route or full Fields2Cover output.",
    }


def build_decision_fusion_summary(
    search_priority_result=None,
    damage_impact_result=None,
    coverage_result=None,
    detection_bridge_result=None,
):
    """Combine lightweight decision references into a human-readable summary."""
    scores = []
    messages = []
    recommended_actions = []
    human_review_required = True

    if search_priority_result and search_priority_result.get("success"):
        stats = search_priority_result.get("priority_statistics", {})
        score = float(stats.get("max_priority", 0.0)) * 10.0
        scores.append(score)
        messages.append(f"搜索优先级图峰值为 {stats.get('max_priority', 0.0)}。")
        if float(stats.get("high_priority_area_ratio", 0.0)) > 0.1:
            recommended_actions.append("优先复核高优先级搜索区域")

    if damage_impact_result and damage_impact_result.get("success"):
        impact_score = float(damage_impact_result.get("impact_score", 0.0))
        scores.append(impact_score)
        messages.append(f"灾损影响评分为 {impact_score}，等级为 {damage_impact_result.get('impact_level')}.")
        if damage_impact_result.get("impact_level") in {"high", "critical"}:
            recommended_actions.append("优先核查 damaged/destroyed building 附近目标")

    if coverage_result and coverage_result.get("success"):
        coverage_score = float(coverage_result.get("coverage_score", 0.0))
        scores.append(coverage_score)
        messages.append(f"覆盖评估得分为 {coverage_score}。")
        if float(coverage_result.get("unsearched_high_priority_ratio", 0.0)) > 0.2:
            recommended_actions.append("对未覆盖高优先级区域补充无人机巡检")

    if detection_bridge_result:
        if detection_bridge_result.get("consensus"):
            consensus_summary = detection_bridge_result["consensus"].get("consensus_summary", "")
            messages.append(consensus_summary)
            if detection_bridge_result["consensus"].get("matched_pairs"):
                recommended_actions.append("对 Transformer-only human_candidate 做人工复核")
        if detection_bridge_result.get("human_review_required", True):
            human_review_required = True

    if not scores:
        return {
            "success": False,
            "decision_fusion_score": 0.0,
            "decision_fusion_level": "low",
            "summary_markdown": "未获得足够的决策层输入，无法生成融合评分。",
            "recommended_actions": ["先运行检测、分割和路径模块后再生成融合结果"],
            "human_review_required": True,
            "truthfulness_note": "Decision fusion requires available lightweight inputs. It is an auxiliary decision summary, not an automatic rescue decision.",
        }

    fusion_score = float(np.clip(np.mean(scores), 0.0, 100.0))
    if fusion_score < 25:
        level = "low"
    elif fusion_score < 50:
        level = "medium"
    elif fusion_score < 75:
        level = "high"
    else:
        level = "critical"

    summary_lines = [
        "## 决策层参考融合摘要",
        f"- 综合评分：{round(fusion_score, 4)}",
        f"- 综合等级：{level}",
        "- 说明：本摘要是对搜索优先级、灾损影响、覆盖评估与检测一致性的轻量融合，不是自动救援决策。",
    ]
    summary_lines.extend([f"- {msg}" for msg in messages if msg])
    if recommended_actions:
        summary_lines.append("### 建议动作")
        for action in dict.fromkeys(recommended_actions):
            summary_lines.append(f"- {action}")
    else:
        summary_lines.append("### 建议动作")
        summary_lines.append("- 继续保持人工复核，补充更多决策输入。")
    return {
        "success": True,
        "decision_fusion_score": round(fusion_score, 4),
        "decision_fusion_level": level,
        "summary_markdown": "\n".join(summary_lines),
        "recommended_actions": list(dict.fromkeys(recommended_actions)) or ["继续保持人工复核，补充更多决策输入。"],
        "human_review_required": human_review_required,
        "truthfulness_note": "This is a lightweight image-plane decision fusion summary inspired by external references. It does not replace human rescue judgment and does not claim full GIS or automatic rescue decision capability.",
    }


def render_priority_map_overlay(base_image, priority_map, output_path=None):
    """Render an image-plane priority overlay for visualization only."""
    base = np.asarray(base_image)
    if base.ndim == 2:
        base = np.stack([base, base, base], axis=-1)
    if base.shape[-1] == 4:
        base = base[:, :, :3]
    priority = np.asarray(priority_map, dtype=np.float32)
    if priority.shape[:2] != base.shape[:2]:
        priority = cv2.resize(priority, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    if float(np.max(priority)) > 0:
        normalized = priority / float(np.max(priority))
    else:
        normalized = priority
    heat = (255 * np.clip(normalized, 0, 1)).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base.astype(np.uint8), 0.6, heat_color, 0.4, 0)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(output_path)
        return str(output_path)
    return overlay
