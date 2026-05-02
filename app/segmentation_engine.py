from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

from environment_risk import (
    CLASS_DISPLAY_NAMES,
    get_environment_risk_score,
    describe_environment_classes,
)


RESCUENET_CLASSES = OrderedDict(
    [
        (0, "background"),
        (1, "water"),
        (2, "no_damage_building"),
        (3, "minor_damage"),
        (4, "major_damage"),
        (5, "destroyed_building"),
        (6, "vehicle"),
        (7, "road_clear"),
        (8, "road_blocked"),
        (9, "tree"),
        (10, "pool"),
    ]
)

RESCUENET_COLORS = OrderedDict(
    [
        (0, (0, 0, 0)),
        (1, (61, 230, 250)),
        (2, (180, 120, 120)),
        (3, (235, 255, 7)),
        (4, (255, 184, 6)),
        (5, (255, 0, 0)),
        (6, (255, 0, 245)),
        (7, (140, 140, 140)),
        (8, (160, 150, 20)),
        (9, (4, 250, 7)),
        (10, (255, 235, 0)),
    ]
)

SUMMARY_CLASS_ORDER = [
    "water",
    "road_blocked",
    "major_damage",
    "destroyed_building",
    "minor_damage",
    "tree",
    "vehicle",
    "road_clear",
    "no_damage_building",
    "pool",
    "background",
]


def _as_rgb_array(image):
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        return np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return array.astype(np.uint8)


def _rgb_to_class_id_mask(rgb_mask):
    rgb = rgb_mask.astype(np.int32)
    palette = np.array(list(RESCUENET_COLORS.values()), dtype=np.int32)
    class_ids = np.array(list(RESCUENET_COLORS.keys()), dtype=np.uint8)

    distances = np.sum((rgb[:, :, None, :] - palette[None, None, :, :]) ** 2, axis=-1)
    nearest = np.argmin(distances, axis=-1)
    return class_ids[nearest].astype(np.uint8)


def load_segmentation_mask(mask_path):
    image = Image.open(mask_path)

    if image.mode in {"L", "I", "I;16", "P"}:
        mask = np.array(image)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask.astype(np.uint8)

    rgb_mask = np.array(image.convert("RGB"))
    return _rgb_to_class_id_mask(rgb_mask)


def resize_segmentation_mask(mask, width, height):
    if mask is None:
        return None
    if mask.shape[:2] == (height, width):
        return mask
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def create_segmentation_overlay(image, mask, alpha=0.45):
    if image is None or mask is None:
        return None

    image_rgb = _as_rgb_array(image)
    height, width = image_rgb.shape[:2]
    aligned_mask = resize_segmentation_mask(mask, width, height)

    color_mask = np.zeros_like(image_rgb, dtype=np.uint8)
    for class_id, color in RESCUENET_COLORS.items():
        color_mask[aligned_mask == class_id] = color

    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, color_mask, alpha, 0)
    return overlay


def summarize_segmentation(mask):
    if mask is None:
        return {}

    total_pixels = max(int(mask.size), 1)
    summary = {}

    for class_id, class_name in RESCUENET_CLASSES.items():
        ratio = float(np.count_nonzero(mask == class_id)) / total_pixels
        if ratio > 0:
            summary[class_name] = round(ratio, 4)

    return {
        class_name: summary[class_name]
        for class_name in SUMMARY_CLASS_ORDER
        if class_name in summary
    }


def _dominant_class(mask_region):
    if mask_region.size == 0:
        return "background"
    values, counts = np.unique(mask_region, return_counts=True)
    dominant_id = int(values[int(np.argmax(counts))])
    return RESCUENET_CLASSES.get(dominant_id, "background")


def _target_neighborhood(target, mask, scale=0.35):
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in target.get("bbox", [0, 0, 0, 0])]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad_x = max(12.0, box_width * scale)
    pad_y = max(12.0, box_height * scale)

    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))

    return mask[top:bottom, left:right]


def get_environment_context_for_target(target, mask):
    if mask is None:
        return {
            "near_water": False,
            "near_blocked_road": False,
            "near_destroyed_building": False,
            "dominant_area_class": "unknown",
            "environment_risk_score": 0.0,
            "environment_reason": "当前未接入灾区语义分割结果。",
        }

    region = _target_neighborhood(target, mask)
    dominant_class = _dominant_class(region)
    present_ids = set(int(value) for value in np.unique(region))
    present_classes = {RESCUENET_CLASSES.get(class_id, "background") for class_id in present_ids}

    near_water = bool({"water", "pool"} & present_classes)
    near_blocked_road = "road_blocked" in present_classes
    near_destroyed_building = bool({"major_damage", "destroyed_building"} & present_classes)

    risk_classes = [
        class_name
        for class_name in present_classes
        if class_name not in {"background", "road_clear", "no_damage_building"}
    ]
    dominant_score = get_environment_risk_score(dominant_class)
    nearby_score = max((get_environment_risk_score(class_name) for class_name in risk_classes), default=0.0)
    environment_risk_score = round(max(dominant_score, nearby_score), 2)

    if risk_classes:
        environment_reason = (
            f"目标附近存在{describe_environment_classes(sorted(risk_classes))}，"
            f"主导环境为{CLASS_DISPLAY_NAMES.get(dominant_class, dominant_class)}。"
        )
    else:
        environment_reason = (
            f"目标附近主导环境为{CLASS_DISPLAY_NAMES.get(dominant_class, dominant_class)}，"
            "未发现明显高风险环境因素。"
        )

    return {
        "near_water": near_water,
        "near_blocked_road": near_blocked_road,
        "near_destroyed_building": near_destroyed_building,
        "dominant_area_class": dominant_class,
        "environment_risk_score": environment_risk_score,
        "environment_reason": environment_reason,
    }
