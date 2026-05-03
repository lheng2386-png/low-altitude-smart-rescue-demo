"""Black-background disaster segmentation visualization helpers."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = {
    0: "background",
    1: "water",
    2: "no_damage_building",
    3: "minor_damage",
    4: "major_damage",
    5: "destroyed_building",
    6: "vehicle",
    7: "road_clear",
    8: "road_blocked",
    9: "tree",
    10: "pool",
}

COLOR_MAP = {
    0: (0, 0, 0),          # black
    1: (61, 230, 250),     # cyan
    2: (178, 130, 120),    # dusty pink / pale brown-pink
    3: (255, 235, 0),      # bright yellow
    4: (255, 165, 0),      # orange
    5: (255, 0, 0),        # red
    6: (255, 0, 255),      # magenta
    7: (140, 140, 140),    # gray
    8: (154, 160, 28),     # olive / yellow-green
    9: (0, 255, 0),        # bright green
    10: (0, 102, 255),     # deep blue
}


def _load_array(data):
    if data is None:
        return None
    if isinstance(data, (str, Path)):
        array = np.array(Image.open(data))
    elif isinstance(data, Image.Image):
        array = np.array(data)
    else:
        array = np.asarray(data)
    return array


def _to_class_id_mask(mask):
    array = _load_array(mask)
    if array is None:
        return None
    if array.ndim == 2:
        return array.astype(np.uint8)
    if array.ndim == 3:
        if array.shape[2] == 1:
            return array[:, :, 0].astype(np.uint8)
        rgb = array[:, :, :3].astype(np.int16)
        palette = np.array(list(COLOR_MAP.values()), dtype=np.int16)
        diff = rgb[:, :, None, :] - palette[None, None, :, :]
        dist = np.sum(diff * diff, axis=-1)
        return np.argmin(dist, axis=-1).astype(np.uint8)
    return None


def _as_rgb(image):
    array = _load_array(image)
    if array is None:
        return None
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return array.astype(np.uint8)


def render_segmentation_mask(mask):
    """Render a 2D class-id mask as a black-background RGB image."""
    class_mask = _to_class_id_mask(mask)
    if class_mask is None:
        return None

    color = np.zeros((class_mask.shape[0], class_mask.shape[1], 3), dtype=np.uint8)
    for class_id, rgb in COLOR_MAP.items():
        color[class_mask == class_id] = rgb
    return color


def create_segmentation_panel(image, color_mask):
    """Create a simple vertical panel: original image on top, color mask below."""
    image_rgb = _as_rgb(image)
    mask_rgb = _as_rgb(color_mask)
    if image_rgb is None or mask_rgb is None:
        return None

    if image_rgb.shape[:2] != mask_rgb.shape[:2]:
        mask_rgb = cv2.resize(mask_rgb, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    divider = np.zeros((8, image_rgb.shape[1], 3), dtype=np.uint8)
    divider[:] = (24, 24, 24)
    return np.vstack([image_rgb, divider, mask_rgb])


def compute_damage_statistics(mask):
    """Compute pixel counts and area ratios for all classes."""
    class_mask = _to_class_id_mask(mask)
    if class_mask is None:
        return {
            "total_pixels": 0,
            "class_pixel_counts": {},
            "class_area_ratios": {},
            "no_damage_area": 0,
            "minor_damage_area": 0,
            "major_damage_area": 0,
            "destroyed_building_area": 0,
            "road_clear_area": 0,
            "road_blocked_area": 0,
            "water_area": 0,
            "tree_area": 0,
            "vehicle_area": 0,
        }

    total_pixels = int(class_mask.size)
    class_pixel_counts = {}
    class_area_ratios = {}
    for class_id, class_name in CLASS_NAMES.items():
        pixels = int(np.count_nonzero(class_mask == class_id))
        class_pixel_counts[class_name] = pixels
        class_area_ratios[class_name] = pixels / max(total_pixels, 1)

    stats = {
        "total_pixels": total_pixels,
        "class_pixel_counts": class_pixel_counts,
        "class_area_ratios": class_area_ratios,
        "no_damage_area": class_pixel_counts["no_damage_building"],
        "minor_damage_area": class_pixel_counts["minor_damage"],
        "major_damage_area": class_pixel_counts["major_damage"],
        "destroyed_building_area": class_pixel_counts["destroyed_building"],
        "road_clear_area": class_pixel_counts["road_clear"],
        "road_blocked_area": class_pixel_counts["road_blocked"],
        "water_area": class_pixel_counts["water"],
        "tree_area": class_pixel_counts["tree"],
        "vehicle_area": class_pixel_counts["vehicle"],
    }
    return stats


def classify_damage_level(stats):
    """Classify overall damage severity."""
    if not stats:
        return "Superficial Damage"

    no_damage = float(stats.get("no_damage_area", 0))
    minor = float(stats.get("minor_damage_area", 0))
    major = float(stats.get("major_damage_area", 0))
    destroyed = float(stats.get("destroyed_building_area", 0))
    denom = max(1.0, no_damage + minor + major + destroyed)
    damage_score = (minor + 2.0 * major + 3.0 * destroyed) / denom

    if damage_score < 0.5:
        return "Superficial Damage"
    if damage_score < 1.5:
        return "Medium Damage"
    return "Major Damage"


def create_legend_image(width=520, row_height=34):
    """Create a simple legend image for the segmentation colors."""
    entries = [(CLASS_NAMES[class_id], COLOR_MAP[class_id]) for class_id in sorted(CLASS_NAMES)]
    height = row_height * len(entries) + 16
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None

    for index, (name, color) in enumerate(entries):
        top = 8 + index * row_height
        draw.rectangle([16, top, 16 + 24, top + 24], fill=tuple(int(v) for v in color))
        draw.rectangle([16, top, 16 + 24, top + 24], outline=(0, 0, 0))
        draw.text((56, top + 4), name, fill=(0, 0, 0), font=font)

    return np.array(image)
