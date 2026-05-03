"""Smoke test for the black-background segmentation visualization helpers."""

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from damage_segmentation_visualizer import (  # noqa: E402
    CLASS_NAMES,
    COLOR_MAP,
    classify_damage_level,
    compute_damage_statistics,
    create_legend_image,
    create_segmentation_panel,
    render_segmentation_mask,
)


def make_synthetic_mask():
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[0:16, 0:10] = 1
    mask[0:16, 10:20] = 2
    mask[0:16, 20:30] = 3
    mask[0:16, 30:40] = 4
    mask[0:16, 40:50] = 5
    mask[0:16, 50:60] = 6
    mask[0:16, 60:70] = 7
    mask[0:16, 70:80] = 8
    mask[0:16, 80:90] = 9
    mask[0:16, 90:100] = 10
    mask[40:70, 10:35] = 2
    mask[40:70, 35:55] = 4
    mask[40:70, 55:80] = 5
    mask[40:70, 80:95] = 7
    mask[60:78, 0:18] = 8
    return mask


def main():
    mask = make_synthetic_mask()
    image = np.full((80, 100, 3), 180, dtype=np.uint8)

    color_mask = render_segmentation_mask(mask)
    assert color_mask is not None
    assert color_mask.shape == (80, 100, 3)
    assert tuple(color_mask[2, 2]) == COLOR_MAP[1]
    assert tuple(color_mask[2, 65]) == COLOR_MAP[7]
    assert tuple(color_mask[50, 40]) == COLOR_MAP[4]

    panel = create_segmentation_panel(image, color_mask)
    assert panel is not None
    assert panel.shape[1] == 100
    assert panel.shape[0] > 80

    stats = compute_damage_statistics(mask)
    assert stats["total_pixels"] == 8000
    assert stats["no_damage_area"] > 0
    assert stats["major_damage_area"] > 0
    assert stats["road_clear_area"] > 0
    assert stats["road_blocked_area"] > 0
    assert "water" in stats["class_pixel_counts"]
    assert "road_clear" in stats["class_area_ratios"]

    level = classify_damage_level(stats)
    assert level in {"Superficial Damage", "Medium Damage", "Major Damage"}

    legend = create_legend_image()
    assert legend is not None
    assert legend.ndim == 3

    assert len(CLASS_NAMES) == 11
    print("AeroRescue-AI damage segmentation visualizer smoke test passed.")


if __name__ == "__main__":
    main()
