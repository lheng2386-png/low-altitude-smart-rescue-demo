"""Smoke test for auto segmentation inference and visualization."""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from damage_segmentation_visualizer import (  # noqa: E402
    classify_damage_level,
    compute_damage_statistics,
    create_segmentation_panel,
    render_segmentation_mask,
)
from segmentation_model_service import predict_segmentation, resolve_allowed_checkpoint  # noqa: E402


def main():
    image = Image.fromarray(np.full((64, 64, 3), 160, dtype=np.uint8))

    pred, status, metadata = predict_segmentation(image, checkpoint_path=None, img_size=64)
    assert pred is None
    assert "checkpoint_path" in metadata
    assert status
    outside_checkpoint, outside_error = resolve_allowed_checkpoint(Path(tempfile.gettempdir()) / "unsafe_checkpoint.pth")
    assert outside_checkpoint is None
    assert "outside allowed project directories" in outside_error

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:24, 8:24] = 2
    mask[24:40, 24:40] = 4
    mask[40:56, 40:56] = 7
    color = render_segmentation_mask(mask)
    assert color.shape == (64, 64, 3)
    panel = create_segmentation_panel(image, color)
    assert panel.shape[0] > 64

    stats = compute_damage_statistics(mask)
    assert stats["building_damage"]["major_damage_area"] > 0
    assert stats["road_stats"]["road_clear_area"] > 0
    level = classify_damage_level(stats)
    assert level in {"Superficial Damage", "Medium Damage", "Major Damage", "Unknown"}

    print("AeroRescue-AI auto segmentation smoke test passed.")


if __name__ == "__main__":
    main()
