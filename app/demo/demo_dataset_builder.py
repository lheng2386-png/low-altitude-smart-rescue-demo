"""Build a small explicit demo dataset for one-click workflow runs.

The generated data is synthetic/demo-only and must never be treated as
operational disaster evidence, real model inference, real thermal measurement,
or real georeferenced mapping.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DEMO_DATA_TRUTHFULNESS_NOTE = "Demo data is for workflow demonstration only and is not operational disaster evidence."
MOCK_DETECTION_NOTE = "Mock/imported detections are not real model inference results."
SIMULATED_THERMAL_NOTE = "Simulated Thermal is not real temperature measurement."
IMAGE_PLANE_ROUTE_NOTE = "Image-plane path is not GPS navigation."


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _draw_mapping_image(path, offset=0):
    image = Image.new("RGB", (220, 160), (182, 196, 188))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 118, 220, 160], fill=(92, 106, 96))
    draw.rectangle([20 + offset, 20, 82 + offset, 76], fill=(139, 112, 96))
    draw.rectangle([110 + offset, 24, 168 + offset, 92], fill=(155, 145, 120))
    draw.polygon([(0, 95), (72, 78), (220, 95), (220, 116), (0, 118)], fill=(86, 113, 145))
    draw.line([(0, 130), (220, 105)], fill=(210, 210, 190), width=7)
    draw.line([(0, 128), (220, 103)], fill=(70, 70, 70), width=2)
    draw.rectangle([145, 106, 184, 126], fill=(102, 82, 64))
    image.save(path, quality=92)


def _draw_local_recon_image(path):
    image = Image.new("RGB", (220, 180), (170, 178, 166))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 140, 220, 180], fill=(92, 94, 90))
    draw.rectangle([104, 82, 194, 142], fill=(120, 104, 94))
    draw.rectangle([35, 40, 70, 110], fill=(210, 190, 145))
    draw.ellipse([44, 28, 61, 45], fill=(205, 165, 125))
    draw.rectangle([120, 95, 180, 140], fill=(58, 83, 120))
    draw.ellipse([128, 132, 145, 149], fill=(35, 35, 35))
    draw.ellipse([157, 132, 174, 149], fill=(35, 35, 35))
    draw.line([(0, 155), (220, 138)], fill=(220, 215, 180), width=3)
    image.save(path, quality=92)


def _draw_thermal_like_image(path):
    width, height = 220, 180
    y, x = np.mgrid[0:height, 0:width]
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[..., 2] = 60
    hot1 = np.exp(-(((x - 52) ** 2 + (y - 75) ** 2) / (2 * 16**2)))
    hot2 = np.exp(-(((x - 150) ** 2 + (y - 118) ** 2) / (2 * 20**2)))
    heat = np.clip((hot1 + 0.7 * hot2) * 255, 0, 255).astype(np.uint8)
    base[..., 0] = heat
    base[..., 1] = np.clip(heat * 0.45, 0, 255).astype(np.uint8)
    Image.fromarray(base, mode="RGB").save(path, quality=92)


def _draw_macro_mask(path):
    mask = np.zeros((160, 220), dtype=np.uint8)
    mask[0:160, 0:220] = 7
    mask[64:100, 0:220] = 1
    mask[104:124, 0:220] = 8
    mask[20:78, 30:88] = 4
    mask[18:88, 120:176] = 5
    Image.fromarray(mask, mode="L").save(path)


def ensure_demo_dataset(output_dir):
    """Generate and return a minimal runnable demo dataset manifest."""
    root = Path(output_dir)
    mapping_dir = _ensure_dir(root / "mapping")
    local_dir = _ensure_dir(root / "local_recon")
    thermal_dir = _ensure_dir(root / "thermal")
    masks_dir = _ensure_dir(root / "masks")
    metadata_dir = _ensure_dir(root / "metadata")

    mapping_paths = []
    for index, offset in enumerate([0, 8, 16], start=1):
        path = mapping_dir / f"map_{index:03d}.jpg"
        _draw_mapping_image(path, offset=offset)
        mapping_paths.append(str(path))

    local_rgb_path = local_dir / "area_A_rgb.jpg"
    thermal_path = thermal_dir / "area_A_thermal_like.jpg"
    mask_path = masks_dir / "macro_mask.png"
    _draw_local_recon_image(local_rgb_path)
    _draw_thermal_like_image(thermal_path)
    _draw_macro_mask(mask_path)

    manifest = {
        "dataset_type": "one_click_demo_dataset",
        "truthfulness_note": DEMO_DATA_TRUTHFULNESS_NOTE,
        "mock_detection_note": MOCK_DETECTION_NOTE,
        "simulated_thermal_note": SIMULATED_THERMAL_NOTE,
        "route_note": IMAGE_PLANE_ROUTE_NOTE,
        "mapping_images": mapping_paths,
        "local_recon_image": str(local_rgb_path),
        "thermal_like_image": str(thermal_path),
        "macro_mask": str(mask_path),
        "notes": [
            "Mapping images are synthetic overlapping RGB demo images.",
            "macro_mask.png is a demo mask using RescueNet class IDs and is not automatic model segmentation.",
            "area_A_thermal_like.jpg is an RGB thermal-like visualization and not radiometric thermal data.",
            "These detections are mock detections for workflow demonstration only.",
        ],
    }
    manifest_path = metadata_dir / "demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_demo_detections():
    """Return mock detections for workflow demonstration only."""
    return [
        {
            "class_name": "person",
            "confidence": 0.88,
            "bbox": [35, 40, 70, 110],
            "center": [52, 75],
            "area": 2450,
            "truthfulness_note": "These detections are mock detections for workflow demonstration only. Mock/imported detections are not real model inference results.",
        },
        {
            "class_name": "vehicle",
            "confidence": 0.76,
            "bbox": [120, 95, 180, 140],
            "center": [150, 117],
            "area": 2700,
            "truthfulness_note": "These detections are mock detections for workflow demonstration only. Mock/imported detections are not real model inference results.",
        },
    ]


def build_demo_thermal_result():
    """Return a simulated thermal result without real temperature measurement."""
    return {
        "thermal_mode": "simulated",
        "temperature_matrix": None,
        "hotspot_count": 2,
        "hotspot_area_ratio": 0.035,
        "risk_level": "Medium",
        "is_real_temperature_measurement": False,
        "truthfulness_note": SIMULATED_THERMAL_NOTE,
    }


def build_demo_route_result(candidate_id="C001", target_id="T001"):
    """Return a demo image-plane route result that is not GPS navigation."""
    return {
        "found": True,
        "path_type": "image_plane_path",
        "is_gps_navigation": False,
        "start": [10, 180],
        "goal": [52, 75],
        "path_length": 120,
        "total_cost": 260.5,
        "target_id": str(target_id or "T001"),
        "candidate_id": str(candidate_id or "C001"),
        "message": "Demo image-plane route generated for workflow demonstration. Image-plane path is not GPS navigation.",
    }
