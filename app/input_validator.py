"""Mission input inspection and module capability classification.

This module does not run detection, segmentation, thermal analysis, ODM, or
reconstruction. It only checks provided input references and states what the
prototype may truthfully enable for a mission.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PIL import ExifTags, Image
except Exception:  # pragma: no cover - Pillow is expected in the app runtime.
    ExifTags = None
    Image = None


RGB_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
THERMAL_EXTENSIONS = {".rjpeg", ".tiff", ".tif", ".raw", ".csv", ".json", ".npy"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _as_existing_paths(paths):
    """Normalize optional path input into existing pathlib.Path objects."""
    if not paths:
        return []
    return [Path(path) for path in paths if path and Path(path).exists()]


def inspect_image_metadata(image_path):
    """Inspect basic image metadata and EXIF tags without interpreting pixels."""
    image_path = Path(image_path)
    metadata = {
        "path": str(image_path),
        "exists": image_path.exists(),
        "suffix": image_path.suffix.lower(),
        "width": None,
        "height": None,
        "has_exif": False,
        "has_gps_exif": False,
        "error": "",
    }
    if not image_path.exists() or Image is None:
        return metadata

    try:
        with Image.open(image_path) as image:
            metadata["width"], metadata["height"] = image.size
            exif = image.getexif()
            metadata["has_exif"] = bool(exif)
            metadata["has_gps_exif"] = _exif_has_gps(exif)
    except Exception as exc:
        metadata["error"] = str(exc)
    return metadata


def _exif_has_gps(exif):
    """Return True when a PIL EXIF object contains a GPSInfo block."""
    if not exif:
        return False
    if ExifTags is None:
        return False
    gps_tag = None
    for tag_id, tag_name in ExifTags.TAGS.items():
        if tag_name == "GPSInfo":
            gps_tag = tag_id
            break
    return bool(gps_tag and exif.get(gps_tag))


def has_gps_exif(image_paths):
    """Check whether any provided image contains GPS EXIF metadata."""
    return any(inspect_image_metadata(path).get("has_gps_exif") for path in image_paths or [])


def _is_radiometric_thermal_candidate(path):
    """Conservatively identify files that may contain real thermal data."""
    path = Path(path)
    return path.suffix.lower() in {".rjpeg", ".tiff", ".tif", ".raw", ".csv", ".json", ".npy"}


def classify_available_modules(
    rgb_image_count=0,
    thermal_image_count=0,
    real_temperature_available=False,
    mask_available=False,
    video_available=False,
    gps_available=False,
    multi_view_available=False,
    odm_enabled=False,
    segmentation_checkpoint_available=False,
):
    """Classify truthful module availability from inspected input features."""
    available_modules = []
    disabled_modules = []
    truthfulness_boundaries = []

    if rgb_image_count > 0:
        available_modules.extend(
            [
                "object_detection",
                "simulated_thermal_risk",
                "target_only_terp",
                "image_plane_path_planning",
                "template_report_generation",
            ]
        )
        truthfulness_boundaries.extend(
            [
                "RGB images can support visual detection but cannot provide real temperature_matrix.",
                "Simulated Thermal is generated from RGB/gray intensity and is not real thermal measurement.",
                "Image-plane path planning is not GPS navigation.",
                "Human/civilian detections are AI candidates and not confirmed civilians.",
                "System outputs are decision-support results and not final rescue conclusions.",
            ]
        )
    else:
        disabled_modules.append({"module": "object_detection", "reason": "No RGB image input was provided."})
        disabled_modules.append({"module": "image_plane_path_planning", "reason": "No image plane is available."})

    if real_temperature_available:
        available_modules.append("real_temperature_analysis")
        truthfulness_boundaries.append(
            "Radiometric thermal results are valid only when a real temperature matrix is successfully parsed."
        )
    else:
        disabled_modules.append(
            {"module": "real_temperature_analysis", "reason": "No radiometric thermal input was provided."}
        )

    if mask_available:
        available_modules.append("uploaded_mask_segmentation")
        truthfulness_boundaries.append("Uploaded Mask / Demo Mask is not automatic model segmentation.")
    elif segmentation_checkpoint_available:
        available_modules.append("auto_segmentation_model")
    else:
        disabled_modules.append(
            {
                "module": "semantic_segmentation",
                "reason": "No segmentation checkpoint or uploaded mask was provided; fallback only.",
            }
        )

    if gps_available:
        available_modules.append("geospatial_reference")
    else:
        disabled_modules.append(
            {"module": "gps_navigation", "reason": "No GPS/EXIF/georeferencing data was detected."}
        )

    if odm_enabled and multi_view_available and gps_available:
        available_modules.append("real_orthomosaic")
    else:
        if multi_view_available:
            available_modules.append("orthomosaic_preview")
        disabled_modules.append(
            {
                "module": "real_orthomosaic",
                "reason": "No ODM task or georeferenced multi-view image set is available.",
            }
        )
        truthfulness_boundaries.append(
            "Fast Preview / OpenCV Stitch / ORB Homography is not a real ODM georeferenced orthomosaic."
        )

    if video_available:
        available_modules.append("reconstruction_preview")
        truthfulness_boundaries.append(
            "Lightweight Reconstruction Preview is not real SfM/MVS or surveying-grade 3D reconstruction."
        )
    else:
        disabled_modules.append(
            {"module": "real_3d_reconstruction", "reason": "No calibrated SfM/MVS dataset was provided."}
        )

    if thermal_image_count and not real_temperature_available:
        truthfulness_boundaries.append(
            "Thermal-like image files without parsed radiometric data must not be treated as real temperature matrices."
        )

    return {
        "available_modules": _dedupe_strings(available_modules),
        "disabled_modules": _dedupe_disabled(disabled_modules),
        "truthfulness_boundaries": _dedupe_strings(truthfulness_boundaries),
    }


def validate_mission_inputs(
    rgb_images=None,
    thermal_images=None,
    mask_files=None,
    video_file=None,
    odm_enabled=False,
    segmentation_checkpoint_path=None,
):
    """Validate mission inputs and return structured module capability data."""
    rgb_paths = _as_existing_paths(rgb_images)
    thermal_paths = _as_existing_paths(thermal_images)
    mask_paths = _as_existing_paths(mask_files)
    video_path = Path(video_file) if video_file else None
    video_available = bool(video_path and video_path.exists() and video_path.suffix.lower() in VIDEO_EXTENSIONS)
    checkpoint_path = Path(segmentation_checkpoint_path) if segmentation_checkpoint_path else None

    image_metadata = [inspect_image_metadata(path) for path in rgb_paths]
    gps_available = any(item.get("has_gps_exif") for item in image_metadata)
    real_temperature_available = any(_is_radiometric_thermal_candidate(path) for path in thermal_paths)
    multi_view_available = len(rgb_paths) >= 2
    segmentation_checkpoint_available = bool(checkpoint_path and checkpoint_path.exists())

    classification = classify_available_modules(
        rgb_image_count=len(rgb_paths),
        thermal_image_count=len(thermal_paths),
        real_temperature_available=real_temperature_available,
        mask_available=bool(mask_paths),
        video_available=video_available,
        gps_available=gps_available,
        multi_view_available=multi_view_available,
        odm_enabled=odm_enabled,
        segmentation_checkpoint_available=segmentation_checkpoint_available,
    )

    if segmentation_checkpoint_available:
        segmentation_source = "auto_model"
    elif mask_paths:
        segmentation_source = "uploaded_mask"
    else:
        segmentation_source = "none"

    return {
        "rgb_image_available": bool(rgb_paths),
        "rgb_image_count": len(rgb_paths),
        "thermal_image_available": bool(thermal_paths),
        "thermal_image_count": len(thermal_paths),
        "real_temperature_available": real_temperature_available,
        "mask_available": bool(mask_paths),
        "mask_count": len(mask_paths),
        "video_available": video_available,
        "gps_available": gps_available,
        "multi_view_available": multi_view_available,
        "odm_enabled": bool(odm_enabled),
        "segmentation_checkpoint_available": segmentation_checkpoint_available,
        "segmentation_source": segmentation_source,
        "image_metadata": image_metadata,
        "available_modules": classification["available_modules"],
        "disabled_modules": classification["disabled_modules"],
        "truthfulness_boundaries": classification["truthfulness_boundaries"],
    }


def _dedupe_strings(items):
    """Deduplicate strings while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _dedupe_disabled(items):
    """Deduplicate disabled module dictionaries by module name."""
    seen = set()
    result = []
    for item in items:
        module = item.get("module")
        if module not in seen:
            seen.add(module)
            result.append(item)
    return result
