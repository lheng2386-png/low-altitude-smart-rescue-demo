"""Convenience runner for the first three AeroRescue-AI rescue stages."""

from __future__ import annotations

try:
    from ..stages import run_area_tasking_stage, run_global_mapping_stage, run_macro_analysis_stage
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from stages import run_area_tasking_stage, run_global_mapping_stage, run_macro_analysis_stage


def run_early_rescue_workflow(
    mission,
    mission_dir,
    image_files=None,
    segmentation_mask_path=None,
    use_real_odm=False,
):
    """Run S1 global mapping, S2 macro analysis, and S3 area tasking."""
    mission, global_mapping = run_global_mapping_stage(
        mission,
        mission_dir,
        image_files=image_files,
        use_real_odm=use_real_odm,
    )
    mission, macro_analysis = run_macro_analysis_stage(
        mission,
        mission_dir,
        map_image_path=global_mapping.get("base_map_path") or None,
        segmentation_mask_path=segmentation_mask_path,
        segmentation_source="uploaded_mask" if segmentation_mask_path else "none",
    )
    mission, area_tasking = run_area_tasking_stage(
        mission,
        mission_dir,
        macro_analysis_result=macro_analysis,
    )
    return {
        "mission": mission,
        "global_mapping": global_mapping,
        "macro_analysis": macro_analysis,
        "area_tasking": area_tasking,
    }
