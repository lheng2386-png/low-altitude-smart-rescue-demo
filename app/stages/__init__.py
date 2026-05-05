"""Workflow stage wrappers for AeroRescue-AI."""

from .global_mapping_stage import run_global_mapping_stage
from .macro_analysis_stage import run_macro_analysis_stage
from .area_tasking_stage import run_area_tasking_stage
from .local_recon_stage import prepare_direct_local_recon_context, run_local_recon_stage
from .target_verification_stage import run_target_verification_stage
from .thermal_check_stage import run_thermal_check_stage
from .decision_fusion_stage import run_decision_fusion_stage
from .rescue_recommendation_stage import run_rescue_recommendation_stage
from .evidence_report_stage import run_evidence_report_stage

__all__ = [
    "run_global_mapping_stage",
    "run_macro_analysis_stage",
    "run_area_tasking_stage",
    "prepare_direct_local_recon_context",
    "run_local_recon_stage",
    "run_target_verification_stage",
    "run_thermal_check_stage",
    "run_decision_fusion_stage",
    "run_rescue_recommendation_stage",
    "run_evidence_report_stage",
]
