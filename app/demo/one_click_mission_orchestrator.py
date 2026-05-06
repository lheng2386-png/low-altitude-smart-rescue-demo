"""One-click demo mission orchestration for S1-S9 workflow validation.

The orchestrator uses explicit demo/mock/imported artifacts so a full mission
can be demonstrated without YOLO weights, real thermal hardware, ODM/Docker,
GIS/GPS, or LLM APIs.
"""

from __future__ import annotations

from pathlib import Path

try:
    from ..evidence_ledger import create_ledger, save_ledger
    from ..mission_schema import create_mission, ensure_mission_dirs, save_mission, update_mission_status
    from ..stages import (
        run_area_tasking_stage,
        run_decision_fusion_stage,
        run_evidence_report_stage,
        run_global_mapping_stage,
        run_local_recon_stage,
        run_macro_analysis_stage,
        run_rescue_recommendation_stage,
        run_target_verification_stage,
        run_thermal_check_stage,
    )
    from ..workflow.workflow_orchestrator import fail_stage, initialize_rescue_workflow
    from ..workflow.workflow_state import summarize_workflow_state
    from .demo_dataset_builder import (
        DEMO_DATA_TRUTHFULNESS_NOTE,
        IMAGE_PLANE_ROUTE_NOTE,
        MOCK_DETECTION_NOTE,
        SIMULATED_THERMAL_NOTE,
        build_demo_detections,
        build_demo_route_result,
        build_demo_thermal_result,
        ensure_demo_dataset,
    )
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from evidence_ledger import create_ledger, save_ledger
    from mission_schema import create_mission, ensure_mission_dirs, save_mission, update_mission_status
    from stages import (
        run_area_tasking_stage,
        run_decision_fusion_stage,
        run_evidence_report_stage,
        run_global_mapping_stage,
        run_local_recon_stage,
        run_macro_analysis_stage,
        run_rescue_recommendation_stage,
        run_target_verification_stage,
        run_thermal_check_stage,
    )
    from workflow.workflow_orchestrator import fail_stage, initialize_rescue_workflow
    from workflow.workflow_state import summarize_workflow_state
    from demo.demo_dataset_builder import (
        DEMO_DATA_TRUTHFULNESS_NOTE,
        IMAGE_PLANE_ROUTE_NOTE,
        MOCK_DETECTION_NOTE,
        SIMULATED_THERMAL_NOTE,
        build_demo_detections,
        build_demo_route_result,
        build_demo_thermal_result,
        ensure_demo_dataset,
    )


DEMO_TRUTHFULNESS_BOUNDARIES = [
    DEMO_DATA_TRUTHFULNESS_NOTE,
    MOCK_DETECTION_NOTE,
    SIMULATED_THERMAL_NOTE,
    "Fast Preview is not a real ODM georeferenced orthomosaic.",
    "Uploaded/Demo Mask is not automatic model segmentation.",
    IMAGE_PLANE_ROUTE_NOTE,
    "AI candidates are not confirmed civilians.",
    "Final Report is an AI-assisted decision-support report and not a final rescue conclusion.",
]


def _append_boundaries(mission, boundaries):
    mission.setdefault("truthfulness_boundaries", [])
    for boundary in boundaries:
        if boundary not in mission["truthfulness_boundaries"]:
            mission["truthfulness_boundaries"].append(boundary)
    return mission


def _failed_result(stage_key, exc):
    return {
        "stage_key": stage_key,
        "status": "failed",
        "error": str(exc),
        "truthfulness_note": f"{DEMO_DATA_TRUTHFULNESS_NOTE} Stage failed during demo orchestration: {exc}",
        "human_review_required": True,
    }


def _run_stage(all_stage_results, mission, mission_dir, stage_key, func, *args, **kwargs):
    try:
        mission, result = func(mission, mission_dir, *args, **kwargs)
        all_stage_results[stage_key] = result
        return mission, result
    except Exception as exc:
        result = _failed_result(stage_key, exc)
        all_stage_results[stage_key] = result
        try:
            mission = fail_stage(mission, mission_dir, stage_key, str(exc))
        except Exception:
            pass
        return mission, result


def _create_demo_mission(missions_root, mission_name):
    mission = create_mission(mission_name=mission_name)
    mission = initialize_rescue_workflow(mission)
    mission = update_mission_status(mission, "running")
    mission["workflow_mode"] = "one_click_demo"
    mission["global_context_available"] = False
    mission["map_registration_available"] = False
    _append_boundaries(mission, DEMO_TRUTHFULNESS_BOUNDARIES)

    mission_dir = Path(missions_root) / mission["mission_id"]
    paths = ensure_mission_dirs(mission_dir)
    ledger_path = paths["evidence"] / "ledger.json"
    mission["evidence_ledger_path"] = str(ledger_path)
    save_ledger(create_ledger(mission["mission_id"]), ledger_path)
    save_mission(mission, mission_dir)
    return mission, mission_dir


def run_one_click_demo_mission(
    missions_root,
    mission_name="灾情感知及影响评估 One-Click Demo Mission",
    demo_output_root=None,
    workflow_mode="full_demo",
    run_mapping_preview=True,
    run_macro_analysis=True,
    run_route_suggestion=True,
):
    """Run a complete S1-S9 demo mission using explicit demo/mock data."""
    missions_root = Path(missions_root)
    demo_output_root = Path(demo_output_root or missions_root / "_demo_dataset")
    manifest = ensure_demo_dataset(demo_output_root)
    mission, mission_dir = _create_demo_mission(missions_root, mission_name)
    mission["workflow_mode"] = "one_click_demo"
    mission["demo_workflow_mode"] = str(workflow_mode or "full_demo")
    save_mission(mission, mission_dir)

    stage_results = {}
    mapping_images = manifest.get("mapping_images", [])
    map_image_path = mapping_images[0] if mapping_images else ""

    if run_mapping_preview:
        mission, s1 = _run_stage(
            stage_results,
            mission,
            mission_dir,
            "global_mapping",
            run_global_mapping_stage,
            image_files=mapping_images,
            use_real_odm=False,
        )
        if s1.get("base_map_path"):
            mission["global_context_available"] = True
            map_image_path = s1.get("base_map_path")
        else:
            mission["global_context_available"] = False
    else:
        stage_results["global_mapping"] = {
            "stage_key": "global_mapping",
            "status": "skipped",
            "truthfulness_note": f"{DEMO_DATA_TRUTHFULNESS_NOTE} Mapping preview was disabled by demo options.",
            "human_review_required": True,
        }

    if run_macro_analysis:
        mission, s2 = _run_stage(
            stage_results,
            mission,
            mission_dir,
            "macro_analysis",
            run_macro_analysis_stage,
            map_image_path=map_image_path,
            segmentation_mask_path=manifest.get("macro_mask"),
            segmentation_source="demo_mask",
        )
    else:
        s2 = {
            "stage_key": "macro_analysis",
            "status": "degraded",
            "macro_zones": [],
            "truthfulness_note": f"{DEMO_DATA_TRUTHFULNESS_NOTE} Macro analysis was disabled by demo options.",
            "human_review_required": True,
        }
        stage_results["macro_analysis"] = s2

    mission, s3 = _run_stage(
        stage_results,
        mission,
        mission_dir,
        "area_tasking",
        run_area_tasking_stage,
        macro_analysis_result=s2,
    )
    area_tasks = s3.get("area_tasks") or []
    area_task = area_tasks[0] if area_tasks else None

    mission, s4 = _run_stage(
        stage_results,
        mission,
        mission_dir,
        "local_recon",
        run_local_recon_stage,
        local_rgb_images=[manifest.get("local_recon_image")],
        area_task=area_task,
        area_id="A",
        detections=build_demo_detections(),
        detection_backend="mock",
        global_context_available=bool(mission.get("global_context_available")),
        map_registration_available=False,
    )

    mission, s5 = _run_stage(
        stage_results,
        mission,
        mission_dir,
        "target_verification",
        run_target_verification_stage,
        local_recon_result=s4,
        review_actions={
            "C001": {
                "review_status": "confirmed_candidate",
                "review_note": "Demo review: candidate kept for thermal support demonstration.",
                "reviewer": "demo_operator",
            }
        },
    )

    mission, s6 = _run_stage(
        stage_results,
        mission,
        mission_dir,
        "thermal_check",
        run_thermal_check_stage,
        target_verification_result=s5,
        thermal_images=[manifest.get("thermal_like_image")],
        thermal_results=build_demo_thermal_result(),
        run_thermal_analysis=False,
    )

    mission, s7 = _run_stage(
        stage_results,
        mission,
        mission_dir,
        "decision_fusion",
        run_decision_fusion_stage,
        local_recon_result=s4,
        target_verification_result=s5,
        thermal_check_result=s6,
        macro_analysis_result=s2,
    )

    route_results = {}
    if run_route_suggestion:
        route_results = {
            "C001": build_demo_route_result(candidate_id="C001", target_id="T001"),
        }
    mission, s8 = _run_stage(
        stage_results,
        mission,
        mission_dir,
        "rescue_recommendation",
        run_rescue_recommendation_stage,
        decision_fusion_result=s7,
        route_results=route_results,
        run_path_planning=False,
    )

    mission, s9 = _run_stage(
        stage_results,
        mission,
        mission_dir,
        "evidence_report",
        run_evidence_report_stage,
        stage_results={
            "global_mapping": stage_results.get("global_mapping", {}),
            "macro_analysis": stage_results.get("macro_analysis", {}),
            "area_tasking": stage_results.get("area_tasking", {}),
            "local_recon": stage_results.get("local_recon", {}),
            "target_verification": stage_results.get("target_verification", {}),
            "thermal_check": stage_results.get("thermal_check", {}),
            "decision_fusion": stage_results.get("decision_fusion", {}),
            "rescue_recommendation": stage_results.get("rescue_recommendation", {}),
        },
    )

    return {
        "mission": mission,
        "mission_dir": str(mission_dir),
        "demo_dataset_dir": str(demo_output_root),
        "stage_results": stage_results,
        "final_report_markdown_path": s9.get("report_markdown_path", ""),
        "final_report_json_path": s9.get("report_json_path", ""),
        "evidence_ledger_path": mission.get("evidence_ledger_path", ""),
        "workflow_summary": summarize_workflow_state(mission.get("workflow_state")),
        "truthfulness_note": DEMO_DATA_TRUTHFULNESS_NOTE,
    }
