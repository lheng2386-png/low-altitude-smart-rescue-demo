"""Full actual integration dashboard for external repositories.

This module tracks the long-term engineering plan for external repository
integration. It does not clone repositories, download datasets, load weights,
or claim that planned adapters are executable.
"""

import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DASHBOARD_PATH = ROOT_DIR / "outputs" / "full_actual_integration" / "status_dashboard.json"
DEFAULT_EXTERNAL_ROOT = ROOT_DIR / "external_integrations"

VALID_STATES = {
    "planned",
    "reproduced_official_demo",
    "adapter_created",
    "project_io_connected",
    "evaluated",
    "report_integrated",
    "blocked_by_dependency",
    "blocked_by_checkpoint",
    "blocked_by_dataset",
}

EXECUTABLE_SUCCESS_STATES = {"project_io_connected", "evaluated", "report_integrated"}

TRUTHFULNESS_POLICY = {
    "no_fake_checkpoints": True,
    "no_fake_metrics": True,
    "no_fake_gps_routes": True,
    "no_fake_full_integration_claims": True,
    "human_review_required": True,
}

COMMON_LIMITATIONS = [
    "No fake checkpoints, datasets, metrics, GPS routes, or full-integration claims are allowed.",
    "Human review is required for all rescue-support outputs.",
    "planned / adapter_created states are not evaluated executable integrations.",
]

INTEGRATION_TARGETS = [
    {
        "key": "yolo_rescue_targets",
        "repository_name": "灾情感知及影响评估 local YOLO Rescue Targets",
        "repository_url": "local models/<variant>/best.pt",
        "family": "detection",
        "directory": "detection/yolo_rescue_targets",
        "target_final_state": "executable_integration",
        "current_state": "blocked_by_checkpoint",
        "dependency_status": "optional",
        "checkpoint_status": "missing",
        "dataset_status": "to_prepare",
        "adapter_file": "app/detection_runtime_service.py",
        "expected_inputs": ["RGB UAV image", "local YOLO best.pt weights"],
        "expected_outputs": ["rescue target bboxes", "confidence", "class_name", "source_backend metadata"],
        "next_actions": ["Verify local best.pt variants", "Run YOLO smoke inference", "Record validation metrics only from a real validation split"],
        "truthfulness_limitations": COMMON_LIMITATIONS
        + ["YOLO is executable only when local best.pt weights exist; missing weights cannot be replaced by fake detections."],
    },
    {
        "key": "transformer_rescuedet_family",
        "repository_name": "ARGUS / Transformer RescueDet family",
        "repository_url": "https://github.com/RoblabWh/argus; https://github.com/RoblabWh/transformer_pipeline; RoblabWhGe/rescuedet-*",
        "family": "detection",
        "directory": "detection/transformer_rescuedet",
        "target_final_state": "executable_integration",
        "current_state": "adapter_created",
        "dependency_status": "optional",
        "checkpoint_status": "missing",
        "dataset_status": "not_required",
        "adapter_file": "app/transformer_detection_service.py",
        "expected_inputs": ["RGB UAV image", "local cached Hugging Face model or explicit allow_download=True workflow"],
        "expected_outputs": ["human_candidate", "vehicle", "fire", "human_review_required metadata"],
        "next_actions": ["Verify local cache or explicit model path", "Run official/demo inference without startup auto-download", "Compare with YOLO output"],
        "truthfulness_limitations": COMMON_LIMITATIONS
        + ["human_candidate is not confirmed civilian; dependency availability does not mean model inference succeeded."],
    },
    {
        "key": "qazi_disaster_management",
        "repository_name": "qazi0/real-time-disaster-management",
        "repository_url": "https://github.com/qazi0/real-time-disaster-management",
        "family": "detection",
        "directory": "detection/qazi_disaster_management",
        "target_final_state": "adapter_wrapped_reference_or_executable_if_reproduced",
        "current_state": "planned",
        "dependency_status": "unknown",
        "checkpoint_status": "unknown",
        "dataset_status": "unknown",
        "adapter_file": None,
        "expected_inputs": ["disaster scene image/video after official repo reproduction"],
        "expected_outputs": ["disaster context detections or labels with source metadata"],
        "next_actions": ["Inspect official dependencies", "Reproduce official demo in isolated environment", "Define adapter only after demo reproduction"],
        "truthfulness_limitations": COMMON_LIMITATIONS + ["Currently not an executable 灾情感知及影响评估 backend."],
    },
    {
        "key": "air_sar_detection",
        "repository_name": "Accenture/AIR",
        "repository_url": "https://github.com/Accenture/AIR",
        "family": "detection",
        "directory": "detection/air_sar_detection",
        "target_final_state": "adapter_wrapped_reference_or_executable_if_reproduced",
        "current_state": "planned",
        "dependency_status": "unknown",
        "checkpoint_status": "unknown",
        "dataset_status": "unknown",
        "adapter_file": None,
        "expected_inputs": ["SAR/search image after official repo reproduction"],
        "expected_outputs": ["person candidate detections with review-required metadata"],
        "next_actions": ["Reproduce official setup", "Identify model/dataset licensing", "Create person-candidate adapter only after real output exists"],
        "truthfulness_limitations": COMMON_LIMITATIONS + ["AIR outputs are not current 灾情感知及影响评估 results."],
    },
    {
        "key": "vtsar_dataset",
        "repository_name": "zxq309/VTSaR",
        "repository_url": "https://github.com/zxq309/VTSaR",
        "family": "dataset",
        "directory": "detection/vtsar_dataset",
        "target_final_state": "dataset_validation_support",
        "current_state": "planned",
        "dependency_status": "not_required",
        "checkpoint_status": "not_required",
        "dataset_status": "to_prepare",
        "adapter_file": None,
        "expected_inputs": ["official dataset files if license and storage allow"],
        "expected_outputs": ["dataset split manifest", "evaluation-ready annotations"],
        "next_actions": ["Review dataset access/license", "Prepare manifest only after data is available", "Use for human_candidate validation, not fake metrics"],
        "truthfulness_limitations": COMMON_LIMITATIONS + ["Dataset reference is not a trained model and cannot produce detections by itself."],
    },
    {
        "key": "python_robotics",
        "repository_name": "AtsushiSakai/PythonRobotics",
        "repository_url": "https://github.com/AtsushiSakai/PythonRobotics",
        "family": "planning",
        "directory": "planning/python_robotics",
        "target_final_state": "algorithm_adapter_and_comparison",
        "current_state": "adapter_created",
        "dependency_status": "available",
        "checkpoint_status": "not_required",
        "dataset_status": "not_required",
        "adapter_file": "app/path_planner.py",
        "expected_inputs": ["image-plane cost map", "start point", "target point"],
        "expected_outputs": ["image-plane reference path", "planner metadata", "not GPS navigation boundary"],
        "next_actions": ["Add Dijkstra/RRT comparison adapters", "Record path cost comparisons", "Keep GPS=false metadata"],
        "truthfulness_limitations": COMMON_LIMITATIONS + ["Image-plane path is not GPS/GIS navigation."],
    },
    {
        "key": "fields2cover",
        "repository_name": "Fields2Cover/Fields2Cover",
        "repository_url": "https://github.com/Fields2Cover/Fields2Cover",
        "family": "planning",
        "directory": "planning/fields2cover",
        "target_final_state": "coverage_planning_adapter_if_dependency_available",
        "current_state": "planned",
        "dependency_status": "missing",
        "checkpoint_status": "not_required",
        "dataset_status": "not_required",
        "adapter_file": None,
        "expected_inputs": ["field polygon / obstacle geometry if C++ dependency is installed"],
        "expected_outputs": ["coverage route candidate with non-GPS boundary unless geospatial data is provided"],
        "next_actions": ["Evaluate C++/Python binding feasibility", "Do not add heavy dependency until isolated demo is reproduced", "Map outputs into coverage score schema"],
        "truthfulness_limitations": COMMON_LIMITATIONS + ["Current coverage score is lightweight image-plane adaptation, not Fields2Cover output."],
    },
    {
        "key": "sarenv",
        "repository_name": "namurproject/SAREnv",
        "repository_url": "https://github.com/namurproject/SAREnv",
        "family": "planning",
        "directory": "planning/sarenv",
        "target_final_state": "search_evaluation_adapter_if_reproduced",
        "current_state": "planned",
        "dependency_status": "unknown",
        "checkpoint_status": "not_required",
        "dataset_status": "to_prepare",
        "adapter_file": None,
        "expected_inputs": ["search environment configuration", "UAV/search path candidate"],
        "expected_outputs": ["search probability / coverage evaluation outputs"],
        "next_actions": ["Reproduce official environment", "Map SAREnv scores to Decision Fusion fields", "Separate geospatial SAREnv results from image-plane approximations"],
        "truthfulness_limitations": COMMON_LIMITATIONS + ["Current search priority map is inspired by SAREnv but is not full SAREnv runtime."],
    },
    {
        "key": "skai",
        "repository_name": "google-research/skai",
        "repository_url": "https://github.com/google-research/skai",
        "family": "damage_impact",
        "directory": "damage_impact/skai",
        "target_final_state": "damage_assessment_adapter_if_model_and_data_available",
        "current_state": "planned",
        "dependency_status": "unknown",
        "checkpoint_status": "missing",
        "dataset_status": "to_prepare",
        "adapter_file": None,
        "expected_inputs": ["pre/post disaster imagery and SKAI-compatible model/data"],
        "expected_outputs": ["building damage predictions with explicit model/source metadata"],
        "next_actions": ["Study official SKAI pipeline", "Identify required imagery and checkpoint availability", "Do not label segmentation heuristic as SKAI output"],
        "truthfulness_limitations": COMMON_LIMITATIONS + ["Current building damage score is segmentation-based, not SKAI model output."],
    },
    {
        "key": "inasafe",
        "repository_name": "inasafe/inasafe",
        "repository_url": "https://github.com/inasafe/inasafe",
        "family": "damage_impact",
        "directory": "damage_impact/inasafe",
        "target_final_state": "gis_impact_adapter_if_qgis_stack_available",
        "current_state": "planned",
        "dependency_status": "missing",
        "checkpoint_status": "not_required",
        "dataset_status": "to_prepare",
        "adapter_file": None,
        "expected_inputs": ["hazard layer", "exposure layer", "vulnerability assumptions"],
        "expected_outputs": ["GIS impact report if QGIS/InaSAFE stack is actually run"],
        "next_actions": ["Keep InaSAFE isolated due to QGIS/GIS dependency weight", "Define schema bridge for impact report", "Do not call image-plane score a GIS result"],
        "truthfulness_limitations": COMMON_LIMITATIONS + ["Current impact score is image-plane lightweight adaptation, not full InaSAFE/QGIS analysis."],
    },
]


class FullActualIntegrationDashboardError(Exception):
    pass


def list_full_actual_integration_targets():
    """Return all integration target records."""
    return [dict(item) for item in INTEGRATION_TARGETS]


def classify_integration_state(record):
    """Classify a target state conservatively."""
    state = record.get("current_state", "planned")
    if state not in VALID_STATES:
        state = "planned"
    return {
        "current_state": state,
        "is_evaluated": state == "evaluated" or state == "report_integrated",
        "is_executable_success": state in EXECUTABLE_SUCCESS_STATES,
        "blocked": state.startswith("blocked_by_"),
        "truthfulness_note": "planned and adapter_created are not evaluated executable integrations.",
    }


def build_status_dashboard(root_dir=None):
    """Build the machine-readable full actual integration dashboard."""
    repositories = []
    for item in INTEGRATION_TARGETS:
        record = dict(item)
        record.update(classify_integration_state(record))
        repositories.append(record)
    return {
        "module": "full_actual_integration_dashboard",
        "project_goal": "Actual multi-repository engineering integration for low-altitude disaster rescue decision support.",
        "truthfulness_policy": dict(TRUTHFULNESS_POLICY),
        "repositories": repositories,
    }


def save_status_dashboard(output_path=None, root_dir=None):
    """Save dashboard JSON. This writes runtime output only; it does not imply integration success."""
    path = Path(output_path) if output_path else DEFAULT_DASHBOARD_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    dashboard = build_status_dashboard(root_dir=root_dir)
    path.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"success": True, "dashboard_path": str(path), "dashboard": dashboard}


def scan_full_actual_integration_status(root_dir=None):
    """Scanner-style summary for the integration dashboard."""
    dashboard = build_status_dashboard(root_dir=root_dir)
    summary = {state: 0 for state in sorted(VALID_STATES)}
    for repo in dashboard["repositories"]:
        summary[repo["current_state"]] = summary.get(repo["current_state"], 0) + 1
    return {
        "success": True,
        "summary": summary,
        "repositories": dashboard["repositories"],
        "truthfulness_note": "This scanner reports integration engineering states. It does not treat planned, adapter_created, or blocked repositories as evaluated executable integrations.",
    }


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _directory_readme(record):
    return f"""# {record['repository_name']}

Repository: {record['repository_url']}

Family: `{record['family']}`

Current state: `{record['current_state']}`

Target final state: `{record['target_final_state']}`

This directory is an engineering integration scaffold. It does not mean the external repository has been reproduced, evaluated, or connected to 灾情感知及影响评估 runtime.
"""


def _integration_plan(record):
    actions = "\n".join(f"- {item}" for item in record.get("next_actions", []))
    return f"""# Integration Plan

## Reproducibility Plan
- Reproduce the official repository demo in an isolated environment.
- Record exact dependency versions, command lines, and required local files.
- Do not claim success until official outputs are generated locally.

## Adapter Plan
- Expected adapter file: `{record.get('adapter_file') or 'to be defined after reproduction'}`
- Wrap inputs and outputs into 灾情感知及影响评估 schemas only after real outputs exist.
- Preserve source metadata and truthfulness notes in every output.

## Roadmap
{actions}
"""


def _limitations(record):
    limitations = "\n".join(f"- {item}" for item in record.get("truthfulness_limitations", []))
    return f"""# Limitations

{limitations}

- Current state `{record['current_state']}` must not be described as evaluated unless the dashboard state is updated by real reproduction/evaluation evidence.
"""


def _io_schema(record):
    return {
        "repository_key": record["key"],
        "expected_input_schema": record.get("expected_inputs", []),
        "expected_output_schema": record.get("expected_outputs", []),
        "required_metadata": [
            "repository_name",
            "adapter_version",
            "dependency_status",
            "checkpoint_status",
            "dataset_status",
            "truthfulness_note",
            "human_review_required",
        ],
        "truthfulness_boundary": record.get("truthfulness_limitations", []),
    }


def write_external_integration_scaffold(root_dir=None):
    """Create external_integrations directory scaffolds for all target families."""
    root = Path(root_dir) if root_dir else ROOT_DIR
    base = root / "external_integrations"
    written = []
    for record in INTEGRATION_TARGETS:
        directory = base / record["directory"]
        _write_text(directory / "README.md", _directory_readme(record))
        _write_text(directory / "integration_plan.md", _integration_plan(record))
        _write_json(
            directory / "status.json",
            {
                "repository_key": record["key"],
                "repository_name": record["repository_name"],
                "current_state": record["current_state"],
                "target_final_state": record["target_final_state"],
                "dependency_status": record["dependency_status"],
                "checkpoint_status": record["checkpoint_status"],
                "dataset_status": record["dataset_status"],
                "truthfulness_limitations": record["truthfulness_limitations"],
            },
        )
        _write_json(directory / "expected_io_schema.json", _io_schema(record))
        _write_text(directory / "limitations.md", _limitations(record))
        written.append(str(directory))
    return {"success": True, "external_root": str(base), "directories": written}


if __name__ == "__main__":
    scaffold = write_external_integration_scaffold()
    saved = save_status_dashboard()
    print(f"Wrote {len(scaffold['directories'])} integration scaffolds.")
    print(saved["dashboard_path"])
