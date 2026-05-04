"""Module execution status scanner for AeroRescue-AI.

This scanner infers module status from outputs/ artifacts and JSON metadata.
It does not assume code existence means execution success.
"""

import json
from datetime import datetime
from pathlib import Path


class ModuleStatusScannerError(Exception):
    pass


MODULE_STATUS = {
    "not_run": "not_run",
    "implemented_but_not_run": "implemented_but_not_run",
    "executed_success": "executed_success",
    "executed_failed": "executed_failed",
    "dependency_missing": "dependency_missing",
    "reference_only": "reference_only",
    "simulated_result": "simulated_result",
    "real_model_output": "real_model_output",
    "real_measurement": "real_measurement",
    "preview_only": "preview_only",
    "unknown": "unknown",
}


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT_DIR / "outputs"


MODULE_SCAN_TARGETS = {
    "detection": {
        "module_name": "detection",
        "display_name": "目标检测运行层",
        "expected_outputs": [
            "outputs/detection/detection_result.json",
            "outputs/detection/detection_metadata.json",
            "outputs/detection/detection_overlay.png",
            "outputs/detection/dual_detection_result.json",
            "outputs/detection/dual_detection_consensus.json",
        ],
        "result_json_candidates": [
            "outputs/detection/detection_result.json",
            "outputs/detection/dual_detection_result.json",
        ],
        "metadata_json_candidates": [
            "outputs/detection/detection_metadata.json",
        ],
        "implementation_files": [
            "app/detection_runtime_service.py",
            "app/app.py",
        ],
        "truthfulness_note": "Detection status is inferred from local detection result artifacts. It does not assume the presence of code means a successful run.",
    },
    "transformer_detection": {
        "module_name": "transformer_detection",
        "display_name": "Transformer 辅助检测后端",
        "expected_outputs": [
            "outputs/detection/transformer_detection_result.json",
            "outputs/detection/transformer_detection_metadata.json",
        ],
        "result_json_candidates": [
            "outputs/detection/transformer_detection_result.json",
        ],
        "metadata_json_candidates": [
            "outputs/detection/transformer_detection_metadata.json",
        ],
        "implementation_files": [
            "app/transformer_detection_service.py",
            "app/detection_runtime_service.py",
        ],
        "truthfulness_note": "Transformer status is inferred from local auxiliary detection artifacts, dependency evidence, and structured error codes.",
    },
    "segmentation": {
        "module_name": "segmentation",
        "display_name": "灾后场景语义分割与损毁评估",
        "expected_outputs": [
            "outputs/segmentation_inference/segmentation_result.json",
            "outputs/segmentation_inference/damage_summary.json",
            "outputs/segmentation_inference/segmentation_source.json",
            "outputs/segmentation_inference/segmentation_overlay.png",
        ],
        "result_json_candidates": [
            "outputs/segmentation_inference/segmentation_result.json",
            "outputs/segmentation_inference/damage_summary.json",
            "outputs/segmentation_inference/segmentation_source.json",
        ],
        "metadata_json_candidates": [
            "outputs/segmentation_inference/segmentation_source.json",
        ],
        "implementation_files": [
            "app/segmentation_engine.py",
            "app/segmentation_model.py",
            "app/segmentation_model_service.py",
            "app/damage_segmentation_visualizer.py",
            "app/segmentation_source_metadata.py",
            "app/segmentation_source_tracking.py",
            "app/app.py",
        ],
        "truthfulness_note": "Segmentation status is inferred from segmentation result metadata. Uploaded/demo masks are not treated as model predictions.",
    },
    "terp": {
        "module_name": "terp",
        "display_name": "TERP 目标—环境—可达性优先级",
        "expected_outputs": [
            "outputs/decision_fusion/terp_ranking.json",
            "outputs/decision_fusion/terp_summary.json",
        ],
        "result_json_candidates": [
            "outputs/decision_fusion/terp_ranking.json",
            "outputs/decision_fusion/terp_summary.json",
        ],
        "metadata_json_candidates": [],
        "implementation_files": [
            "app/terp_engine.py",
            "app/priority_ranker.py",
        ],
        "truthfulness_note": "TERP is a lightweight local decision model. It is not a GIS or GPS routing system.",
    },
    "path_planning": {
        "module_name": "path_planning",
        "display_name": "风险感知路径规划",
        "expected_outputs": [
            "outputs/decision_fusion/path_planning_result.json",
            "outputs/decision_fusion/path_comparison.json",
            "outputs/decision_fusion/path_overlay.png",
        ],
        "result_json_candidates": [
            "outputs/decision_fusion/path_planning_result.json",
            "outputs/decision_fusion/path_comparison.json",
        ],
        "metadata_json_candidates": [],
        "implementation_files": [
            "app/path_planner.py",
            "app/scene_mode_and_entry_service.py",
            "app/scene_applicability_gate.py",
        ],
        "truthfulness_note": "Path planning status is inferred from local image-plane route artifacts. It is not real GPS navigation.",
    },
    "decision_fusion": {
        "module_name": "decision_fusion",
        "display_name": "决策融合层",
        "expected_outputs": [
            "outputs/decision_fusion/decision_fusion_summary.json",
            "outputs/decision_fusion/damage_impact_result.json",
            "outputs/decision_fusion/coverage_score_result.json",
            "outputs/decision_fusion/search_priority_map.npy",
            "outputs/decision_fusion/search_priority_overlay.png",
        ],
        "result_json_candidates": [
            "outputs/decision_fusion/decision_fusion_summary.json",
        ],
        "metadata_json_candidates": [],
        "implementation_files": [
            "app/decision_fusion_adapter.py",
            "app/detection_decision_bridge.py",
            "app/app.py",
        ],
        "truthfulness_note": "Decision fusion is a lightweight image-plane adaptation inspired by external references. It is not a full GIS or automated rescue decision system.",
    },
    "thermal": {
        "module_name": "thermal",
        "display_name": "热红外分析",
        "expected_outputs": [
            "outputs/thermal/thermal_result.json",
            "outputs/thermal/thermal_heatmap.jpg",
            "outputs/thermal/thermal_overlay.jpg",
            "outputs/thermal/hotspot_mask.jpg",
            "outputs/thermal/radiometric_thermal_result.json",
            "outputs/thermal/temperature_matrix.npy",
        ],
        "result_json_candidates": [
            "outputs/thermal/radiometric_thermal_result.json",
            "outputs/thermal/thermal_result.json",
        ],
        "metadata_json_candidates": [],
        "implementation_files": [
            "app/thermal_service.py",
            "app/radiometric_thermal_service.py",
        ],
        "truthfulness_note": "Thermal status distinguishes simulated hotspot analysis from real radiometric temperature measurement.",
    },
    "orthomosaic": {
        "module_name": "orthomosaic",
        "display_name": "正射影像 / 航测拼接预览",
        "expected_outputs": [
            "outputs/orthomosaic/processing_log.json",
            "outputs/orthomosaic/orthomosaic_result.jpg",
            "outputs/odm/odm_orthophoto.tif",
        ],
        "result_json_candidates": [
            "outputs/orthomosaic/processing_log.json",
        ],
        "metadata_json_candidates": [],
        "implementation_files": [
            "app/orthomosaic_service.py",
            "app/odm_service.py",
        ],
        "truthfulness_note": "Orthomosaic status distinguishes OpenCV preview from real ODM orthophoto output.",
    },
    "odm": {
        "module_name": "odm",
        "display_name": "OpenDroneMap 真正正射处理",
        "expected_outputs": [
            "outputs/odm/**/odm_orthophoto.tif",
            "outputs/odm/**/odm_run.log",
            "outputs/odm/**/orthophoto_preview.jpg",
            "outputs/odm/**/odm_georeferencing/odm_georeferenced_model.ply",
            "outputs/odm/**/odm_georeferencing/odm_georeferenced_model.laz",
        ],
        "result_json_candidates": [
            "outputs/odm/**/odm_orthophoto.tif",
            "outputs/odm/**/odm_run.log",
        ],
        "metadata_json_candidates": [],
        "implementation_files": [
            "app/odm_service.py",
        ],
        "truthfulness_note": "ODM status is inferred from actual ODM artifacts. Real ODM requires Docker and a generated odm_orthophoto.tif.",
    },
    "reconstruction": {
        "module_name": "reconstruction",
        "display_name": "360° 视频 / 三维重建预处理",
        "expected_outputs": [
            "outputs/reconstruction/reconstruction_result.json",
            "outputs/reconstruction/keyframes_preview.jpg",
            "outputs/reconstruction/features_preview.jpg",
            "outputs/reconstruction/matches_preview.jpg",
            "outputs/reconstruction/camera_trajectory.jpg",
            "outputs/reconstruction/point_cloud.ply",
        ],
        "result_json_candidates": [
            "outputs/reconstruction/reconstruction_result.json",
        ],
        "metadata_json_candidates": [],
        "implementation_files": [
            "app/reconstruction_service.py",
        ],
        "truthfulness_note": "Reconstruction status reflects lightweight preview preprocessing rather than full SfM/MVS reconstruction.",
    },
    "scene_description": {
        "module_name": "scene_description",
        "display_name": "AI 灾情描述",
        "expected_outputs": [
            "outputs/reports/scene_description.md",
        ],
        "result_json_candidates": [
            "outputs/reports/scene_description.md",
        ],
        "metadata_json_candidates": [],
        "implementation_files": [
            "app/scene_description_service.py",
        ],
        "truthfulness_note": "Scene description is a generated markdown report or rule-based narrative, not a claim of automatic rescue understanding.",
    },
    "report_export": {
        "module_name": "report_export",
        "display_name": "综合报告导出",
        "expected_outputs": [
            "outputs/reports/final_report.md",
            "outputs/reports/final_report.html",
        ],
        "result_json_candidates": [
            "outputs/reports/final_report.md",
            "outputs/reports/final_report.html",
        ],
        "metadata_json_candidates": [],
        "implementation_files": [
            "app/report_export_service.py",
        ],
        "truthfulness_note": "Report export is a document assembly step based on existing local artifacts. It does not imply all upstream modules succeeded.",
    },
    "detection_backend_registry": {
        "module_name": "detection_backend_registry",
        "display_name": "检测后端注册表",
        "expected_outputs": [],
        "result_json_candidates": [],
        "metadata_json_candidates": [],
        "implementation_files": [
            "app/detection_backend_registry.py",
        ],
        "truthfulness_note": "This registry manages detection capability metadata. It is not a runtime result.",
        "reference_only": True,
    },
    "decision_reference_registry": {
        "module_name": "decision_reference_registry",
        "display_name": "决策参考注册表",
        "expected_outputs": [],
        "result_json_candidates": [],
        "metadata_json_candidates": [],
        "implementation_files": [
            "app/decision_reference_registry.py",
        ],
        "truthfulness_note": "This registry manages decision-layer references. It is not a runtime result.",
        "reference_only": True,
    },
}


_JSON_ERROR_KEY = "_json_error"


def _normalize_root(root_dir=None):
    return Path(root_dir).resolve() if root_dir else ROOT_DIR


def _resolve_paths(root_dir, relative_path):
    root = _normalize_root(root_dir)
    rel = str(relative_path).replace("\\", "/")
    if any(token in rel for token in ["*", "?", "["]):
        return sorted([path for path in root.glob(rel) if path.exists()])
    return [root / rel] if (root / rel).exists() else []


def _file_exists_any(root_dir, relative_path):
    return bool(_resolve_paths(root_dir, relative_path))


def file_exists(root_dir, relative_path):
    """Return True if a relative or absolute path exists."""
    path = Path(relative_path)
    if path.is_absolute():
        return path.exists()
    return _file_exists_any(root_dir, relative_path)


def safe_read_json(path):
    """Read JSON safely and return None or a structured parse error."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            _JSON_ERROR_KEY: True,
            "error": str(exc),
            "path": str(path),
        }


def _select_json_candidate(root_dir, candidate_paths):
    first_error = None
    for candidate in candidate_paths:
        for path in _resolve_paths(root_dir, candidate):
            data = safe_read_json(path)
            if data is None:
                continue
            if isinstance(data, dict) and data.get(_JSON_ERROR_KEY):
                if first_error is None:
                    first_error = (path, data)
                continue
            return path, data
    if first_error is not None:
        return first_error
    return None, None


def _collect_existing_paths(root_dir, candidate_paths):
    seen = []
    for candidate in candidate_paths:
        for path in _resolve_paths(root_dir, candidate):
            rel = str(path)
            if rel not in seen:
                seen.append(rel)
    return seen


def _module_implementation_exists(module_key, root_dir=None):
    config = MODULE_SCAN_TARGETS[module_key]
    for candidate in config.get("implementation_files", []):
        if file_exists(root_dir, candidate):
            return True
    return False


def _build_missing_expected_outputs(root_dir, config):
    missing = []
    for candidate in config.get("expected_outputs", []):
        if not _file_exists_any(root_dir, candidate):
            missing.append(candidate)
    return missing


def _default_not_run_result(module_key, root_dir=None):
    config = MODULE_SCAN_TARGETS[module_key]
    executed = False
    status = MODULE_STATUS["not_run"]
    if _module_implementation_exists(module_key, root_dir=root_dir):
        status = MODULE_STATUS["implemented_but_not_run"]
    return {
        "module_key": module_key,
        "module_name": config["module_name"],
        "display_name": config["display_name"],
        "status": status,
        "executed": executed,
        "success": None,
        "evidence_files": _collect_existing_paths(root_dir, config.get("expected_outputs", [])),
        "missing_expected_outputs": _build_missing_expected_outputs(root_dir, config),
        "result_json_path": None,
        "metadata_json_path": None,
        "error_code": None,
        "message": "No runtime artifact found under outputs/.",
        "capability_tags": [],
        "truthfulness_note": config["truthfulness_note"],
        "raw_result_summary": None,
    }


def _error_status_from_code(error_code):
    if not error_code:
        return MODULE_STATUS["executed_failed"]
    code = str(error_code).upper()
    missing_tokens = [
        "DEPENDENCY_MISSING",
        "MODEL_UNAVAILABLE",
        "DJI_SDK_NOT_AVAILABLE",
        "ODM_NOT_AVAILABLE",
        "YOLO_WEIGHTS_MISSING",
        "TRANSFORMER_SERVICE_MISSING",
        "MISSING",
    ]
    if any(token in code for token in missing_tokens):
        return MODULE_STATUS["dependency_missing"]
    return MODULE_STATUS["executed_failed"]


def _scan_detection(root_dir, config):
    result_path, result = _select_json_candidate(root_dir, config["result_json_candidates"])
    metadata_path, metadata = _select_json_candidate(root_dir, config["metadata_json_candidates"])
    if result is None:
        return _default_not_run_result("detection", root_dir=root_dir)

    if isinstance(result, dict) and result.get(_JSON_ERROR_KEY):
        data = _default_not_run_result("detection", root_dir=root_dir)
        data["status"] = MODULE_STATUS["executed_failed"]
        data["executed"] = True
        data["success"] = False
        data["result_json_path"] = str(result_path)
        data["message"] = f"JSON parse error: {result.get('error', '')}"
        data["raw_result_summary"] = result
        data["capability_tags"] = ["json_parse_error"]
        return data

    success = bool(result.get("success"))
    error_code = result.get("error_code")
    status = MODULE_STATUS["executed_success"] if success else _error_status_from_code(error_code)
    capability_tags = ["model_detection_output"] if result.get("is_model_output") else []
    if result.get("detection_mode") == "dual_backend_compare":
        capability_tags.append("dual_backend_consensus")
    if result.get("backend_key") == "transformer_rescuedet_argus":
        capability_tags.append("auxiliary_transformer_detection")
    return {
        "module_key": "detection",
        "module_name": config["module_name"],
        "display_name": config["display_name"],
        "status": MODULE_STATUS["real_model_output"] if success and result.get("is_model_output") else status,
        "executed": True,
        "success": success,
        "evidence_files": _collect_existing_paths(root_dir, config.get("expected_outputs", [])),
        "missing_expected_outputs": _build_missing_expected_outputs(root_dir, config),
        "result_json_path": str(result_path) if result_path else None,
        "metadata_json_path": str(metadata_path) if metadata_path else None,
        "error_code": error_code,
        "message": result.get("message", ""),
        "capability_tags": capability_tags,
        "truthfulness_note": result.get("truthfulness_note") or config["truthfulness_note"],
        "raw_result_summary": result,
    }


def _scan_transformer_detection(root_dir, config):
    result_path, result = _select_json_candidate(root_dir, config["result_json_candidates"])
    metadata_path, metadata = _select_json_candidate(root_dir, config["metadata_json_candidates"])
    if result is None:
        if _module_implementation_exists("transformer_detection", root_dir=root_dir):
            data = _default_not_run_result("transformer_detection", root_dir=root_dir)
            data["status"] = MODULE_STATUS["implemented_but_not_run"]
            data["truthfulness_note"] = config["truthfulness_note"]
            return data
        return _default_not_run_result("transformer_detection", root_dir=root_dir)

    if isinstance(result, dict) and result.get(_JSON_ERROR_KEY):
        data = _default_not_run_result("transformer_detection", root_dir=root_dir)
        data["status"] = MODULE_STATUS["executed_failed"]
        data["executed"] = True
        data["success"] = False
        data["result_json_path"] = str(result_path)
        data["message"] = f"JSON parse error: {result.get('error', '')}"
        data["raw_result_summary"] = result
        data["capability_tags"] = ["json_parse_error"]
        return data

    success = bool(result.get("success"))
    error_code = result.get("error_code")
    status = MODULE_STATUS["real_model_output"] if success else _error_status_from_code(error_code)
    capability_tags = ["auxiliary_transformer_detection"]
    if any(target.get("class_name") == "human_candidate" for target in result.get("targets", [])):
        capability_tags.append("human_candidate_requires_review")
    return {
        "module_key": "transformer_detection",
        "module_name": config["module_name"],
        "display_name": config["display_name"],
        "status": status,
        "executed": True,
        "success": success,
        "evidence_files": _collect_existing_paths(root_dir, config.get("expected_outputs", [])),
        "missing_expected_outputs": _build_missing_expected_outputs(root_dir, config),
        "result_json_path": str(result_path) if result_path else None,
        "metadata_json_path": str(metadata_path) if metadata_path else None,
        "error_code": error_code,
        "message": result.get("message", ""),
        "capability_tags": capability_tags,
        "truthfulness_note": result.get("truthfulness_note") or config["truthfulness_note"],
        "raw_result_summary": result,
    }


def _scan_segmentation(root_dir, config):
    result_path, result = _select_json_candidate(root_dir, config["result_json_candidates"])
    metadata_path, metadata = _select_json_candidate(root_dir, config["metadata_json_candidates"])
    if result is None:
        if _module_implementation_exists("segmentation", root_dir=root_dir):
            data = _default_not_run_result("segmentation", root_dir=root_dir)
            data["status"] = MODULE_STATUS["implemented_but_not_run"]
            return data
        return _default_not_run_result("segmentation", root_dir=root_dir)

    if isinstance(result, dict) and result.get(_JSON_ERROR_KEY):
        data = _default_not_run_result("segmentation", root_dir=root_dir)
        data["status"] = MODULE_STATUS["executed_failed"]
        data["executed"] = True
        data["success"] = False
        data["result_json_path"] = str(result_path)
        data["message"] = f"JSON parse error: {result.get('error', '')}"
        data["raw_result_summary"] = result
        data["capability_tags"] = ["json_parse_error"]
        return data

    source = result.get("segmentation_source") or result.get("source_metadata") or {}
    source_type = str(source.get("source_type") or source.get("source") or "").lower()
    is_model_prediction = bool(source.get("is_model_prediction"))
    success = bool(result.get("success", True))
    error_code = result.get("error_code")
    if source_type in {"uploaded_mask", "demo_fallback"}:
        status = MODULE_STATUS["executed_success"] if success is not False else _error_status_from_code(error_code)
        capability_tags = ["uploaded_or_demo_mask_not_model_prediction"]
    elif source_type == "auto_model" and is_model_prediction:
        status = MODULE_STATUS["real_model_output"] if success is not False else _error_status_from_code(error_code)
        capability_tags = ["auto_segmentation_model_output"]
    elif source_type == "auto_model" and not is_model_prediction:
        status = MODULE_STATUS["executed_failed"] if success is False else MODULE_STATUS["implemented_but_not_run"]
        capability_tags = ["auto_segmentation_not_verified"]
    else:
        status = MODULE_STATUS["executed_success"] if success else _error_status_from_code(error_code)
        capability_tags = []
    return {
        "module_key": "segmentation",
        "module_name": config["module_name"],
        "display_name": config["display_name"],
        "status": status,
        "executed": status not in {MODULE_STATUS["not_run"], MODULE_STATUS["implemented_but_not_run"], MODULE_STATUS["reference_only"]},
        "success": None if status in {MODULE_STATUS["not_run"], MODULE_STATUS["implemented_but_not_run"], MODULE_STATUS["reference_only"]} else bool(success),
        "evidence_files": _collect_existing_paths(root_dir, config.get("expected_outputs", [])),
        "missing_expected_outputs": _build_missing_expected_outputs(root_dir, config),
        "result_json_path": str(result_path) if result_path else None,
        "metadata_json_path": str(metadata_path) if metadata_path else None,
        "error_code": error_code,
        "message": result.get("message", ""),
        "capability_tags": capability_tags,
        "truthfulness_note": source.get("truthfulness_note") or result.get("truthfulness_note") or config["truthfulness_note"],
        "raw_result_summary": result,
    }


def _scan_thermal(root_dir, config):
    result_path, result = _select_json_candidate(root_dir, config["result_json_candidates"])
    if result is None:
        if _module_implementation_exists("thermal", root_dir=root_dir):
            data = _default_not_run_result("thermal", root_dir=root_dir)
            data["status"] = MODULE_STATUS["implemented_but_not_run"]
            return data
        return _default_not_run_result("thermal", root_dir=root_dir)

    if isinstance(result, dict) and result.get(_JSON_ERROR_KEY):
        data = _default_not_run_result("thermal", root_dir=root_dir)
        data["status"] = MODULE_STATUS["executed_failed"]
        data["executed"] = True
        data["success"] = False
        data["result_json_path"] = str(result_path)
        data["message"] = f"JSON parse error: {result.get('error', '')}"
        data["raw_result_summary"] = result
        data["capability_tags"] = ["json_parse_error"]
        return data

    mode = str(result.get("thermal_mode", "")).lower()
    is_real = bool(result.get("is_real_temperature_measurement"))
    success = bool(result.get("success", True))
    error_code = result.get("error_code")
    if mode in {"simulated", "simulated_thermal"}:
        status = MODULE_STATUS["simulated_result"]
        capability_tags = ["not_real_temperature"]
    elif mode in {"radiometric", "radiometric_thermal"} and is_real:
        temp_path = result.get("temperature_matrix_path")
        if temp_path and not file_exists(root_dir, temp_path):
            status = MODULE_STATUS["executed_failed"]
            capability_tags = ["real_temperature_matrix_missing_artifact"]
            error_code = error_code or "MISSING_OUTPUT_ARTIFACT"
            success = False
        else:
            status = MODULE_STATUS["real_measurement"]
            capability_tags = ["real_temperature_matrix"]
    elif mode in {"radiometric", "radiometric_thermal"}:
        status = _error_status_from_code(error_code)
        capability_tags = ["radiometric_attempt_failed"]
    elif mode == "infrared_detection":
        status = MODULE_STATUS["preview_only"]
        capability_tags = ["infrared_detection_placeholder"]
    else:
        status = MODULE_STATUS["executed_success"] if success else _error_status_from_code(error_code)
        capability_tags = []
    return {
        "module_key": "thermal",
        "module_name": config["module_name"],
        "display_name": config["display_name"],
        "status": status,
        "executed": status not in {MODULE_STATUS["not_run"], MODULE_STATUS["implemented_but_not_run"], MODULE_STATUS["reference_only"]},
        "success": None if status in {MODULE_STATUS["not_run"], MODULE_STATUS["implemented_but_not_run"], MODULE_STATUS["reference_only"]} else bool(success),
        "evidence_files": _collect_existing_paths(root_dir, config.get("expected_outputs", [])),
        "missing_expected_outputs": _build_missing_expected_outputs(root_dir, config),
        "result_json_path": str(result_path) if result_path else None,
        "metadata_json_path": None,
        "error_code": error_code,
        "message": result.get("error") or result.get("message", ""),
        "capability_tags": capability_tags,
        "truthfulness_note": result.get("truthfulness_note") or config["truthfulness_note"],
        "raw_result_summary": result,
    }


def _scan_orthomosaic(root_dir, config):
    result_path, result = _select_json_candidate(root_dir, config["result_json_candidates"])
    if result is None:
        if _resolve_paths(root_dir, "outputs/odm/**/odm_orthophoto.tif"):
            return {
                "module_key": "orthomosaic",
                "module_name": config["module_name"],
                "display_name": config["display_name"],
                "status": MODULE_STATUS["real_measurement"],
                "executed": True,
                "success": True,
                "evidence_files": _collect_existing_paths(root_dir, config.get("expected_outputs", [])),
                "missing_expected_outputs": _build_missing_expected_outputs(root_dir, config),
                "result_json_path": None,
                "metadata_json_path": None,
                "error_code": None,
                "message": "Real ODM orthophoto detected.",
                "capability_tags": ["real_odm_orthophoto"],
                "truthfulness_note": config["truthfulness_note"],
                "raw_result_summary": {"real_odm_orthophoto": True},
            }
        if _module_implementation_exists("orthomosaic", root_dir=root_dir):
            data = _default_not_run_result("orthomosaic", root_dir=root_dir)
            data["status"] = MODULE_STATUS["implemented_but_not_run"]
            return data
        return _default_not_run_result("orthomosaic", root_dir=root_dir)

    if isinstance(result, dict) and result.get(_JSON_ERROR_KEY):
        data = _default_not_run_result("orthomosaic", root_dir=root_dir)
        data["status"] = MODULE_STATUS["executed_failed"]
        data["executed"] = True
        data["success"] = False
        data["result_json_path"] = str(result_path)
        data["message"] = f"JSON parse error: {result.get('error', '')}"
        data["raw_result_summary"] = result
        data["capability_tags"] = ["json_parse_error"]
        return data

    success = bool(result.get("success", True))
    fallback_reason = str(result.get("fallback_reason", "")).lower()
    mode = str(result.get("mode") or result.get("method") or "").lower()
    truthfulness_note = str(result.get("truthfulness_note", "")).lower()
    has_real_odm = bool(_resolve_paths(root_dir, "outputs/odm/**/odm_orthophoto.tif"))
    is_preview = (
        "fast" in mode
        or "preview" in mode
        or "preview" in fallback_reason
        or "预览" in fallback_reason
        or "单张图像" in fallback_reason
        or "not a real odm" in truthfulness_note
    )
    if is_preview:
        status = MODULE_STATUS["preview_only"]
        capability_tags = ["not_real_orthomosaic"]
    elif has_real_odm or bool(result.get("real_odm_success")):
        status = MODULE_STATUS["real_measurement"]
        capability_tags = ["real_odm_orthophoto"]
    else:
        status = _error_status_from_code(result.get("error_code"))
        capability_tags = []
    return {
        "module_key": "orthomosaic",
        "module_name": config["module_name"],
        "display_name": config["display_name"],
        "status": status,
        "executed": status not in {MODULE_STATUS["not_run"], MODULE_STATUS["implemented_but_not_run"], MODULE_STATUS["reference_only"]},
        "success": None if status in {MODULE_STATUS["not_run"], MODULE_STATUS["implemented_but_not_run"], MODULE_STATUS["reference_only"]} else bool(success),
        "evidence_files": _collect_existing_paths(root_dir, config.get("expected_outputs", [])),
        "missing_expected_outputs": _build_missing_expected_outputs(root_dir, config),
        "result_json_path": str(result_path) if result_path else None,
        "metadata_json_path": None,
        "error_code": result.get("error_code"),
        "message": result.get("message", "") or result.get("fallback_reason", ""),
        "capability_tags": capability_tags,
        "truthfulness_note": result.get("truthfulness_note") or config["truthfulness_note"],
        "raw_result_summary": result,
    }


def _scan_odm(root_dir, config):
    odf_paths = _resolve_paths(root_dir, "outputs/odm/**/odm_orthophoto.tif")
    log_paths = _resolve_paths(root_dir, "outputs/odm/**/odm_run.log")
    if odf_paths:
        first = odf_paths[0]
        return {
            "module_key": "odm",
            "module_name": config["module_name"],
            "display_name": config["display_name"],
            "status": MODULE_STATUS["executed_success"],
            "executed": True,
            "success": True,
            "evidence_files": [str(p) for p in odf_paths] + [str(p) for p in log_paths],
            "missing_expected_outputs": [p for p in config.get("expected_outputs", []) if not _resolve_paths(root_dir, p)],
            "result_json_path": str(first),
            "metadata_json_path": None,
            "error_code": None,
            "message": "odm_orthophoto.tif exists.",
            "capability_tags": ["real_odm_orthophoto"],
            "truthfulness_note": config["truthfulness_note"],
            "raw_result_summary": {"orthophoto_tif": str(first)},
        }

    if log_paths:
        log_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in log_paths if path.exists())
        lower = log_text.lower()
        if "docker 不可用" in log_text or "docker unavailable" in lower:
            status = MODULE_STATUS["dependency_missing"]
            error_code = "DOCKER_UNAVAILABLE"
        elif "permission denied" in lower:
            status = MODULE_STATUS["dependency_missing"]
            error_code = "DOCKER_PERMISSION_DENIED"
        elif "odm 运行失败" in log_text or "failed" in lower:
            status = MODULE_STATUS["executed_failed"]
            error_code = "ODM_RUN_FAILED"
        else:
            status = MODULE_STATUS["executed_failed"]
            error_code = "ODM_NOT_AVAILABLE"
        return {
            "module_key": "odm",
            "module_name": config["module_name"],
            "display_name": config["display_name"],
            "status": status,
            "executed": True,
            "success": False,
            "evidence_files": [str(p) for p in odf_paths] + [str(p) for p in log_paths],
            "missing_expected_outputs": [p for p in config.get("expected_outputs", []) if not _resolve_paths(root_dir, p)],
            "result_json_path": str(log_paths[0]),
            "metadata_json_path": None,
            "error_code": error_code,
            "message": log_text[:600],
            "capability_tags": [],
            "truthfulness_note": config["truthfulness_note"],
            "raw_result_summary": {"odm_run_log": str(log_paths[0])},
        }

    if _module_implementation_exists("odm", root_dir=root_dir):
        data = _default_not_run_result("odm", root_dir=root_dir)
        data["status"] = MODULE_STATUS["implemented_but_not_run"]
        return data
    return _default_not_run_result("odm", root_dir=root_dir)


def _scan_reconstruction(root_dir, config):
    result_path, result = _select_json_candidate(root_dir, config["result_json_candidates"])
    if result is None:
        if _module_implementation_exists("reconstruction", root_dir=root_dir):
            data = _default_not_run_result("reconstruction", root_dir=root_dir)
            data["status"] = MODULE_STATUS["implemented_but_not_run"]
            return data
        return _default_not_run_result("reconstruction", root_dir=root_dir)

    if isinstance(result, dict) and result.get(_JSON_ERROR_KEY):
        data = _default_not_run_result("reconstruction", root_dir=root_dir)
        data["status"] = MODULE_STATUS["executed_failed"]
        data["executed"] = True
        data["success"] = False
        data["result_json_path"] = str(result_path)
        data["message"] = f"JSON parse error: {result.get('error', '')}"
        data["raw_result_summary"] = result
        data["capability_tags"] = ["json_parse_error"]
        return data

    success = result.get("status", "").lower().startswith("completed") or bool(result.get("success", True))
    status = MODULE_STATUS["preview_only"] if success else _error_status_from_code(result.get("error_code"))
    return {
        "module_key": "reconstruction",
        "module_name": config["module_name"],
        "display_name": config["display_name"],
        "status": status,
        "executed": status not in {MODULE_STATUS["not_run"], MODULE_STATUS["implemented_but_not_run"], MODULE_STATUS["reference_only"]},
        "success": None if status in {MODULE_STATUS["not_run"], MODULE_STATUS["implemented_but_not_run"], MODULE_STATUS["reference_only"]} else bool(success),
        "evidence_files": _collect_existing_paths(root_dir, config.get("expected_outputs", [])),
        "missing_expected_outputs": _build_missing_expected_outputs(root_dir, config),
        "result_json_path": str(result_path) if result_path else None,
        "metadata_json_path": None,
        "error_code": result.get("error_code"),
        "message": result.get("status", "") or result.get("message", ""),
        "capability_tags": ["preview_only", "not_full_3d_reconstruction"] if success else [],
        "truthfulness_note": result.get("truthfulness_note") or config["truthfulness_note"],
        "raw_result_summary": result,
    }


def _scan_scene_description(root_dir, config):
    path = _resolve_paths(root_dir, config["result_json_candidates"][0] if config["result_json_candidates"] else [])
    if path:
        return {
            "module_key": "scene_description",
            "module_name": config["module_name"],
            "display_name": config["display_name"],
            "status": MODULE_STATUS["executed_success"],
            "executed": True,
            "success": True,
            "evidence_files": [str(p) for p in path],
            "missing_expected_outputs": [],
            "result_json_path": str(path[0]),
            "metadata_json_path": None,
            "error_code": None,
            "message": "scene_description.md exists.",
            "capability_tags": ["markdown_scene_description"],
            "truthfulness_note": config["truthfulness_note"],
            "raw_result_summary": {"scene_description_path": str(path[0])},
        }
    if _module_implementation_exists("scene_description", root_dir=root_dir):
        data = _default_not_run_result("scene_description", root_dir=root_dir)
        data["status"] = MODULE_STATUS["implemented_but_not_run"]
        return data
    return _default_not_run_result("scene_description", root_dir=root_dir)


def _scan_report_export(root_dir, config):
    md_paths = _resolve_paths(root_dir, "outputs/reports/final_report.md")
    html_paths = _resolve_paths(root_dir, "outputs/reports/final_report.html")
    if md_paths or html_paths:
        return {
            "module_key": "report_export",
            "module_name": config["module_name"],
            "display_name": config["display_name"],
            "status": MODULE_STATUS["executed_success"],
            "executed": True,
            "success": True,
            "evidence_files": [str(p) for p in md_paths + html_paths],
            "missing_expected_outputs": [p for p in config.get("expected_outputs", []) if not _resolve_paths(root_dir, p)],
            "result_json_path": str(md_paths[0] if md_paths else html_paths[0]),
            "metadata_json_path": None,
            "error_code": None,
            "message": "final_report markdown/html exists.",
            "capability_tags": ["final_report_artifact"],
            "truthfulness_note": config["truthfulness_note"],
            "raw_result_summary": {"final_report_md": str(md_paths[0]) if md_paths else "", "final_report_html": str(html_paths[0]) if html_paths else ""},
        }
    if _module_implementation_exists("report_export", root_dir=root_dir):
        data = _default_not_run_result("report_export", root_dir=root_dir)
        data["status"] = MODULE_STATUS["implemented_but_not_run"]
        return data
    return _default_not_run_result("report_export", root_dir=root_dir)


def _scan_path_planning(root_dir, config):
    result_path, result = _select_json_candidate(root_dir, config["result_json_candidates"])
    if result is None:
        if _module_implementation_exists("path_planning", root_dir=root_dir):
            data = _default_not_run_result("path_planning", root_dir=root_dir)
            data["status"] = MODULE_STATUS["implemented_but_not_run"]
            return data
        return _default_not_run_result("path_planning", root_dir=root_dir)

    if isinstance(result, dict) and result.get(_JSON_ERROR_KEY):
        data = _default_not_run_result("path_planning", root_dir=root_dir)
        data["status"] = MODULE_STATUS["executed_failed"]
        data["executed"] = True
        data["success"] = False
        data["result_json_path"] = str(result_path)
        data["message"] = f"JSON parse error: {result.get('error', '')}"
        data["raw_result_summary"] = result
        data["capability_tags"] = ["json_parse_error"]
        return data

    success = bool(result.get("success", True))
    error_code = result.get("error_code")
    status = MODULE_STATUS["executed_success"] if success else _error_status_from_code(error_code)
    capability_tags = ["image_plane_reference_path"]
    if result.get("force_path_planning"):
        capability_tags.append("forced_debug_path")
    if result.get("path") or result.get("path_points") or result.get("path_result"):
        capability_tags.append("path_coordinates_available")
    return {
        "module_key": "path_planning",
        "module_name": config["module_name"],
        "display_name": config["display_name"],
        "status": status,
        "executed": True,
        "success": success,
        "evidence_files": _collect_existing_paths(root_dir, config.get("expected_outputs", [])),
        "missing_expected_outputs": _build_missing_expected_outputs(root_dir, config),
        "result_json_path": str(result_path) if result_path else None,
        "metadata_json_path": None,
        "error_code": error_code,
        "message": result.get("message", "") or result.get("summary", ""),
        "capability_tags": capability_tags,
        "truthfulness_note": result.get("truthfulness_note") or config["truthfulness_note"],
        "raw_result_summary": result,
    }


def _scan_decision_fusion(root_dir, config):
    result_path, result = _select_json_candidate(root_dir, config["result_json_candidates"])
    if result is None:
        if _module_implementation_exists("decision_fusion", root_dir=root_dir):
            data = _default_not_run_result("decision_fusion", root_dir=root_dir)
            data["status"] = MODULE_STATUS["implemented_but_not_run"]
            return data
        return _default_not_run_result("decision_fusion", root_dir=root_dir)

    if isinstance(result, dict) and result.get(_JSON_ERROR_KEY):
        data = _default_not_run_result("decision_fusion", root_dir=root_dir)
        data["status"] = MODULE_STATUS["executed_failed"]
        data["executed"] = True
        data["success"] = False
        data["result_json_path"] = str(result_path)
        data["message"] = f"JSON parse error: {result.get('error', '')}"
        data["raw_result_summary"] = result
        data["capability_tags"] = ["json_parse_error"]
        return data

    success = bool(result.get("success", True))
    error_code = result.get("error_code")
    status = MODULE_STATUS["executed_success"] if success else _error_status_from_code(error_code)
    capability_tags = ["image_plane_decision_fusion"]
    if result.get("search_priority_map") is not None:
        capability_tags.append("search_priority_map")
    if result.get("coverage_score") is not None:
        capability_tags.append("coverage_score")
    if result.get("damage_impact_score") is not None:
        capability_tags.append("damage_impact_score")
    return {
        "module_key": "decision_fusion",
        "module_name": config["module_name"],
        "display_name": config["display_name"],
        "status": status,
        "executed": True,
        "success": success,
        "evidence_files": _collect_existing_paths(root_dir, config.get("expected_outputs", [])),
        "missing_expected_outputs": _build_missing_expected_outputs(root_dir, config),
        "result_json_path": str(result_path) if result_path else None,
        "metadata_json_path": None,
        "error_code": error_code,
        "message": result.get("message", "") or result.get("summary_markdown", ""),
        "capability_tags": capability_tags,
        "truthfulness_note": result.get("truthfulness_note") or config["truthfulness_note"],
        "raw_result_summary": result,
    }


def _scan_registry_only(module_key, root_dir=None):
    config = MODULE_SCAN_TARGETS[module_key]
    evidence = _collect_existing_paths(root_dir, config.get("expected_outputs", []))
    return {
        "module_key": module_key,
        "module_name": config["module_name"],
        "display_name": config["display_name"],
        "status": MODULE_STATUS["reference_only"],
        "executed": False,
        "success": None,
        "evidence_files": evidence,
        "missing_expected_outputs": _build_missing_expected_outputs(root_dir, config),
        "result_json_path": None,
        "metadata_json_path": None,
        "error_code": None,
        "message": "Registry/Reference management module. Not a runtime result.",
        "capability_tags": ["reference_management"],
        "truthfulness_note": config["truthfulness_note"],
        "raw_result_summary": None,
    }


def scan_single_module(module_key, root_dir=None):
    """Scan one module status from outputs and JSON artifacts."""
    if module_key not in MODULE_SCAN_TARGETS:
        raise ModuleStatusScannerError(f"Unknown module key: {module_key}")
    config = MODULE_SCAN_TARGETS[module_key]
    root_dir = _normalize_root(root_dir)

    if module_key in {"detection_backend_registry", "decision_reference_registry"} or config.get("reference_only"):
        return _scan_registry_only(module_key, root_dir=root_dir)
    if module_key == "detection":
        return _scan_detection(root_dir, config)
    if module_key == "transformer_detection":
        return _scan_transformer_detection(root_dir, config)
    if module_key == "segmentation":
        return _scan_segmentation(root_dir, config)
    if module_key == "thermal":
        return _scan_thermal(root_dir, config)
    if module_key == "orthomosaic":
        return _scan_orthomosaic(root_dir, config)
    if module_key == "odm":
        return _scan_odm(root_dir, config)
    if module_key == "reconstruction":
        return _scan_reconstruction(root_dir, config)
    if module_key == "scene_description":
        return _scan_scene_description(root_dir, config)
    if module_key == "report_export":
        return _scan_report_export(root_dir, config)
    if module_key == "path_planning":
        return _scan_path_planning(root_dir, config)
    if module_key == "decision_fusion":
        return _scan_decision_fusion(root_dir, config)

    # Lightweight modules with no dedicated outputs yet.
    result_paths = _collect_existing_paths(root_dir, config.get("result_json_candidates", []))
    metadata_paths = _collect_existing_paths(root_dir, config.get("metadata_json_candidates", []))
    if not result_paths and not metadata_paths:
        return _default_not_run_result(module_key, root_dir=root_dir)
    return _default_not_run_result(module_key, root_dir=root_dir)


def scan_all_modules(root_dir=None):
    """Scan all known modules and summarize their status."""
    root_dir = _normalize_root(root_dir)
    modules = {}
    summary = {
        "executed_success_count": 0,
        "executed_failed_count": 0,
        "not_run_count": 0,
        "dependency_missing_count": 0,
        "simulated_result_count": 0,
        "real_model_output_count": 0,
        "real_measurement_count": 0,
        "preview_only_count": 0,
        "reference_only_count": 0,
    }

    success_like = {
        MODULE_STATUS["executed_success"],
        MODULE_STATUS["real_model_output"],
        MODULE_STATUS["real_measurement"],
        MODULE_STATUS["simulated_result"],
        MODULE_STATUS["preview_only"],
    }
    for module_key in MODULE_SCAN_TARGETS:
        module_result = scan_single_module(module_key, root_dir=root_dir)
        modules[module_key] = module_result
        status = module_result.get("status", MODULE_STATUS["unknown"])
        if status in success_like:
            summary["executed_success_count"] += 1
        if status == MODULE_STATUS["executed_failed"]:
            summary["executed_failed_count"] += 1
        if status == MODULE_STATUS["not_run"]:
            summary["not_run_count"] += 1
        if status == MODULE_STATUS["dependency_missing"]:
            summary["dependency_missing_count"] += 1
        if status == MODULE_STATUS["simulated_result"]:
            summary["simulated_result_count"] += 1
        if status == MODULE_STATUS["real_model_output"]:
            summary["real_model_output_count"] += 1
        if status == MODULE_STATUS["real_measurement"]:
            summary["real_measurement_count"] += 1
        if status == MODULE_STATUS["preview_only"]:
            summary["preview_only_count"] += 1
        if status == MODULE_STATUS["reference_only"]:
            summary["reference_only_count"] += 1
        if status == MODULE_STATUS["implemented_but_not_run"]:
            summary["not_run_count"] += 1

    return {
        "success": True,
        "root_dir": str(root_dir),
        "modules": modules,
        "summary": summary,
        "truthfulness_note": "Module status is inferred from local output artifacts and result metadata, not from code existence alone.",
    }


def _format_module_block(module_result):
    evidence = module_result.get("evidence_files", [])
    evidence_text = "\n".join(f"- {item}" for item in evidence) if evidence else "- 无证据文件"
    error_text = module_result.get("error_code") or module_result.get("message") or "无错误信息"
    truthfulness = module_result.get("truthfulness_note") or "未提供真实性说明。"
    return (
        f"### {module_result.get('display_name', module_result.get('module_key'))}\n"
        f"- 模块键：{module_result.get('module_key')}\n"
        f"- 状态：{module_result.get('status')}\n"
        f"- 是否执行：{'是' if module_result.get('executed') else '否'}\n"
        f"- 是否成功：{'是' if module_result.get('success') is True else ('否' if module_result.get('success') is False else '未知')}\n"
        f"- 证据文件：\n{evidence_text}\n"
        f"- 错误信息：{error_text}\n"
        f"- 真实性说明：{truthfulness}\n"
    )


def format_module_status_markdown(scan_result):
    """Format a module scan result or a full scan into Chinese Markdown."""
    if not isinstance(scan_result, dict):
        return "## 模块执行状态扫描\n\n扫描结果格式无效。"

    if "modules" not in scan_result:
        return "## 模块执行状态扫描\n\n" + _format_module_block(scan_result)

    modules = scan_result.get("modules", {})
    groups = {
        "### 已成功执行": [],
        "### 执行失败 / 依赖缺失": [],
        "### 模拟 / 预览结果": [],
        "### 未执行": [],
        "### 参考 / 注册表模块": [],
    }
    for module_result in modules.values():
        status = module_result.get("status", MODULE_STATUS["unknown"])
        if status in {MODULE_STATUS["executed_success"], MODULE_STATUS["real_model_output"], MODULE_STATUS["real_measurement"]}:
            groups["### 已成功执行"].append(module_result)
        elif status in {MODULE_STATUS["executed_failed"], MODULE_STATUS["dependency_missing"], MODULE_STATUS["unknown"]}:
            groups["### 执行失败 / 依赖缺失"].append(module_result)
        elif status in {MODULE_STATUS["simulated_result"], MODULE_STATUS["preview_only"]}:
            groups["### 模拟 / 预览结果"].append(module_result)
        elif status in {MODULE_STATUS["not_run"], MODULE_STATUS["implemented_but_not_run"]}:
            groups["### 未执行"].append(module_result)
        else:
            groups["### 参考 / 注册表模块"].append(module_result)

    lines = [
        "## 模块执行状态扫描",
        f"- 扫描根目录：{scan_result.get('root_dir', '')}",
        f"- 成功执行计数：{scan_result.get('summary', {}).get('executed_success_count', 0)}",
        f"- 执行失败计数：{scan_result.get('summary', {}).get('executed_failed_count', 0)}",
        f"- 未执行计数：{scan_result.get('summary', {}).get('not_run_count', 0)}",
        f"- 依赖缺失计数：{scan_result.get('summary', {}).get('dependency_missing_count', 0)}",
        f"- 模拟结果计数：{scan_result.get('summary', {}).get('simulated_result_count', 0)}",
        f"- 真实模型输出计数：{scan_result.get('summary', {}).get('real_model_output_count', 0)}",
        f"- 真实测温/实测计数：{scan_result.get('summary', {}).get('real_measurement_count', 0)}",
        f"- 预览结果计数：{scan_result.get('summary', {}).get('preview_only_count', 0)}",
        f"- 参考模块计数：{scan_result.get('summary', {}).get('reference_only_count', 0)}",
    ]
    for title, items in groups.items():
        if not items:
            continue
        lines.append("")
        lines.append(title)
        for item in items:
            lines.append(_format_module_block(item))
    lines.append("")
    lines.append("以上状态根据 outputs/ 运行产物和 JSON 元数据推断，不根据代码文件存在与否判断。")
    return "\n".join(lines)


def save_module_status_report(scan_result=None, output_dir=None):
    """Save module status report markdown and json."""
    if scan_result is None:
        scan_result = scan_all_modules()
    output_root = Path(output_dir) if output_dir else OUTPUT_ROOT / "reports"
    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "module_status_report.md"
    json_path = output_root / "module_status_report.json"
    markdown_path.write_text(format_module_status_markdown(scan_result), encoding="utf-8")
    json_path.write_text(json.dumps(scan_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "success": True,
        "markdown_path": str(markdown_path),
        "json_path": str(json_path),
    }
