"""External source-level impact assessment status helpers.

This module never fabricates SKAI or InaSAFE outputs. It only reports real
source/dependency/output availability and keeps legacy internal scores out of
the final S2-S3 display path.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT_DIR / "outputs" / "external_impact_assessment"


SKAI_SOURCE_CANDIDATES = [
    Path(os.environ.get("SKAI_REPO_PATH", "")),
    ROOT_DIR / "external_sources" / "skai",
    ROOT_DIR / "third_party" / "skai",
    ROOT_DIR / "external_integrations" / "source_repos" / "skai",
]

INASAFE_SOURCE_CANDIDATES = [
    Path(os.environ.get("INASAFE_REPO_PATH", "")),
    ROOT_DIR / "external_sources" / "inasafe",
    ROOT_DIR / "third_party" / "inasafe",
    ROOT_DIR / "external_integrations" / "source_repos" / "inasafe",
]
INASAFE_PYTHON_CANDIDATES = [
    Path(os.environ.get("INASAFE_PYTHON", "")),
    Path.home() / "miniconda3" / "envs" / "aerorescue-inasafe" / "bin" / "python",
    Path.home() / "anaconda3" / "envs" / "aerorescue-inasafe" / "bin" / "python",
]
INASAFE_CONDA_ENV_NAME = os.environ.get("INASAFE_CONDA_ENV", "aerorescue-inasafe")

SKAI_REQUIRED_SOURCE_FILES = [
    "src/skai/buildings.py",
    "src/skai/generate_examples.py",
    "src/skai/model/inference.py",
    "src/skai/model/inference_lib.py",
]

INASAFE_REQUIRED_SOURCE_FILES = [
    "safe/impact_function/impact_function.py",
    "safe/impact_function/impact_function_utilities.py",
    "safe/report/impact_report.py",
    "safe/definitions/hazard.py",
    "safe/definitions/exposure.py",
]

SKAI_OUTPUT_PATTERNS = [
    "predictions/*.geojson",
    "predictions/*.json",
    "predictions/*.csv",
    "*.geojson",
    "*.json",
    "*.csv",
]
SKAI_CONFIG_CANDIDATES = [
    Path(os.environ.get("SKAI_CONFIG_PATH", "")),
    ROOT_DIR / "configs" / "skai_config.json",
    ROOT_DIR / "external_integrations" / "damage_impact" / "skai" / "runner_config.json",
]
SKAI_CHECKPOINT_CANDIDATES = [
    Path(os.environ.get("SKAI_CHECKPOINT_PATH", "")),
    ROOT_DIR / "models" / "skai" / "checkpoint",
    ROOT_DIR / "models" / "skai" / "model.ckpt",
    ROOT_DIR / "models" / "skai" / "saved_model",
]
SKAI_INPUT_CANDIDATES = [
    Path(os.environ.get("SKAI_INPUT_PATH", "")),
    ROOT_DIR / "data" / "skai",
    ROOT_DIR / "outputs" / "external_impact_assessment" / "skai" / "inputs",
]
SKAI_PYTHON_CANDIDATES = [
    Path(os.environ.get("SKAI_PYTHON", "")),
    Path.home() / "miniconda3" / "envs" / "aerorescue-skai" / "bin" / "python",
    Path.home() / "anaconda3" / "envs" / "aerorescue-skai" / "bin" / "python",
]

INASAFE_OUTPUT_PATTERNS = [
    "reports/*.pdf",
    "reports/*.html",
    "reports/*.json",
    "impact_layers/*.gpkg",
    "impact_layers/*.geojson",
    "*.pdf",
    "*.html",
    "*.json",
    "*.gpkg",
    "*.geojson",
]


def _valid_candidates(candidates):
    return [path for path in candidates if str(path) not in {"", "."} and path.exists()]


def _first_repo_with_files(candidates, required_files):
    missing_by_candidate = []
    for candidate in _valid_candidates(candidates):
        missing = [rel for rel in required_files if not (candidate / rel).exists()]
        if not missing:
            return candidate.resolve(), []
        missing_by_candidate.append({"path": str(candidate), "missing": missing})
    return None, missing_by_candidate


def _find_outputs(output_dir, patterns):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    files = []
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.is_file() and path.stat().st_size > 0:
                files.append(str(path))
    return sorted(set(files))


def _python_module_available(module_name):
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _python_module_available_with_interpreter(module_name, python_path=None):
    if not python_path:
        return _python_module_available(module_name)
    try:
        result = subprocess.run(
            [str(python_path), "-c", f"__import__({module_name!r})"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        return result.returncode == 0
    except Exception:
        return False


def _first_python_with_modules(candidates, module_names):
    interpreter_candidates = [Path(sys.executable)] + [path for path in candidates if str(path) not in {"", "."}]
    for python_path in interpreter_candidates:
        if not python_path.exists():
            continue
        status = {
            module_name: _python_module_available_with_interpreter(module_name, python_path)
            for module_name in module_names
        }
        if all(status.values()):
            return python_path.resolve(), status
    fallback = Path(sys.executable)
    return fallback, {
        module_name: _python_module_available_with_interpreter(module_name, fallback)
        for module_name in module_names
    }


def _conda_env_module_status(env_name, module_names):
    conda = shutil.which("conda")
    if not conda or not env_name:
        return None, {}
    code = (
        "import json\n"
        f"mods={module_names!r}\n"
        "status={}\n"
        "for m in mods:\n"
        "    try:\n"
        "        __import__(m)\n"
        "        status[m]=True\n"
        "    except Exception:\n"
        "        status[m]=False\n"
        "print(json.dumps(status))\n"
    )
    try:
        result = subprocess.run(
            [conda, "run", "-n", env_name, "python", "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            return None, {}
        output = (result.stdout or "").strip().splitlines()[-1]
        return f"conda:{env_name}", json.loads(output)
    except Exception:
        return None, {}


def _available_path(candidates):
    for path in candidates:
        if str(path) not in {"", "."} and path.exists():
            return path.resolve()
    return None


def _cn_available(path):
    return "可用" if path else "缺失"


def _cn_dependency_status(dependency_status):
    return "可用" if dependency_status and all(dependency_status.values()) else "缺失"


def _runner_executed(output_dir):
    output_dir = Path(output_dir)
    marker = output_dir / "runner_status.json"
    if marker.exists():
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
            return bool(data.get("executed") or data.get("success") or data.get("completed"))
        except Exception:
            return True
    return False


def _safe_int(value):
    try:
        if value in {"", None, "unavailable"}:
            return "unavailable"
        return int(value)
    except Exception:
        return "unavailable"


def _damage_level(total, damaged, severe, destroyed):
    if total in {"unavailable", 0}:
        return "unavailable"
    ratio = float(damaged or 0) / max(float(total), 1.0)
    severe_ratio = float((severe or 0) + (destroyed or 0)) / max(float(total), 1.0)
    if severe_ratio >= 0.35 or ratio >= 0.65:
        return "Critical"
    if severe_ratio >= 0.18 or ratio >= 0.40:
        return "High"
    if ratio >= 0.15:
        return "Medium"
    return "Low"


def _load_skai_building_summary(outputs):
    """Read a SKAI-produced summary if present; otherwise return unavailable fields."""
    unavailable = {
        "building_total": "unavailable",
        "damaged_building_count": "unavailable",
        "severe_damage_building_count": "unavailable",
        "destroyed_building_count": "unavailable",
        "building_damage_level": "unavailable",
    }
    for output in outputs or []:
        path = Path(output)
        if path.suffix.lower() != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = data.get("building_damage_summary", data) if isinstance(data, dict) else {}
        total = _safe_int(summary.get("building_total", summary.get("total_buildings")))
        damaged = _safe_int(summary.get("damaged_building_count", summary.get("damaged_buildings")))
        severe = _safe_int(summary.get("severe_damage_building_count", summary.get("major_damage_buildings")))
        destroyed = _safe_int(summary.get("destroyed_building_count", summary.get("destroyed_buildings")))
        level = summary.get("building_damage_level") or _damage_level(total, damaged, severe, destroyed)
        return {
            "building_total": total,
            "damaged_building_count": damaged,
            "severe_damage_building_count": severe,
            "destroyed_building_count": destroyed,
            "building_damage_level": level,
        }
    return unavailable


def _load_inasafe_impact_summary(outputs):
    """Read an InaSAFE-produced summary if present; otherwise return unavailable fields."""
    unavailable = {
        "affected_roads": "unavailable",
        "affected_buildings": "unavailable",
        "affected_points": "unavailable",
        "high_risk_impact_areas": "unavailable",
        "output_report": "unavailable",
    }
    report_files = [
        str(path)
        for path in outputs or []
        if Path(path).suffix.lower() in {".pdf", ".html"}
    ]
    for output in outputs or []:
        path = Path(output)
        if path.suffix.lower() != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = data.get("impact_assessment_summary", data) if isinstance(data, dict) else {}
        return {
            "affected_roads": _safe_int(summary.get("affected_roads", summary.get("affected_road_count"))),
            "affected_buildings": _safe_int(summary.get("affected_buildings", summary.get("affected_building_count"))),
            "affected_points": _safe_int(summary.get("affected_points", summary.get("affected_point_count"))),
            "high_risk_impact_areas": _safe_int(summary.get("high_risk_impact_areas", summary.get("high_risk_area_count"))),
            "output_report": summary.get("output_report") or (report_files[0] if report_files else "unavailable"),
        }
    if report_files:
        unavailable["output_report"] = report_files[0]
    return unavailable


def assess_skai_source_level_status(output_dir=None):
    """Return conservative SKAI source-level integration status."""
    output_dir = Path(output_dir or OUTPUT_ROOT / "skai")
    source_root, missing_sources = _first_repo_with_files(SKAI_SOURCE_CANDIDATES, SKAI_REQUIRED_SOURCE_FILES)
    outputs = _find_outputs(output_dir, SKAI_OUTPUT_PATTERNS)
    config_path = _available_path(SKAI_CONFIG_CANDIDATES + [output_dir / "runner_config.json", output_dir / "config.json"])
    checkpoint_path = _available_path(SKAI_CHECKPOINT_CANDIDATES + [output_dir / "checkpoint", output_dir / "saved_model"])
    input_path = _available_path(SKAI_INPUT_CANDIDATES + [output_dir / "inputs"])
    runner_executed = _runner_executed(output_dir)
    dependency_python, dependency_status = _first_python_with_modules(
        SKAI_PYTHON_CANDIDATES,
        ["tensorflow", "geopandas", "rasterio", "apache_beam"],
    )
    unavailable_reasons = []
    if source_root is None:
        unavailable_reasons.append("SKAI source repository not found or required source files are missing.")
    missing_deps = [name for name, available in dependency_status.items() if not available]
    if missing_deps:
        unavailable_reasons.append(f"Missing SKAI Python dependencies: {', '.join(missing_deps)}.")
    if config_path is None:
        unavailable_reasons.append("SKAI runner config file is missing.")
    if checkpoint_path is None:
        unavailable_reasons.append("SKAI checkpoint/model artifact is missing.")
    if input_path is None:
        unavailable_reasons.append("SKAI-compatible input data is missing.")
    if not runner_executed:
        unavailable_reasons.append("SKAI adapter runner has not executed.")
    if not outputs:
        unavailable_reasons.append("No verified SKAI prediction output file was found.")
    real_output = (
        source_root is not None
        and not missing_deps
        and config_path is not None
        and checkpoint_path is not None
        and input_path is not None
        and runner_executed
        and bool(outputs)
    )
    building_summary = _load_skai_building_summary(outputs) if real_output else _load_skai_building_summary([])
    return {
        "module": "SKAI 外部源码级建筑灾损评估",
        "repository": "google-research/skai",
        "integration_method": "外部源码级集成 + adapter runner",
        "task_role": "建筑级灾损评估",
        "source_root": str(source_root) if source_root else "",
        "config_path": str(config_path) if config_path else "",
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "input_path": str(input_path) if input_path else "",
        "required_source_files": SKAI_REQUIRED_SOURCE_FILES,
        "missing_source_candidates": missing_sources,
        "dependency_status": dependency_status,
        "dependency_python": str(dependency_python),
        "output_dir": str(output_dir),
        "verified_output_files": outputs,
        "runner_executed": bool(runner_executed),
        "status": "real_output_verified" if real_output else "unavailable",
        "is_real_skai_output": bool(real_output),
        "run_status_display": {
            "skai_source": "已集成" if source_root else "缺失",
            "dependency_environment": _cn_dependency_status(dependency_status),
            "config_file": _cn_available(config_path),
            "checkpoint": _cn_available(checkpoint_path),
            "input_data": _cn_available(input_path),
            "runner": "已执行" if runner_executed else "未执行",
            "real_skai_output": "已产生" if real_output else "未产生",
        },
        "building_damage_result": building_summary,
        "truthfulness_note": (
            "Only verified files produced by a real SKAI source-level run may be labeled as SKAI output. "
            "灾情感知及影响评估 does not substitute segmentation statistics or legacy image-plane scores as SKAI results."
        ),
        "unavailable_reasons": unavailable_reasons,
    }


def assess_inasafe_source_level_status(output_dir=None):
    """Return conservative InaSAFE source-level integration status."""
    output_dir = Path(output_dir or OUTPUT_ROOT / "inasafe")
    source_root, missing_sources = _first_repo_with_files(INASAFE_SOURCE_CANDIDATES, INASAFE_REQUIRED_SOURCE_FILES)
    outputs = _find_outputs(output_dir, INASAFE_OUTPUT_PATTERNS)
    dependency_python, python_dependency_status = _conda_env_module_status(
        INASAFE_CONDA_ENV_NAME,
        ["qgis.core", "osgeo.gdal"],
    )
    if not dependency_python:
        dependency_python, python_dependency_status = _first_python_with_modules(
            INASAFE_PYTHON_CANDIDATES,
            ["qgis.core", "osgeo.gdal"],
        )
    inasafe_env_bin = Path.home() / "miniconda3" / "envs" / INASAFE_CONDA_ENV_NAME / "bin"
    qgis_cli_available = (
        shutil.which("qgis") is not None
        or shutil.which("qgis_process") is not None
        or (inasafe_env_bin / "qgis").exists()
        or (inasafe_env_bin / "qgis_process").exists()
        or (Path(str(dependency_python)).exists() and (Path(str(dependency_python)).parent / "qgis_process").exists())
    )
    dependency_status = {
        "qgis_cli": qgis_cli_available,
        "pyqgis": python_dependency_status.get("qgis.core", False),
        "gdal": python_dependency_status.get("osgeo.gdal", False),
    }
    unavailable_reasons = []
    if source_root is None:
        unavailable_reasons.append("InaSAFE source repository not found or required source files are missing.")
    missing_deps = [name for name, available in dependency_status.items() if not available]
    if missing_deps:
        unavailable_reasons.append(f"Missing InaSAFE/QGIS/GIS dependencies: {', '.join(missing_deps)}.")
    if not outputs:
        unavailable_reasons.append("No verified InaSAFE impact report or impact layer output file was found.")
    real_output = source_root is not None and not missing_deps and bool(outputs)
    impact_summary = _load_inasafe_impact_summary(outputs) if real_output else _load_inasafe_impact_summary([])
    return {
        "module": "InaSAFE 外部源码级灾害影响评估",
        "repository": "inasafe/inasafe",
        "integration_method": "外部源码级集成 + QGIS/GIS impact runner",
        "task_role": "灾害影响评估",
        "source_root": str(source_root) if source_root else "",
        "required_source_files": INASAFE_REQUIRED_SOURCE_FILES,
        "missing_source_candidates": missing_sources,
        "dependency_status": dependency_status,
        "dependency_python": str(dependency_python),
        "output_dir": str(output_dir),
        "verified_output_files": outputs,
        "status": "real_output_verified" if real_output else "unavailable",
        "is_real_inasafe_output": bool(real_output),
        "run_status_display": {
            "inasafe_source": "已集成" if source_root else "缺失",
            "dependency_environment": _cn_dependency_status(dependency_status),
            "real_inasafe_output": "已产生" if real_output else "未产生",
        },
        "impact_assessment_result": impact_summary,
        "truthfulness_note": (
            "Only verified files produced by a real InaSAFE/QGIS source-level run may be labeled as InaSAFE output. "
            "灾情感知及影响评估 does not substitute segmentation statistics or legacy image-plane scores as InaSAFE results."
        ),
        "unavailable_reasons": unavailable_reasons,
    }


def build_external_impact_assessment_status(output_dir=None):
    output_dir = Path(output_dir or OUTPUT_ROOT)
    return {
        "success": True,
        "module": "灾情感知与外部影响评估（高级深度版）",
        "segmentation_contract": (
            "S2-S3 uses the local trained semantic segmentation model to generate one pred_mask. "
            "Overlay, black-background color mask, legend, statistics, and downstream status are derived from that same pred_mask."
        ),
        "skai": assess_skai_source_level_status(output_dir / "skai"),
        "inasafe": assess_inasafe_source_level_status(output_dir / "inasafe"),
        "legacy_internal_fallback": {
            "key": "lightweight_skai_inasafe_adaptation",
            "status": "legacy_internal_only",
            "shown_in_final_s2s3": False,
            "truthfulness_note": "This legacy/internal score must not be displayed as final S2-S3 SKAI or InaSAFE output.",
        },
        "human_review_required": True,
        "truthfulness_note": "Outputs are auxiliary decision support and require human review.",
    }


def format_external_impact_assessment_status(status):
    """Format external status for the Gradio UI and reports."""
    if not status:
        status = build_external_impact_assessment_status()
    lines = [
        "灾情感知与外部影响评估（高级深度版）",
        "",
        "统一口径：已训练语义分割模型生成 pred_mask；SKAI / InaSAFE 只接受外部源码级真实运行输出。",
        "",
    ]
    for key in ["skai", "inasafe"]:
        item = status.get(key, {}) or {}
        lines.extend(
            [
                f"{item.get('module', key)}",
                f"- repository: {item.get('repository', '')}",
                f"- status: {item.get('status', 'unavailable')}",
                f"- source_root: {item.get('source_root') or 'unavailable'}",
                f"- output_dir: {item.get('output_dir', '')}",
                f"- verified_outputs: {len(item.get('verified_output_files', []) or [])}",
            ]
        )
        reasons = item.get("unavailable_reasons", []) or []
        if reasons:
            lines.append("- unavailable_reasons:")
            lines.extend([f"  - {reason}" for reason in reasons])
        lines.append(f"- truthfulness: {item.get('truthfulness_note', '')}")
        lines.append("")
    legacy = status.get("legacy_internal_fallback", {}) or {}
    lines.append(
        f"legacy/internal fallback: {legacy.get('key')} = {legacy.get('status')}; "
        "不得作为最终 S2-S3 高级深度版主展示。"
    )
    return "\n".join(lines).strip()


def format_skai_building_damage_panel(skai_status):
    """Format the final S2-S3 SKAI panel in the agreed product wording."""
    item = skai_status or assess_skai_source_level_status()
    run_status = item.get("run_status_display", {}) or {}
    result = item.get("building_damage_result", {}) or {}
    lines = [
        "SKAI 外部源码级建筑灾损评估",
        "",
        "来源仓库：google-research/skai",
        "集成方式：外部源码级集成 + adapter runner",
        "任务定位：建筑级灾损评估",
        "",
        "运行状态：",
        f"- SKAI 源码：{run_status.get('skai_source', '缺失')}",
        f"- 依赖环境：{run_status.get('dependency_environment', '缺失')}",
        f"- 依赖 Python：{item.get('dependency_python', 'unavailable')}",
        f"- 配置文件：{run_status.get('config_file', '缺失')}",
        f"- Checkpoint：{run_status.get('checkpoint', '缺失')}",
        f"- 输入数据：{run_status.get('input_data', '缺失')}",
        f"- Runner：{run_status.get('runner', '未执行')}",
        f"- 真实 SKAI 输出：{run_status.get('real_skai_output', '未产生')}",
        "",
        "建筑灾损结果：",
        f"- 建筑总数：{result.get('building_total', 'unavailable')}",
        f"- 受损建筑数：{result.get('damaged_building_count', 'unavailable')}",
        f"- 严重损毁建筑数：{result.get('severe_damage_building_count', 'unavailable')}",
        f"- 完全毁坏建筑数：{result.get('destroyed_building_count', 'unavailable')}",
        f"- 建筑灾损等级：{result.get('building_damage_level', 'unavailable')}",
        "",
        "说明：",
        "只有当 SKAI 源码真实运行并验证到输出文件时，本区域才显示为真实 SKAI 输出。",
    ]
    reasons = item.get("unavailable_reasons", []) or []
    if reasons:
        lines.extend(["", "unavailable 原因："])
        lines.extend([f"- {reason}" for reason in reasons])
    return "\n".join(lines)


def format_inasafe_impact_assessment_panel(inasafe_status):
    """Format the final S2-S3 InaSAFE panel in the agreed product wording."""
    item = inasafe_status or assess_inasafe_source_level_status()
    result = item.get("impact_assessment_result", {}) or {}
    lines = [
        "InaSAFE 外部源码级灾害影响评估",
        "",
        "来源仓库：inasafe/inasafe",
        "集成方式：外部源码级集成 + QGIS/GIS impact runner",
        "任务定位：灾害影响评估",
        "",
        "InaSAFE 影响评估结果：",
        f"- 受影响道路：{result.get('affected_roads', 'unavailable')}",
        f"- 受影响建筑：{result.get('affected_buildings', 'unavailable')}",
        f"- 受影响目标点：{result.get('affected_points', 'unavailable')}",
        f"- 高风险影响区域：{result.get('high_risk_impact_areas', 'unavailable')}",
        f"- 输出报告：{result.get('output_report', 'unavailable')}",
        "",
        "说明：",
        "只有当 InaSAFE/QGIS 真实运行并验证到输出文件时，本区域才显示为真实 InaSAFE 输出。",
    ]
    reasons = item.get("unavailable_reasons", []) or []
    if reasons:
        lines.extend(["", "unavailable 原因："])
        lines.extend([f"- {reason}" for reason in reasons])
    return "\n".join(lines)


def format_external_output_file_summary(status):
    """Format verified external output files from SKAI and InaSAFE."""
    status = status or build_external_impact_assessment_status()
    lines = ["外部输出文件摘要："]
    any_file = False
    for key, title in [("skai", "SKAI"), ("inasafe", "InaSAFE")]:
        item = status.get(key, {}) or {}
        files = item.get("verified_output_files", []) or []
        lines.append(f"{title}：{len(files)} 个已验证输出文件")
        for path in files[:8]:
            any_file = True
            lines.append(f"- {path}")
        if len(files) > 8:
            lines.append(f"- ... 另有 {len(files) - 8} 个文件")
    if not any_file:
        lines.append("当前未发现可验证的 SKAI / InaSAFE 外部输出文件。")
    return "\n".join(lines)


def format_external_unavailable_reasons(status):
    """Format unavailable reasons and truthfulness boundaries for external modules."""
    status = status or build_external_impact_assessment_status()
    lines = ["不可用原因 / 真实性边界："]
    for key, title in [("skai", "SKAI"), ("inasafe", "InaSAFE")]:
        item = status.get(key, {}) or {}
        lines.append(f"{title} status：{item.get('status', 'unavailable')}")
        reasons = item.get("unavailable_reasons", []) or []
        if reasons:
            lines.extend([f"- {reason}" for reason in reasons])
        else:
            lines.append("- 当前未记录不可用原因。")
        lines.append(f"- truthfulness：{item.get('truthfulness_note', '')}")
    legacy = status.get("legacy_internal_fallback", {}) or {}
    lines.append(
        f"legacy/internal fallback：{legacy.get('key')} = {legacy.get('status')}；不得作为最终主展示。"
    )
    return "\n".join(lines)


def save_external_impact_assessment_status(status=None, output_path=None):
    status = status or build_external_impact_assessment_status()
    path = Path(output_path or OUTPUT_ROOT / "status.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
