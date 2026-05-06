"""Evidence-driven Final Report 2.0 helpers for S9.

This module builds a structured AI-assisted decision-support report from
mission state, workflow state, stage outputs, and Evidence Ledger entries. It
does not run models, call LLM APIs, or invent missing mission results.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from ..workflow.stage_definitions import list_stage_keys
    from ..workflow.workflow_state import summarize_workflow_context, summarize_workflow_state
except ImportError:  # pragma: no cover - supports direct app/ path imports.
    from workflow.stage_definitions import list_stage_keys
    from workflow.workflow_state import summarize_workflow_context, summarize_workflow_state


FINAL_REPORT_NOTICE = "Final Report is an AI-assisted decision-support report and not a final rescue conclusion."
NO_INVENTION_NOTE = "The report summarizes available evidence and missing context; it must not invent mission results."
FIELD_REVIEW_NOTE = "Human review and field commander judgment are required before field action."

REPORT_TRUTHFULNESS_BOUNDARIES = [
    FINAL_REPORT_NOTICE,
    NO_INVENTION_NOTE,
    "AI candidates are not confirmed civilians.",
    "Thermal support is auxiliary evidence and not confirmation of life.",
    "Image-plane path is not GPS navigation.",
    "Fast Preview is not a real ODM georeferenced orthomosaic.",
    "S2-S3 final display uses the trained local semantic segmentation model pred_mask; uploaded/demo masks are not final main-display sources.",
    FIELD_REVIEW_NOTE,
]


STAGE_RESULT_CANDIDATES = {
    "global_mapping": [
        ("outputs", "orthomosaic", "processing_log.json"),
        ("outputs", "orthomosaic", "global_mapping_result.json"),
        ("outputs", "odm", "odm_result.json"),
    ],
    "macro_analysis": [
        ("outputs", "segmentation", "macro_analysis_result.json"),
        ("outputs", "segmentation", "segmentation_result.json"),
    ],
    "area_tasking": [
        ("outputs", "area_tasking", "area_tasking_result.json"),
    ],
    "local_recon": [
        ("outputs", "detection", "local_recon_result.json"),
    ],
    "target_verification": [
        ("outputs", "target_verification", "target_verification_result.json"),
    ],
    "thermal_check": [
        ("outputs", "thermal_check", "thermal_check_result.json"),
    ],
    "decision_fusion": [
        ("outputs", "priority", "decision_fusion_result.json"),
        ("outputs", "decision_fusion", "decision_fusion_result.json"),
        ("outputs", "decision_fusion", "decision_fusion_summary.json"),
    ],
    "rescue_recommendation": [
        ("outputs", "path", "rescue_recommendation_result.json"),
    ],
}


def safe_load_json(path, default=None):
    """Load JSON from path, returning default when missing or invalid."""
    if default is None:
        default = {}
    try:
        if not path:
            return default
        path = Path(path)
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _mission_output_refs(mission, stage_key):
    outputs = (mission or {}).get("outputs", {}) or {}
    keys = [stage_key, f"workflow:{stage_key}"]
    refs = []
    for key in keys:
        value = outputs.get(key)
        if isinstance(value, dict) and value.get("output_ref"):
            refs.append(value.get("output_ref"))
        elif isinstance(value, str):
            refs.append(value)
    return refs


def load_stage_result_from_outputs(mission_dir, stage_key, mission=None):
    """Load the best available JSON result for a workflow stage."""
    mission_dir = Path(mission_dir)
    for output_ref in _mission_output_refs(mission or {}, stage_key):
        loaded = safe_load_json(output_ref, default=None)
        if isinstance(loaded, dict):
            return loaded

    for parts in STAGE_RESULT_CANDIDATES.get(stage_key, []):
        candidate_path = mission_dir.joinpath(*parts)
        loaded = safe_load_json(candidate_path, default=None)
        if isinstance(loaded, dict):
            return loaded
    return {}


def collect_available_stage_results(mission, mission_dir):
    """Collect available S1-S8 stage result JSON documents."""
    return {
        stage_key: load_stage_result_from_outputs(mission_dir, stage_key, mission=mission)
        for stage_key in list_stage_keys()
        if stage_key != "evidence_report"
    }


def build_evidence_summary(ledger):
    """Summarize Evidence Ledger coverage and limitations."""
    entries = list((ledger or {}).get("entries", []) or [])
    module_counts = {}
    source_type_counts = {}
    limitations = []
    truthfulness_note_count = 0
    human_review_required_count = 0
    for entry in entries:
        module = entry.get("module", "unknown")
        source_type = entry.get("source_type", "unknown")
        module_counts[module] = module_counts.get(module, 0) + 1
        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
        if entry.get("human_review_required"):
            human_review_required_count += 1
        note = entry.get("truthfulness_note")
        if note:
            truthfulness_note_count += 1
        limitation = entry.get("limitation")
        if limitation and limitation not in limitations:
            limitations.append(limitation)

    return {
        "total_evidence_count": len(entries),
        "human_review_required_count": human_review_required_count,
        "module_counts": module_counts,
        "source_type_counts": source_type_counts,
        "truthfulness_note_count": truthfulness_note_count,
        "limitations": limitations,
    }


def build_workflow_report_summary(mission, workflow_state):
    """Build mission and workflow execution summary for the final report."""
    workflow_summary = summarize_workflow_state(workflow_state)
    workflow_context = summarize_workflow_context(workflow_state)
    stages = (workflow_state or {}).get("stages", {}) or {}
    missing_context_notes = list(workflow_context.get("missing_context_notes", []) or [])

    if (mission or {}).get("workflow_mode") == "direct_local_recon":
        direct_notes = [
            "No verified global map is connected.",
            "Local detections are not georeferenced unless map registration is provided.",
        ]
        for note in direct_notes:
            if note not in missing_context_notes:
                missing_context_notes.append(note)

    return {
        "mission_id": (mission or {}).get("mission_id", ""),
        "mission_name": (mission or {}).get("mission_name", ""),
        "workflow_mode": (mission or {}).get("workflow_mode", "standard"),
        "status": (mission or {}).get("status", ""),
        "completed_stage_count": workflow_summary.get("completed_stage_count", 0),
        "failed_stage_count": workflow_summary.get("failed_stage_count", 0),
        "skipped_stage_count": sum(1 for item in stages.values() if item.get("status") == "skipped"),
        "current_stage_key": workflow_summary.get("current_stage_key", ""),
        "global_context_available": bool((mission or {}).get("global_context_available", workflow_context.get("global_mapping_available", False))),
        "map_registration_available": bool((mission or {}).get("map_registration_available", False)),
        "missing_context_notes": missing_context_notes,
    }


def _stage_status(stage_result):
    if not stage_result:
        return "missing"
    return stage_result.get("status", "available")


def _brief_stage_summary(stage_key, result):
    summary = {"status": _stage_status(result), "available": bool(result)}
    if not result:
        summary["note"] = "missing or unavailable"
        return summary

    if stage_key == "global_mapping":
        summary.update(
            {
                "base_map_type": result.get("base_map_type", ""),
                "base_map_path": result.get("base_map_path", ""),
                "input_image_count": result.get("input_image_count", 0),
            }
        )
    elif stage_key == "macro_analysis":
        summary.update(
            {
                "segmentation_source": result.get("segmentation_source", ""),
                "macro_zone_count": len(result.get("macro_zones", []) or []),
            }
        )
    elif stage_key == "area_tasking":
        summary.update({"area_count": result.get("area_count", len(result.get("area_tasks", []) or []))})
    elif stage_key == "local_recon":
        summary.update({"candidate_count": result.get("candidate_count", 0), "detection_backend": result.get("detection_backend", "")})
    elif stage_key == "target_verification":
        verification_summary = result.get("verification_summary", {}) or {}
        summary.update({"candidate_count": result.get("candidate_count", 0), "verification_summary": verification_summary})
    elif stage_key == "thermal_check":
        summary.update({"thermal_summary": result.get("thermal_summary", {}) or {}})
    elif stage_key == "decision_fusion":
        summary.update(
            {
                "decision_summary": result.get("decision_summary", {}) or {},
                "top_priority_candidate": result.get("top_priority_candidate", {}) or {},
                "decision_candidate_count": len(result.get("decision_candidates", []) or []),
            }
        )
    elif stage_key == "rescue_recommendation":
        summary.update({"recommendation_summary": result.get("recommendation_summary", {}) or {}})

    if result.get("truthfulness_note"):
        summary["truthfulness_note"] = result.get("truthfulness_note")
    return summary


def _collect_key_findings(stage_results, evidence_summary):
    findings = []
    local_recon = stage_results.get("local_recon", {}) or {}
    if local_recon:
        findings.append(f"S4 local_recon generated {local_recon.get('candidate_count', 0)} rescue candidates for review.")
    target_verification = stage_results.get("target_verification", {}) or {}
    if target_verification:
        summary = target_verification.get("verification_summary", {}) or {}
        findings.append(f"S5 target_verification produced {summary.get('verification_count', 0)} visual verification records.")
    thermal_check = stage_results.get("thermal_check", {}) or {}
    if thermal_check:
        summary = thermal_check.get("thermal_summary", {}) or {}
        findings.append(f"S6 thermal_check produced {summary.get('thermal_check_count', 0)} thermal support records.")
    decision_fusion = stage_results.get("decision_fusion", {}) or {}
    if decision_fusion:
        candidate_count = len(decision_fusion.get("decision_candidates", []) or [])
        findings.append(f"S7 decision_fusion produced {candidate_count} EC-TERP priority candidates.")
    rescue_recommendation = stage_results.get("rescue_recommendation", {}) or {}
    if rescue_recommendation:
        summary = rescue_recommendation.get("recommendation_summary", {}) or {}
        findings.append(f"S8 rescue_recommendation produced {summary.get('recommendation_count', 0)} route/task recommendations.")
    if evidence_summary.get("total_evidence_count", 0):
        findings.append(f"Evidence Ledger contains {evidence_summary.get('total_evidence_count', 0)} evidence entries.")
    return findings


def _collect_priority_recommendations(stage_results):
    decision_fusion = stage_results.get("decision_fusion", {}) or {}
    candidates = decision_fusion.get("decision_candidates") or []
    if candidates:
        return list(candidates)
    top = decision_fusion.get("top_priority_candidate")
    return [top] if isinstance(top, dict) and top else []


def _collect_route_recommendations(stage_results):
    rescue_recommendation = stage_results.get("rescue_recommendation", {}) or {}
    return list(rescue_recommendation.get("recommendations", []) or [])


def _collect_human_review_items(stage_results, ledger):
    items = []
    for entry in (ledger or {}).get("entries", []) or []:
        if entry.get("human_review_required"):
            items.append(
                {
                    "source": "evidence_ledger",
                    "evidence_id": entry.get("evidence_id", ""),
                    "module": entry.get("module", ""),
                    "reason": entry.get("truthfulness_note", "") or entry.get("limitation", ""),
                }
            )

    for record in (stage_results.get("target_verification", {}) or {}).get("verification_records", []) or []:
        if record.get("human_review_required"):
            items.append(
                {
                    "source": "target_verification",
                    "candidate_id": record.get("candidate_id", ""),
                    "review_status": record.get("review_status", ""),
                    "reason": record.get("truthfulness_note", ""),
                }
            )
    for record in (stage_results.get("thermal_check", {}) or {}).get("thermal_records", []) or []:
        if record.get("human_review_required"):
            items.append(
                {
                    "source": "thermal_check",
                    "candidate_id": record.get("candidate_id", ""),
                    "thermal_support_level": record.get("thermal_support_level", ""),
                    "reason": record.get("truthfulness_note", ""),
                }
            )
    for item in _collect_priority_recommendations(stage_results):
        if item.get("human_review_required", True):
            items.append(
                {
                    "source": "decision_fusion",
                    "candidate_id": item.get("candidate_id", ""),
                    "priority_level": item.get("priority_level", ""),
                    "reason": item.get("truthfulness_note", "EC-TERP priority ranking requires commander review."),
                }
            )
    for item in _collect_route_recommendations(stage_results):
        if item.get("human_review_required", True):
            items.append(
                {
                    "source": "rescue_recommendation",
                    "candidate_id": item.get("candidate_id", ""),
                    "priority_level": item.get("priority_level", ""),
                    "reason": item.get("truthfulness_note", ""),
                }
            )
    return items


def build_final_report_data(mission, mission_dir, stage_results=None, ledger=None):
    """Build structured Final Report 2.0 data from mission evidence."""
    stage_results = stage_results if stage_results is not None else collect_available_stage_results(mission, mission_dir)
    stage_results = {key: dict(stage_results.get(key, {}) or {}) for key in list_stage_keys() if key != "evidence_report"}
    ledger = ledger or {}
    evidence_summary = build_evidence_summary(ledger)
    workflow_summary = build_workflow_report_summary(mission, (mission or {}).get("workflow_state", {}))
    stage_summaries = {key: _brief_stage_summary(key, result) for key, result in stage_results.items()}
    limitations = list(evidence_summary.get("limitations", []) or [])
    for note in workflow_summary.get("missing_context_notes", []) or []:
        if note not in limitations:
            limitations.append(note)
    if not any(stage_results.values()) and evidence_summary.get("total_evidence_count", 0) == 0:
        limitations.append("No stage evidence is available; the report is limited.")

    truthfulness_boundaries = []
    for boundary in REPORT_TRUTHFULNESS_BOUNDARIES + list((mission or {}).get("truthfulness_boundaries", []) or []):
        if boundary and boundary not in truthfulness_boundaries:
            truthfulness_boundaries.append(boundary)

    return {
        "report_title": "灾情感知及影响评估 Final Report 2.0",
        "report_type": "AI-assisted decision-support report",
        "mission_summary": {
            "mission_id": (mission or {}).get("mission_id", ""),
            "mission_name": (mission or {}).get("mission_name", ""),
            "status": (mission or {}).get("status", ""),
            "evidence_ledger_path": (mission or {}).get("evidence_ledger_path", ""),
        },
        "workflow_summary": workflow_summary,
        "stage_summaries": stage_summaries,
        "key_findings": _collect_key_findings(stage_results, evidence_summary),
        "priority_recommendations": _collect_priority_recommendations(stage_results),
        "route_recommendations": _collect_route_recommendations(stage_results),
        "human_review_items": _collect_human_review_items(stage_results, ledger),
        "evidence_summary": evidence_summary,
        "truthfulness_boundaries": truthfulness_boundaries,
        "limitations": limitations,
        "final_notice": FINAL_REPORT_NOTICE,
    }


def _yes_no(value):
    return "是" if value else "否"


def _append_stage_section(lines, number, title, summary):
    lines.extend([f"## {number}. {title}", ""])
    if not summary or not summary.get("available"):
        lines.extend(["- 状态：missing / unavailable", "- 说明：当前未连接该阶段结果。", ""])
        return
    lines.append(f"- 状态：{summary.get('status', '')}")
    for key, value in summary.items():
        if key in {"available", "status"}:
            continue
        lines.append(f"- {key}: {value}")
    lines.append("")


def _append_items(lines, items, empty_message):
    if not items:
        lines.append(empty_message)
        lines.append("")
        return
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            label = item.get("candidate_id") or item.get("target_id") or item.get("evidence_id") or f"Item {index}"
            lines.append(f"{index}. {label}: {item}")
        else:
            lines.append(f"{index}. {item}")
    lines.append("")


def render_final_report_markdown(report_data):
    """Render structured report data as Chinese Markdown."""
    report_data = report_data or {}
    mission_summary = report_data.get("mission_summary", {}) or {}
    workflow_summary = report_data.get("workflow_summary", {}) or {}
    stage_summaries = report_data.get("stage_summaries", {}) or {}
    evidence_summary = report_data.get("evidence_summary", {}) or {}

    lines = [
        "# 灾情感知及影响评估 Final Report 2.0",
        "",
        "> 本报告为 AI 辅助决策报告，不构成最终救援结论；所有候选目标、路径建议和优先级排序均需由现场指挥人员与救援队人工复核。",
        "",
        "## 1. 任务基本信息",
        "",
        f"- Mission ID：{mission_summary.get('mission_id', '')}",
        f"- Mission Name：{mission_summary.get('mission_name', '')}",
        f"- Mission Status：{mission_summary.get('status', '')}",
        f"- Report Type：{report_data.get('report_type', '')}",
        f"- Evidence Ledger：{mission_summary.get('evidence_ledger_path', '')}",
        "",
        "## 2. Workflow 执行摘要",
        "",
        f"- Workflow Mode：{workflow_summary.get('workflow_mode', '')}",
        f"- 当前阶段：{workflow_summary.get('current_stage_key', '')}",
        f"- 已完成阶段数：{workflow_summary.get('completed_stage_count', 0)}",
        f"- 失败阶段数：{workflow_summary.get('failed_stage_count', 0)}",
        f"- 跳过阶段数：{workflow_summary.get('skipped_stage_count', 0)}",
        f"- 全局地图上下文可用：{_yes_no(workflow_summary.get('global_context_available'))}",
        f"- 地图配准可用：{_yes_no(workflow_summary.get('map_registration_available'))}",
        "",
        "缺失上下文：",
    ]
    _append_items(lines, workflow_summary.get("missing_context_notes", []) or [], "无额外缺失上下文说明。")

    stage_titles = [
        ("global_mapping", "3", "高空建图结果"),
        ("macro_analysis", "4", "宏观灾情分析"),
        ("area_tasking", "5", "重点区域划分"),
        ("local_recon", "6", "局部目标检测结果"),
        ("target_verification", "7", "目标复核证据"),
        ("thermal_check", "8", "热红外辅助证据"),
        ("decision_fusion", "9", "EC-TERP 决策融合与优先级排序"),
        ("rescue_recommendation", "10", "路径与任务建议"),
    ]
    for key, number, title in stage_titles:
        _append_stage_section(lines, number, title, stage_summaries.get(key, {}))

    lines.extend(["## 11. 人工复核清单", ""])
    _append_items(lines, report_data.get("human_review_items", []) or [], "当前报告未发现已连接的人工复核条目。")

    lines.extend(
        [
            "## 12. Evidence Ledger 摘要",
            "",
            f"- 证据条目总数：{evidence_summary.get('total_evidence_count', 0)}",
            f"- 需要人工复核条目数：{evidence_summary.get('human_review_required_count', 0)}",
            f"- 模块统计：{evidence_summary.get('module_counts', {})}",
            f"- 来源类型统计：{evidence_summary.get('source_type_counts', {})}",
            "",
            "## 13. 真实性边界与局限性",
            "",
        ]
    )
    _append_items(lines, report_data.get("truthfulness_boundaries", []) or [], "无真实性边界记录。")
    lines.append("局限性：")
    _append_items(lines, report_data.get("limitations", []) or [], "无额外局限性记录。")

    lines.extend(
        [
            "## 14. 最终声明",
            "",
            "本报告为 AI 辅助决策报告，不构成最终救援结论；所有候选目标、路径建议和优先级排序均需由现场指挥人员与救援队人工复核。",
            "",
            report_data.get("final_notice", FINAL_REPORT_NOTICE),
        ]
    )
    return "\n".join(lines)


def save_final_report_outputs(report_data, output_dir):
    """Save Final Report 2.0 JSON and Markdown outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "final_report_data.json"
    markdown_path = output_dir / "final_report.md"
    markdown = render_final_report_markdown(report_data)
    json_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return {
        "report_json_path": str(json_path),
        "report_markdown_path": str(markdown_path),
        "saved": True,
    }
