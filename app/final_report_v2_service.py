"""Final Report 2.0 for 灾情感知及影响评估.

The report is driven by the mission evidence ledger, not by code existence.
"""

import html
import json
from pathlib import Path

from mission_evidence_ledger import (
    build_mission_evidence_ledger,
    get_decision_supporting_evidence,
    get_human_review_items,
)


class FinalReportV2Error(Exception):
    pass


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT_DIR / "outputs"
REPORT_DIR = OUTPUT_ROOT / "reports"


def _ensure_record_list(records):
    return list(records) if isinstance(records, list) else []


def _normalize_record_list(records):
    normalized = []
    for record in _ensure_record_list(records):
        if isinstance(record, dict):
            normalized.append(record)
    return normalized


def build_report_context(ledger=None, root_dir=None):
    """Build report context from a mission evidence ledger."""
    try:
        if ledger is None:
            ledger = build_mission_evidence_ledger(root_dir=root_dir)
        if not isinstance(ledger, dict):
            raise FinalReportV2Error("ledger must be a dict.")
        summary = ledger.get("summary", {}) or {}
        sections = ledger.get("report_sections", {}) or {}
        decision_supporting_evidence = get_decision_supporting_evidence(ledger)
        human_review_items = get_human_review_items(ledger)
        return {
            "success": True,
            "root_dir": str(Path(root_dir) if root_dir else ROOT_DIR),
            "ledger": ledger,
            "summary": summary,
            "sections": sections,
            "decision_supporting_evidence": decision_supporting_evidence,
            "human_review_items": human_review_items,
            "global_truthfulness_note": ledger.get("global_truthfulness_note", "")
            or "Final Report 2.0 is derived from Mission Evidence Ledger and local artifacts, not from code existence alone.",
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "ledger": ledger if isinstance(ledger, dict) else {},
            "summary": {},
            "sections": {},
            "decision_supporting_evidence": [],
            "human_review_items": [],
            "global_truthfulness_note": "Final Report 2.0 generation failed before report assembly.",
        }


def classify_report_sections(context):
    """Classify evidence records into report sections."""
    if not isinstance(context, dict):
        return {
            "main_model_evidence": [],
            "real_measurement_evidence": [],
            "auxiliary_decision_evidence": [],
            "simulated_or_preview_results": [],
            "failed_or_unavailable_modules": [],
            "not_run_modules": [],
            "reference_or_registry_modules": [],
            "human_review_items": [],
        }

    ledger = context.get("ledger", {}) or {}
    records = ledger.get("evidence_records", {}) or {}
    sections = {
        "main_model_evidence": [],
        "real_measurement_evidence": [],
        "auxiliary_decision_evidence": [],
        "simulated_or_preview_results": [],
        "failed_or_unavailable_modules": [],
        "not_run_modules": [],
        "reference_or_registry_modules": [],
        "human_review_items": _normalize_record_list(context.get("human_review_items", [])),
    }

    for record in records.values():
        if not isinstance(record, dict):
            continue
        section_name = record.get("recommended_report_section", "")
        evidence_level = str(record.get("evidence_level", "none"))
        evidence_type = str(record.get("evidence_type", "not_available"))
        scanner_status = str(record.get("scanner_status", "unknown"))

        if section_name == "主要模型输出证据" or evidence_level == "strong" and evidence_type == "model_output":
            sections["main_model_evidence"].append(record)
        elif section_name == "真实测量 / 真实产物证据" or evidence_type == "real_measurement":
            sections["real_measurement_evidence"].append(record)
        elif section_name == "辅助决策证据" or evidence_level == "medium":
            sections["auxiliary_decision_evidence"].append(record)
        elif section_name == "模拟 / 预览结果" or evidence_type == "simulated_or_preview":
            sections["simulated_or_preview_results"].append(record)
        elif section_name == "执行失败 / 不可用模块" or scanner_status in {"executed_failed", "dependency_missing"}:
            sections["failed_or_unavailable_modules"].append(record)
        elif section_name == "未执行模块" or scanner_status in {"not_run", "implemented_but_not_run"}:
            sections["not_run_modules"].append(record)
        elif section_name == "参考 / 注册表模块" or evidence_type == "reference_only":
            sections["reference_or_registry_modules"].append(record)
        else:
            # Keep conservative: unknown items stay in auxiliary if they can support decision,
            # otherwise they are grouped into not_run / failed later by scanner status.
            if record.get("can_support_decision"):
                sections["auxiliary_decision_evidence"].append(record)
            else:
                sections["not_run_modules"].append(record)

    return sections


def _count_levels(records):
    counts = {"strong": 0, "medium": 0, "weak": 0, "none": 0}
    for record in records:
        level = str(record.get("evidence_level", "none"))
        if level in counts:
            counts[level] += 1
    return counts


def build_executive_summary(context, sections):
    """Build the executive summary markdown."""
    summary = context.get("summary", {}) if isinstance(context, dict) else {}
    all_records = []
    for key in [
        "main_model_evidence",
        "real_measurement_evidence",
        "auxiliary_decision_evidence",
        "simulated_or_preview_results",
        "failed_or_unavailable_modules",
        "not_run_modules",
        "reference_or_registry_modules",
    ]:
        all_records.extend(_normalize_record_list(sections.get(key, [])))

    counts = _count_levels(all_records)
    strong_count = counts["strong"]
    medium_count = counts["medium"]
    weak_count = counts["weak"]
    none_count = counts["none"]
    decision_support_count = int(summary.get("decision_support_count", 0))
    human_review_count = int(summary.get("human_review_required_count", 0))

    if strong_count >= 1 and medium_count >= 1:
        evidence_sentence = "当前任务已形成一定的感知与辅助决策证据链。"
    elif strong_count == 0 and medium_count > 0:
        evidence_sentence = "当前任务主要依赖规则与图像平面辅助决策，缺少强证据。"
    elif strong_count == 0 and medium_count == 0:
        evidence_sentence = "当前任务缺少可支持决策的模型或测量证据。"
    else:
        evidence_sentence = "当前任务具有有限证据链，需要进一步核查。"

    return (
        "## 任务摘要\n\n"
        "本报告由证据链自动生成，不根据代码文件存在判断模块状态。\n\n"
        f"- strong 证据数量：{strong_count}\n"
        f"- medium 证据数量：{medium_count}\n"
        f"- weak 证据数量：{weak_count}\n"
        f"- none 证据数量：{none_count}\n"
        f"- 可支持决策模块数量：{decision_support_count}\n"
        f"- 需要人工复核模块数量：{human_review_count}\n\n"
        f"{evidence_sentence}\n\n"
        "本报告是辅助决策报告，仍需人工复核，不能替代现场救援判断。"
    )


def format_evidence_section(title, records, empty_message):
    """Format one report section in markdown."""
    records = _normalize_record_list(records)
    if not records:
        return f"## {title}\n\n{empty_message}"
    lines = [f"## {title}", ""]
    for record in records:
        evidence_files = record.get("evidence_files", []) or []
        limitations = record.get("limitations", []) or []
        lines.extend(
            [
                f"### {record.get('display_name', record.get('module_key', 'unknown'))}",
                f"- 模块键：{record.get('module_key', 'unknown')}",
                f"- 证据等级：{record.get('evidence_level', 'none')}",
                f"- 证据类型：{record.get('evidence_type', 'not_available')}",
                f"- 是否支持决策：{'是' if record.get('can_support_decision') else '否'}",
                f"- 是否需要人工复核：{'是' if record.get('human_review_required') else '否'}",
                f"- 证据文件：",
            ]
        )
        if evidence_files:
            lines.extend([f"  - {item}" for item in evidence_files])
        else:
            lines.append("  - 无证据文件")
        lines.append("- 局限说明：")
        if limitations:
            lines.extend([f"  - {item}" for item in limitations])
        else:
            lines.append("  - 无额外局限说明")
        lines.append(f"- 真实性说明：{record.get('truthfulness_note', '')}")
        lines.append("")
    return "\n".join(lines)


def _load_first_existing_json(paths):
    for item in paths or []:
        path = Path(item)
        if not path.exists() or path.suffix.lower() != ".json":
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def _load_ec_terp_visual_metadata(context):
    root_dir = Path(context.get("root_dir") or ROOT_DIR)
    candidates = [
        root_dir / "outputs" / "ec_terp_visuals" / "ec_terp_visuals_metadata.json",
        ROOT_DIR / "outputs" / "ec_terp_visuals" / "ec_terp_visuals_metadata.json",
    ]
    return _load_first_existing_json(candidates)


def build_ec_terp_section(context, sections):
    """Build an EC-TERP ranking section when runtime artifacts exist."""
    records = []
    for record in sections.get("auxiliary_decision_evidence", []) or []:
        if record.get("module_key") == "ec_terp_ranking":
            records.append(record)
    for record in sections.get("simulated_or_preview_results", []) or []:
        if record.get("module_key") == "ec_terp_ranking":
            records.append(record)

    lines = [
        "## EC-TERP 救援辅助优先级排序",
        "",
        "### 算法说明",
        "- 公式：`EC-TERP = αT + βE + γR + δC + λQ - μU`",
        "- T：Target urgency，目标紧急度",
        "- E：Environment risk，环境风险",
        "- R：Route accessibility，路径可达性",
        "- C：Coverage gap，覆盖缺口",
        "- Q：Evidence quality，证据质量",
        "- U：Uncertainty penalty，不确定性惩罚，属于扣分项",
        "",
    ]

    if not records:
        lines.extend(
            [
                "当前没有 EC-TERP runtime ranking 产物。",
                "",
                "### 真实性边界",
                "- EC-TERP 是辅助优先级排序算法，不是自动救援决策系统。",
                "- Image-plane path planning is not GPS navigation.",
                "- Synthetic demo cases are not real rescue data.",
            ]
        )
        return "\n".join(lines)

    record = records[0]
    ranking_payload = _load_first_existing_json(
        [path for path in record.get("evidence_files", []) if str(path).endswith("ec_terp_rankings.json")]
    )
    rankings = []
    if isinstance(ranking_payload, dict):
        rankings = ranking_payload.get("rankings", []) or []

    lines.extend(["### Top-K Ranking Table"])
    if rankings:
        lines.extend(
            [
                "| Rank | Target ID | Target Type | EC-TERP Score | Evidence | Human Review | Key Reason |",
                "| --- | --- | --- | ---: | --- | --- | --- |",
            ]
        )
        for item in rankings[:10]:
            explanation = str(item.get("explanation", "")).replace("\n", " ")
            lines.append(
                f"| {item.get('rank', '')} | {item.get('target_id', '')} | {item.get('target_type', '')} | "
                f"{item.get('ec_terp_score', 0.0)} | {item.get('evidence_level', '')} | "
                f"{'是' if item.get('human_review_required') else '否'} | {explanation[:120]} |"
            )
    else:
        lines.append("当前 EC-TERP 模块已记录，但未找到可展示 ranking items。")

    lines.extend(["", "### 解释文本"])
    if rankings:
        top = rankings[0]
        lines.append(
            f"- 当前最高优先级目标为 `{top.get('target_id')}`，类型 `{top.get('target_type')}`，EC-TERP 分数为 {top.get('ec_terp_score')}。"
        )
        lines.append("- Top-1 排名由目标紧急度、环境风险、路径可达性、覆盖缺口、证据质量和不确定性惩罚共同决定。")
        lines.append("- Top-3 主要排序原因：")
        for item in rankings[:3]:
            components = item.get("score_components", {}) or {}
            positive_components = {
                "target_urgency": components.get("target_urgency", 0.0),
                "environment_risk": components.get("environment_risk", 0.0),
                "route_accessibility": components.get("route_accessibility", 0.0),
                "coverage_gap": components.get("coverage_gap", 0.0),
                "evidence_quality": components.get("evidence_quality", 0.0),
            }
            top_factor = max(positive_components, key=lambda key: float(positive_components.get(key) or 0.0))
            uncertainty = float(components.get("uncertainty_penalty", 0.0) or 0.0)
            lines.append(
                f"  - Rank {item.get('rank')}: `{item.get('target_id')}` 的主要贡献项为 `{top_factor}`，"
                f"不确定性惩罚 U={uncertainty}。"
            )
        review_targets = [
            item.get("target_id")
            for item in rankings
            if item.get("human_review_required") or str(item.get("evidence_level", "")).lower() in {"weak", "none"}
        ]
        if review_targets:
            lines.append(f"- 需要重点人工复核的目标：{', '.join(str(item) for item in review_targets[:8])}。")
    lines.append("- 所有 EC-TERP ranking items 均需要人工复核。")

    visual_metadata = _load_ec_terp_visual_metadata(context)
    lines.extend(["", "### 图表引用"])
    if isinstance(visual_metadata, dict) and visual_metadata.get("generated_figures"):
        for figure in visual_metadata.get("generated_figures", []):
            if isinstance(figure, dict) and figure.get("path"):
                lines.append(f"- {figure.get('name', 'figure')}: `{figure.get('path')}`")
        for limitation in visual_metadata.get("limitations", []) or []:
            lines.append(f"- 图表 limitation：{limitation}")
    else:
        lines.append("- 当前未检测到 `outputs/ec_terp_visuals/` 下的 EC-TERP 图表产物；报告仅展示表格和文字解释。")

    lines.extend(
        [
            "",
            "### 真实性边界",
            "- EC-TERP is an assistive priority ranking algorithm.",
            "- It is not an automatic rescue decision system.",
            "- Image-plane path planning is not GPS navigation.",
            "- Synthetic demo cases are not real rescue data.",
            "- human_candidate 不等于 confirmed civilian。",
        ]
    )
    return "\n".join(lines)


def build_human_review_section(human_review_items):
    """Build the human review section."""
    human_review_items = _normalize_record_list(human_review_items)
    if not human_review_items:
        return (
            "## 十、人工复核清单\n\n"
            "当前证据链未记录明确的人工复核项，但实际救援仍需人工确认。"
        )
    lines = ["## 十、人工复核清单", ""]
    for item in human_review_items:
        lines.extend(
            [
                f"### {item.get('display_name', item.get('module_key', 'unknown'))}",
                f"- 模块键：{item.get('module_key', 'unknown')}",
                f"- 复核原因 / limitations：{item.get('reason', '需要人工复核。')}",
                f"- 真实性说明：{item.get('truthfulness_note', '')}",
                "",
            ]
        )
        for detail in item.get("limitations", []) or []:
            lines.append(f"- {detail}")
        lines.append("")
    return "\n".join(lines).strip()


def build_rescue_recommendations(context, sections):
    """Build conservative rescue recommendations."""
    lines = ["## 十一、综合救援辅助建议", ""]

    main_records = sections.get("main_model_evidence", [])
    aux_records = sections.get("auxiliary_decision_evidence", [])
    simulated_records = sections.get("simulated_or_preview_results", [])
    real_measurement_records = sections.get("real_measurement_evidence", [])
    weak_or_none = not main_records and not aux_records and not real_measurement_records

    if main_records:
        lines.append("- 建议优先人工复核检测出的救援目标。")
    if any(rec.get("evidence_type") in {"image_plane_decision", "rule_based_decision"} for rec in aux_records):
        lines.append("- 可参考已训练语义分割模型、外部源码级影响评估状态和覆盖缺口，但仍需人工确认。")
    if any(rec.get("module_key") == "path_planning" for rec in aux_records):
        lines.append("- 路径应作为图像平面参考路线，不应视为 GPS 导航路线。")
    if any(rec.get("module_key") == "thermal" and rec.get("evidence_level") == "weak" for rec in simulated_records):
        lines.append("- 热红外模拟结果仅可用于热点模拟参考，不能作为真实温度依据。")
    if real_measurement_records:
        lines.append("- 真实测量证据可用于辅助复核异常区域，但仍需结合现场条件解释。")
    if weak_or_none:
        lines.append("- 当前缺少足够的强证据或中等证据，建议补充图像、模型推理或人工标注结果。")
    lines.extend(
        [
            "",
            "以下内容仅是辅助建议，不代表自动救援决策：",
            "- 不能输出“系统已自动完成救援决策”。",
            "- 不能输出“可直接按路径执行救援”。",
            "- 不能输出“已确认被困人员”。",
            "- 不能输出“已生成真实 GPS 导航路线”。",
        ]
    )
    return "\n".join(lines)


def build_truthfulness_boundary_section(context, sections):
    """Build the truthfulness boundary section."""
    return (
        "## 十二、真实性边界说明\n\n"
        "- 图像平面路径不是 GPS 导航。\n"
        "- Simulated Thermal 不是真测温。\n"
        "- Fast Preview 不是真 ODM。\n"
        "- S2-S3 最终版固定使用已训练语义分割模型生成 pred_mask；Uploaded/Demo Mask 不作为最终主展示来源。\n"
        "- Registry / Reference 模块不是运行结果。\n"
        "- Transformer human_candidate 不是已核实人员身份，不能确认幸存者或平民状态。\n"
        "- SKAI 外部源码级建筑灾损评估和 InaSAFE 外部源码级灾害影响评估只有在真实调用源码并验证输出文件后才标记为真实输出。\n"
        "- legacy/internal lightweight_skai_inasafe_adaptation 不得作为最终 S2-S3 灾情感知及影响评估主展示，也不得称为真实 SKAI 或 InaSAFE 结果。\n"
        "- 报告不能替代现场人工判断。"
    )


def _format_section_by_records(title, records, empty_message):
    return format_evidence_section(title, records, empty_message)


def build_final_report_v2(ledger=None, root_dir=None):
    """Build Final Report 2.0 from mission evidence ledger."""
    context = build_report_context(ledger=ledger, root_dir=root_dir)
    if not context.get("success"):
        return {
            "success": False,
            "report_markdown": "# 灾情感知及影响评估 证据链驱动综合救援报告\n\n报告生成失败。",
            "report_json": {
                "summary": {},
                "sections": {},
                "truthfulness_note": context.get("global_truthfulness_note", ""),
                "error": context.get("error", "unknown"),
            },
            "context": context,
            "sections": {},
            "truthfulness_note": context.get("global_truthfulness_note", ""),
        }

    sections = classify_report_sections(context)
    executive_summary = build_executive_summary(context, sections)
    report_sections = {
        "main_model_evidence": sections["main_model_evidence"],
        "real_measurement_evidence": sections["real_measurement_evidence"],
        "auxiliary_decision_evidence": sections["auxiliary_decision_evidence"],
        "ec_terp_ranking": [
            record
            for record in sections["auxiliary_decision_evidence"] + sections["simulated_or_preview_results"]
            if record.get("module_key") == "ec_terp_ranking"
        ],
        "simulated_or_preview_results": sections["simulated_or_preview_results"],
        "failed_or_unavailable_modules": sections["failed_or_unavailable_modules"],
        "not_run_modules": sections["not_run_modules"],
        "reference_or_registry_modules": sections["reference_or_registry_modules"],
        "human_review_items": sections["human_review_items"],
    }

    markdown_parts = [
        "# 灾情感知及影响评估 证据链驱动综合救援报告",
        "",
        "## 一、任务报告说明",
        "- 报告生成机制：由 Mission Evidence Ledger 自动驱动生成。",
        "- 辅助决策定位：该报告用于辅助决策，不替代人工救援判断。",
        "- 证据链来源：Module Execution Status Scanner 的扫描结果和本地 outputs/ 运行产物。",
        "",
        "## 二、证据链总览",
        f"- strong / medium / weak / none 数量：{context.get('summary', {}).get('strong_count', 0)} / {context.get('summary', {}).get('medium_count', 0)} / {context.get('summary', {}).get('weak_count', 0)} / {context.get('summary', {}).get('none_count', 0)}",
        f"- 可支持决策模块数量：{context.get('summary', {}).get('decision_support_count', 0)}",
        f"- 人工复核模块数量：{context.get('summary', {}).get('human_review_required_count', 0)}",
        "",
        executive_summary,
        "",
        _format_section_by_records("三、主要模型输出证据", sections["main_model_evidence"], "当前没有主要模型输出证据。"),
        "",
        _format_section_by_records("四、真实测量 / 真实产物证据", sections["real_measurement_evidence"], "当前没有真实测量 / 真实产物证据。"),
        "",
        _format_section_by_records("五、辅助决策证据", sections["auxiliary_decision_evidence"], "当前没有辅助决策证据。"),
        "",
        build_ec_terp_section(context, sections),
        "",
        _format_section_by_records("六、模拟 / 预览结果", sections["simulated_or_preview_results"], "当前没有模拟 / 预览结果。"),
        "",
        _format_section_by_records("七、执行失败 / 依赖缺失模块", sections["failed_or_unavailable_modules"], "当前没有执行失败 / 依赖缺失模块。"),
        "",
        _format_section_by_records("八、未执行模块", sections["not_run_modules"], "当前没有未执行模块。"),
        "",
        _format_section_by_records("九、参考 / 注册表模块", sections["reference_or_registry_modules"], "当前没有参考 / 注册表模块。"),
        "",
        build_human_review_section(sections["human_review_items"]),
        "",
        build_rescue_recommendations(context, sections),
        "",
        build_truthfulness_boundary_section(context, sections),
        "",
        "## 十三、全局说明",
        "- Evidence Ledger 来源：Mission Evidence Ledger 基于 Scanner 的状态和 outputs/ 本地产物生成。",
        "- 不根据代码文件存在判断模块执行状态。",
        "- 不替代人工救援判断。",
    ]

    report_markdown = "\n".join(part for part in markdown_parts if part is not None)
    report_json = {
        "summary": context.get("summary", {}),
        "sections": report_sections,
        "truthfulness_note": context.get("global_truthfulness_note", ""),
    }
    return {
        "success": True,
        "report_markdown": report_markdown,
        "report_json": report_json,
        "context": context,
        "sections": report_sections,
        "truthfulness_note": context.get("global_truthfulness_note", ""),
    }


def markdown_to_simple_html(markdown_text):
    """Convert markdown text to a very simple HTML page."""
    escaped = html.escape(markdown_text or "")
    return (
        "<html><head><meta charset='utf-8'><title>灾情感知及影响评估 Final Report 2.0</title></head>"
        "<body><pre style='white-space: pre-wrap; font-family: -apple-system, BlinkMacSystemFont, sans-serif;'>"
        + escaped
        + "</pre></body></html>"
    )


def save_final_report_v2(report=None, root_dir=None, output_dir=None):
    """Save Final Report 2.0 to markdown/html/json files."""
    if report is None:
        report = build_final_report_v2(root_dir=root_dir)
    if not isinstance(report, dict):
        raise FinalReportV2Error("report must be a dict or None.")

    output_root = Path(output_dir) if output_dir else REPORT_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "final_report_v2.md"
    html_path = output_root / "final_report_v2.html"
    json_path = output_root / "final_report_v2.json"

    markdown_text = report.get("report_markdown", "") if report.get("success") else report.get("report_markdown", "")
    json_text = json.dumps(report.get("report_json", {}), ensure_ascii=False, indent=2)

    markdown_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(markdown_to_simple_html(markdown_text), encoding="utf-8")
    json_path.write_text(json_text, encoding="utf-8")

    return {
        "success": True,
        "markdown_path": str(markdown_path),
        "html_path": str(html_path),
        "json_path": str(json_path),
        "message": "Final Report 2.0 已生成。",
    }
