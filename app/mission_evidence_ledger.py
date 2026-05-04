"""Mission evidence ledger for AeroRescue-AI.

This layer converts scanner status into evidence value.
It does not execute models or infer runtime success from code existence alone.
"""

import json
from pathlib import Path

from module_status_scanner import MODULE_SCAN_TARGETS, MODULE_STATUS, scan_all_modules


class MissionEvidenceLedgerError(Exception):
    pass


EVIDENCE_LEVELS = {
    "strong": "strong",
    "medium": "medium",
    "weak": "weak",
    "none": "none",
}


EVIDENCE_TYPES = {
    "model_output": "model_output",
    "real_measurement": "real_measurement",
    "rule_based_decision": "rule_based_decision",
    "image_plane_decision": "image_plane_decision",
    "simulated_or_preview": "simulated_or_preview",
    "uploaded_or_demo_input": "uploaded_or_demo_input",
    "reference_only": "reference_only",
    "not_available": "not_available",
}


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT_DIR / "outputs"
REPORT_DIR = OUTPUT_ROOT / "reports"

_LEVEL_PRIORITY = {
    EVIDENCE_LEVELS["strong"]: 3,
    EVIDENCE_LEVELS["medium"]: 2,
    EVIDENCE_LEVELS["weak"]: 1,
    EVIDENCE_LEVELS["none"]: 0,
}


def _as_path_list(value):
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return [str(value)]


def _unique_extend(base, items):
    for item in items:
        if item and item not in base:
            base.append(item)


def _make_record(
    module_scan_result,
    evidence_level,
    evidence_type,
    can_support_decision,
    can_enter_final_report,
    human_review_required,
    recommended_report_section,
    limitations,
    message,
):
    return {
        "module_key": module_scan_result.get("module_key", "unknown"),
        "module_name": module_scan_result.get("module_name", "unknown"),
        "display_name": module_scan_result.get("display_name", "unknown"),
        "scanner_status": module_scan_result.get("status", MODULE_STATUS["unknown"]),
        "evidence_level": evidence_level,
        "evidence_type": evidence_type,
        "can_support_decision": bool(can_support_decision),
        "can_enter_final_report": bool(can_enter_final_report),
        "human_review_required": bool(human_review_required),
        "evidence_files": _as_path_list(module_scan_result.get("evidence_files", [])),
        "limitations": limitations,
        "recommended_report_section": recommended_report_section,
        "truthfulness_note": module_scan_result.get("truthfulness_note", "") or "未提供真实性说明。",
        "message": message,
    }


def build_evidence_record(module_scan_result):
    """Convert a single scanner result into an evidence record."""
    if not isinstance(module_scan_result, dict):
        raise MissionEvidenceLedgerError("module_scan_result must be a dict.")

    module_key = str(module_scan_result.get("module_key", "unknown"))
    status = str(module_scan_result.get("status", MODULE_STATUS["unknown"]))
    capability_tags = set(module_scan_result.get("capability_tags", []) or [])
    truthfulness_note = module_scan_result.get("truthfulness_note", "") or "未提供真实性说明。"
    scanner_message = module_scan_result.get("message", "") or ""
    error_code = module_scan_result.get("error_code")
    message = scanner_message or (error_code or "")
    limitations = []

    def with_status_note(note):
        if note:
            limitations.append(note)

    if status in {MODULE_STATUS["not_run"], MODULE_STATUS["implemented_but_not_run"]}:
        with_status_note("该模块没有本次运行产物，不能作为当前任务证据。")
        if status == MODULE_STATUS["implemented_but_not_run"]:
            with_status_note("该模块已实现但本次未产生产物。")
        return _make_record(
            module_scan_result,
            EVIDENCE_LEVELS["none"],
            EVIDENCE_TYPES["not_available"],
            False,
            True,
            False,
            "未执行模块",
            limitations,
            message or "No runtime artifact found.",
        )

    if status == MODULE_STATUS["executed_failed"]:
        with_status_note("该模块运行失败，不能作为当前任务证据。")
        if error_code or scanner_message:
            with_status_note(f"错误信息：{error_code or scanner_message}")
        return _make_record(
            module_scan_result,
            EVIDENCE_LEVELS["none"],
            EVIDENCE_TYPES["not_available"],
            False,
            True,
            True,
            "执行失败 / 不可用模块",
            limitations,
            message or "Module execution failed.",
        )

    if status == MODULE_STATUS["dependency_missing"]:
        with_status_note("该模块由于依赖缺失未能形成可用证据。")
        if error_code or scanner_message:
            with_status_note(f"依赖/运行信息：{error_code or scanner_message}")
        return _make_record(
            module_scan_result,
            EVIDENCE_LEVELS["none"],
            EVIDENCE_TYPES["not_available"],
            False,
            True,
            False,
            "执行失败 / 不可用模块",
            limitations,
            message or "Dependency missing.",
        )

    if status == MODULE_STATUS["reference_only"]:
        with_status_note("该模块用于能力或参考资源管理，不是本次运行结果。")
        return _make_record(
            module_scan_result,
            EVIDENCE_LEVELS["none"],
            EVIDENCE_TYPES["reference_only"],
            False,
            True,
            False,
            "参考 / 注册表模块",
            limitations,
            message or "Reference-only module.",
        )

    if status == MODULE_STATUS["simulated_result"]:
        with_status_note("模拟结果不能作为真实测量依据。")
        return _make_record(
            module_scan_result,
            EVIDENCE_LEVELS["weak"],
            EVIDENCE_TYPES["simulated_or_preview"],
            False,
            True,
            True,
            "模拟 / 预览结果",
            limitations,
            message or "Simulated result.",
        )

    if status == MODULE_STATUS["preview_only"]:
        evidence_level = EVIDENCE_LEVELS["weak"]
        evidence_type = EVIDENCE_TYPES["simulated_or_preview"]
        can_support_decision = False
        if module_key == "path_planning" or "image_plane_reference_path" in capability_tags:
            evidence_level = EVIDENCE_LEVELS["medium"]
            evidence_type = EVIDENCE_TYPES["image_plane_decision"]
            can_support_decision = True
            with_status_note("路径规划为图像平面参考路径，不是真实 GPS 导航。")
        else:
            with_status_note("Fast Preview / 预览结果不能作为真实测量或真实产物证据。")
        return _make_record(
            module_scan_result,
            evidence_level,
            evidence_type,
            can_support_decision,
            True,
            True,
            "模拟 / 预览结果",
            limitations,
            message or "Preview result.",
        )

    if status == MODULE_STATUS["real_measurement"]:
        with_status_note("该结果属于真实测量或真实产物证据。")
        with_status_note("真实测量结果仍需结合现场条件解释。")
        return _make_record(
            module_scan_result,
            EVIDENCE_LEVELS["strong"],
            EVIDENCE_TYPES["real_measurement"],
            True,
            True,
            True,
            "真实测量 / 真实产物证据",
            limitations,
            message or "Real measurement artifact.",
        )

    if status == MODULE_STATUS["real_model_output"]:
        if "auxiliary_transformer_detection" in capability_tags:
            with_status_note("Transformer human_candidate 是辅助候选，不等于 confirmed civilian。")
            with_status_note("该模型输出属于辅助检测证据，需要人工复核。")
            return _make_record(
                module_scan_result,
                EVIDENCE_LEVELS["medium"],
                EVIDENCE_TYPES["model_output"],
                True,
                True,
                True,
                "辅助决策证据",
                limitations,
                message or "Auxiliary transformer model output.",
            )
        with_status_note("模型输出需要人工复核，不能替代真实救援判断。")
        return _make_record(
            module_scan_result,
            EVIDENCE_LEVELS["strong"],
            EVIDENCE_TYPES["model_output"],
            True,
            True,
            True,
            "主要模型输出证据",
            limitations,
            message or "Model output artifact.",
        )

    if status == MODULE_STATUS["executed_success"]:
        if module_key in {"odm", "orthomosaic"} and "real_odm_orthophoto" in capability_tags:
            with_status_note("该结果属于真实 ODM 正射产物证据。")
            with_status_note("真实测量结果仍需结合现场条件解释。")
            return _make_record(
                module_scan_result,
                EVIDENCE_LEVELS["strong"],
                EVIDENCE_TYPES["real_measurement"],
                True,
                True,
                True,
                "真实测量 / 真实产物证据",
                limitations,
                message or "Real ODM orthophoto artifact.",
            )

        evidence_level = EVIDENCE_LEVELS["medium"]
        evidence_type = EVIDENCE_TYPES["rule_based_decision"]
        can_support_decision = True
        can_enter_final_report = True
        human_review_required = True
        recommended_report_section = "辅助决策证据"

        if module_key == "terp":
            with_status_note("TERP 是规则/加权辅助优先级模型，不是自动救援决策。")
        elif module_key == "path_planning":
            evidence_type = EVIDENCE_TYPES["image_plane_decision"]
            with_status_note("路径规划为图像平面参考路径，不是真实 GPS 导航。")
        elif module_key == "decision_fusion":
            evidence_type = EVIDENCE_TYPES["image_plane_decision"]
            with_status_note("Decision Fusion 是轻量 image-plane adaptation，不是完整 GIS / SAREnv / SKAI / InaSAFE 输出。")
        elif module_key == "report_export":
            evidence_type = EVIDENCE_TYPES["rule_based_decision"]
            can_support_decision = False
            with_status_note("报告是结果汇总，不是独立感知证据。")
        elif module_key == "scene_description":
            with_status_note("场景描述为规则/模板汇总结果，需要结合底层证据理解。")
        elif module_key == "segmentation" and "uploaded_or_demo_mask_not_model_prediction" in capability_tags:
            evidence_type = EVIDENCE_TYPES["uploaded_or_demo_input"]
            with_status_note("Uploaded/Demo mask 可用于环境风险演示，但不是自动模型预测。")
        elif module_key == "transformer_detection" or "auxiliary_transformer_detection" in capability_tags:
            evidence_level = EVIDENCE_LEVELS["medium"]
            evidence_type = EVIDENCE_TYPES["model_output"]
            with_status_note("Transformer human_candidate 是辅助候选，不等于 confirmed civilian。")
        else:
            with_status_note("该模块属于规则/加权辅助证据，需要结合其他模块综合判断。")

        if "uploaded_or_demo_mask_not_model_prediction" in capability_tags and module_key != "segmentation":
            evidence_level = EVIDENCE_LEVELS["medium"]
            evidence_type = EVIDENCE_TYPES["uploaded_or_demo_input"]
            with_status_note("Uploaded/Demo mask 可用于环境风险演示，但不是自动模型预测。")

        if "image_plane_reference_path" in capability_tags and module_key == "path_planning":
            evidence_level = EVIDENCE_LEVELS["medium"]
            evidence_type = EVIDENCE_TYPES["image_plane_decision"]

        return _make_record(
            module_scan_result,
            evidence_level,
            evidence_type,
            can_support_decision,
            can_enter_final_report,
            human_review_required,
            recommended_report_section,
            limitations,
            message or "Executed successfully.",
        )

    with_status_note("模块状态未知，当前不应作为可靠证据使用。")
    return _make_record(
        module_scan_result,
        EVIDENCE_LEVELS["none"],
        EVIDENCE_TYPES["not_available"],
        False,
        True,
        True,
        "状态未知模块",
        limitations,
        message or "Unknown module status.",
    )


def build_mission_evidence_ledger(scan_result=None, root_dir=None):
    """Build the mission evidence ledger from a full module scan."""
    if scan_result is None:
        scan_result = scan_all_modules(root_dir=root_dir)
    if not isinstance(scan_result, dict):
        raise MissionEvidenceLedgerError("scan_result must be a dict or None.")

    modules = scan_result.get("modules", {})
    if not isinstance(modules, dict):
        modules = {}

    evidence_records = {}
    section_buckets = {
        "主要模型输出证据": [],
        "真实测量 / 真实产物证据": [],
        "辅助决策证据": [],
        "模拟 / 预览结果": [],
        "执行失败 / 不可用模块": [],
        "未执行模块": [],
        "参考 / 注册表模块": [],
        "状态未知模块": [],
    }

    summary = {
        "strong_count": 0,
        "medium_count": 0,
        "weak_count": 0,
        "none_count": 0,
        "decision_support_count": 0,
        "human_review_required_count": 0,
        "final_report_entry_count": 0,
    }

    for module_key, module_scan_result in modules.items():
        try:
            record = build_evidence_record(module_scan_result)
        except Exception as exc:
            fallback_scan = module_scan_result if isinstance(module_scan_result, dict) else {"module_key": module_key, "display_name": module_key, "module_name": module_key, "status": MODULE_STATUS["unknown"], "truthfulness_note": ""}
            record = _make_record(
                fallback_scan,
                EVIDENCE_LEVELS["none"],
                EVIDENCE_TYPES["not_available"],
                False,
                True,
                True,
                "状态未知模块",
                [f"证据记录生成失败：{exc}"],
                str(exc),
            )
        evidence_records[module_key] = record
        level = record.get("evidence_level", EVIDENCE_LEVELS["none"])
        summary_key = f"{level}_count"
        if summary_key in summary:
            summary[summary_key] += 1
        if record.get("can_support_decision"):
            summary["decision_support_count"] += 1
        if record.get("human_review_required"):
            summary["human_review_required_count"] += 1
        if record.get("can_enter_final_report"):
            summary["final_report_entry_count"] += 1
        section = record.get("recommended_report_section", "状态未知模块")
        if section not in section_buckets:
            section = "状态未知模块"
            section_buckets.setdefault(section, [])
        section_buckets[section].append(module_key)

    return {
        "success": True,
        "root_dir": scan_result.get("root_dir") or str(Path(root_dir).resolve() if root_dir else ROOT_DIR),
        "evidence_records": evidence_records,
        "summary": summary,
        "report_sections": section_buckets,
        "global_truthfulness_note": "Mission evidence is derived from module execution scan results and local artifacts, not from code existence alone.",
    }


def get_decision_supporting_evidence(ledger):
    """Return evidence records that can support decisions, strongest first."""
    if not isinstance(ledger, dict):
        return []
    records = list((ledger.get("evidence_records") or {}).values())
    records = [record for record in records if record.get("can_support_decision")]
    return sorted(records, key=lambda record: (-_LEVEL_PRIORITY.get(record.get("evidence_level"), 0), record.get("module_key", "")))


def get_human_review_items(ledger):
    """Return evidence records that require human review."""
    if not isinstance(ledger, dict):
        return []
    records = []
    for record in (ledger.get("evidence_records") or {}).values():
        if not record.get("human_review_required"):
            continue
        records.append(
            {
                "module_key": record.get("module_key"),
                "display_name": record.get("display_name"),
                "reason": record.get("message") or "需要人工复核。",
                "limitations": record.get("limitations", []),
                "truthfulness_note": record.get("truthfulness_note", ""),
            }
        )
    return records


def format_evidence_record_markdown(record):
    """Format a single evidence record into Chinese markdown."""
    if not isinstance(record, dict):
        return "### 证据记录\n\n记录格式无效。"
    evidence_files = record.get("evidence_files", []) or []
    evidence_lines = "\n".join(f"- {item}" for item in evidence_files) if evidence_files else "- 无证据文件"
    limitations = record.get("limitations", []) or []
    limitation_lines = "\n".join(f"- {item}" for item in limitations) if limitations else "- 无限制说明"
    return (
        f"### {record.get('display_name', record.get('module_key', 'unknown'))}\n"
        f"- 模块键：{record.get('module_key', 'unknown')}\n"
        f"- 证据等级：{record.get('evidence_level', 'none')}\n"
        f"- 证据类型：{record.get('evidence_type', 'not_available')}\n"
        f"- 是否支持决策：{'是' if record.get('can_support_decision') else '否'}\n"
        f"- 是否进入最终报告：{'是' if record.get('can_enter_final_report') else '否'}\n"
        f"- 是否需要人工复核：{'是' if record.get('human_review_required') else '否'}\n"
        f"- 证据文件：\n{evidence_lines}\n"
        f"- 局限说明：\n{limitation_lines}\n"
        f"- 真实性说明：{record.get('truthfulness_note', '')}\n"
    )


def format_mission_evidence_ledger_markdown(ledger):
    """Format the full mission evidence ledger into markdown."""
    if not isinstance(ledger, dict):
        return "# AeroRescue-AI 任务证据链总账\n\n证据总账格式无效。"

    sections = ledger.get("report_sections") or {}
    records = ledger.get("evidence_records") or {}

    lines = [
        "# AeroRescue-AI 任务证据链总账",
        "",
        "## 总览",
        f"- 强证据数量：{ledger.get('summary', {}).get('strong_count', 0)}",
        f"- 中等证据数量：{ledger.get('summary', {}).get('medium_count', 0)}",
        f"- 弱证据数量：{ledger.get('summary', {}).get('weak_count', 0)}",
        f"- 无证据数量：{ledger.get('summary', {}).get('none_count', 0)}",
        f"- 可支持决策模块数量：{ledger.get('summary', {}).get('decision_support_count', 0)}",
        f"- 需要人工复核模块数量：{ledger.get('summary', {}).get('human_review_required_count', 0)}",
        "",
    ]

    section_order = [
        "主要模型输出证据",
        "真实测量 / 真实产物证据",
        "辅助决策证据",
        "模拟 / 预览结果",
        "执行失败 / 不可用模块",
        "未执行模块",
        "参考 / 注册表模块",
        "状态未知模块",
    ]
    for section in section_order:
        items = sections.get(section, [])
        lines.append(f"## {section}")
        if not items:
            lines.append("- 无")
        for module_key in items:
            record = records.get(module_key)
            if record:
                lines.append(format_evidence_record_markdown(record))
        lines.append("")

    lines.extend(
        [
            "## 全局真实性说明",
            "证据链基于 Module Execution Status Scanner 的扫描结果和 outputs/ 本地运行产物生成，不根据代码文件存在与否判断模块是否执行成功。",
        ]
    )
    return "\n".join(lines)


def save_mission_evidence_ledger(ledger=None, root_dir=None, output_dir=None):
    """Save mission evidence ledger JSON and markdown."""
    if ledger is None:
        ledger = build_mission_evidence_ledger(root_dir=root_dir)
    output_root = Path(output_dir) if output_dir else REPORT_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "mission_evidence_ledger.json"
    markdown_path = output_root / "mission_evidence_ledger.md"
    json_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(format_mission_evidence_ledger_markdown(ledger), encoding="utf-8")
    return {
        "success": True,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }
