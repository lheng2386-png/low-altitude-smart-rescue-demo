import html
import json
from datetime import datetime
from pathlib import Path

from decision_reference_registry import format_decision_reference_summary_for_report
from mission_evidence_ledger import build_mission_evidence_ledger, format_mission_evidence_ledger_markdown
from final_report_v2_service import build_final_report_v2, save_final_report_v2
from module_status_scanner import format_module_status_markdown, scan_all_modules


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT_DIR / "outputs"
REPORT_DIR = OUTPUT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _read(path):
    path = Path(path)
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return "该模块尚未执行。"


def _exists_text(path):
    path = Path(path)
    return str(path) if path.exists() else "该模块尚未执行。"


def _file_index():
    lines = []
    for directory in ["orthomosaic", "thermal", "detection", "reconstruction", "reports"]:
        base = OUTPUT_ROOT / directory
        if not base.exists():
            lines.append(f"- outputs/{directory}/：该目录尚未生成")
            continue
        files = [p for p in sorted(base.rglob("*")) if p.is_file()]
        if not files:
            lines.append(f"- outputs/{directory}/：暂无输出文件")
        for path in files:
            lines.append(f"- {path.relative_to(ROOT_DIR)}")
    return "\n".join(lines)


def _thermal_truthfulness_summary(thermal_result_text):
    try:
        data = json.loads(thermal_result_text)
    except Exception:
        return "热红外模块尚未执行或结果不是 JSON。"

    mode = data.get("thermal_mode", "unknown")
    is_real = bool(data.get("is_real_temperature_measurement"))
    note = data.get("truthfulness_note", "")
    if mode in {"simulated", "simulated_thermal"}:
        return "该结果为普通图像灰度热点分析，不是真实热红外测温。\n" + note
    if mode in {"radiometric", "radiometric_thermal"} and is_real:
        stats = data.get("statistics", {})
        return (
            "该结果来自 radiometric thermal 文件解析出的 temperature matrix。\n"
            f"file_type: {data.get('file_type', '')}\n"
            f"parser: {data.get('parser', '')}\n"
            f"max_temperature: {stats.get('max_temperature')}\n"
            f"mean_temperature: {stats.get('mean_temperature')}\n"
            f"hotspot_threshold: {stats.get('hotspot_threshold')}\n"
            f"hotspot_area_ratio: {stats.get('hotspot_area_ratio')}\n"
            f"{note}"
        )
    if mode in {"radiometric", "radiometric_thermal"}:
        return f"Radiometric Thermal 解析失败：{data.get('error', '')}\n未解析出真实 temperature matrix，不能自动 fallback 成假温度。\n{note}"
    if mode == "infrared_detection":
        return "红外目标检测不等于真实温度测量。\n" + note
    return note or "未提供热红外真实性说明。"


def format_decision_fusion_for_report(decision_fusion_result):
    if not decision_fusion_result:
        return "决策层参考融合尚未执行。"
    lines = [
        f"决策融合评分：{decision_fusion_result.get('decision_fusion_score', 0)}",
        f"决策融合等级：{decision_fusion_result.get('decision_fusion_level', 'unknown')}",
        f"人工复核：{'是' if decision_fusion_result.get('human_review_required', True) else '否'}",
        f"真实性说明：{decision_fusion_result.get('truthfulness_note', '')}",
    ]
    if decision_fusion_result.get("recommended_actions"):
        lines.append("建议动作：")
        for action in decision_fusion_result.get("recommended_actions", []):
            lines.append(f"- {action}")
    return "\n".join(lines)


def format_module_execution_status_for_report(root_dir=None):
    """Return a markdown summary of module execution status for future reports."""
    scan_result = scan_all_modules(root_dir=root_dir)
    return format_module_status_markdown(scan_result)


def format_mission_evidence_for_report(root_dir=None):
    """Return the mission evidence ledger markdown for future reports."""
    try:
        ledger = build_mission_evidence_ledger(root_dir=root_dir)
        return format_mission_evidence_ledger_markdown(ledger)
    except Exception as exc:
        return f"# AeroRescue-AI 任务证据链总账\n\n证据链生成失败：{exc}"


def export_final_report_v2(root_dir=None, output_dir=None):
    """Export the evidence-chain-driven final report v2."""
    try:
        report = build_final_report_v2(root_dir=root_dir)
        saved = save_final_report_v2(report=report, root_dir=root_dir, output_dir=output_dir)
        return (
            "Final Report 2.0 已生成。",
            saved["markdown_path"],
            saved["html_path"],
            saved["json_path"],
            report.get("report_markdown", ""),
        )
    except Exception as exc:
        return (
            f"Final Report 2.0 生成失败：{exc}",
            None,
            None,
            None,
            "",
        )


def export_final_report():
    """Export final markdown and HTML reports from existing module outputs."""
    orthomosaic_log = _read(OUTPUT_ROOT / "orthomosaic" / "processing_log.json")
    thermal_result = _read(OUTPUT_ROOT / "thermal" / "thermal_result.json")
    reconstruction_result = _read(OUTPUT_ROOT / "reconstruction" / "reconstruction_result.json")
    scene_description = _read(REPORT_DIR / "scene_description.md")

    md = f"""# AeroRescue-AI 综合救援报告

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 正射影像结果

```json
{orthomosaic_log}
```

## 热红外分析结果

{_thermal_truthfulness_summary(thermal_result)}

```json
{thermal_result}
```

## 目标检测与综合决策摘要

目标检测与综合决策模块在 Gradio 页面内生成检测图、风险排序、TERP 排名、路径规划摘要与中文救援报告。若需要纳入最终报告，请先在“AI 灾情描述”Tab 中粘贴该报告文本。

{format_decision_reference_summary_for_report()}

本阶段已预留 `outputs/decision_fusion/` 的后续接入能力，可在未来汇总 search priority、damage impact、coverage score 与 detection runtime evidence。

## 任务证据链总账

{format_mission_evidence_for_report()}

## 三维重建结果

```json
{reconstruction_result}
```

## AI 灾情描述

{scene_description}

## 输出文件索引

{_file_index()}
"""
    md_path = REPORT_DIR / "final_report.md"
    html_path = REPORT_DIR / "final_report.html"
    md_path.write_text(md, encoding="utf-8")
    html_body = "<html><head><meta charset='utf-8'><title>AeroRescue-AI 综合报告</title></head><body>"
    html_body += "<pre style='white-space: pre-wrap; font-family: -apple-system, BlinkMacSystemFont, sans-serif;'>"
    html_body += html.escape(md)
    html_body += "</pre></body></html>"
    html_path.write_text(html_body, encoding="utf-8")
    status = "综合报告已生成。未执行的模块已在报告中标记为“该模块尚未执行”。"
    return status, str(md_path), str(html_path), md
