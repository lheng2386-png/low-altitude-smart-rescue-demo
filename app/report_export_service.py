import html
import json
from datetime import datetime
from pathlib import Path


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

```json
{thermal_result}
```

## 目标检测与综合决策摘要

目标检测与综合决策模块在 Gradio 页面内生成检测图、风险排序、TERP 排名、路径规划摘要与中文救援报告。若需要纳入最终报告，请先在“AI 灾情描述”Tab 中粘贴该报告文本。

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

