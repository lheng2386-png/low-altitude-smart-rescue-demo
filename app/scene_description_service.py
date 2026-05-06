import json
from datetime import datetime
from pathlib import Path

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT_DIR / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _read_text_or_json(value):
    if not value:
        return None
    path = None
    if isinstance(value, str):
        path = value
    elif isinstance(value, dict):
        path = value.get("path") or value.get("name")
    elif hasattr(value, "name"):
        path = value.name
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    return str(value)


def _risk_level_from_text(*texts):
    joined = "\n".join(text or "" for text in texts)
    if any(word in joined for word in ["High", "高风险", "极高", "热点"]):
        return "高"
    if any(word in joined for word in ["Medium", "中风险"]):
        return "中"
    return "低-中"


def _try_ollama(prompt, ollama_url, model):
    try:
        response = requests.post(
            f"{ollama_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=45,
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception:
        return ""
    return ""


def generate_scene_description(
    task_name,
    scene_note,
    detection_report_text,
    thermal_json_file=None,
    reconstruction_json_file=None,
    orthomosaic_json_file=None,
    use_ollama=False,
    ollama_url="http://127.0.0.1:11434",
    ollama_model="llama3.2",
):
    """Generate a markdown disaster-scene description with Ollama fallback."""
    task_name = task_name or "灾情感知及影响评估 应急救援任务"
    scene_note = scene_note or "未填写人工场景说明。"
    detection_text = detection_report_text or "尚未提供目标检测与综合决策报告。"
    thermal_text = _read_text_or_json(thermal_json_file) or "热红外分析模块尚未执行。"
    reconstruction_text = _read_text_or_json(reconstruction_json_file) or "三维重建模块尚未执行。"
    orthomosaic_text = _read_text_or_json(orthomosaic_json_file) or "正射影像生成模块尚未执行。"
    risk_level = _risk_level_from_text(detection_text, thermal_text, reconstruction_text, orthomosaic_text)

    base_md = f"""# AI 灾情描述

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 场景概述

任务名称：{task_name}

人工场景说明：{scene_note}

## 航测结果

{orthomosaic_text}

## 热红外风险

{thermal_text}

## 目标检测与救援优先级

{detection_text}

## 路径可达性

目标检测与综合决策报告中的 TERP、普通 A* 与风险感知 A* 对比结果作为路径可达性依据。

## 三维重建观察

{reconstruction_text}

## 综合风险等级

当前综合风险等级：{risk_level}

## 建议行动

- 优先复核 TERP 排名靠前目标。
- 若热红外热点明显，优先确认是否存在火源、人员或危险热源。
- 若正射/重建信息不足，应补充更多航测图像或视频。
- 路径建议仅为图像平面辅助参考，现场行动需结合真实道路、地形和指挥要求。
"""

    if use_ollama:
        prompt = (
            "请基于以下救援任务信息生成中文灾情分析报告，保持专业、简洁、面向应急救援：\n\n"
            + base_md
        )
        llm_text = _try_ollama(prompt, ollama_url, ollama_model)
        if llm_text:
            base_md += "\n## 本地大模型补充描述\n\n" + llm_text + "\n"
        else:
            base_md += "\n## 本地大模型状态\n\nOllama 不可用或未返回有效内容，已使用规则模板 fallback。\n"

    output_path = REPORT_DIR / "scene_description.md"
    output_path.write_text(base_md, encoding="utf-8")
    return base_md, str(output_path)

