import os
import json
import sys
import gradio as gr
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import shutil
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from fastapi import HTTPException, Request

ROOT_DIR_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_DIR_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR_FOR_IMPORT))

from priority_ranker import rank_targets
from report_generator import generate_report
from environment_risk import CLASS_DISPLAY_NAMES
from damage_assessment_service import (
    assess_damage_and_entry,
    format_damage_summary,
    format_entry_suggestion,
    format_scene_mode,
)
from path_planner import create_path_overlay, plan_rescue_path
from path_planner import (
    compare_path_plans,
    create_dual_path_overlay,
    plan_baseline_path,
    plan_risk_aware_path,
)
from segmentation_engine import (
    create_segmentation_overlay,
    load_segmentation_mask,
    resize_segmentation_mask,
    summarize_segmentation,
    validate_segmentation_mask,
    get_environment_context_for_target,
)
from segmentation_model import (
    get_default_segmentation_weights,
    get_segmentation_model_status,
    load_segmentation_model,
    predict_segmentation_mask,
)
from scene_applicability_gate import evaluate_scene_applicability
from terp_engine import rank_targets_by_terp
from orthomosaic_service import process_orthomosaic
from odm_service import check_odm_environment, run_odm_task
from thermal_service import analyze_thermal
from reconstruction_service import process_reconstruction
from scene_description_service import generate_scene_description
from report_export_service import export_final_report
from segmentation_model_service import predict_segmentation as predict_trained_segmentation
from backend.llm.report_assistant import generate_mission_report
from backend.llm.mission_copilot import answer_mission_copilot_question
from backend.llm.mission_planner import execute_mission_planner
from backend.llm.auditor import run_evidence_audit
from llm_report_panel import attach_llm_report_panel
from mission_copilot_panel import attach_mission_copilot_panel
from mission_planner_panel import attach_mission_planner_panel
from evidence_audit_panel import attach_evidence_audit_panel
from ui.mission_dashboard_panel import attach_mission_dashboard_panel
try:
    from ui.mission_control_panel import attach_mission_control_panel
    MISSION_CONTROL_PANEL_IMPORT_ERROR = ""
except Exception as exc:
    attach_mission_control_panel = None
    MISSION_CONTROL_PANEL_IMPORT_ERROR = str(exc)
try:
    from ui.validation_roadmap_panel import attach_validation_roadmap_panel
    VALIDATION_ROADMAP_PANEL_IMPORT_ERROR = ""
except Exception as exc:
    attach_validation_roadmap_panel = None
    VALIDATION_ROADMAP_PANEL_IMPORT_ERROR = str(exc)
from damage_segmentation_visualizer import (
    classify_damage_level,
    compute_damage_statistics,
    create_legend_image,
    create_segmentation_panel,
    render_segmentation_mask,
)
from external_impact_assessment import (
    build_external_impact_assessment_status,
    format_external_output_file_summary,
    format_external_impact_assessment_status,
    format_external_unavailable_reasons,
    format_inasafe_impact_assessment_panel,
    format_skai_building_damage_panel,
    save_external_impact_assessment_status,
)
from segmentation_source_metadata import (
    build_segmentation_source_metadata,
    format_segmentation_source_status,
    segmentation_visualization_note,
)
from scene_mode_and_entry_service import (
    analyze_scene_mode,
    build_path_planning_gate_result,
    build_path_planning_reliability_status,
    find_rescue_entry_point,
    format_path_planning_reliability_status,
)
from transformer_detection_service import (
    TRANSFORMER_DETECTION_MODELS,
    compare_yolo_and_transformer_detections,
    run_transformer_detection,
)
from detection_backend_registry import summarize_detection_backend_capabilities
from mission_orchestrator import initialize_mission_from_inputs, record_module_result, finalize_mission
from modules.reconstruction_3d.reconstruction_workflow import (
    check_reconstruction_dependencies,
    run_reconstruction_workflow,
)

TARGET_CLASS_DISPLAY_NAMES = {
    "civilian": "平民",
    "rescuer": "救援人员",
    "dog": "犬",
    "cat": "猫",
    "horse": "马",
    "cow": "牛",
    "person": "平民",
    "people": "平民",
    "rescuers": "救援人员",
}


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
STATIC_VIDEO_PATH = ROOT_DIR / "static" / "video" / "rescuer.mp4"
for output_name in ["orthomosaic", "thermal", "detection", "target_verification", "reconstruction", "reports"]:
    (ROOT_DIR / "outputs" / output_name).mkdir(parents=True, exist_ok=True)
(ROOT_DIR / "outputs" / "odm").mkdir(parents=True, exist_ok=True)
(ROOT_DIR / "outputs" / "reconstruction_3d").mkdir(parents=True, exist_ok=True)
(ROOT_DIR / "outputs" / "segmentation_inference").mkdir(parents=True, exist_ok=True)
MODEL_CACHE = {}
SEGMENTATION_MODEL_CACHE = {}
VIDEO_CLASS_MIN_CONF = {
    "dog": 0.45,
    "cat": 0.45,
}
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

TEXT = {
    "zh": {
        "page_title": "AeroRescue-AI 低空应急救援智能感知与辅助决策系统",
        "page_description": "YOLO 灾害目标检测、环境风险融合、TERP 优先级排序、风险感知 A* 路径规划与中文救援报告。",
        "image_tab": "图像",
        "gallery_tab": "展示",
        "video_tab": "视频",
        "language_label": "语言",
        "upload_image": "上传图像",
        "segmentation_source": "语义分割来源",
        "uploaded_mask": "上传掩码",
        "auto_segmentation_model": "自动分割模型",
        "none": "无分割",
        "optional_segmentation_mask": "可选语义分割掩码上传",
        "rescue_start_x": "救援起点 X",
        "rescue_start_y": "救援起点 Y（-1 表示使用底部默认值）",
        "confidence_threshold": "置信度阈值",
        "select_model": "选择模型",
        "select_model_info": "选择要使用的 YOLOv11 模型版本。",
        "process_image": "处理图像",
        "processed_image": "处理后图像",
        "segmentation_overlay": "分割叠加图",
        "path_overlay": "路径规划叠加图",
        "segmentation_status": "语义分割来源状态",
        "segmentation_status_placeholder": "语义分割来源状态会显示在这里……",
        "detection_details": "检测详情",
        "segmentation_summary": "语义分割汇总",
        "risk_ranking": "风险排序",
        "terp_ranking": "TERP 排名",
        "path_summary": "路径规划摘要",
        "path_comparison": "A* 路径对比",
        "rescue_report": "生成的救援报告",
        "example_images": "示例图像",
        "demo_gallery_title": "AeroRescue-AI 演示画廊",
        "workflow_title": "工作流程",
        "workflow_note": "UAV 图像 / 视频 → YOLOv11 检测 → 语义分割来源 → 环境风险融合 → TERP 优先级 → 救援排序 → A* 路径规划 → 中文救援报告。",
        "current_capability": "当前能力说明",
        "current_capability_note": "本地 Gradio 原型支持完整决策链路。",
        "local_assets": "本地 AeroRescue-AI 资源",
        "generated_outputs": "生成的演示案例输出",
        "segmentation_legend": "语义分割类别图例",
        "upload_video": "上传视频",
        "frame_skip": "帧跳过（越大越快）",
        "max_frames": "最大处理帧数（0 = 全视频）",
        "processed_video": "处理后视频",
        "predictions": "预测结果",
        "process_video": "处理视频",
        "example_videos": "示例视频",
        "all_target_none": "未检测到目标。",
        "no_detections": "未检测到目标。",
        "limited_frames": "（仅处理前 {max_frames} 帧）",
        "uploaded_mask_ok": "上传的掩码已成功读取。",
        "uploaded_mask_missing": "已选择上传掩码模式，但未上传掩码文件，已回退到无分割模式。",
        "auto_weights_missing": "未找到自动分割权重。已回退到无分割模式。请训练分割模型或上传掩码。",
        "auto_prediction_failed": "自动分割预测失败，已回退到无分割模式。",
        "auto_prediction_ok": "自动分割预测完成。",
        "auto_model_unavailable": "自动分割模型不可用，已回退到无分割模式。",
        "no_seg_selected": "未选择语义分割。",
        "fallback_no_seg": "风险评分和路径规划将回退到仅基于目标和默认代价的模式。",
    },
    "en": {
        "page_title": "AeroRescue-AI Low-Altitude Emergency Rescue Intelligence System",
        "page_description": "YOLO disaster target detection, environment-risk fusion, TERP ranking, Risk-Aware A* path planning, and English rescue reports.",
        "image_tab": "Image",
        "gallery_tab": "Gallery",
        "video_tab": "Video",
        "language_label": "Language",
        "upload_image": "Upload an Image",
        "segmentation_source": "Segmentation Source",
        "uploaded_mask": "Uploaded Mask",
        "auto_segmentation_model": "Auto Segmentation Model",
        "none": "None",
        "optional_segmentation_mask": "Optional Segmentation Mask Upload",
        "rescue_start_x": "Rescue Start X",
        "rescue_start_y": "Rescue Start Y (-1 means bottom default)",
        "confidence_threshold": "Confidence Threshold",
        "select_model": "Select Model",
        "select_model_info": "Select the YOLOv11 model variant to use.",
        "process_image": "Process Image",
        "processed_image": "Processed Image",
        "segmentation_overlay": "Segmentation Overlay",
        "path_overlay": "Path Planning Overlay",
        "segmentation_status": "Segmentation Source Status",
        "segmentation_status_placeholder": "Segmentation source status will appear here...",
        "detection_details": "Detection Details",
        "segmentation_summary": "Segmentation Summary",
        "risk_ranking": "Risk Ranking",
        "terp_ranking": "TERP Ranking",
        "path_summary": "Path Planning Summary",
        "path_comparison": "A* Path Comparison",
        "rescue_report": "Generated Rescue Report",
        "example_images": "Example Images",
        "demo_gallery_title": "AeroRescue-AI Demo Gallery",
        "workflow_title": "Workflow",
        "workflow_note": "UAV image / video → YOLOv11 detection → segmentation source → environment-risk fusion → TERP priority → rescue ranking → A* path planning → English rescue report.",
        "current_capability": "Current Capability Notes",
        "current_capability_note": "本地 Gradio 原型支持完整决策链路。",
        "local_assets": "Local AeroRescue-AI Assets",
        "generated_outputs": "生成的演示案例输出",
        "segmentation_legend": "Segmentation Class Legend",
        "upload_video": "Upload a Video",
        "frame_skip": "Frame Skip (higher = faster)",
        "max_frames": "Max Processed Frames (0 = full video)",
        "processed_video": "Processed Video",
        "predictions": "Predictions",
        "process_video": "Process Video",
        "example_videos": "Example Videos",
        "all_target_none": "No detections.",
        "no_detections": "No detections.",
        "limited_frames": "(limited to first {max_frames} frames)",
        "uploaded_mask_ok": "Uploaded mask loaded successfully.",
        "uploaded_mask_missing": "Uploaded Mask mode selected, but no mask file was uploaded. Falling back to no segmentation mask.",
        "auto_weights_missing": "Automatic segmentation weights not found. Falling back to no segmentation mask. Please train a segmentation model or upload a mask.",
        "auto_prediction_failed": "Auto segmentation prediction failed. Falling back to no segmentation mask.",
        "auto_prediction_ok": "Auto segmentation prediction completed.",
        "auto_model_unavailable": "Auto segmentation model is unavailable. Falling back to no segmentation mask.",
        "no_seg_selected": "No segmentation selected.",
        "fallback_no_seg": "Risk scoring and path planning will fall back to target-only/default-cost mode.",
    },
}

TABLE_HEADERS = {
    "zh": {
        "target": ["编号", "类别", "置信度", "边框", "中心点", "面积"],
        "summary": ["类别名", "显示名称", "面积占比(%)"],
        "ranking": ["排名", "目标ID", "类别", "置信度", "边框", "风险分数", "风险等级", "环境分数", "主导环境", "原因"],
        "terp": ["排名", "目标ID", "类别", "TERP 分数", "TERP 等级", "目标分数", "环境分数", "可达性分数", "原因"],
    },
    "en": {
        "target": ["id", "class", "confidence", "bbox", "center", "area"],
        "summary": ["class_name", "display_name", "area_percent"],
        "ranking": ["rank", "target_id", "class", "confidence", "bbox", "risk_score", "risk_level", "environment_score", "dominant_environment", "reason"],
        "terp": ["rank", "target_id", "class", "terp_score", "terp_level", "target_score", "environment_score", "accessibility_score", "reason"],
    },
}

RISK_LEVEL_LABELS = {
    "zh": {"Low": "低", "Medium": "中", "High": "高", "Critical": "极高"},
    "en": {"Low": "Low", "Medium": "Medium", "High": "High", "Critical": "Critical"},
}

CLASS_DISPLAY_NAMES_EN = {
    "background": "background",
    "water": "water",
    "no_damage_building": "no_damage_building",
    "minor_damage": "minor_damage",
    "major_damage": "major_damage",
    "destroyed_building": "destroyed_building",
    "vehicle": "vehicle",
    "road_clear": "road_clear",
    "road_blocked": "road_blocked",
    "tree": "tree",
    "pool": "pool",
}


def get_model_path(model_variant):
    weights_path = MODELS_DIR / model_variant / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}. "
            f"Place the trained file at models/{model_variant}/best.pt."
        )
    return str(weights_path)


def get_model(model_variant):
    if model_variant not in MODEL_CACHE:
        MODEL_CACHE[model_variant] = YOLO(get_model_path(model_variant))
    return MODEL_CACHE[model_variant]


def get_segmentation_weights_path():
    return get_default_segmentation_weights()


def get_auto_segmentation_model(weights_path):
    cache_key = str(weights_path)
    if cache_key not in SEGMENTATION_MODEL_CACHE:
        model, status = load_segmentation_model(weights_path)
        SEGMENTATION_MODEL_CACHE[cache_key] = (model, status)
    return SEGMENTATION_MODEL_CACHE[cache_key]


def lang_key(language):
    return "zh"


def t(language, key, **kwargs):
    text = TEXT[lang_key(language)][key]
    if kwargs:
        return text.format(**kwargs)
    return text


def zh_path_message(message):
    """Translate common internal path-planning status text for the Chinese UI."""
    if not message:
        return ""
    replacements = {
        "A* path planning succeeded.": "A* 路径规划成功。",
        "No target is available for path planning.": "当前没有可用于路径规划的目标。",
        "Path planning failed.": "路径规划失败。",
        "Path comparison is limited because one or both paths were not found.": "由于一条或两条路径未能生成，路径对比结果有限。",
        "No segmentation mask, path comparison is limited; baseline and risk-aware routes use equivalent assumptions.": "当前没有语义分割掩码，路径对比仅作演示，普通路径和风险感知路径使用近似假设。",
        "Risk-aware path reduces high-risk exposure compared with baseline.": "与普通 A* 相比，风险感知 A* 降低了高风险区域暴露。",
        "Risk-aware path does not reduce high-risk exposure in this case.": "本案例中风险感知 A* 未明显降低高风险区域暴露。",
        "Path environment risk calculated.": "路径环境风险已计算。",
    }
    text = str(message)
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("baseline", "普通 A*")
    text = text.replace("Baseline", "普通 A*")
    text = text.replace("Risk-Aware", "风险感知")
    text = text.replace("risk-aware", "风险感知")
    text = text.replace("segmentation mask", "语义分割掩码")
    text = text.replace("automatic segmentation", "自动语义分割")
    return text


def display_class_name(class_name, language):
    if lang_key(language) == "en":
        return class_name
    return TARGET_CLASS_DISPLAY_NAMES.get(class_name, CLASS_DISPLAY_NAMES.get(class_name, class_name))


def display_risk_level(risk_level, language):
    return RISK_LEVEL_LABELS[lang_key(language)].get(risk_level, risk_level)


def get_chinese_font(size=18):
    for font_path in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def image_header_html(language):
    return """
        <h1 style='text-align: center'>AeroRescue-AI</h1>
        <p style='text-align: center'>使用 Gradio（本地网页界面）构建 · YOLO（目标识别模型）灾害目标检测、可选语义分割（按区域理解道路/水域/建筑）风险融合与 A*（自动寻路算法）图像平面路径规划</p>
    """


def normalize_segmentation_source(value):
    text = str(value or "").strip().lower()
    if text in {"uploaded mask", "上传掩码"}:
        return "uploaded"
    if text in {"auto segmentation model", "自动分割模型"}:
        return "auto"
    return "none"


def demo_gallery_markdown(language):
    return """
## AeroRescue-AI 演示画廊

**工作流程**

无人机图像 / 视频  
→ YOLOv11 目标检测  
→ 已训练语义分割模型生成 pred_mask  
→ 环境风险融合  
→ TERP 优先级模型  
→ 场景适用性门控  
→ 救援优先级排序  
→ 普通 A* / 风险感知 A* 路径规划  
→ 中文救援报告

**Full Deep Fusion 展示层**

- ARGUS 风格平台工作流：任务、图像、检测、报告、案例归档。
- urban-disaster-monitor 风格检测模块：YOLOv11 灾害目标检测、示例输入与检测效果图。
- Detection-Models 风格模型对比：DINO、Faster R-CNN 等参考结构和 reference benchmark 展示。
- RescueNet 风格语义分割：11 类灾区环境类别、参考 mask 图、手工演示 mask。

**当前能力说明**

- 灾情感知：固定使用本地已训练语义分割模型，不生成伪造分割结果。
- TERP：融合目标、环境与路径可达性优先级。
- 场景适用性门控：目标或模型权重不足时明确提示，不夸大能力。
- 风险感知 A*：对比均匀代价基线路径与分割代价路径。
- 路径规划：图像平面参考路径，不是真实定位路线。

**核心创新**

- TERP 目标—环境—可达性联合救援优先级模型。
- 场景适用性门控。
- 风险感知 A* 图像平面救援路径规划。
- 感知-决策-报告闭环。
- 多仓库深度融合的低空无人机救援平台工作流。

**演示案例**

- 案例 1 洪涝平民救援：水域风险 + TERP + 风险感知 A*。
- 案例 2 建筑坍塌：严重损毁 / 完全毁坏建筑风险。
- 案例 3 道路阻断：阻断道路代价地图与绕行路径。
- 案例 4 多目标优先级：平民、动物、救援人员的 TERP 排序。
- 案例 5 无目标 / 低置信度：安全回退，不乱给路径。

在仓库根目录运行 `python scripts/generate_demo_cases.py` 即可生成完整本地展示输出。
    """


def segmentation_legend_markdown(language):
    return """
## 语义分割类别图例

| ID | 类别 | 风险含义 | 路径代价含义 |
| --- | --- | --- | --- |
| 1 | 水域 | 高风险 | 代价很高 |
| 7 | 可通行道路 | 低风险 | 低代价 |
| 8 | 道路阻断 | 高风险 | 高代价 |
| 4 | 严重损毁建筑 | 高风险 | 高代价 |
| 5 | 完全毁坏建筑 | 高风险 | 代价很高 |
    """


def custom_bounding_box(image, results, language="zh"):
    annotated_image = image.copy()
    pil_image = Image.fromarray(annotated_image)
    draw = ImageDraw.Draw(pil_image)
    font = get_chinese_font(18)

    class_names = results[0].names

    class_colors = {
        0: (255, 0, 0),      # vermelho
        1: (255, 255, 0),    # amarelo
        2: (0, 255, 0),      # verde
        3: (0, 0, 255),      # azul
        4: (255, 0, 255),    # magenta
        5: (0, 255, 255),    # ciano
    }

    font_colors = {
        0: (255, 255, 255),  # branco
        1: (0, 0, 0),        # preto
        2: (0, 0, 0),        # preto
        3: (255, 255, 255),  # branco
        4: (255, 255, 255),  # branco
        5: (0, 0, 0),        # preto
    }

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_color = class_colors.get(int(box.cls[0]), (255, 255, 255))  # fallback: branco
        text_color = font_colors.get(int(box.cls[0]), (0, 0, 0))        # fallback: preto
        class_label = display_class_name(class_names[int(box.cls[0])], language)
        label = f"{class_label} {float(box.conf[0]):.2f}"

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = max(x1, 0)
        text_y = max(y1 - text_h - 6, 0)

        draw.rectangle([text_x, text_y, text_x + text_w + 6, text_y + text_h + 6], fill=box_color)
        draw.text((text_x + 3, text_y + 2), label, fill=text_color, font=font)
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

    return np.array(pil_image)


def extract_targets(results):
    class_names = results[0].names
    targets = []

    for index, box in enumerate(results[0].boxes, start=1):
        cls_id = int(box.cls[0])
        confidence = round(float(box.conf[0]), 4)
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height

        targets.append(
            {
                "id": f"T{index:03d}",
                "class_name": class_names[cls_id],
                "confidence": confidence,
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "center": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
                "area": round(area, 2),
            }
        )

    return targets


def annotate_transformer_targets(image_rgb, targets, language="zh"):
    annotated_image = image_rgb.copy()
    pil_image = Image.fromarray(annotated_image)
    draw = ImageDraw.Draw(pil_image)
    font = get_chinese_font(18)
    box_color = (0, 180, 255)
    text_color = (0, 0, 0)

    for target in targets or []:
        x1, y1, x2, y2 = [int(round(float(value))) for value in target.get("bbox", [0, 0, 0, 0])[:4]]
        label = f"{display_class_name(target.get('class_name', 'human_candidate'), language)} {float(target.get('confidence', 0.0)):.2f}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = max(x1, 0)
        text_y = max(y1 - text_h - 6, 0)
        draw.rectangle([text_x, text_y, text_x + text_w + 6, text_y + text_h + 6], fill=box_color)
        draw.text((text_x + 3, text_y + 2), label, fill=text_color, font=font)
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

    return np.array(pil_image)


def target_table_rows(targets, language="zh"):
    return [
        [
            target["id"],
            display_class_name(target["class_name"], language),
            target["confidence"],
            target["bbox"],
            target["center"],
            target["area"],
        ]
        for target in targets
    ]


def ranking_table_rows(ranked_targets, language="zh"):
    return [
        [
            target["rank"],
            target["target_id"],
            display_class_name(target["class_name"], language),
            target["confidence"],
            target["bbox"],
            target["risk_score"],
            display_risk_level(target["risk_level"], language),
            target.get("environment_score", 0.0),
            display_class_name(target.get("environment", "not_available"), language)
            if target.get("environment", "not_available") != "not_available"
            else ("not_available" if lang_key(language) == "en" else "未提供"),
            target["reason"],
        ]
        for target in ranked_targets
    ]


def terp_table_rows(terp_rankings, language="zh"):
    return [
        [
            target["rank"],
            target["target_id"],
            display_class_name(target["class_name"], language),
            target["terp_score"],
            display_risk_level(target["terp_level"], language),
            target["target_score"],
            target["environment_score"],
            target["accessibility_score"],
            target["reason"],
        ]
        for target in terp_rankings
    ]


def segmentation_summary_rows(segmentation_summary, language="zh"):
    return [
        [
            class_name,
            display_class_name(class_name, language),
            round(float(ratio) * 100, 2),
        ]
        for class_name, ratio in segmentation_summary.items()
    ]


def segmentation_validation_lines(validation, language="zh"):
    message = validation.get("message", "Unknown validation result.")
    if lang_key(language) == "zh":
        message_map = {
            "No segmentation mask is available.": "当前没有可用的语义分割掩码。",
            "Mask contains only background; environment risk may be limited.": "掩码仅包含背景，环境风险可能较低。",
            "Segmentation mask is valid.": "语义分割掩码有效。",
        }
        if message.startswith("Segmentation mask contains unknown class ids:"):
            message = message.replace("Segmentation mask contains unknown class ids:", "语义分割掩码包含未知类别 ID：")
        else:
            message = message_map.get(message, message)
    else:
        message_map = {
            "当前没有可用的语义分割掩码。": "No segmentation mask is available.",
            "掩码仅包含背景，环境风险可能较低。": "Mask contains only background; environment risk may be limited.",
            "语义分割掩码有效。": "Segmentation mask is valid.",
        }
        if message.startswith("语义分割掩码包含未知类别 ID："):
            message = message.replace("语义分割掩码包含未知类别 ID：", "Segmentation mask contains unknown class ids:")
        else:
            message = message_map.get(message, message)
    if lang_key(language) == "en":
        return [
            f"Mask validation result: {message}",
            f"Mask size: {validation.get('width', 0)}x{validation.get('height', 0)}",
            f"Unique class ids: {validation.get('unique_class_ids', [])}",
            f"Unknown class ids: {validation.get('unknown_class_ids', [])}",
        ]
    return [
        f"掩码验证结果：{message}",
        f"掩码尺寸：{validation.get('width', 0)}x{validation.get('height', 0)}",
        f"唯一类别 ID：{validation.get('unique_class_ids', [])}",
        f"未知类别 ID：{validation.get('unknown_class_ids', [])}",
    ]


def prepare_valid_segmentation_mask(mask, image_width, image_height, segmentation_status, language="zh"):
    """Resize and validate a mask. Invalid masks fall back to no segmentation."""
    if mask is None:
        validation = validate_segmentation_mask(None)
        segmentation_status.extend(segmentation_validation_lines(validation, language))
        return None, {}

    aligned_mask = resize_segmentation_mask(mask, image_width, image_height)
    validation = validate_segmentation_mask(aligned_mask)
    segmentation_status.extend(segmentation_validation_lines(validation, language))
    if not validation.get("valid"):
        segmentation_status.append(
            "Invalid segmentation mask. Falling back to no segmentation mask."
            if lang_key(language) == "en"
            else "掩码无效，已回退到无分割模式。"
        )
        return None, {}

    return aligned_mask, summarize_segmentation(aligned_mask)


def gallery_image_items(language="zh"):
    """Return available local demo and reference images."""
    candidates = [
        (ROOT_DIR / "static" / "images" / "showcase" / "aerorescue_gradio_interface.png", "AeroRescue-AI Gradio interface"),
        (ROOT_DIR / "static" / "images" / "showcase" / "flood_demo_input.jpg", "Flood demo input"),
        (ROOT_DIR / "static" / "images" / "showcase" / "flood_demo_detection.webp", "Flood demo detection preview"),
        (ROOT_DIR / "static" / "images" / "reference" / "detection_result_reference.png", "Detection reference asset"),
        (ROOT_DIR / "static" / "images" / "reference" / "detection_metrics_reference.png", "Detection metrics reference asset"),
        (ROOT_DIR / "static" / "images" / "capa1.webp", "Disaster response scenario"),
        (ROOT_DIR / "static" / "images" / "capa2.webp", "Low-altitude rescue context"),
        (ROOT_DIR / "static" / "images" / "app_gradio.png", "AeroRescue-AI interface"),
        (ROOT_DIR / "static" / "images" / "230714-india-flooding-mb-0831-d3a66d.jpg", "Local demo input"),
        (ROOT_DIR / "static" / "images" / "230714-india-flooding-mb-0831-d3a66d_annotated.webp", "Local detection output"),
        (ROOT_DIR / "static" / "images" / "modelo-customizado.png", "Rescue-class detector output"),
        (ROOT_DIR / "static" / "images" / "modelo-coco.png", "Generic detector comparison"),
        (ROOT_DIR / "static" / "images" / "metricas0.5.png", "mAP@0.5 comparison"),
        (ROOT_DIR / "static" / "images" / "metricas-classes.png", "Class-level metrics"),
    ]
    zh_captions = {
        "AeroRescue-AI Gradio interface": "AeroRescue-AI Gradio 界面",
        "Flood demo input": "洪涝示例输入",
        "Flood demo detection preview": "洪涝检测预览",
        "Detection reference asset": "检测参考素材",
        "Detection metrics reference asset": "检测指标参考素材",
        "Disaster response scenario": "灾害响应场景",
        "Low-altitude rescue context": "低空救援场景",
        "AeroRescue-AI interface": "AeroRescue-AI 界面",
        "Local demo input": "本地示例输入",
        "Local detection output": "本地检测输出",
        "Rescue-class detector output": "救援类别检测输出",
        "Generic detector comparison": "通用检测器对比",
        "mAP@0.5 comparison": "mAP@0.5 对比",
        "Class-level metrics": "类别级指标",
    }
    items = [(str(path), zh_captions.get(caption, caption)) for path, caption in candidates if path.exists()]

    reference_dirs = [
        (ROOT_DIR / "static" / "images" / "reference" / "argus", "ARGUS 风格平台参考"),
        (ROOT_DIR / "static" / "images" / "reference" / "urban_disaster_monitor", "灾害目标检测参考"),
        (ROOT_DIR / "static" / "images" / "reference" / "detection_models", "模型对比参考"),
        (ROOT_DIR / "static" / "images" / "reference" / "rescuenet", "语义分割参考"),
        (ROOT_DIR / "static" / "images" / "showcase" / "aerorescue_outputs", "AeroRescue-AI 本项目输出"),
    ]
    for directory, label in reference_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            caption = f"{label}：{path.stem.replace('_', ' ')}"
            items.append((str(path), caption))
    return items


def demo_case_gallery_items(language="zh"):
    """Return generated demo case showcase images when available."""
    showcase_dir = ROOT_DIR / "static" / "images" / "showcase"
    preferred_names = [
        "input.jpg",
        "detection_overlay.png",
        "segmentation_overlay.png",
        "risk_aware_path_overlay.png",
        "dual_path_overlay.png",
    ]
    items = []
    for case_dir in sorted(showcase_dir.glob("case_*")):
        if not case_dir.is_dir():
            continue
        case_label = case_dir.name.replace("_", " ").title()
        if lang_key(language) == "zh":
            case_label = case_label.replace("Case 1 Flood Civilian Rescue", "案例 1 · 洪涝平民救援")
            case_label = case_label.replace("Case 2 Building Collapse", "案例 2 · 建筑坍塌")
            case_label = case_label.replace("Case 3 Road Blocked", "案例 3 · 道路阻断")
            case_label = case_label.replace("Case 4 Multi Target", "案例 4 · 多目标优先级")
            case_label = case_label.replace("Case 5 No Target Or Fallback", "案例 5 · 无目标 / 回退")
        for name in preferred_names:
            path = case_dir / name
            if path.exists():
                artifact_name = {
                    "input.jpg": "输入图像",
                    "detection_overlay.png": "检测结果图",
                    "segmentation_overlay.png": "语义分割叠加图",
                    "risk_aware_path_overlay.png": "风险感知路径图",
                    "dual_path_overlay.png": "双路径对比图",
                }.get(name, name)
                caption = f"{case_label}：{artifact_name}"
                items.append((str(path), caption))
    return items


def _resolve_video_path(video_path):
    if video_path is None:
        return None
    if isinstance(video_path, dict):
        return video_path.get("path") or video_path.get("name")
    if hasattr(video_path, "name"):
        return video_path.name
    return str(video_path)


def _create_video_writer(path, fps, width, height):
    """Create a browser-friendly video writer with codec fallback."""
    for fourcc_name in ("VP80", "VP90", "avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_name), fps, (width, height))
        if writer.isOpened():
            return writer
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def path_summary_text(path_result, has_segmentation_mask, language="zh"):
    if not path_result or not path_result.get("found"):
        if lang_key(language) == "en":
            return f"Path planning result: {path_result.get('message', 'No valid path could be generated.')}" if path_result else "Path planning result: No valid path could be generated."
        return f"路径规划结果：{path_result.get('message', '当前未能生成有效路径。')}" if path_result else "路径规划结果：当前未能生成有效路径。"

    start = path_result.get("start", [0, 0])
    goal = path_result.get("goal", [0, 0])
    target_id = path_result.get("target_id", "unknown")
    target_class = path_result.get("target_class", "unknown")
    target_class_display = display_class_name(target_class, language)
    if lang_key(language) == "en":
        summary = [
            "Path planning result: A* path planning succeeded.",
            f"Start S: ({start[0]}, {start[1]})",
            f"Goal T: {target_id} / {target_class_display} @ ({goal[0]}, {goal[1]})",
            f"Path length: {path_result.get('path_length', 0)} pixels",
            f"Total path cost: {path_result.get('total_cost', 0.0):.2f}",
        ]
    else:
        summary = [
            "路径规划结果：A* 路径规划成功。",
            f"起点 S：({start[0]}, {start[1]})",
            f"终点 T：{target_id} / {target_class_display} @ ({goal[0]}, {goal[1]})",
            f"路径长度：{path_result.get('path_length', 0)} 个像素点",
            f"累计路径代价：{path_result.get('total_cost', 0.0):.2f}",
        ]
    if has_segmentation_mask:
        summary.append(
            "The path planning has incorporated segmentation-mask or automatic-segmentation environment costs."
            if lang_key(language) == "en"
            else "当前路径规划已结合语义分割掩码或自动分割结果的环境代价。"
        )
    else:
        summary.append(
            "No segmentation mask was uploaded, so path planning only uses the default image-plane cost map."
            if lang_key(language) == "en"
            else "当前未上传语义分割掩码，路径规划仅基于图像平面默认代价地图。"
        )
    summary.append(f"说明：{zh_path_message(path_result.get('message', 'A* 路径规划成功。'))}")
    return "\n".join(summary)


def path_comparison_text(comparison, language="zh"):
    if not comparison:
        return "Path comparison result: no path comparison could be generated." if lang_key(language) == "en" else "路径对比结果：当前无法生成路径对比。"

    if lang_key(language) == "en":
        lines = [
            "A* Path Comparison:",
            f"Baseline path length: {comparison.get('baseline_length', 0)}",
            f"Baseline total cost: {comparison.get('baseline_cost', 0.0):.2f}",
            f"Risk-Aware path length: {comparison.get('risk_aware_length', 0)}",
            f"Risk-Aware total cost: {comparison.get('risk_aware_cost', 0.0):.2f}",
            f"Baseline high-risk ratio: {comparison.get('baseline_environment_risk', 0.0) * 100:.2f}%",
            f"Risk-Aware high-risk ratio: {comparison.get('risk_aware_environment_risk', 0.0) * 100:.2f}%",
            f"Path risk reduction: {comparison.get('risk_reduction', 0.0) * 100:.2f}%",
            f"Explanation: {comparison.get('message', '')}",
        ]
    else:
        lines = [
            "A* 路径对比：",
            f"普通 A* 路径长度：{comparison.get('baseline_length', 0)}",
            f"普通 A* 累计代价：{comparison.get('baseline_cost', 0.0):.2f}",
            f"风险感知 A* 路径长度：{comparison.get('risk_aware_length', 0)}",
            f"风险感知 A* 累计代价：{comparison.get('risk_aware_cost', 0.0):.2f}",
            f"普通 A* 高风险区域比例：{comparison.get('baseline_environment_risk', 0.0) * 100:.2f}%",
            f"风险感知 A* 高风险区域比例：{comparison.get('risk_aware_environment_risk', 0.0) * 100:.2f}%",
            f"路径环境风险降低：{comparison.get('risk_reduction', 0.0) * 100:.2f}%",
            f"说明：{zh_path_message(comparison.get('message', ''))}",
        ]
    return "\n".join(lines)


def disabled_path_result(message):
    """Create a consistent disabled path-planning result."""
    return {
        "found": False,
        "start": None,
        "goal": None,
        "target_id": None,
        "target_class": None,
        "path": [],
        "total_cost": 0.0,
        "path_length": 0,
        "mode": "disabled",
        "message": message,
    }


def scene_gate_status_text(scene_gate):
    """Format scene applicability gate state for the Gradio UI."""
    level = scene_gate.get("level", "Fallback")
    level_labels = {
        "Full": "Full：目标 + 语义分割 + 环境风险 + 风险感知路径规划",
        "Target Only": "Target Only：仅目标检测 + 默认图像平面路径规划",
        "Fallback": "Fallback：自动能力不可用，已回退到可运行基础流程",
        "No Target": "No Target：未检测到明确救援目标",
    }
    return f"{level_labels.get(level, level)}\n{scene_gate.get('message', '')}"


def run_orthomosaic_mode(image_files, processing_mode, odm_task_name, odm_max_images, odm_fast_orthophoto):
    """Dispatch orthomosaic UI requests to preview or real ODM mode."""
    if str(processing_mode or "").startswith("Real ODM"):
        max_images = None
        if odm_max_images is not None and int(odm_max_images) > 0:
            max_images = int(odm_max_images)
        return run_odm_task(
            image_files,
            task_name=odm_task_name or "aerorescue_odm_task",
            max_images=max_images,
            fast_orthophoto=bool(odm_fast_orthophoto),
        )

    preview_path, status, log_json = process_orthomosaic(image_files)
    status = "Fast Preview / OpenCV 拼接预览：这是快速航测拼接预览，不是专业正射影像。\n" + status
    log_text = "当前模式未调用 OpenDroneMap / ODM。\n" + log_json
    return preview_path, status, log_json, log_text


def _upload_path(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)


def _prepare_reconstruction_image_dir(image_files, run_id):
    """Copy uploaded images into a stable workflow input directory."""
    files = image_files or []
    if not isinstance(files, list):
        files = [files]
    input_dir = ROOT_DIR / "outputs" / "reconstruction_3d" / "ui_inputs" / run_id / "images"
    input_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    used_names = set()
    for index, item in enumerate(files, start=1):
        raw_path = _upload_path(item)
        if not raw_path:
            continue
        source = Path(raw_path)
        if not source.exists():
            continue
        candidate = source.name
        if candidate in used_names or (input_dir / candidate).exists():
            candidate = f"{source.stem}_{index:04d}{source.suffix.lower()}"
        used_names.add(candidate)
        shutil.copy2(source, input_dir / candidate)
        copied += 1
    return str(input_dir) if copied else None


def _workflow_status_text(result):
    lines = [
        f"Workflow status: {result.get('status')}",
        f"Success: {result.get('success')}",
        f"Mode: {result.get('mode')}",
        f"Workflow status JSON: {result.get('workflow_status_path')}",
        "",
        "Step summary:",
    ]
    for step in result.get("steps", []):
        marker = "OK" if step.get("success") else "NO"
        lines.append(f"- [{marker}] {step.get('name')}: {step.get('status')} {step.get('message') or ''}".rstrip())
    lines.extend(
        [
            "",
            "Truthfulness boundaries:",
            "- Fast Preview is not a real ODM orthophoto.",
            "- 360 panorama preview is not true 3D reconstruction.",
            "- No GPS/GCP/EXIF geotags means outputs are relative-scale or non-georeferenced.",
            "- Outputs are auxiliary spatial evidence and require human review.",
        ]
    )
    return "\n".join(lines)


def run_reconstruction_dependency_check(panorama_sfm_script, odm_docker_image):
    status = check_reconstruction_dependencies(
        include_colmap=True,
        include_odm=True,
        odm_image=odm_docker_image or "opendronemap/odm",
        panorama_sfm_script=panorama_sfm_script or None,
    )
    text = [
        "依赖状态 / Dependency status:",
        f"- OpenCV/cv2: {status.get('opencv', {}).get('status')}",
        f"- COLMAP: {status.get('colmap', {}).get('status')}",
        f"- panorama_sfm.py: {status.get('panorama_sfm_script', {}).get('status')}",
        f"- Docker: {status.get('docker', {}).get('status')}",
        f"- ODM image: {status.get('odm_image', {}).get('status')}",
        "",
        "缺失依赖不会触发伪造输出；真实重建只在外部工具运行并生成实际文件后才标记成功。",
    ]
    return "\n".join(text), json.dumps(status, ensure_ascii=False, indent=2)


def run_real_reconstruction_workflow_ui(
    mode,
    video_file,
    image_files,
    fps,
    run_quality_filter,
    blur_threshold,
    brightness_threshold,
    colmap_matcher,
    colmap_run_dense,
    colmap_run_mesher,
    panorama_sfm_script,
    odm_project_name,
    odm_docker_image,
    odm_camera_lens,
    odm_feature_quality,
    odm_pc_quality,
    odm_dsm,
    odm_dtm,
    odm_fast_orthophoto,
    odm_auto_pull,
    timeout_seconds,
):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT_DIR / "outputs" / "reconstruction_3d" / "workflow" / run_id
    video_path = _upload_path(video_file)
    image_dir = _prepare_reconstruction_image_dir(image_files, run_id) if image_files else None
    if not video_path and not image_dir and mode != "report_only":
        result = {
            "success": False,
            "status": "invalid_input",
            "mode": mode,
            "message": "请上传视频或图片文件夹图像；report_only 模式除外。",
        }
        return _workflow_status_text(result), json.dumps(result, ensure_ascii=False, indent=2), None, None, None

    result = run_reconstruction_workflow(
        mode=mode,
        output_dir=str(output_dir),
        video_path=video_path,
        image_dir=image_dir,
        fps=float(fps or 1.0),
        run_quality_filter=bool(run_quality_filter),
        blur_threshold=float(blur_threshold or 100.0),
        brightness_threshold=float(brightness_threshold or 30.0),
        colmap_matcher=colmap_matcher or "sequential",
        colmap_run_dense=bool(colmap_run_dense),
        colmap_run_mesher=bool(colmap_run_mesher),
        panorama_sfm_script=panorama_sfm_script or None,
        odm_project_name=odm_project_name or "aerorescue_odm",
        odm_docker_image=odm_docker_image or "opendronemap/odm",
        odm_camera_lens=odm_camera_lens or "auto",
        odm_feature_quality=odm_feature_quality or "medium",
        odm_pc_quality=odm_pc_quality or "medium",
        odm_dsm=bool(odm_dsm),
        odm_dtm=bool(odm_dtm),
        odm_use_fast_orthophoto=bool(odm_fast_orthophoto),
        odm_auto_pull=bool(odm_auto_pull),
        timeout=int(timeout_seconds) if timeout_seconds and int(timeout_seconds) > 0 else None,
    )
    report_paths = (result.get("report") or {}).get("paths") or {}
    return (
        _workflow_status_text(result),
        json.dumps(result, ensure_ascii=False, indent=2),
        result.get("workflow_status_path"),
        report_paths.get("json"),
        report_paths.get("markdown"),
    )


def _target_route_item(target):
    return {
        "target_id": target.get("id"),
        "class_name": target.get("class_name"),
        "center": target.get("center"),
        "bbox": target.get("bbox"),
    }


def run_damage_segmentation_inference(image, segmentation_mode, mask_file, checkpoint_path, img_size):
    """Run standalone disaster-scene segmentation and damage assessment."""
    if image is None:
        metadata = build_segmentation_source_metadata("none", fallback_reason="未上传图像")
        return None, None, None, "{}", "Unknown", create_legend_image(), format_segmentation_source_status(metadata), "请先上传一张图像。"

    mode = str(segmentation_mode or "Uploaded Mask")
    status_lines = []
    mask = None

    if mode.startswith("Auto") or mode.startswith("自动"):
        checkpoint = str(checkpoint_path).strip() if checkpoint_path else None
        if not checkpoint:
            checkpoint = None
        mask, status_text, metadata = predict_trained_segmentation(
            image,
            checkpoint_path=checkpoint,
            img_size=int(img_size or 512),
        )
        source_metadata = build_segmentation_source_metadata(
            "auto_model",
            checkpoint_path=metadata.get("checkpoint_path") if isinstance(metadata, dict) else checkpoint,
            model_available=bool(metadata.get("ok")) if isinstance(metadata, dict) else False,
            prediction_success=mask is not None,
            fallback_reason=None if mask is not None else "未找到 checkpoint 或推理失败",
        )
        status_lines.append(status_text)
        status_lines.append(f"元数据：{json.dumps(metadata, ensure_ascii=False)}")
        if mask is None:
            stats = {
                "message": "当前未检测到可用的语义分割模型 checkpoint，无法执行自动分割。请上传 mask 或先训练模型。",
                "no_fake_result": True,
                "segmentation_source": source_metadata,
            }
            status_lines.append("没有 checkpoint 时不会生成假分割图。")
            return image, None, None, json.dumps(stats, ensure_ascii=False, indent=2), "Unknown", create_legend_image(), format_segmentation_source_status(source_metadata), "\n".join(status_lines)
    elif mode.startswith("Uploaded") or mode.startswith("上传"):
        if not mask_file:
            source_metadata = build_segmentation_source_metadata("uploaded_mask", fallback_reason="未上传 mask")
            stats = {"message": "未上传 segmentation mask。", "segmentation_source": source_metadata}
            return image, None, None, json.dumps(stats, ensure_ascii=False, indent=2), "Unknown", create_legend_image(), format_segmentation_source_status(source_metadata), "Uploaded Mask 模式需要上传 class-id mask。"
        mask_path = mask_file.name if hasattr(mask_file, "name") else mask_file
        mask = load_segmentation_mask(mask_path)
        mask = resize_segmentation_mask(mask, image.size[0], image.size[1])
        source_metadata = build_segmentation_source_metadata(
            "uploaded_mask",
            mask_path=mask_path,
            prediction_success=True,
        )
        status_lines.append("已加载用户上传的 segmentation mask。该结果来自上传掩码，不代表自动模型预测。")
    else:
        source_metadata = build_segmentation_source_metadata(
            "demo_fallback",
            fallback_reason="Demo / Fallback 模式不生成模型预测结果",
        )
        stats = {
            "message": "Demo / Fallback 模式不会生成伪造 mask。请上传 mask 或提供训练好的 checkpoint。",
            "segmentation_source": source_metadata,
        }
        return image, None, None, json.dumps(stats, ensure_ascii=False, indent=2), "Unknown", create_legend_image(), format_segmentation_source_status(source_metadata), stats["message"]

    validation = validate_segmentation_mask(mask)
    status_lines.append(f"掩码验证：{validation.get('message')}")
    if not validation.get("valid"):
        source_metadata["fallback_reason"] = "mask 验证失败"
        stats = {"validation": validation, "message": "mask 无效，未继续进行损毁评估。", "segmentation_source": source_metadata}
        return image, None, None, json.dumps(stats, ensure_ascii=False, indent=2), "Unknown", create_legend_image(), format_segmentation_source_status(source_metadata), "\n".join(status_lines)

    color_mask = render_segmentation_mask(mask)
    panel = create_segmentation_panel(image, color_mask)
    stats = compute_damage_statistics(mask)
    damage_level = classify_damage_level(stats)
    stats["overall_damage_level"] = damage_level
    stats["segmentation_source"] = source_metadata
    stats["visualization_note"] = segmentation_visualization_note(source_metadata)
    return (
        image,
        color_mask,
        panel,
        json.dumps(stats, ensure_ascii=False, indent=2),
        damage_level,
        create_legend_image(),
        format_segmentation_source_status(source_metadata),
        "\n".join(status_lines),
    )


def image_detection(
    image,
    detection_backend,
    transformer_model_key,
    segmentation_source,
    segmentation_mask_path,
    start_x,
    start_y,
    use_manual_start,
    force_path_planning,
    conf_threshold,
    model_variant,
    language="zh",
):
    if image is None:
        message = "Please upload an image first." if lang_key(language) == "en" else "请先上传一张图像。"
        no_target_gate = {
            "level": "No Target",
            "message": "当前未上传图像，无法进行目标检测和路径规划。",
        }
        return None, None, None, message, "", scene_gate_status_text(no_target_gate), "", "", "", "", "", [], [], [], [], "", "", message

    image_width, image_height = image.size
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detection_backend_text = str(detection_backend or "YOLO Rescue Targets")

    if detection_backend_text.startswith("Transformer"):
        transformer_result = run_transformer_detection(
            image,
            model_key=transformer_model_key,
            confidence_threshold=conf_threshold,
        )
        transformer_targets = transformer_result.get("targets", []) if transformer_result.get("success") else []
        annotated_image = annotate_transformer_targets(image_rgb, transformer_targets, language=language)
        transformer_summary_text = json.dumps(transformer_result, ensure_ascii=False, indent=2)
        if not transformer_result.get("success"):
            transformer_summary_text += "\n\nTransformer 后端未成功运行；没有生成伪检测结果。"
        gate_status = scene_gate_status_text(
            {
                "level": "Target Only" if transformer_targets else "No Target",
                "message": (
                    "Transformer RescueDet 仅作为辅助候选检测运行；human_candidate 不会自动进入 TERP、路径规划或确认平民判断。"
                    if transformer_targets
                    else "Transformer RescueDet 未产生可展示候选，系统不会回退到 YOLO 或伪造检测。"
                ),
            }
        )
        path_disabled = disabled_path_result("Transformer-only 模式不会进入 TERP 或路径规划；请使用 YOLO 或双后端对比作为主检测流程。")
        report = (
            "Transformer RescueDet 辅助检测完成。\n\n"
            "当前模式只展示 Transformer 候选结果；human_candidate 需要人工复核，不能自动升级为 confirmed civilian，"
            "也不会进入救援优先级排名或路径规划。"
        )
        return (
            annotated_image,
            None,
            None,
            "Transformer-only 模式未执行语义分割；未生成环境风险或路径规划输入。",
            transformer_summary_text,
            gate_status,
            "Transformer-only 模式未执行损毁评估。",
            "场景模式：Transformer-only auxiliary detection\n说明：未执行 YOLO 主检测，未生成场景模式或入口建议。",
            "是否找到入口：否\n说明：Transformer-only 模式不生成救援入口。",
            "路径规划启用：否\n说明：Transformer-only 模式不会进入路径规划。",
            path_disabled.get("message", ""),
            target_table_rows(transformer_targets, language=language),
            [],
            [],
            [],
            path_summary_text(path_disabled, False, language=language),
            path_comparison_text({}, language=language),
            report,
        )

    model_image = get_model(model_variant)
    
    results = model_image(image_bgr, conf=conf_threshold)

    annotated_image = custom_bounding_box(image_rgb, results, language=language)

    targets = extract_targets(results)
    transformer_summary_text = "备用检测模型未启用。YOLO 主检测模型仍为当前目标识别方式。"
    if "Transformer Compare" in detection_backend_text:
        transformer_result = run_transformer_detection(
            image,
            model_key=transformer_model_key,
            confidence_threshold=conf_threshold,
        )
        comparison = compare_yolo_and_transformer_detections(targets, transformer_result.get("targets", []))
        transformer_summary_text = (
            f"{comparison.get('consensus_summary')}\n"
            f"Transformer success: {transformer_result.get('success')}\n"
            f"Transformer error_code: {transformer_result.get('error_code')}\n"
            f"Transformer targets: {transformer_result.get('target_count', 0)}\n"
            f"Matched pairs: {len(comparison.get('matched_pairs', []))}\n"
            f"Truthfulness: {comparison.get('truthfulness_note')}\n\n"
            + json.dumps(
                {
                    "transformer_result": transformer_result,
                    "comparison": comparison,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    segmentation_mask = None
    segmentation_overlay = None
    segmentation_summary = {}
    segmentation_status = []

    segmentation_source_key = normalize_segmentation_source(segmentation_source)
    auto_model_available = False
    segmentation_source_metadata = build_segmentation_source_metadata("none")

    if segmentation_source_key == "uploaded":
        if segmentation_mask_path:
            mask_path = (
                segmentation_mask_path.name
                if hasattr(segmentation_mask_path, "name")
                else segmentation_mask_path
            )
            segmentation_status.append(t(language, "uploaded_mask_ok"))
            loaded_mask = load_segmentation_mask(mask_path)
            segmentation_mask, segmentation_summary = prepare_valid_segmentation_mask(
                loaded_mask,
                image_width,
                image_height,
                segmentation_status,
                language,
            )
            segmentation_source_metadata = build_segmentation_source_metadata(
                "uploaded_mask",
                mask_path=mask_path,
                prediction_success=segmentation_mask is not None,
                fallback_reason=None if segmentation_mask is not None else "上传 mask 无效或验证失败",
            )
        else:
            segmentation_status.append(t(language, "uploaded_mask_missing"))
            segmentation_source_metadata = build_segmentation_source_metadata(
                "uploaded_mask",
                fallback_reason="Uploaded Mask 模式未上传 mask",
            )
    elif segmentation_source_key == "auto":
        weights_path = get_segmentation_weights_path()
        model_status = get_segmentation_model_status(weights_path)
        auto_model_available = bool(model_status.get("available"))
        if model_status["available"]:
            auto_model, load_status = get_auto_segmentation_model(weights_path)
            if lang_key(language) == "en":
                segmentation_status.append(load_status["message"])
            else:
                if load_status["available"]:
                    segmentation_status.append("自动分割模型加载成功。")
                else:
                    segmentation_status.append("自动分割模型加载失败，已回退到无分割模式。")
            if auto_model is not None:
                predicted_mask = predict_segmentation_mask(image, auto_model)
                if predicted_mask is None:
                    segmentation_status.append(t(language, "auto_prediction_failed"))
                    segmentation_source_metadata = build_segmentation_source_metadata(
                        "auto_model",
                        checkpoint_path=weights_path,
                        model_available=True,
                        prediction_success=False,
                        fallback_reason="自动分割推理失败",
                    )
                else:
                    segmentation_status.append(t(language, "auto_prediction_ok"))
                    segmentation_mask, segmentation_summary = prepare_valid_segmentation_mask(
                        predicted_mask,
                        image_width,
                        image_height,
                        segmentation_status,
                        language,
                    )
                    segmentation_source_metadata = build_segmentation_source_metadata(
                        "auto_model",
                        checkpoint_path=weights_path,
                        model_available=True,
                        prediction_success=segmentation_mask is not None,
                        fallback_reason=None if segmentation_mask is not None else "自动分割预测结果无效",
                    )
            else:
                segmentation_status.append(t(language, "auto_model_unavailable"))
                segmentation_source_metadata = build_segmentation_source_metadata(
                    "auto_model",
                    checkpoint_path=weights_path,
                    model_available=False,
                    prediction_success=False,
                    fallback_reason="自动分割模型加载失败",
                )
        else:
            segmentation_status.append(t(language, "auto_weights_missing"))
            segmentation_source_metadata = build_segmentation_source_metadata(
                "auto_model",
                checkpoint_path=weights_path,
                model_available=False,
                prediction_success=False,
                fallback_reason="当前未检测到可用的语义分割模型 checkpoint，无法执行自动分割",
            )
    else:
        segmentation_status.append(t(language, "no_seg_selected"))
        segmentation_source_metadata = build_segmentation_source_metadata(
            "none",
            fallback_reason="用户选择无分割",
        )

    has_segmentation_mask = segmentation_mask is not None
    if has_segmentation_mask:
        segmentation_overlay = create_segmentation_overlay(image_rgb, segmentation_mask)
    else:
        segmentation_status.append(t(language, "fallback_no_seg"))

    scene_gate = evaluate_scene_applicability(
        targets,
        segmentation_mask=segmentation_mask,
        segmentation_source=segmentation_source_key,
        auto_model_available=auto_model_available,
    )
    gate_status = scene_gate_status_text(scene_gate)
    segmentation_status.append(f"场景适用性门控：{scene_gate['level']}。{scene_gate['message']}")
    segmentation_status.append(format_segmentation_source_status(segmentation_source_metadata))

    ranked_targets = rank_targets(targets, image_width, image_height, segmentation_mask, language=language)
    top_for_entry = ranked_targets[0] if ranked_targets else (targets[0] if targets else None)
    damage_assessment = assess_damage_and_entry(
        segmentation_mask,
        targets=targets,
        image_width=image_width,
        image_height=image_height,
        top_target=top_for_entry,
    )

    if start_y is None or float(start_y) < 0:
        start_y = image_height - 20
    if start_x is None:
        start_x = 20

    manual_start_point = (start_x, start_y)
    target_point = None
    if top_for_entry:
        target_point = top_for_entry.get("center")
        if not target_point and top_for_entry.get("bbox"):
            x1, y1, x2, y2 = [float(v) for v in top_for_entry.get("bbox", [0, 0, 0, 0])[:4]]
            target_point = [(x1 + x2) / 2, (y1 + y2) / 2]

    scene_mode_result = analyze_scene_mode(
        image,
        segmentation_mask=segmentation_mask,
        detections=targets,
    )
    entry_result = (
        find_rescue_entry_point(segmentation_mask, target_point=target_point)
        if scene_mode_result.get("scene_mode") == "wide_area_assessment"
        else {
            "entry_found": False,
            "entry_point": None,
            "entry_reason": scene_mode_result.get("reason", "当前场景不适合生成救援入口。"),
            "candidate_count": 0,
        }
    )
    path_gate = build_path_planning_gate_result(
        scene_mode_result,
        entry_result,
        use_manual_start=bool(use_manual_start),
        manual_start_point=manual_start_point,
        force_path_planning=bool(force_path_planning),
    )
    path_start_point = tuple(path_gate["start_point"]) if path_gate.get("start_point") is not None else manual_start_point
    path_enabled = bool(path_gate.get("path_enabled")) and bool(ranked_targets)

    entry = {
        "entry_found": entry_result.get("entry_found"),
        "entry_point_x": entry_result.get("entry_point", [None, None])[0] if entry_result.get("entry_point") else None,
        "entry_point_y": entry_result.get("entry_point", [None, None])[1] if entry_result.get("entry_point") else None,
        "entry_reason": entry_result.get("entry_reason"),
        "candidate_count": entry_result.get("candidate_count", 0),
    }
    damage_assessment.update(
        {
            "scene_mode": {
                "local_reconnaissance": "Local Reconnaissance",
                "wide_area_assessment": "Wide-area Assessment",
                "unknown": "Unknown",
            }.get(scene_mode_result.get("scene_mode"), "Unknown"),
            "scene_mode_key": scene_mode_result.get("scene_mode"),
            "scene_mode_label": scene_mode_result.get("scene_mode_label"),
            "scene_mode_reason": scene_mode_result.get("reason"),
            "scene_mode_evidence": scene_mode_result.get("evidence", {}),
            "entry": entry,
            "path_planning_enabled": path_enabled,
            "path_planning_reason": path_gate.get("gate_reason"),
            "path_planning_gate": path_gate,
            "force_path_planning": bool(force_path_planning),
            "use_manual_start": bool(use_manual_start),
            "path_start_source": path_gate.get("start_source"),
        }
    )

    damage_summary_text = format_damage_summary(damage_assessment)
    scene_mode_text = (
        f"场景模式：{scene_mode_result.get('scene_mode_label')}\n"
        f"说明：{scene_mode_result.get('reason')}\n"
        f"证据：{json.dumps(scene_mode_result.get('evidence', {}), ensure_ascii=False, indent=2)}"
    )
    entry_suggestion_text = (
        f"是否找到入口：{'是' if entry_result.get('entry_found') else '否'}\n"
        f"入口坐标：{entry_result.get('entry_point')}\n"
        f"候选入口数量：{entry_result.get('candidate_count', 0)}\n"
        f"说明：{entry_result.get('entry_reason')}"
    )
    path_gate_text = (
        f"路径规划启用：{'是' if path_enabled else '否'}\n"
        f"起点：{path_gate.get('start_point')}\n"
        f"起点来源：{path_gate.get('start_source')}\n"
        f"说明：{path_gate.get('display_message')}"
    )
    reliability_status = build_path_planning_reliability_status(
        scene_mode_result,
        entry_result,
        path_gate,
        segmentation_source_metadata=segmentation_source_metadata,
        force_path_planning=bool(force_path_planning),
    )
    reliability_text = format_path_planning_reliability_status(reliability_status)
    damage_assessment["path_planning_reliability"] = reliability_status
    if path_enabled:
        baseline_path_result = plan_baseline_path(
            ranked_targets,
            image_width,
            image_height,
            start_point=path_start_point,
        )
        path_result = plan_risk_aware_path(
            ranked_targets,
            segmentation_mask,
            image_width,
            image_height,
            start_point=path_start_point,
        )
        path_comparison = compare_path_plans(baseline_path_result, path_result, segmentation_mask)
    else:
        disable_reason = path_gate.get("gate_reason") or "当前场景不适合生成路径建议。"
        baseline_path_result = disabled_path_result(disable_reason)
        path_result = disabled_path_result(disable_reason)
        path_comparison = {}

    environment_contexts = {}
    target_path_results = {}
    for target in targets:
        target_id = target.get("id")
        if segmentation_mask is not None:
            environment_contexts[target_id] = get_environment_context_for_target(target, segmentation_mask, language=language)
        target_path_results[target_id] = plan_risk_aware_path(
            [_target_route_item(target)],
            segmentation_mask,
            image_width,
            image_height,
            start_point=path_start_point,
        ) if path_enabled else disabled_path_result(damage_assessment.get("path_planning_reason", "路径规划未启用。"))

    terp_rankings = rank_targets_by_terp(
        targets,
        image_width,
        image_height,
        environment_contexts=environment_contexts,
        path_results=target_path_results,
        language=language,
    )

    base_path_image = segmentation_overlay if segmentation_overlay is not None else image_rgb
    if baseline_path_result and baseline_path_result.get("found") and path_result and path_result.get("found"):
        path_overlay = create_dual_path_overlay(base_path_image, baseline_path_result, path_result)
    else:
        path_overlay = create_path_overlay(base_path_image, path_result) if path_enabled else None
    report = generate_report(
        targets,
        ranked_targets,
        segmentation_summary,
        path_result,
        terp_rankings,
        path_comparison,
        damage_assessment=damage_assessment,
        segmentation_source_metadata=segmentation_source_metadata,
        language=language,
    )
    summary_text = path_summary_text(path_result, has_segmentation_mask, language=language)
    comparison_text = path_comparison_text(path_comparison, language=language)

    return (
        annotated_image,
        segmentation_overlay,
        path_overlay,
        "\n".join(segmentation_status),
        transformer_summary_text,
        gate_status,
        damage_summary_text,
        scene_mode_text,
        entry_suggestion_text,
        path_gate_text,
        reliability_text,
        target_table_rows(targets, language=language),
        segmentation_summary_rows(segmentation_summary, language=language),
        ranking_table_rows(ranked_targets, language=language),
        terp_table_rows(terp_rankings, language=language),
        summary_text,
        comparison_text,
        report,
    )


def target_detection_only(
    image,
    detection_backend,
    transformer_model_key,
    conf_threshold,
    model_variant,
    language="zh",
):
    if image is None:
        message = "请先上传一张图像。"
        return None, "", [], message

    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detection_backend_text = str(detection_backend or "YOLO Rescue Targets")

    if detection_backend_text.startswith("Transformer"):
        transformer_result = run_transformer_detection(
            image,
            model_key=transformer_model_key,
            confidence_threshold=conf_threshold,
        )
        transformer_targets = transformer_result.get("targets", []) if transformer_result.get("success") else []
        annotated_image = annotate_transformer_targets(image_rgb, transformer_targets, language=language)
        summary = json.dumps(transformer_result, ensure_ascii=False, indent=2)
        if not transformer_result.get("success"):
            summary += "\n\nTransformer 后端未成功运行；没有生成伪检测结果。"
        report = (
            "Transformer RescueDet 辅助检测完成。\n\n"
            "当前结果只作为候选检测线索，需要人工复核；不会自动进入救援优先级或路径规划。"
        )
        return annotated_image, summary, target_table_rows(transformer_targets, language=language), report

    model_image = get_model(model_variant)
    results = model_image(image_bgr, conf=conf_threshold)
    annotated_image = custom_bounding_box(image_rgb, results, language=language)
    targets = extract_targets(results)
    summary = "备用检测模型未启用。YOLO 主检测模型为当前目标识别方式。"
    if "Transformer Compare" in detection_backend_text:
        transformer_result = run_transformer_detection(
            image,
            model_key=transformer_model_key,
            confidence_threshold=conf_threshold,
        )
        comparison = compare_yolo_and_transformer_detections(targets, transformer_result.get("targets", []))
        summary = (
            f"{comparison.get('consensus_summary')}\n"
            f"Transformer success: {transformer_result.get('success')}\n"
            f"Transformer error_code: {transformer_result.get('error_code')}\n"
            f"Transformer targets: {transformer_result.get('target_count', 0)}\n"
            f"Matched pairs: {len(comparison.get('matched_pairs', []))}\n"
            f"Truthfulness: {comparison.get('truthfulness_note')}\n\n"
            + json.dumps(
                {
                    "transformer_result": transformer_result,
                    "comparison": comparison,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    report = (
        f"目标检测完成。\n\n检测目标数：{len(targets)}\n"
        "本页只展示目标检测结果；如需语义分割与环境风险，请进入“灾情感知”Tab；如需 TERP 排序和路径规划，请进入“综合决策”Tab。"
    )
    return annotated_image, summary, target_table_rows(targets, language=language), report


def disaster_perception_only(
    image,
    detection_backend,
    transformer_model_key,
    segmentation_source,
    segmentation_mask_path,
    conf_threshold,
    model_variant,
    language="zh",
):
    legend_image = create_legend_image()
    external_status = build_external_impact_assessment_status()
    try:
        status_path = save_external_impact_assessment_status(external_status)
    except Exception:
        status_path = ""
    external_text = format_external_impact_assessment_status(external_status)
    unavailable_external = (
        "SKAI 外部源码级建筑灾损评估：unavailable\n"
        "InaSAFE 外部源码级灾害影响评估：unavailable"
    )
    if image is None:
        return (
            None,
            None,
            legend_image,
            "模型来源：已训练语义分割模型\n状态：unavailable\n原因：请先上传图像。",
            external_text,
            "灾情感知与外部影响评估（高级深度版）：unavailable",
            "未生成 pred_mask，无法计算统计信息。",
            unavailable_external,
            "辅助决策，人工复核。未生成替代假结果。",
            [],
            [],
            [],
        )

    _detection_image, transformer_summary, target_rows, _detection_report = target_detection_only(
        image,
        detection_backend,
        transformer_model_key,
        conf_threshold,
        model_variant,
        language=language,
    )

    weights_path = get_segmentation_weights_path()
    model_status = get_segmentation_model_status(weights_path)
    checkpoint_path = str(model_status.get("path") or weights_path)
    if not model_status.get("available"):
        return (
            None,
            None,
            legend_image,
            "模型来源：已训练语义分割模型\n"
            f"Checkpoint 路径：{checkpoint_path}\n"
            "状态：unavailable\n"
            "原因：未找到已训练语义分割模型 checkpoint。\n"
            "不会展示 uploaded/demo/none/来源选择，也不会生成替代假分割。",
            external_text,
            "灾情感知与外部影响评估（高级深度版）：unavailable",
            "未生成 pred_mask，无法计算统计信息。",
            unavailable_external,
            "辅助决策，人工复核。未生成替代假结果。",
            target_rows,
            [],
            [],
        )

    pred_mask, predict_message, predict_metadata = predict_trained_segmentation(
        image,
        checkpoint_path=checkpoint_path,
        img_size=512,
    )
    if pred_mask is None:
        return (
            None,
            None,
            legend_image,
            "模型来源：已训练语义分割模型\n"
            f"Checkpoint 路径：{checkpoint_path}\n"
            "状态：unavailable\n"
            f"原因：{predict_message}\n"
            "模型推理失败时不生成替代假结果。",
            external_text,
            "灾情感知与外部影响评估（高级深度版）：unavailable",
            "模型推理失败，未生成 pred_mask。",
            unavailable_external,
            "辅助决策，人工复核。未生成替代假结果。",
            target_rows,
            [],
            [],
        )

    validation = validate_segmentation_mask(pred_mask)
    if not validation.get("valid"):
        return (
            None,
            None,
            legend_image,
            "模型来源：已训练语义分割模型\n"
            f"Checkpoint 路径：{checkpoint_path}\n"
            "状态：unavailable\n"
            f"原因：{validation.get('message', 'pred_mask 验证失败。')}\n"
            "无效 pred_mask 不进入外部影响评估主展示。",
            external_text,
            "灾情感知与外部影响评估（高级深度版）：unavailable",
            "pred_mask 验证失败，未计算统计信息。",
            unavailable_external,
            "辅助决策，人工复核。未生成替代假结果。",
            target_rows,
            [],
            [],
        )

    overlay = create_segmentation_overlay(image, pred_mask)
    color_mask = render_segmentation_mask(pred_mask)
    stats = compute_damage_statistics(pred_mask)
    damage_level = classify_damage_level(stats)
    source_metadata = build_segmentation_source_metadata(
        "auto_model",
        checkpoint_path=checkpoint_path,
        model_available=True,
        prediction_success=True,
    )
    segmentation_summary = stats.get("class_area_ratios", {})
    status_text = (
        "模型来源：已训练语义分割模型\n"
        f"Checkpoint 路径：{checkpoint_path}\n"
        f"推理尺寸：{int(predict_metadata.get('img_size') or 512) if isinstance(predict_metadata, dict) else 512}\n"
        "状态：pred_mask generated\n"
        "同一个 pred_mask 已用于覆盖图、黑底彩色分割图、图例对应关系和统计信息。\n"
        f"真实性边界：{segmentation_visualization_note(source_metadata)}\n"
        f"外部评估状态文件：{status_path or '未写入'}"
    )
    external_summary = (
        f"SKAI 外部源码级建筑灾损评估：{external_status['skai']['status']}\n"
        f"InaSAFE 外部源码级灾害影响评估：{external_status['inasafe']['status']}\n"
        "只有真实调用外部源码并验证到输出文件，才标记为真实 SKAI / InaSAFE 输出。"
    )
    return (
        overlay,
        color_mask,
        legend_image,
        status_text,
        external_text if not transformer_summary else f"{transformer_summary}\n\n{external_text}",
        f"灾情感知与外部影响评估（高级深度版）：pred_mask ready；整体损毁等级：{damage_level}",
        _build_damage_area_stats_text(stats),
        external_summary,
        _build_segmentation_impact_text(stats, damage_level),
        target_rows,
        segmentation_summary_rows(segmentation_summary, language=language),
        [],
    )


def _normalize_ui_image_value(image_value):
    """Convert a Gradio image value, file path, or numpy array into a PIL image."""
    if image_value is None:
        return None
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    if isinstance(image_value, np.ndarray):
        return Image.fromarray(image_value).convert("RGB")
    image_path = _upload_path(image_value)
    if image_path and Path(image_path).exists():
        return Image.open(image_path).convert("RGB")
    return image_value


def disaster_perception_with_source(
    image_source,
    shared_image,
    s1_preview_image,
    stage_uploaded_image,
    detection_backend,
    transformer_model_key,
    conf_threshold,
    model_variant,
):
    """Run S2/S3 disaster perception from S1 preview, shared input, or local upload."""
    source_text = str(image_source or "首页照片")
    if source_text.startswith("S1") or source_text.startswith("S1预览") or source_text.startswith("预览"):
        selected_image = s1_preview_image
    elif source_text.startswith("本地") or source_text.startswith("上传"):
        selected_image = stage_uploaded_image
    else:
        selected_image = shared_image
    return disaster_perception_only(
        _normalize_ui_image_value(selected_image),
        detection_backend,
        transformer_model_key,
        "自动分割模型",
        None,
        conf_threshold,
        model_variant,
    )


def _format_segmentation_ratio(value):
    return f"{float(value or 0.0) * 100:.2f}%"


def _build_damage_area_stats_text(stats):
    area_ratios = stats.get("class_area_ratios", {}) if isinstance(stats, dict) else {}
    lines = [
        f"水域占比：{_format_segmentation_ratio(area_ratios.get('water'))}",
        f"积水/水池占比：{_format_segmentation_ratio(area_ratios.get('pool'))}",
        f"可通行道路占比：{_format_segmentation_ratio(area_ratios.get('road_clear'))}",
        f"道路阻断占比：{_format_segmentation_ratio(area_ratios.get('road_blocked'))}",
        f"严重损毁建筑占比：{_format_segmentation_ratio(area_ratios.get('major_damage'))}",
        f"完全毁坏建筑占比：{_format_segmentation_ratio(area_ratios.get('destroyed_building'))}",
        f"树木占比：{_format_segmentation_ratio(area_ratios.get('tree'))}",
        f"车辆占比：{_format_segmentation_ratio(area_ratios.get('vehicle'))}",
    ]
    return "\n".join(lines)


def _build_damage_risk_summary_text(stats, damage_level):
    road_stats = stats.get("road_stats", {}) if isinstance(stats, dict) else {}
    environment_stats = stats.get("environment_stats", {}) if isinstance(stats, dict) else {}
    major_damage = float(stats.get("major_damage_area", 0) or 0)
    destroyed = float(stats.get("destroyed_building_area", 0) or 0)
    road_blocked = float(road_stats.get("road_blocked_ratio", 0.0) or 0.0)
    water_ratio = float(environment_stats.get("water_ratio", 0.0) or 0.0)

    lines = [f"整体损毁等级：{damage_level}"]
    if damage_level == "Major Damage":
        lines.append("当前区域整体损毁较重，通常意味着救援通行和现场判断都需要优先复核。")
    elif damage_level == "Medium Damage":
        lines.append("当前区域存在明显损毁，应结合道路和积水情况继续人工复核。")
    else:
        lines.append("当前区域损毁相对较轻，但仍需结合环境风险和现场证据判断。")

    if road_blocked > 0.02:
        lines.append("道路阻断比例较高，可能影响救援队进入与回撤。")
    if water_ratio > 0.03:
        lines.append("水域/积水占比明显，建议将涉水通行作为重点风险项。")
    if destroyed > 0 or major_damage > 0:
        lines.append("严重损毁和完全毁坏建筑会抬高环境风险权重。")

    return "\n".join(lines)


def _build_segmentation_impact_text(stats, damage_level):
    road_stats = stats.get("road_stats", {}) if isinstance(stats, dict) else {}
    environment_stats = stats.get("environment_stats", {}) if isinstance(stats, dict) else {}
    road_clear = float(road_stats.get("road_clear_ratio", 0.0) or 0.0)
    road_blocked = float(road_stats.get("road_blocked_ratio", 0.0) or 0.0)
    water_ratio = float(environment_stats.get("water_ratio", 0.0) or 0.0)
    destroyed = float(stats.get("destroyed_building_area", 0) or 0)

    lines = [
        "对 TERP：该区域的环境风险会影响候选目标的优先级排序，积水、阻断道路和严重损毁建筑通常会提高风险权重。",
        "对路径规划：可通行道路比例越低，绕行代价越高；道路阻断和积水区域应优先视为高代价或禁入区域。",
        "对 Final Report：该结果应作为辅助决策证据写入报告，并标记为需要人工复核的灾情感知结果。",
    ]
    if road_blocked > 0.02 or water_ratio > 0.03 or destroyed > 0:
        lines.insert(0, f"当前 {damage_level} 场景下，环境风险不宜直接转化为现场行动命令。")
    if road_clear > 0.15:
        lines.append("存在一定比例的可通行道路，但仍需结合局部阻断和积水情况判断实际可达性。")
    return "\n".join(lines)


def _format_external_module_box(item):
    lines = [
        f"模块：{item.get('module', '')}",
        f"仓库：{item.get('repository', '')}",
        f"状态：{item.get('status', 'unavailable')}",
        f"源码路径：{item.get('source_root') or 'unavailable'}",
        f"输出目录：{item.get('output_dir', '')}",
        f"已验证输出文件数：{len(item.get('verified_output_files', []) or [])}",
    ]
    dependency_status = item.get("dependency_status", {}) or {}
    if dependency_status:
        dep_text = ", ".join(f"{name}={'ok' if ok else 'missing'}" for name, ok in dependency_status.items())
        lines.append(f"依赖状态：{dep_text}")
    reasons = item.get("unavailable_reasons", []) or []
    if reasons:
        lines.append("unavailable 原因：")
        lines.extend([f"- {reason}" for reason in reasons])
    verified_outputs = item.get("verified_output_files", []) or []
    if verified_outputs:
        lines.append("已验证输出：")
        lines.extend([f"- {path}" for path in verified_outputs])
    lines.append(f"真实性说明：{item.get('truthfulness_note', '')}")
    return "\n".join(lines)


def _format_skai_run_status(item):
    item = item or {}
    run_status = item.get("run_status_display", {}) or {}
    lines = [
        "SKAI 运行状态：",
        f"- status：{item.get('status', 'unavailable')}",
        f"- SKAI 源码：{run_status.get('skai_source', '缺失')}",
        f"- 依赖环境：{run_status.get('dependency_environment', '缺失')}",
        f"- 配置文件：{run_status.get('config_file', '缺失')}",
        f"- Checkpoint：{run_status.get('checkpoint', '缺失')}",
        f"- 输入数据：{run_status.get('input_data', '缺失')}",
        f"- Runner：{run_status.get('runner', '未执行')}",
        f"- 真实 SKAI 输出：{run_status.get('real_skai_output', '未产生')}",
    ]
    return "\n".join(lines)


def _format_inasafe_run_status(item):
    item = item or {}
    run_status = item.get("run_status_display", {}) or {}
    dependency_status = item.get("dependency_status", {}) or {}
    dep_text = ", ".join(
        f"{name}={'ok' if ok else 'missing'}" for name, ok in dependency_status.items()
    ) or "unavailable"
    lines = [
        "InaSAFE 运行状态：",
        f"- status：{item.get('status', 'unavailable')}",
        f"- InaSAFE 源码：{run_status.get('inasafe_source', '缺失')}",
        f"- 依赖环境：{run_status.get('dependency_environment', '缺失')}",
        f"- 依赖明细：{dep_text}",
        f"- 真实 InaSAFE 输出：{run_status.get('real_inasafe_output', '未产生')}",
    ]
    return "\n".join(lines)


def _s2s3_truthfulness_text(extra=""):
    base = (
        "统一真实性边界：\n"
        "- S2-S3 最终模块为：灾情感知与外部影响评估（高级深度版）。\n"
        "- 固定使用本地已训练语义分割模型生成 pred_mask。\n"
        "- 覆盖图、黑底彩色分割图、图例和统计信息必须来自同一个 pred_mask。\n"
        "- SKAI 只有真实调用 google-research/skai 外部源码并验证输出文件后，才标记为真实 SKAI 输出。\n"
        "- InaSAFE 只有真实调用 inasafe/inasafe 外部源码并验证输出文件后，才标记为真实 InaSAFE 输出。\n"
        "- 依赖、权重、输入或 QGIS/GIS 环境缺失时只显示 unavailable，不生成替代假结果。\n"
        "- legacy/internal lightweight_skai_inasafe_adaptation 不作为最终主展示，也不得称为真实 SKAI 或 InaSAFE 结果。\n"
        "- 所有结果均为辅助决策，必须人工复核。"
    )
    return f"{base}\n- {extra}" if extra else base


def _s2s3_response(
    overlay,
    color_mask,
    legend,
    model_status_text,
    perception_summary,
    skai_text,
    inasafe_text,
    skai_run_status,
    inasafe_run_status,
    external_files_text,
    unavailable_text,
    downstream_text,
    truthfulness_text,
    run_status_text,
):
    return (
        overlay,
        color_mask,
        legend,
        model_status_text,
        perception_summary,
        skai_text,
        inasafe_text,
        skai_run_status,
        inasafe_run_status,
        external_files_text,
        unavailable_text,
        downstream_text,
        truthfulness_text,
        run_status_text,
    )


def run_damage_segmentation_analysis(image, img_size=512, language="zh"):
    """Run disaster perception with the locally trained segmentation checkpoint only."""
    external_status = build_external_impact_assessment_status()
    try:
        external_status_path = save_external_impact_assessment_status(external_status)
    except Exception:
        external_status_path = ""
    skai_text = format_skai_building_damage_panel(external_status.get("skai", {}))
    inasafe_text = format_inasafe_impact_assessment_panel(external_status.get("inasafe", {}))
    skai_run_status = _format_skai_run_status(external_status.get("skai", {}))
    inasafe_run_status = _format_inasafe_run_status(external_status.get("inasafe", {}))
    external_files_text = format_external_output_file_summary(external_status)
    unavailable_text = format_external_unavailable_reasons(external_status)
    if image is None:
        return _s2s3_response(
            None,
            None,
            None,
            "语义分割模型状态：unavailable\n模型来源：已训练语义分割模型\nCheckpoint 路径：未提供\n原因：请先上传图像。",
            "灾情感知摘要：unavailable\n原因：请先上传图像；未生成 pred_mask。",
            skai_text,
            inasafe_text,
            skai_run_status,
            inasafe_run_status,
            external_files_text,
            unavailable_text,
            "下游决策建议：请先上传图像并成功生成 pred_mask。",
            _s2s3_truthfulness_text("请先上传图像。"),
            "请先上传一张图像。",
        )

    weights_path = get_segmentation_weights_path()
    model_status = get_segmentation_model_status(weights_path)
    checkpoint_path = str(model_status.get("path") or weights_path)
    if not model_status.get("available"):
        return _s2s3_response(
            None,
            None,
            None,
            "语义分割模型状态：unavailable\n"
            "模型来源：已训练语义分割模型\n"
            f"Checkpoint 路径：{checkpoint_path}\n"
            "原因：未找到已训练语义分割模型 checkpoint，无法执行灾情感知分析。",
            "灾情感知摘要：unavailable\n原因：未找到已训练语义分割模型 checkpoint；未生成 pred_mask。",
            skai_text,
            inasafe_text,
            skai_run_status,
            inasafe_run_status,
            external_files_text,
            unavailable_text,
            "下游决策建议：未生成 pred_mask，不进入灾情统计、SKAI 或 InaSAFE 真实输出主展示。",
            _s2s3_truthfulness_text("缺少 checkpoint 时不会生成替代假结果。"),
            "未找到已训练语义分割模型 checkpoint，无法执行灾情感知分析。",
        )

    pred_mask, predict_message, predict_metadata = predict_trained_segmentation(
        image,
        checkpoint_path=checkpoint_path,
        img_size=int(img_size or 512),
    )
    if pred_mask is None:
        status_text = predict_metadata.get("message") if isinstance(predict_metadata, dict) else predict_message
        return _s2s3_response(
            None,
            None,
            None,
            "语义分割模型状态：unavailable\n"
            "模型来源：已训练语义分割模型\n"
            f"Checkpoint 路径：{checkpoint_path}\n"
            f"原因：{status_text or '模型推理失败，未生成 pred_mask。'}",
            "灾情感知摘要：unavailable\n原因：模型推理失败；未生成 pred_mask。",
            skai_text,
            inasafe_text,
            skai_run_status,
            inasafe_run_status,
            external_files_text,
            unavailable_text,
            "下游决策建议：模型推理失败，未生成 pred_mask，不生成替代假结果。",
            _s2s3_truthfulness_text("模型推理失败时不会生成替代假结果。"),
            status_text or "自动分割预测失败。",
        )

    validation = validate_segmentation_mask(pred_mask)
    if not validation.get("valid"):
        return _s2s3_response(
            None,
            None,
            None,
            "语义分割模型状态：unavailable\n"
            "模型来源：已训练语义分割模型\n"
            f"Checkpoint 路径：{checkpoint_path}\n"
            f"原因：{validation.get('message', 'pred_mask 验证失败。')}",
            "灾情感知摘要：unavailable\n原因：pred_mask 验证失败；未计算灾情统计。",
            skai_text,
            inasafe_text,
            skai_run_status,
            inasafe_run_status,
            external_files_text,
            unavailable_text,
            "下游决策建议：pred_mask 验证失败，未继续评估。",
            _s2s3_truthfulness_text("无效 pred_mask 不进入外部影响评估主展示。"),
            validation.get("message", "分割结果无效。"),
        )

    overlay = create_segmentation_overlay(image, pred_mask)
    color_mask = render_segmentation_mask(pred_mask)
    legend = create_legend_image()
    stats = compute_damage_statistics(pred_mask)
    damage_level = classify_damage_level(stats)
    stats["overall_damage_level"] = damage_level
    source_metadata = build_segmentation_source_metadata(
        "auto_model",
        checkpoint_path=checkpoint_path,
        model_available=True,
        prediction_success=True,
    )
    stats["segmentation_source"] = source_metadata
    stats["visualization_note"] = segmentation_visualization_note(source_metadata)
    area_text = _build_damage_area_stats_text(stats)
    perception_summary = (
        "灾情感知摘要：pred_mask ready\n"
        f"整体损毁等级：{damage_level}\n"
        f"{area_text}\n"
        "说明：以上统计来自本地已训练语义分割模型输出的同一个 pred_mask，需要人工复核。"
    )
    impact_text = (
        f"整体损毁等级：{damage_level}\n"
        f"{area_text}\n\n"
        f"{_build_damage_risk_summary_text(stats, damage_level)}\n\n"
        f"{_build_segmentation_impact_text(stats, damage_level)}"
    )
    status_text = (
        "语义分割模型状态：pred_mask generated\n"
        "模型来源：已训练语义分割模型\n"
        f"Checkpoint 路径：{checkpoint_path}\n"
        f"推理尺寸：{int(predict_metadata.get('img_size') or 512) if isinstance(predict_metadata, dict) else 512}\n"
        f"整体损毁等级：{damage_level}\n"
        "同一个 pred_mask 已用于覆盖图、黑底彩色分割图、图例和统计信息。\n"
        f"外部影响评估状态文件：{external_status_path or '未写入'}"
    )
    return _s2s3_response(
        overlay,
        color_mask,
        legend,
        status_text,
        perception_summary,
        skai_text,
        inasafe_text,
        skai_run_status,
        inasafe_run_status,
        external_files_text,
        unavailable_text,
        impact_text,
        _s2s3_truthfulness_text(),
        "灾情感知与外部影响评估（高级深度版）运行完成。",
    )


def target_detection_with_source(
    image_source,
    shared_image,
    stage_image,
    detection_backend,
    transformer_model_key,
    conf_threshold,
    model_variant,
):
    """Run S4 image detection from shared input or this tab's upload."""
    selected_image = stage_image if str(image_source or "").startswith(("本阶段", "本地", "上传")) else shared_image
    return target_detection_only(
        _normalize_ui_image_value(selected_image),
        detection_backend,
        transformer_model_key,
        conf_threshold,
        model_variant,
    )


def video_detection_with_source(
    video_source,
    shared_video,
    stage_video,
    conf_threshold,
    model_variant,
    frame_skip=15,
    max_frames=0,
):
    """Run S4 video detection from shared input or this tab's upload."""
    selected_video = stage_video if str(video_source or "").startswith(("本阶段", "本地", "上传")) else shared_video
    return video_detection(selected_video, conf_threshold, model_variant, frame_skip=frame_skip, max_frames=max_frames)


def decision_detection_with_source(
    image_source,
    shared_image,
    stage_image,
    detection_backend,
    transformer_model_key,
    segmentation_source,
    shared_segmentation_mask_path,
    stage_segmentation_mask_path,
    start_x,
    start_y,
    use_manual_start,
    force_path_planning,
    conf_threshold,
    model_variant,
):
    """Run S7/S8 decision workflow from shared input or this tab's upload."""
    selected_image = stage_image if str(image_source or "").startswith(("本阶段", "本地", "上传")) else shared_image
    return image_detection(
        _normalize_ui_image_value(selected_image),
        detection_backend,
        transformer_model_key,
        segmentation_source,
        stage_segmentation_mask_path or shared_segmentation_mask_path,
        start_x,
        start_y,
        use_manual_start,
        force_path_planning,
        conf_threshold,
        model_variant,
    )


def video_detection(video_path, conf_threshold, model_variant, frame_skip=15, max_frames=0, language="zh"):
    video_path = _resolve_video_path(video_path)
    if not video_path:
        return None, "Please upload a video first." if lang_key(language) == "en" else "请先上传一个视频。"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Unable to open video file: {video_path}" if lang_key(language) == "en" else f"无法打开视频文件：{video_path}"

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if width <= 0 or height <= 0:
        cap.release()
        return None, "Unable to read video dimensions." if lang_key(language) == "en" else "无法读取视频尺寸。"

    temp_video_fd, temp_video_path = tempfile.mkstemp(suffix=".webm")
    os.close(temp_video_fd)
    out = _create_video_writer(temp_video_path, fps, width, height)
    if not out.isOpened():
        cap.release()
        return None, "Unable to create output video writer." if lang_key(language) == "en" else "无法创建输出视频写入器。"

    all_classes = set()
    frame_count = 0
    last_annotated_frame = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_skip = max(1, int(frame_skip or 1))
    max_frames = int(max_frames or 0)
    model_video = get_model(model_variant)

    def _video_box_allowed(box, names):
        class_name = names[int(box.cls[0])]
        min_conf = VIDEO_CLASS_MIN_CONF.get(class_name, conf_threshold)
        return float(box.conf[0]) >= min_conf

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames > 0 and frame_count >= max_frames:
            break

        if frame_count % frame_skip == 0:
            results = model_video(frame, conf=conf_threshold, verbose=False)
            filtered_boxes = [box for box in results[0].boxes if _video_box_allowed(box, results[0].names)]
            filtered_result = type(
                "FilteredVideoResult",
                (),
                {"names": results[0].names, "boxes": filtered_boxes},
            )()
            annotated_frame = custom_bounding_box(frame, [filtered_result], language=language)
        
            for box in filtered_boxes:
                class_name = results[0].names[int(box.cls[0])]
                all_classes.add(class_name)
                
            last_annotated_frame = np.ascontiguousarray(annotated_frame)
        else:
            if last_annotated_frame is not None:
                annotated_frame = last_annotated_frame.copy()
            else:
                annotated_frame = np.ascontiguousarray(frame)
        
        out.write(np.ascontiguousarray(annotated_frame))
        frame_count += 1

    cap.release()
    out.release()

    if all_classes:
        predictions = ", ".join(sorted(display_class_name(class_name, language) for class_name in all_classes))
    else:
        predictions = t(language, "no_detections")
    if max_frames > 0 and total_frames > max_frames:
        predictions += f" {t(language, 'limited_frames', max_frames=max_frames)}"
    return temp_video_path, predictions


MISSION_FIRST_LAYOUT_CSS = """
#aerorescue-mission-app .gradio-container {
    width: 100%;
    max-width: 1320px !important;
}
#aerorescue-mission-app h1 {
    margin-bottom: 18px !important;
}
@media (min-width: 900px) {
    .tabs {
        display: block !important;
    }
    .tabs > .tab-wrapper {
        display: block !important;
        position: relative;
        padding-left: 232px !important;
    }
    .tabs > .tab-wrapper > .tab-container.visually-hidden {
        display: none !important;
    }
    .tabs > .tab-wrapper > .tab-container:not(.visually-hidden) {
        position: absolute !important;
        top: 12px;
        left: 0;
        width: 208px !important;
        height: auto !important;
        min-height: 0 !important;
        max-height: none !important;
        overflow: visible !important;
        border-bottom: 0 !important;
        border-right: 1px solid var(--border-color-primary);
        padding: 8px 10px 8px 0 !important;
        z-index: 10;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        align-content: stretch !important;
        align-items: stretch !important;
        gap: 4px;
    }
    .tabs .overflow-menu {
        display: block !important;
        width: 100% !important;
    }
    .tabs .overflow-menu > button {
        display: none !important;
    }
    .tabs .overflow-dropdown {
        position: static !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 4px;
        visibility: visible !important;
        opacity: 1 !important;
        transform: none !important;
        box-shadow: none !important;
        border: 0 !important;
        background: transparent !important;
        padding: 0 !important;
        min-width: 0 !important;
        height: auto !important;
        max-height: none !important;
    }
    .tabs > .tab-wrapper > .tab-container:not(.visually-hidden) > button,
    .tabs .overflow-dropdown > button {
        position: static !important;
        inset: auto !important;
        transform: none !important;
        display: block !important;
        flex: 0 0 auto !important;
        width: 100%;
        height: auto !important;
        max-height: none !important;
        box-sizing: border-box !important;
        justify-content: flex-start !important;
        text-align: left !important;
        white-space: normal !important;
        line-height: 1.25;
        min-height: 38px;
        border-radius: 8px !important;
        margin: 0 0 3px 0 !important;
        padding: 8px 10px !important;
        font-size: 14px !important;
    }
    .tabs > .tab-wrapper > .tab-container:not(.visually-hidden) > button[aria-selected="true"],
    .tabs .overflow-dropdown > button[aria-selected="true"] {
        border-left: 4px solid var(--color-accent, #f97316) !important;
        background: var(--background-fill-secondary) !important;
    }
    .tabs .tabitem {
        margin-left: 0 !important;
        max-width: 100% !important;
        min-width: 0;
    }
    .tabs [role="tabpanel"] {
        margin-left: 232px !important;
        max-width: calc(100% - 232px) !important;
        min-width: 0;
        padding-bottom: 56px;
    }
}
#aerorescue-mission-app .form {
    gap: 12px !important;
}
#aerorescue-mission-app label {
    font-size: 14px !important;
}
#aerorescue-mission-app textarea,
#aerorescue-mission-app input {
    font-size: 14px !important;
}
#aerorescue-mission-app .prose h2 {
    margin-top: 0 !important;
}
#aerorescue-mission-app .prose p,
#aerorescue-mission-app .prose li {
    line-height: 1.6;
}
.mission-stage-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin: 12px 0 14px 0;
}
.mission-stage-card {
    height: 128px;
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 12px;
    background: var(--background-fill-primary);
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
}
.mission-stage-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 10px;
}
.mission-stage-id {
    font-weight: 700;
    color: var(--color-accent, #f97316);
}
.mission-stage-status {
    font-size: 12px;
    color: var(--body-text-color-subdued);
}
.mission-stage-name {
    font-size: 16px;
    font-weight: 700;
    line-height: 1.35;
}
.mission-stage-action {
    color: var(--body-text-color-subdued);
    font-size: 13px;
    line-height: 1.45;
    margin-top: 8px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.stage-input-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    padding: 10px;
    background: var(--background-fill-primary);
}
.stage-action-panel {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    background: var(--background-fill-primary);
    margin: 8px 0 !important;
    padding: 12px !important;
}
.stage-action-panel .wrap,
.stage-action-panel .form,
.stage-action-panel .block {
    max-width: 100% !important;
}
.stage-action-panel .upload-container {
    min-height: 76px !important;
    padding: 10px !important;
}
.stage-action-panel .image-container,
.stage-action-panel .file-preview,
.stage-action-panel video {
    max-height: 170px !important;
    overflow: hidden !important;
}
.stage-brief {
    border-left: 4px solid var(--color-accent, #f97316);
    padding: 10px 14px;
    border-radius: 8px;
    background: var(--background-fill-secondary);
    margin: 8px 0 12px 0;
}
.stage-run-row {
    align-items: end !important;
    gap: 10px !important;
    margin: 10px 0 10px 0;
    flex-wrap: wrap !important;
}
.stage-run-row button {
    min-height: 42px !important;
    white-space: nowrap !important;
    min-width: 148px !important;
}
.stage-run-row > div {
    min-width: 0 !important;
}
.stage-run-row > div:has(button) {
    flex: 0 0 auto !important;
}
.stage-run-row .compact-status {
    flex: 1 1 360px !important;
}
.stage-toolbar {
    align-items: end !important;
    gap: 10px !important;
}
.stage-toolbar > .form,
.stage-toolbar > div {
    min-width: 0 !important;
}
.stage-toolbar button {
    min-height: 42px !important;
}
.stage-toolbar textarea {
    min-height: 44px !important;
}
.stage-toolbar .upload-container,
.stage-toolbar .file-preview,
.stage-toolbar .image-container,
.stage-toolbar .wrap {
    min-height: 44px !important;
}
.stage-toolbar .upload-container {
    padding: 8px !important;
}
.stage-toolbar .label-wrap {
    margin-bottom: 4px !important;
}
.stage-result-window {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 16px;
    background: var(--background-fill-primary);
    margin: 8px 0 !important;
}
.stage-result-window::before {
    display: none;
}
.compact-status textarea {
    min-height: 56px !important;
}
.stage-action-panel .label-wrap,
.stage-result-window .label-wrap {
    font-size: 14px !important;
}
.stage-result-window textarea {
    font-size: 14px !important;
}
.s2s3-triptych {
    gap: 0 !important;
    margin-top: 8px !important;
    margin-bottom: 8px !important;
}
.s2s3-triptych > div {
    padding-left: 0 !important;
    padding-right: 0 !important;
}
.s2s3-triptych .image-container {
    margin: 0 !important;
    height: 260px !important;
}
.seg-eval-triptych {
    gap: 12px !important;
    margin-top: 10px !important;
    margin-bottom: 10px !important;
    align-items: stretch !important;
}
.seg-eval-triptych > div {
    padding-left: 0 !important;
    padding-right: 0 !important;
}
.seg-eval-triptych .image-container {
    margin: 0 !important;
    height: 280px !important;
}
"""


with gr.Blocks(
    title="AeroRescue-AI 低空应急救援智能感知与辅助决策系统",
    css=MISSION_FIRST_LAYOUT_CSS,
    elem_id="aerorescue-mission-app",
) as app:
    gr.HTML("""
        <h1 style='text-align: center'>AeroRescue-AI</h1>
    """)

    with gr.Tab("任务总览"):
        with gr.Accordion("任务总览", open=False, elem_classes=["stage-action-panel"]):
            attach_mission_dashboard_panel()

    with gr.Tab("一键任务演示"):
        with gr.Accordion("一键任务演示", open=False, elem_classes=["stage-action-panel"]):
            if attach_mission_control_panel is None:
                gr.Markdown(f"一键任务演示面板暂不可用：{MISSION_CONTROL_PANEL_IMPORT_ERROR}")
            else:
                attach_mission_control_panel()

    with gr.Tab("真实能力验证路线图"):
        with gr.Accordion("真实能力验证路线图", open=False, elem_classes=["stage-action-panel"]):
            if attach_validation_roadmap_panel is None:
                gr.Markdown(f"真实能力验证路线图面板暂不可用：{VALIDATION_ROADMAP_PANEL_IMPORT_ERROR}")
            else:
                attach_validation_roadmap_panel()

    with gr.Tab("流程导览"):
        with gr.Accordion("系统说明", open=False, elem_classes=["stage-action-panel"]):
            gr.Markdown(
                """
    ## AeroRescue-AI 系统功能说明

    AeroRescue-AI 是面向低空无人机应急救援场景的智能感知（让系统看懂图像）与辅助决策（给救援排序和路线建议）原型系统。系统围绕“无人机数据输入 → 灾情目标识别 → 环境风险理解 → 救援优先级排序 → 路径规划 → 灾情报告生成”构建完整演示链路，用于展示灾后城市区域的目标发现、风险研判和救援辅助决策能力。

    ### 一、系统总体流程

    1. 数据采集与上传  
       用户上传无人机图像、视频、航测照片、热红外图像或三维重建素材。系统首先完成输入检查、文件整理和基础可用性判断。

    2. 航测与场景预处理  
       对多张航测图像执行快速拼接预览，或在本机 Docker（容器运行环境）与 OpenDroneMap / ODM（开源无人机正射影像处理工具）可用时运行真实正射处理；对 360° 视频或多视角图像提取关键帧、特征点和匹配关系，为后续三维重建预处理提供基础数据。

    3. 热红外与热点分析  
       系统支持普通图像的模拟热力分析，也预留真实 radiometric thermal 文件解析入口。普通 JPG 只生成模拟热点结果，只有成功解析温度矩阵的热红外文件才被标记为真实测温。

    4. 目标检测与灾情感知  
       图像和视频输入进入目标检测模块。系统以 YOLOv11（快速目标识别模型）灾害目标检测为主，同时提供 Transformer RescueDet（另一类深度学习检测模型）和双后端对比模式，用于识别平民、救援人员、动物等救援相关目标，并输出检测框、类别、置信度（模型有多确定）和目标位置。

    5. 已训练语义分割模型与环境风险融合  
       系统使用本地已训练语义分割模型生成 pred_mask，并将水域、道路阻断、建筑损毁、可通行道路等环境类别转化为风险和路径代价，用于解释目标周边环境。

    6. 灾损评估与场景适用性门控  
       系统统计建筑损毁、道路阻断、水域等灾损信息，并判断当前画面属于局部侦察还是广域评估。若缺少图像、目标或模型权重，系统会明确提示证据不足，避免在证据不足时给出过度结论。

    7. TERP 救援优先级排序  
       TERP（目标-环境-可达性联合优先级模型）综合目标类型、检测置信度、环境风险和可达性信息，对多个候选目标进行救援优先级排序。输出包括目标编号、综合分数、风险等级、排序原因和人工复核提示。

    8. 普通 A* 与风险感知 A* 路径规划  
       系统在图像平面内生成参考救援路径，并对比普通 A*（经典自动寻路算法）路径与风险感知 A*（会避开危险区域的寻路算法）路径。风险感知路径会尽量避开水域、阻断道路和严重损毁区域。该路径是图像平面辅助参考，不是真实 GPS 导航路线。

    9. AI 灾情描述与综合报告导出  
       系统汇总检测、分割、路径、热红外、正射影像、三维重建和人工输入信息，生成中文灾情描述、模块摘要、Markdown 报告和 HTML 报告。若本机 Ollama 可用，可作为可选增强；不可用时自动使用规则模板生成稳定输出。

    ### 二、主要功能模块

    | 功能页 | 主要用途 | 输出结果 |
    | --- | --- | --- |
    | 正射影像 / 航测拼接预览 | 检查航测照片质量、重叠关系和 ODM（无人机正射处理工具）环境，生成快速拼接或真实 ODM 输出 | 拼接预览图、处理状态、运行日志、结果 JSON |
    | 模拟热红外 / 红外热点分析 | 对普通图像生成模拟热力图，或解析真实热红外温度矩阵 | 热力图、热点叠加图、真实性说明、温度/热点 JSON |
    | 通用数据输入 | 在系统首页折叠区统一导入照片和视频 | 共享照片、共享视频 |
    | 目标检测 | 识别救援相关目标，输出类别、置信度和检测框 | 检测图、检测详情、后端对比摘要 |
    | 灾情感知 | 融合目标检测、语义分割、环境风险和灾损统计 | 分割叠加图、灾损摘要、场景模式、环境风险排序 |
    | 综合决策 | 汇总灾情感知结果，生成 TERP（救援优先级）排序、路径规划和救援报告 | 路径图、救援优先级排名、路径可靠性、中文救援报告 |
    | 视频目标检测 | 对上传视频抽帧检测并生成标注视频 | 处理后视频、目标类别摘要 |
    | 360°视频 / 三维重建预处理 | 提取关键帧、特征点、匹配关系和简化点云预览 | 关键帧、匹配可视化、相机轨迹、PLY 点云预览 |
    | AI 灾情描述 | 根据已有模块结果和人工输入生成灾情描述 | 灾情描述 Markdown、生成日志 |
    | 综合报告导出 | 汇总所有已执行模块结果形成最终交付材料 | Markdown 报告、HTML 报告、报告摘要 |

    ### 三、核心决策链路

    无人机图像 / 视频  
    → 目标检测  
    → 已训练语义分割模型生成 pred_mask  
    → 环境风险与灾损评估  
    → 场景适用性门控  
    → TERP 救援优先级排序  
    → 救援入口建议  
    → 普通 A*（常规自动寻路）/ 风险感知 A*（避开危险区域的自动寻路）路径对比
    → 中文救援报告  
    → 综合报告导出

    ### 四、能力边界说明

    - 当前系统是本地 Gradio 竞赛原型，不接入真实飞控系统，也不直接控制无人机。
    - 路径规划结果基于图像平面像素坐标，仅用于辅助研判，不等同于真实地理坐标路径。
    - 普通 RGB 图像生成的热力图属于模拟分析，不代表真实温度测量。
    - 灾情感知结果必须来自本地已训练语义分割模型；未找到 checkpoint 时不会生成伪造分割结果。
    - 系统会把各模块产物统一写入 `outputs/` 目录，便于复核、归档和导出报告。
                """
                )
            gallery_items = gallery_image_items("zh")
            if gallery_items:
                with gr.Accordion("系统素材与参考输出", open=False, elem_classes=["stage-action-panel"]):
                    gr.Gallery(value=gallery_items, label="系统素材与参考输出", columns=3, height=320, allow_preview=True)

        with gr.Accordion("通用数据输入（照片 / 视频）", open=False):
            gr.Markdown(
                """
    这里保留共享输入能力，但不再单独占用一个 Tab。S4 局部精查、S2-S3 灾情感知和 S7-S8 综合决策会继续读取这里的照片或视频。
                """
            )
            imported_image = gr.Image(label="导入照片 / 无人机图像", type="pil", elem_classes=["stage-input-card"])
            imported_video = gr.Video(label="导入视频", autoplay=True)
            imported_segmentation_mask = gr.File(
                label="内部兼容输入",
                file_types=[".png", ".jpg", ".jpeg"],
                visible=False,
            )
            selected_features = gr.CheckboxGroup(
                ["目标检测", "灾情感知", "综合决策"],
                label="选择要使用的功能",
                value=["目标检测", "灾情感知", "综合决策"],
            )
            gr.Markdown(
                """
    **推荐流程**

    1. 在这里导入照片或视频。
    2. 进入对应 S 阶段页面点击运行按钮。
    3. 灾情感知阶段会固定使用本地已训练语义分割模型。
                """
            )
            gr.Examples(
                examples=[
                    ["examples/1019715_jpg.rf.58a43da4e0959d4e75f1eceb0d288bd0.jpg"],
                    ["examples/20250924_1153_Vacas_em_Alagamento_simple_compose_01k5y3bzjee4sbbf02c30c2phm1_png.rf.1caa0a0ff7a605e8b84669b0cc6fc364.jpg"],
                    ["examples/230714-india-flooding-mb-0831-d3a66d_jpg.rf.3e607c4f8f121834224f95ab0d44ddd6.jpg"],
                    ["examples/754_jpg.rf.47e7b8cdcfa1ffb020bb1b0588890f78.jpg"],
                    ["examples/775_jpg.rf.d2c4a77e35dd329df2478517c42c1176.jpg"],
                    ["examples/Flood-25_jpg.rf.92d30a193fb4f368a8d92f65f9669244.jpg"],
                ],
                inputs=imported_image,
                label="示例照片",
            )
            gr.Examples(
                examples=[[str(STATIC_VIDEO_PATH)]],
                inputs=imported_video,
                label="示例视频",
            )

    with gr.Tab("S1 高空建图"):
        with gr.Accordion("1. 阶段说明", open=False, elem_classes=["stage-action-panel"]):
            gr.Markdown(
                """
    ## 正射影像 / 航测拼接预览

    该模块提供两种处理模式：

    - Fast Preview / OpenCV 拼接预览：用于快速检查图像质量和重叠关系，不是专业正射影像。
    - Real ODM Orthomosaic / OpenDroneMap 真实正射处理：调用本机 Docker 运行 `opendronemap/odm`，需要 Docker 可用并准备具有 70%-80% 重叠度的无人机航测照片。该模式不会伪造 ODM 输出。
                """
                )
        with gr.Accordion("2. 输入与运行", open=False, elem_classes=["stage-action-panel"]):
            orthomosaic_files = gr.File(
                label="航测图像（可多选）",
                file_count="multiple",
                file_types=[".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"],
            )
            orthomosaic_mode = gr.Radio(
                [
                    "Fast Preview / OpenCV 拼接预览",
                    "Real ODM Orthomosaic / OpenDroneMap 真实正射处理",
                ],
                label="处理模式",
                value="Fast Preview / OpenCV 拼接预览",
            )
            with gr.Accordion("高级 ODM 参数", open=False):
                odm_task_name = gr.Textbox(label="ODM 任务名称", value="aerorescue_odm_task")
                odm_max_images = gr.Number(label="最多使用图像数（0 表示不限制）", value=0, precision=0)
                odm_fast_orthophoto = gr.Checkbox(label="ODM 快速正射参数 --fast-orthophoto", value=False)
            with gr.Row(elem_classes=["stage-run-row"]):
                orthomosaic_btn = gr.Button("运行高空建图", variant="primary")
                odm_env_btn = gr.Button("检查 ODM 环境", variant="secondary")
        with gr.Accordion("3. 核心结果", open=False, elem_classes=["stage-result-window"]):
            orthomosaic_status = gr.Textbox(label="生成提示", lines=2, elem_classes=["compact-status"])
            orthomosaic_image = gr.Image(label="拼接 / 预览图")
            with gr.Accordion("4. 详细日志与数据", open=False):
                orthomosaic_log = gr.Code(label="结果 JSON", language="json", lines=12)
                orthomosaic_run_log = gr.Textbox(label="运行日志", lines=12)
        orthomosaic_btn.click(
            fn=run_orthomosaic_mode,
            inputs=[orthomosaic_files, orthomosaic_mode, odm_task_name, odm_max_images, odm_fast_orthophoto],
            outputs=[orthomosaic_image, orthomosaic_status, orthomosaic_log, orthomosaic_run_log],
        )
        odm_env_btn.click(
            fn=check_odm_environment,
            inputs=[],
            outputs=[orthomosaic_status, orthomosaic_log, orthomosaic_run_log],
        )

    with gr.Tab("S2-S3 灾情感知与外部影响评估（高级深度版）"):
        with gr.Accordion("① 上传图像 + 运行按钮", open=True, elem_classes=["stage-action-panel"]):
            gr.Markdown(
                """
    固定使用本地已训练语义分割模型生成 pred_mask。覆盖图、黑底彩色分割图、图例和统计信息都来自同一个 pred_mask。

    SKAI 作为外部源码级建筑灾损评估模块接入，InaSAFE 作为外部源码级灾害影响评估模块接入；只有真实调用外部源码并验证到输出文件，才标记为真实 SKAI / InaSAFE 输出。依赖、权重、输入或 QGIS/GIS 环境缺失时只显示 unavailable，不生成替代假结果。
                """
            )
            perception_stage_image = gr.Image(label="上传高空图 / UAV 图像", type="pil")
            perception_img_size = gr.Number(label="推理尺寸", value=512, precision=0)
            with gr.Row(elem_classes=["stage-run-row"]):
                perception_btn = gr.Button("运行灾情感知分析", variant="primary")
        with gr.Accordion("② 已训练语义分割模型状态", open=True, elem_classes=["stage-result-window"]):
            perception_segmentation_status = gr.Textbox(label="已训练语义分割模型状态", lines=7)
            perception_model_status = gr.Textbox(label="运行状态", lines=3)
        with gr.Accordion("③ 覆盖图", open=True, elem_classes=["stage-result-window"]):
            perception_segmentation_overlay = gr.Image(label="覆盖图：原图 + pred_mask 半透明叠加")
        with gr.Accordion("④ 黑底彩色分割图", open=True, elem_classes=["stage-result-window"]):
            perception_segmentation_color = gr.Image(label="黑底彩色分割图：pred_mask 类别渲染")
        with gr.Accordion("⑤ 图例", open=True, elem_classes=["stage-result-window"]):
            perception_segmentation_legend = gr.Image(label="图例：类别颜色说明")
        with gr.Accordion("⑥ 灾情感知摘要", open=True, elem_classes=["stage-result-window"]):
            perception_summary = gr.Textbox(label="灾情感知摘要", lines=10)
        with gr.Accordion("⑦ 灾损与影响评估", open=True, elem_classes=["stage-result-window"]):
            with gr.Row():
                perception_skai_assessment = gr.Textbox(label="SKAI 建筑灾损结果", lines=13)
                perception_inasafe_assessment = gr.Textbox(label="InaSAFE 影响评估结果", lines=13)
            perception_skai_run_status = gr.Textbox(label="SKAI 运行状态", lines=8)
            perception_inasafe_run_status = gr.Textbox(label="InaSAFE 运行状态", lines=6)
            perception_external_files = gr.Textbox(label="外部输出文件摘要", lines=7)
            perception_unavailable_reasons = gr.Textbox(label="不可用原因 / 真实性边界", lines=10)
        with gr.Accordion("⑧ 下游决策建议", open=True, elem_classes=["stage-result-window"]):
            perception_downstream_impact = gr.Textbox(label="下游决策建议", lines=10)
        with gr.Accordion("⑨ 统一真实性边界", open=True, elem_classes=["stage-result-window"]):
            perception_truthfulness = gr.Textbox(label="统一真实性边界", lines=10)
        perception_btn.click(
            fn=run_damage_segmentation_analysis,
            inputs=[perception_stage_image, perception_img_size],
            outputs=[
                perception_segmentation_overlay,
                perception_segmentation_color,
                perception_segmentation_legend,
                perception_segmentation_status,
                perception_summary,
                perception_skai_assessment,
                perception_inasafe_assessment,
                perception_skai_run_status,
                perception_inasafe_run_status,
                perception_external_files,
                perception_unavailable_reasons,
                perception_downstream_impact,
                perception_truthfulness,
                perception_model_status,
            ],
        )

    with gr.Tab("S4 局部精查"):
        with gr.Accordion("1. 阶段说明", open=False, elem_classes=["stage-action-panel"]):
            gr.Markdown(
                """
    ## 目标检测

    该页面只负责从无人机图像或视频中识别救援相关目标，并输出检测框（目标在图中的位置）、类别、识别把握度（模型有多确定）和备用模型对比信息。语义分割与环境风险已拆分到“灾情感知”，救援排序和路径规划已拆分到“综合决策”。
                """
                )
            detection_input_mode = gr.Radio(
                ["图片检测", "视频检测"],
                label="检测输入类型",
                value="图片检测",
            )
        with gr.Group(visible=True) as image_detection_group:
            with gr.Accordion("2. 图片检测：输入与运行", open=False, elem_classes=["stage-action-panel"]):
                gr.Markdown("读取局部 RGB 图像，生成候选目标。检测到的人只作为候选目标，需要人工复核。")
                s4_image_source = gr.Radio(
                    ["首页照片", "本地上传"],
                    label="图片来源",
                    value="首页照片",
                )
                s4_stage_image = gr.Image(label="局部 RGB 图像（可选）", type="pil")
                detection_backend = gr.Radio(
                    [
                        "YOLO Rescue Targets",
                        "Transformer RescueDet",
                        "YOLO + Transformer Compare",
                    ],
                    label="目标识别方式",
                    value="YOLO Rescue Targets",
                )
                transformer_model_key = gr.Dropdown(
                    list(TRANSFORMER_DETECTION_MODELS.keys()),
                    label="备用检测模型",
                    value="rescuedet_deformable_detr",
                )
                conf_threshold = gr.Slider(label="最低识别把握度", minimum=0.0, maximum=1.0, step=0.05, value=0.30)
                output_model = gr.Dropdown(["yolov11n", "yolov11s", "yolov11m", "yolov11l"], label="主检测模型大小", info="选择要使用的 YOLOv11 模型版本。", value="yolov11m")
                with gr.Accordion("目标识别方式说明", open=False):
                    gr.Markdown(summarize_detection_backend_capabilities())
                    gr.Markdown(
                        "### 运行说明\n"
                        "- YOLO Rescue Targets：主目标识别模型，用来识别救援候选目标。\n"
                        "- Transformer RescueDet：备用检测模型，用来提供另一套候选结果，方便人工对比。\n"
                        "- YOLO + Transformer Compare：双模型对比模式，用来检查两个模型结果是否一致。\n"
                        "- 综合决策仍以主结果为准，备用模型只作为人工复核线索。"
                    )
                with gr.Row(elem_classes=["stage-run-row"]):
                    btn = gr.Button("运行图片检测", variant="primary")
            with gr.Accordion("3. 图片检测：核心结果", open=False, elem_classes=["stage-result-window"]):
                output_report = gr.Textbox(
                    label="生成提示",
                    lines=2,
                    placeholder="运行后显示生成状态和关键提示。",
                    elem_classes=["compact-status"],
                )
                output_image = gr.Image(label="处理后图像")
                with gr.Accordion("详细数据", open=False):
                    output_transformer_summary = gr.Textbox(
                        label="备用模型对比摘要",
                        lines=8,
                        placeholder="备用检测模型输出、失败原因或双模型一致性分析会显示在这里……",
                    )
                    output_details = gr.Dataframe(
                        headers=["编号", "目标类别", "识别把握度", "检测框位置", "中心点", "目标面积"],
                        label="检测详情",
                        interactive=False,
                    )

            btn.click(
                fn=target_detection_with_source,
                inputs=[
                    s4_image_source,
                    imported_image,
                    s4_stage_image,
                    detection_backend,
                    transformer_model_key,
                    conf_threshold,
                    output_model,
                ],
                outputs=[
                    output_image,
                    output_transformer_summary,
                    output_details,
                    output_report,
                ],
            )
        with gr.Group(visible=False) as video_detection_group:
            with gr.Accordion("2. 视频检测：输入与运行", open=False, elem_classes=["stage-action-panel"]):
                gr.Markdown("读取局部视频并按帧抽样检测，结果仍然只是图像/视频证据，不是现场确认结论。")
                s4_video_source = gr.Radio(
                    ["首页视频", "本地上传"],
                    label="视频来源",
                    value="首页视频",
                )
                s4_stage_video = gr.Video(label="视频（可选）", autoplay=True)
                video_conf_threshold = gr.Slider(label="视频最低识别把握度", minimum=0.0, maximum=1.0, step=0.05, value=0.30)
                frame_skip = gr.Slider(label="抽帧间隔", minimum=1, maximum=60, step=1, value=15)
                max_frames = gr.Slider(label="最大处理帧数（0 = 全视频）", minimum=0, maximum=600, step=30, value=0)
                video_model = gr.Dropdown(
                    ["yolov11n", "yolov11s", "yolov11m", "yolov11l"],
                    label="视频检测模型大小",
                    info="选择要使用的 YOLOv11 模型版本。",
                    value="yolov11m",
                )
                with gr.Row(elem_classes=["stage-run-row"]):
                    video_btn = gr.Button("运行视频检测", variant="primary")
            with gr.Accordion("3. 视频检测：核心结果", open=False, elem_classes=["stage-result-window"]):
                output_predictions = gr.Textbox(label="生成提示", lines=2, placeholder="运行后显示生成状态和目标摘要。", elem_classes=["compact-status"])
                output_video = gr.Video(label="处理后视频", autoplay=True)

            video_btn.click(
                fn=video_detection_with_source,
                inputs=[s4_video_source, imported_video, s4_stage_video, video_conf_threshold, video_model, frame_skip, max_frames],
                outputs=[output_video, output_predictions],
            )
        detection_input_mode.change(
            fn=lambda mode: (
                gr.update(visible=mode == "图片检测"),
                gr.update(visible=mode == "视频检测"),
            ),
            inputs=detection_input_mode,
            outputs=[image_detection_group, video_detection_group],
        )

    with gr.Tab("S5 目标复核"):
        with gr.Accordion("1. 阶段说明与真实性边界", open=False, elem_classes=["stage-action-panel"]):
            gr.Markdown(
                """
    ## S5 低空目标复核

    S5 接收 S4 生成的救援候选目标，将候选目标整理成可人工复核的证据包。真实救援中，中高空局部精查发现疑似目标后，低空无人机需要靠近目标补拍更清晰的 RGB 图像或不同角度照片；系统只负责整理视觉证据，不直接宣布“发现被困人员”。

    **S5 标准输出**

    - 候选目标裁剪图 `target_crop`
    - 周边环境裁剪图 `context_crop`
    - 复核状态 `review_status`
    - 复核备注 `review_note`
    - 是否需要二次巡查 `need_recheck`
    - 是否进入 S6 热红外复查 `thermal_check_required`

    **真实性边界**

    - AI 检测结果只能作为候选目标，不能称为已确认平民或已确认被困人员。
    - 目标复核阶段只提供视觉证据，供人工复核使用，不构成最终救援结论。
    - 即使人工把目标标记为“保留候选目标”，也不等于已经确认救援结果。
    - 裁剪图来自图像像素，可能遗漏裁剪框之外的重要上下文。
    - 系统不能编造候选目标，也不能编造人工复核决定。

    	当前独立 S5 页面先作为流程入口和证据结构说明；完整可运行链路请使用 **一键任务演示**，它会自动执行 S4 → S5 → S6 并生成 `outputs/target_verification/target_verification_result.json`。
    	            """
    	        )
        with gr.Accordion("2. 输入与运行", open=False, elem_classes=["stage-action-panel"]):
            gr.Markdown(
                """
    这里先提供独立上传入口，方便后续把 S4 生成的候选目标 JSON、低空近景 RGB 图像和人工复核记录接入 S5 证据裁剪流程。当前页面不伪造候选目标，也不自动生成复核结论。
                """
            )
            s5_candidate_json = gr.File(label="候选目标 JSON（来自 S4，可选）", file_types=[".json"])
            s5_close_view_images = gr.File(
                label="低空近景 RGB 图像（可多选，可选）",
                file_count="multiple",
                file_types=[".jpg", ".jpeg", ".png", ".webp"],
            )
            s5_review_json = gr.File(label="人工复核记录 JSON（可选）", file_types=[".json"])
            gr.Markdown("提示：后续可将这些输入连接到 S5 target_verification stage，生成目标裁剪图、周边环境图和热红外复查需求。")
        with gr.Accordion("3. 证据字段说明", open=False):
            gr.Dataframe(
                headers=[
                    "字段",
                    "含义",
                    "来源",
                    "真实性边界",
                ],
                value=[
                    [
                        "candidate_id / target_id",
                        "候选目标编号",
                        "S4 局部精查候选目标",
                        "候选目标不是已确认平民或已确认被困人员",
                    ],
                    [
                        "target_crop_path",
                        "目标本体裁剪图",
                        "原始 RGB 图或低空近景图裁剪",
                        "裁剪证据可能遗漏框外上下文",
                    ],
                    [
                        "context_crop_path",
                        "目标周边环境裁剪图",
                        "扩大 bbox 后的上下文区域",
                        "仅为图像像素证据",
                    ],
                    [
                        "review_status",
                        "need_review / confirmed_candidate / rejected_false_positive / need_recheck / urgent_review",
                        "人工复核动作",
                        "保留候选目标不等于已确认平民或已确认被困人员",
                    ],
                    [
                        "thermal_check_required",
                        "是否建议进入 S6 热红外辅助复查",
                        "候选类别与复核状态规则",
                        "热红外仍然只是辅助证据",
                    ],
                ],
                label="S5 目标复核证据结构",
                interactive=False,
            )

    with gr.Tab("S6 热红外复查"):
        with gr.Accordion("1. 阶段说明", open=False, elem_classes=["stage-action-panel"]):
            gr.Markdown(
                """
    ## 模拟热红外 / 红外热点分析

    该模块提供模拟热点分析、真实 radiometric thermal 测温解析入口和红外目标检测预留入口。普通 JPG 不能被当成真实温度矩阵；只有成功解析 radiometric thermal 文件中的 temperature matrix，才属于真实测温。
                """
                )
        with gr.Accordion("2. 输入与运行", open=False, elem_classes=["stage-action-panel"]):
            thermal_mode = gr.Radio(
                [
                    "Simulated Thermal / 模拟热红外",
                    "Radiometric Thermal / 真实热红外测温",
                    "Infrared Detection / 红外目标检测（预留）",
                ],
                label="热红外分析模式",
                value="Simulated Thermal / 模拟热红外",
            )
            thermal_image = gr.File(
                label="热红外 / RGB 图像文件",
                file_types=[".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"],
            )
            thermal_threshold = gr.Number(label="热点阈值 °C（仅真实热红外可选）", value=None, precision=2)
            gr.Markdown(
                """
    - 模拟热红外：只用于流程演示，不代表真实温度。
    - 真实热红外：必须成功解析 radiometric temperature matrix。
    - 红外目标检测：预留入口，不等同于真实测温。
                """
            )
            with gr.Row(elem_classes=["stage-run-row"]):
                thermal_btn = gr.Button("运行热红外复查", variant="primary")
        with gr.Accordion("3. 核心结果", open=False, elem_classes=["stage-result-window"]):
            thermal_status = gr.Textbox(label="生成提示", lines=2, elem_classes=["compact-status"])
            thermal_truthfulness = gr.Textbox(label="真实性说明", lines=2, elem_classes=["compact-status"])
            thermal_overlay = gr.Image(label="热点叠加图")
            with gr.Accordion("4. 热图与详细数据", open=False):
                thermal_heatmap = gr.Image(label="热力图")
                thermal_json = gr.Code(label="分析结果 JSON", language="json", lines=12)
        thermal_btn.click(
            fn=lambda image_file, mode, threshold: (
                lambda result: (
                    result[0],
                    result[1],
                    result[2],
                    json.loads(result[3]).get("truthfulness_note", "未提供真实性说明。") if result[3] else "未提供真实性说明。",
                    result[3],
                )
            )(analyze_thermal(image_file, mode=mode, threshold_celsius=threshold)),
            inputs=[thermal_image, thermal_mode, thermal_threshold],
            outputs=[thermal_heatmap, thermal_overlay, thermal_status, thermal_truthfulness, thermal_json],
            )
    with gr.Tab("S7-S8 决策路径"):
        with gr.Accordion("1. 阶段说明", open=False, elem_classes=["stage-action-panel"]):
            gr.Markdown(
                """
    ## 综合决策

    该页面位于目标检测和灾情感知之后，负责把检测目标、环境风险、灾损评估和场景门控（证据是否足够）结果汇总为救援优先级、路径规划、路径可靠性说明和中文救援报告。
                """
                )
        with gr.Accordion("2. 输入与运行", open=False, elem_classes=["stage-action-panel"]):
            decision_image_source = gr.Radio(
                ["首页照片", "本地上传"],
                label="图像来源",
                value="首页照片",
            )
            decision_stage_image = gr.Image(label="图像（可选）", type="pil")
            decision_stage_mask = gr.File(
                label="内部兼容输入",
                file_types=[".png", ".jpg", ".jpeg"],
                visible=False,
            )
            decision_detection_backend = gr.Radio(
                ["YOLO Rescue Targets", "Transformer RescueDet", "YOLO + Transformer Compare"],
                label="目标识别方式",
                value="YOLO Rescue Targets",
            )
            decision_transformer_model_key = gr.Dropdown(
                list(TRANSFORMER_DETECTION_MODELS.keys()),
                label="备用检测模型",
                value="rescuedet_deformable_detr",
            )
            decision_segmentation_source = gr.State("自动分割模型")
            with gr.Accordion("路径与模型高级参数", open=False):
                decision_start_x = gr.Number(label="救援起点 X", value=20, precision=0)
                decision_start_y = gr.Number(label="救援起点 Y（-1 表示使用底部默认值）", value=-1, precision=0)
                decision_use_manual_start = gr.Checkbox(label="使用手动起点", value=False)
                decision_force_path_planning = gr.Checkbox(label="强制生成路径（调试用）", value=False)
                decision_conf_threshold = gr.Slider(label="最低识别把握度", minimum=0.0, maximum=1.0, step=0.05, value=0.30)
                decision_model = gr.Dropdown(["yolov11n", "yolov11s", "yolov11m", "yolov11l"], label="主检测模型大小", value="yolov11m")
            with gr.Row(elem_classes=["stage-run-row"]):
                decision_btn = gr.Button("运行决策与路径建议", variant="primary")
        with gr.Accordion("3. 核心结果", open=False, elem_classes=["stage-result-window"]):
            decision_scene_gate_status = gr.Textbox(label="生成提示", lines=2, elem_classes=["compact-status"])
            decision_path_reliability = gr.Textbox(label="路径真实性", lines=2, elem_classes=["compact-status"])
            decision_path_overlay = gr.Image(label="路径规划叠加图")
            decision_path_summary = gr.Textbox(label="路径规划摘要", lines=6)
            decision_report = gr.Textbox(label="生成的救援报告", lines=14)
            with gr.Accordion("4. 图像与环境证据", open=False):
                decision_output_image = gr.Image(label="目标检测图")
                decision_segmentation_overlay = gr.Image(label="分割叠加图")
            with gr.Accordion("5. 详细表格与分析", open=False):
                decision_segmentation_status = gr.Textbox(label="模型状态", lines=4)
                decision_transformer_summary = gr.Textbox(label="备用模型对比摘要", lines=6)
                decision_damage_summary = gr.Textbox(label="灾损评估摘要（道路、水域、建筑损毁等统计）", lines=7)
                decision_scene_mode = gr.Textbox(label="场景模式（局部侦察或广域评估）", lines=3)
                decision_rescue_entry = gr.Textbox(label="救援入口建议", lines=4)
                decision_path_gate = gr.Textbox(label="路径规划门控（是否允许生成路径及原因）", lines=5)
                decision_details = gr.Dataframe(headers=["编号", "目标类别", "识别把握度", "检测框位置", "中心点", "目标面积"], label="检测详情", interactive=False)
                decision_segmentation_summary = gr.Dataframe(headers=["区域类别", "中文名称", "面积占比(%)"], label="环境区域汇总", interactive=False)
                decision_ranking = gr.Dataframe(
                    headers=["排名", "目标ID", "目标类别", "识别把握度", "检测框位置", "风险分数", "风险等级", "环境分数", "主要风险环境", "原因"],
                    label="环境风险排序",
                    interactive=False,
                )
                decision_terp_ranking = gr.Dataframe(
                    headers=["排名", "目标ID", "目标类别", "救援优先级分数", "优先级等级", "目标重要性", "环境风险", "可达性", "原因"],
                    label="救援优先级排名（EC-TERP）",
                    interactive=False,
                )
                decision_path_comparison = gr.Textbox(label="自动寻路对比（A*：普通路径 vs 避险路径）", lines=6)
            attach_llm_report_panel(
                [
                    decision_report,
                    decision_transformer_summary,
                    decision_segmentation_status,
                    decision_scene_gate_status,
                    decision_damage_summary,
                    decision_scene_mode,
                    decision_rescue_entry,
                    decision_path_gate,
                    decision_path_reliability,
                    decision_details,
                    decision_segmentation_summary,
                    decision_ranking,
                    decision_terp_ranking,
                    decision_path_summary,
                    decision_path_comparison,
                ]
            )
        decision_btn.click(
            fn=decision_detection_with_source,
            inputs=[
                decision_image_source,
                imported_image,
                decision_stage_image,
                decision_detection_backend,
                decision_transformer_model_key,
                decision_segmentation_source,
                imported_segmentation_mask,
                decision_stage_mask,
                decision_start_x,
                decision_start_y,
                decision_use_manual_start,
                decision_force_path_planning,
                decision_conf_threshold,
                decision_model,
            ],
            outputs=[
                decision_output_image,
                decision_segmentation_overlay,
                decision_path_overlay,
                decision_segmentation_status,
                decision_transformer_summary,
                decision_scene_gate_status,
                decision_damage_summary,
                decision_scene_mode,
                decision_rescue_entry,
                decision_path_gate,
                decision_path_reliability,
                decision_details,
                decision_segmentation_summary,
                decision_ranking,
                decision_terp_ranking,
                decision_path_summary,
                decision_path_comparison,
                decision_report,
            ],
        )
    with gr.Tab("S9 证据报告"):
        with gr.Accordion("1. 阶段说明", open=False, elem_classes=["stage-action-panel"]):
            gr.Markdown(
                """
    ## S9 证据链与辅助决策报告

    这一阶段只做最后汇总：把前面已经生成的阶段结果、证据链、真实性边界和人工复核要求整理成报告。报告用于辅助决策，不是最终救援结论，也不是现场行动命令。
                """
                )
        with gr.Accordion("2. 生成报告", open=False, elem_classes=["stage-action-panel"]):
            gr.Markdown("点击生成报告后，系统只汇总已有结果；缺失阶段会标记为未生成，不会补造结论。")
            with gr.Row(elem_classes=["stage-run-row"]):
                final_report_btn = gr.Button("生成证据报告", variant="primary")
        with gr.Accordion("3. 核心结果", open=False, elem_classes=["stage-result-window"]):
            final_report_status = gr.Textbox(label="生成提示", lines=2, elem_classes=["compact-status"])
            final_report_md = gr.File(label="Markdown 报告下载")
            final_report_html = gr.File(label="HTML 报告下载")
            with gr.Accordion("4. 报告正文预览", open=False):
                final_report_preview = gr.Textbox(label="报告预览", lines=24)
        final_report_btn.click(
            fn=export_final_report,
            inputs=[],
            outputs=[final_report_status, final_report_md, final_report_html, final_report_preview],
        )
        with gr.Accordion("5. 外部证据导入（预留）", open=False, elem_classes=["stage-action-panel"]):
            s9_evidence_ledger_upload = gr.File(
                label="Evidence Ledger JSON（可选）",
                file_types=[".json"],
            )
            s9_stage_result_uploads = gr.File(
                label="阶段结果 JSON（可多选，可选）",
                file_count="multiple",
                file_types=[".json"],
            )
            gr.Markdown("这里仅保留外部证据接入口；当前报告按钮仍按本地已生成结果汇总，不会伪造缺失阶段结果。")
        with gr.Accordion("6. 高级辅助工具（可选）", open=False):
            gr.Markdown("以下工具用于任务草案、复核沟通和证据审计；默认收起，避免干扰 S9 主报告流程。")
            attach_mission_copilot_panel()
            attach_mission_planner_panel()
            attach_evidence_audit_panel()
    with gr.Tab("S1 扩展三维"):
        with gr.Accordion("1. 阶段说明与真实性边界", open=False, elem_classes=["stage-action-panel"]):
            gr.Markdown(
                """
    ## 真实三维重建 / 摄影测量工作流

    该页面现在包含两类能力：

    - Real Workflow：调用 `modules/reconstruction_3d` 的真实 COLMAP / ODM 工作流。缺少 COLMAP、Docker、ODM 镜像或 panorama_sfm.py 时返回透明状态，不伪造结果。
    - Lightweight Preview：保留旧版 ORB 关键帧预览，仅用于快速可视检查，不是完整 SfM/MVS、真实 ODM 正射或 GPS 导航。

    真实性边界：360 全景查看不等于三维重建；无 GPS/GCP/EXIF geotag 时为相对尺度或非地理参考；所有输出仅供人工复核的辅助空间证据。
                """
            )

        with gr.Accordion("2. 真实三维重建与 ODM 验证", open=False, elem_classes=["stage-action-panel"]):
            real_reconstruction_mode = gr.Radio(
                ["standard_uav", "360_panorama", "odm", "report_only"],
                label="工作流模式",
                value="standard_uav",
            )
            real_reconstruction_video = gr.File(
                label="上传视频（可选，系统会先抽帧）",
                file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
            )
            real_reconstruction_images = gr.File(
                label="上传图像序列（可多选；视频和图像二选一）",
                file_count="multiple",
                file_types=[".jpg", ".jpeg", ".png", ".tif", ".tiff"],
            )
            real_fps = gr.Number(label="视频抽帧 FPS", value=1.0, precision=2)
            real_quality_filter = gr.Checkbox(label="启用画质过滤（模糊/过暗/近重复）", value=True)
            real_blur_threshold = gr.Number(label="模糊阈值 Laplacian variance", value=100.0, precision=2)
            real_brightness_threshold = gr.Number(label="亮度阈值", value=30.0, precision=2)
            with gr.Accordion("高级重建参数", open=False):
                colmap_matcher = gr.Radio(["sequential", "exhaustive"], label="COLMAP 匹配方式", value="sequential")
                colmap_run_dense = gr.Checkbox(label="运行 COLMAP 稠密重建 / PatchMatch stereo", value=False)
                colmap_run_mesher = gr.Checkbox(label="运行 COLMAP 网格生成", value=False)
                panorama_sfm_script = gr.Textbox(
                    label="panorama_sfm.py 路径（360 模式必需）",
                    value="third_party/colmap/python/examples/panorama_sfm.py",
                )
                odm_project_name = gr.Textbox(label="ODM 项目名称", value="aerorescue_odm")
                odm_docker_image = gr.Textbox(label="ODM Docker 镜像", value="opendronemap/odm")
                odm_camera_lens = gr.Dropdown(
                    ["auto", "perspective", "fisheye", "spherical", "equirectangular"],
                    label="ODM 相机镜头类型",
                    value="auto",
                )
                odm_feature_quality = gr.Dropdown(["ultra", "high", "medium", "low", "lowest"], label="ODM 特征质量", value="medium")
                odm_pc_quality = gr.Dropdown(["ultra", "high", "medium", "low", "lowest"], label="ODM 点云质量", value="medium")
                odm_dsm = gr.Checkbox(label="生成 DSM 高程模型", value=True)
                odm_dtm = gr.Checkbox(label="生成 DTM 地形模型", value=False)
                odm_fast_orthophoto_real = gr.Checkbox(label="ODM 快速正射参数（不等于完整验证正射）", value=False)
                odm_auto_pull = gr.Checkbox(label="允许自动拉取 ODM 镜像", value=False)
                reconstruction_timeout = gr.Number(label="命令超时秒数（0 表示不设置）", value=0, precision=0)
            with gr.Row(elem_classes=["stage-run-row"]):
                real_dependency_btn = gr.Button("检查真实重建依赖")
                real_reconstruction_btn = gr.Button("运行真实重建工作流", variant="primary")
            real_dependency_status = gr.Textbox(label="依赖状态摘要", lines=9)
            real_workflow_status = gr.Textbox(label="工作流状态摘要", lines=14)
            with gr.Accordion("重建输出文件与 JSON", open=False):
                real_dependency_json = gr.Code(label="依赖状态 JSON", language="json", lines=12)
                real_workflow_json = gr.Code(label="工作流状态 JSON", language="json", lines=18)
                real_workflow_status_file = gr.File(label="workflow_status.json")
                real_report_json_file = gr.File(label="reconstruction_report.json")
                real_report_md_file = gr.File(label="reconstruction_report.md")

            real_dependency_btn.click(
                fn=run_reconstruction_dependency_check,
                inputs=[panorama_sfm_script, odm_docker_image],
                outputs=[real_dependency_status, real_dependency_json],
            )
            real_reconstruction_btn.click(
                fn=run_real_reconstruction_workflow_ui,
                inputs=[
                    real_reconstruction_mode,
                    real_reconstruction_video,
                    real_reconstruction_images,
                    real_fps,
                    real_quality_filter,
                    real_blur_threshold,
                    real_brightness_threshold,
                    colmap_matcher,
                    colmap_run_dense,
                    colmap_run_mesher,
                    panorama_sfm_script,
                    odm_project_name,
                    odm_docker_image,
                    odm_camera_lens,
                    odm_feature_quality,
                    odm_pc_quality,
                    odm_dsm,
                    odm_dtm,
                    odm_fast_orthophoto_real,
                    odm_auto_pull,
                    reconstruction_timeout,
                ],
                outputs=[
                    real_workflow_status,
                    real_workflow_json,
                    real_workflow_status_file,
                    real_report_json_file,
                    real_report_md_file,
                ],
            )

        with gr.Accordion("3. 轻量三维预览（非真实 SfM / ODM）", open=False, elem_classes=["stage-action-panel"]):
            reconstruction_video = gr.File(
                label="上传 360°视频 / 普通重建视频",
                file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
            )
            max_keyframes = gr.Slider(
                label="最多抽取关键帧数",
                minimum=2,
                maximum=20,
                step=1,
                value=20,
            )
            reconstruction_btn = gr.Button("运行轻量三维预览 / ORB 点云预览", variant="secondary")
            reconstruction_output = gr.Image(label="关键帧预览")
            reconstruction_status = gr.Textbox(label="预览状态", lines=5)
            with gr.Accordion("预览细节与文件", open=False):
                reconstruction_features = gr.Image(label="特征点预览")
                reconstruction_matches = gr.Image(label="相邻帧匹配预览")
                reconstruction_trajectory = gr.Image(label="相机轨迹图")
                reconstruction_ply = gr.File(label="预览 PLY 文件（非真实重建成果）")
                reconstruction_json = gr.Code(label="预览结果 JSON", language="json", lines=12)
            reconstruction_btn.click(
                fn=process_reconstruction,
                inputs=[reconstruction_video, max_keyframes],
                outputs=[
            reconstruction_output,
                    reconstruction_features,
                    reconstruction_matches,
                    reconstruction_trajectory,
                    reconstruction_ply,
                    reconstruction_status,
                    reconstruction_json,
                ],
            )

    with gr.Tab("AI 灾情描述"):
        with gr.Accordion("1. 阶段说明", open=False, elem_classes=["stage-action-panel"]):
            gr.Markdown(
                """
    ## AI 灾情描述

    该模块会汇总用户输入、目标检测与综合决策报告、正射影像日志、热红外分析结果和三维重建摘要。若本机 Ollama 可用，可调用本地模型补充描述；否则自动使用规则模板生成 Markdown 灾情描述。
                    """
            )
        with gr.Accordion("2. 输入与运行", open=False, elem_classes=["stage-action-panel"]):
            scene_task_name = gr.Textbox(label="任务名称", value="AeroRescue-AI 应急救援任务")
            scene_note = gr.Textbox(label="人工场景说明", lines=4, placeholder="可填写灾害类型、地点、无人机视角、现场关注点等。")
            detection_report_text = gr.Textbox(
                label="目标检测与综合决策报告文本",
                lines=8,
                placeholder="可粘贴“目标检测与综合决策”页生成的救援报告。",
            )
            thermal_json_file = gr.File(label="热红外结果 JSON（可选）", file_types=[".json"])
            reconstruction_json_file = gr.File(label="三维重建结果 JSON（可选）", file_types=[".json"])
            orthomosaic_json_file = gr.File(label="正射影像处理日志 JSON（可选）", file_types=[".json"])
            with gr.Accordion("本地模型选项（可选）", open=False):
                use_ollama = gr.Checkbox(label="尝试使用本地 Ollama", value=False)
                ollama_url = gr.Textbox(label="本地 Ollama 地址", value="http://127.0.0.1:11434")
                ollama_model = gr.Textbox(label="本地模型名称", value="llama3.2")
            with gr.Row(elem_classes=["stage-run-row"]):
                scene_btn = gr.Button("生成 AI 灾情描述", variant="primary")
        with gr.Accordion("3. 核心结果", open=False, elem_classes=["stage-result-window"]):
            scene_description_output = gr.Markdown(label="灾情描述 Markdown")
            scene_description_file = gr.File(label="灾情描述文件下载")
        scene_btn.click(
            fn=generate_scene_description,
            inputs=[
                scene_task_name,
                scene_note,
                detection_report_text,
                thermal_json_file,
                reconstruction_json_file,
                orthomosaic_json_file,
                use_ollama,
                ollama_url,
                ollama_model,
            ],
            outputs=[scene_description_output, scene_description_file],
        )



    def _authorize_llm_api_request(request: Request):
        """Allow local UI self-calls and optional token-authenticated API calls."""
        client_host = request.client.host if request.client else ""
        if client_host in LOOPBACK_HOSTS:
            return

        expected_token = os.getenv("AERORESCUE_LLM_API_TOKEN", "").strip()
        provided_token = request.headers.get("x-aerorescue-token", "").strip()
        auth_header = request.headers.get("authorization", "").strip()
        if auth_header.lower().startswith("bearer "):
            provided_token = auth_header.split(" ", 1)[1].strip()

        if expected_token and provided_token == expected_token:
            return

        raise HTTPException(
            status_code=403,
            detail="LLM API endpoints accept local requests only unless AERORESCUE_LLM_API_TOKEN is configured and provided.",
        )


    @app.app.post("/api/llm/mission-report")
    async def llm_mission_report_api(payload: dict, request: Request):
        """Generate an optional LLM-assisted post-processing mission report."""
        _authorize_llm_api_request(request)
        mission_result = payload.get("mission_result", payload) if isinstance(payload, dict) else {}
        return generate_mission_report(mission_result)


    @app.app.post("/api/llm/mission-copilot")
    async def llm_mission_copilot_api(payload: dict, request: Request):
        """Answer a mission question using only assembled mission evidence."""
        _authorize_llm_api_request(request)
        payload = payload if isinstance(payload, dict) else {}
        return answer_mission_copilot_question(
            mission_id=payload.get("mission_id", "current_mission"),
            question=payload.get("question", ""),
        )


    @app.app.post("/api/llm/mission-planner")
    async def llm_mission_planner_api(payload: dict, request: Request):
        """Generate, validate, and execute a white-listed mission tool plan."""
        _authorize_llm_api_request(request)
        payload = payload if isinstance(payload, dict) else {}
        return execute_mission_planner(
            mission_id=payload.get("mission_id", "current_mission"),
            user_goal=payload.get("user_goal", ""),
        )


    @app.app.post("/api/llm/evidence-audit")
    async def llm_evidence_audit_api(payload: dict, request: Request):
        """Audit mission outputs for evidence consistency and authenticity boundaries."""
        _authorize_llm_api_request(request)
        payload = payload if isinstance(payload, dict) else {}
        return run_evidence_audit(
            mission_id=payload.get("mission_id", "current_mission"),
            audit_target=payload.get("audit_target", "all"),
        )


    if __name__ == "__main__":
        app.launch(
            server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
            server_port=int(os.environ.get("GRADIO_SERVER_PORT", "9101")),
            share=False,
            allowed_paths=[str(APP_DIR), str(ROOT_DIR / "static"), str(ROOT_DIR / "outputs")],
        )
