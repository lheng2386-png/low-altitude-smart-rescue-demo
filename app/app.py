import os
import gradio as gr
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from priority_ranker import rank_targets
from report_generator import generate_report
from environment_risk import CLASS_DISPLAY_NAMES
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
from thermal_service import analyze_thermal
from reconstruction_service import process_reconstruction
from scene_description_service import generate_scene_description
from report_export_service import export_final_report

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
for output_name in ["orthomosaic", "thermal", "detection", "reconstruction", "reports"]:
    (ROOT_DIR / "outputs" / output_name).mkdir(parents=True, exist_ok=True)
MODEL_CACHE = {}
SEGMENTATION_MODEL_CACHE = {}
VIDEO_CLASS_MIN_CONF = {
    "dog": 0.45,
    "cat": 0.45,
}

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
        <p style='text-align: center'>使用Gradio构建 · YOLO 灾害目标检测、可选语义分割风险融合与 A* 图像平面路径规划</p>
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
→ 语义分割来源  
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

- 上传掩码：支持类别编号掩码或彩色掩码融合。
- 自动分割模型：实验性功能，需要本地权重文件。
- 无分割：可用回退，仅基于目标与默认代价进行风险评分。
- TERP：融合目标、环境与路径可达性优先级。
- 场景适用性门控：目标、掩码或权重不足时自动回退，不夸大能力。
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


def _target_route_item(target):
    return {
        "target_id": target.get("id"),
        "class_name": target.get("class_name"),
        "center": target.get("center"),
        "bbox": target.get("bbox"),
    }


def image_detection(image, segmentation_source, segmentation_mask_path, start_x, start_y, conf_threshold, model_variant, language="zh"):
    if image is None:
        message = "Please upload an image first." if lang_key(language) == "en" else "请先上传一张图像。"
        return None, None, None, message, [], [], [], [], "", "", message

    image_width, image_height = image.size
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    model_image = get_model(model_variant)
    
    results = model_image(image_bgr, conf=conf_threshold)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    annotated_image = custom_bounding_box(image_rgb, results, language=language)

    targets = extract_targets(results)
    segmentation_mask = None
    segmentation_overlay = None
    segmentation_summary = {}
    segmentation_status = []

    segmentation_source_key = normalize_segmentation_source(segmentation_source)
    auto_model_available = False

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
        else:
            segmentation_status.append(t(language, "uploaded_mask_missing"))
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
                else:
                    segmentation_status.append(t(language, "auto_prediction_ok"))
                    segmentation_mask, segmentation_summary = prepare_valid_segmentation_mask(
                        predicted_mask,
                        image_width,
                        image_height,
                        segmentation_status,
                        language,
                    )
            else:
                segmentation_status.append(t(language, "auto_model_unavailable"))
        else:
            segmentation_status.append(t(language, "auto_weights_missing"))
    else:
        segmentation_status.append(t(language, "no_seg_selected"))

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
    segmentation_status.append(f"场景适用性门控：{scene_gate['level']}。{scene_gate['message']}")

    ranked_targets = rank_targets(targets, image_width, image_height, segmentation_mask, language=language)
    if start_y is None or float(start_y) < 0:
        start_y = image_height - 20
    if start_x is None:
        start_x = 20

    baseline_path_result = plan_baseline_path(
        ranked_targets,
        image_width,
        image_height,
        start_point=(start_x, start_y),
    )
    path_result = plan_risk_aware_path(
        ranked_targets,
        segmentation_mask,
        image_width,
        image_height,
        start_point=(start_x, start_y),
    )
    path_comparison = compare_path_plans(baseline_path_result, path_result, segmentation_mask)

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
            start_point=(start_x, start_y),
        )

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
        path_overlay = create_path_overlay(base_path_image, path_result)
    report = generate_report(targets, ranked_targets, segmentation_summary, path_result, terp_rankings, path_comparison, language=language)
    summary_text = path_summary_text(path_result, has_segmentation_mask, language=language)
    comparison_text = path_comparison_text(path_comparison, language=language)

    return (
        annotated_image,
        segmentation_overlay,
        path_overlay,
        "\n".join(segmentation_status),
        target_table_rows(targets, language=language),
        segmentation_summary_rows(segmentation_summary, language=language),
        ranking_table_rows(ranked_targets, language=language),
        terp_table_rows(terp_rankings, language=language),
        summary_text,
        comparison_text,
        report,
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

with gr.Blocks(title="AeroRescue-AI 低空应急救援智能感知与辅助决策系统") as app:
    gr.HTML("""
        <h1 style='text-align: center'>AeroRescue-AI</h1>
        <p style='text-align: center'>使用Gradio构建 · YOLO 灾害目标检测、可选语义分割风险融合与 A* 图像平面路径规划</p>
    """)

    with gr.Tab("系统首页"):
        gr.Markdown(
            """
## AeroRescue-AI 系统首页

当前主系统由 7 个功能页组成：

1. 正射影像生成：多图航测拼接、EXIF/GPS 检查、处理日志输出。
2. 热红外分析：模拟温度矩阵、热力图、热点掩码、风险 JSON。
3. 目标检测与综合决策：YOLOv11、语义分割、TERP、风险感知 A*、中文报告、视频检测。
4. 360°视频 / 三维重建：关键帧、ORB 特征、匹配、相机轨迹、简化 PLY 点云。
5. AI 灾情描述：Ollama 可选，规则模板 fallback。
6. 综合报告导出：汇总各模块输出，生成 Markdown 和 HTML 报告。

所有模块输出统一写入 `outputs/` 目录。
            """
        )
        gallery_items = gallery_image_items("zh")
        if gallery_items:
            gr.Gallery(value=gallery_items, label="系统素材与参考输出", columns=3, height=320, allow_preview=True)

    with gr.Tab("正射影像生成"):
        with gr.Row():
            with gr.Column():
                orthomosaic_files = gr.File(
                    label="上传航测图像（可多选）",
                    file_count="multiple",
                    file_types=[".jpg", ".jpeg", ".png", ".webp"],
                )
                orthomosaic_btn = gr.Button("运行正射影像生成", variant="primary")
            with gr.Column():
                orthomosaic_image = gr.Image(label="正射影像 / 接片结果")
                orthomosaic_status = gr.Textbox(label="状态提示", lines=4)
                orthomosaic_log = gr.Code(label="处理日志 JSON", language="json", lines=12)
        orthomosaic_btn.click(
            fn=process_orthomosaic,
            inputs=[orthomosaic_files],
            outputs=[orthomosaic_image, orthomosaic_status, orthomosaic_log],
        )

    with gr.Tab("热红外分析"):
        with gr.Row():
            with gr.Column():
                thermal_image = gr.File(
                    label="上传 RGB / 灰度 / 热红外图像",
                    file_types=[".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"],
                )
                thermal_btn = gr.Button("运行热红外分析", variant="primary")
            with gr.Column():
                thermal_heatmap = gr.Image(label="热力图")
                thermal_overlay = gr.Image(label="热点叠加图")
                thermal_status = gr.Textbox(label="状态说明", lines=4)
                thermal_json = gr.Code(label="分析结果 JSON", language="json", lines=12)
        thermal_btn.click(
            fn=analyze_thermal,
            inputs=[thermal_image],
            outputs=[thermal_heatmap, thermal_overlay, thermal_status, thermal_json],
        )

    with gr.Tab("目标检测与综合决策"):
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="上传图像", type="pil")
                segmentation_source = gr.Radio(
                    ["上传掩码", "自动分割模型", "无分割"],
                    label="语义分割来源",
                    value="上传掩码",
                )
                segmentation_mask = gr.File(
                    label="可选语义分割掩码上传",
                    file_types=[".png", ".jpg", ".jpeg"],
                )
                start_x = gr.Number(label="救援起点 X", value=20, precision=0)
                start_y = gr.Number(label="救援起点 Y（-1 表示使用底部默认值）", value=-1, precision=0)
                conf_threshold = gr.Slider(label="置信度阈值", minimum=0.0, maximum=1.0, step=0.05, value=0.30)
                output_model = gr.Dropdown(["yolov11n", "yolov11s", "yolov11m", "yolov11l"], label="选择模型", info="选择要使用的 YOLOv11 模型版本。", value="yolov11m")
                btn = gr.Button("处理图像", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="处理后图像")
                output_segmentation_overlay = gr.Image(label="分割叠加图")
                output_path_overlay = gr.Image(label="路径规划叠加图")
                output_segmentation_status = gr.Textbox(
                    label="语义分割来源状态",
                    lines=4,
                    placeholder="语义分割来源状态会显示在这里……",
                )
                output_details = gr.Dataframe(
                    headers=["编号", "类别", "置信度", "边框", "中心点", "面积"],
                    label="检测详情",
                    interactive=False,
                )
                output_segmentation_summary = gr.Dataframe(
                    headers=["类别名", "显示名称", "面积占比(%)"],
                    label="语义分割汇总",
                    interactive=False,
                )
                output_ranking = gr.Dataframe(
                    headers=[
                        "排名",
                        "目标ID",
                        "类别",
                        "置信度",
                        "边框",
                        "风险分数",
                        "风险等级",
                        "环境分数",
                        "主导环境",
                        "原因",
                    ],
                    label="风险排序",
                    interactive=False,
                )
                output_terp_ranking = gr.Dataframe(
                    headers=[
                        "排名",
                        "目标ID",
                        "类别",
                        "TERP 分数",
                        "TERP 等级",
                        "目标分数",
                        "环境分数",
                        "可达性分数",
                        "原因",
                    ],
                    label="TERP 排名",
                    interactive=False,
                )
                output_path_summary = gr.Textbox(
                    label="路径规划摘要",
                    lines=6,
                    placeholder="路径规划摘要会显示在这里……",
                )
                output_path_comparison = gr.Textbox(
                    label="A* 路径对比",
                    lines=6,
                    placeholder="普通 A* 与风险感知 A* 的对比会显示在这里……",
                )
                output_report = gr.Textbox(
                    label="生成的救援报告",
                    lines=14,
                    placeholder="救援报告会显示在这里……",
                )

        btn.click(
            fn=image_detection,
            inputs=[image, segmentation_source, segmentation_mask, start_x, start_y, conf_threshold, output_model],
            outputs=[
                output_image,
                output_segmentation_overlay,
                output_path_overlay,
                output_segmentation_status,
                output_details,
                output_segmentation_summary,
                output_ranking,
                output_terp_ranking,
                output_path_summary,
                output_path_comparison,
                output_report,
            ],
        )
    
        gr.Examples(
            examples=[
                ["examples/1019715_jpg.rf.58a43da4e0959d4e75f1eceb0d288bd0.jpg"],
                ["examples/20250924_1153_Vacas_em_Alagamento_simple_compose_01k5y3bzjee4sbbf02c30c2phm1_png.rf.1caa0a0ff7a605e8b84669b0cc6fc364.jpg"],
                ["examples/230714-india-flooding-mb-0831-d3a66d_jpg.rf.3e607c4f8f121834224f95ab0d44ddd6.jpg"],
                ["examples/754_jpg.rf.47e7b8cdcfa1ffb020bb1b0588890f78.jpg"],
                ["examples/775_jpg.rf.d2c4a77e35dd329df2478517c42c1176.jpg"],
                ["examples/f-banglafloods-a-20190725_jpg.rf.db7b95e9eb7d8294b89644a27cc18166.jpg"],
                ["examples/Flood-25_jpg.rf.92d30a193fb4f368a8d92f65f9669244.jpg"],
                ["examples/Flood-30_jpg.rf.a9d21f122ddb98ee863989f552c0adc4.jpg"],
                ["examples/Flood-46_jpg.rf.1b3bd9e0e51798a4f61a51de0a694c6d.jpg"],
                ["examples/Flood-7_jpg.rf.a71bfe309c707883299f283ca207306b.jpg"],
                ["examples/image_123f58f43036403cb7aab908fe5fc69d_png.rf.b94dda710e99cc3ab85dbbd7f0d196f0.jpg"],
                ["examples/image23_jpeg.rf.20eca34e2be7c8a452a1ab682e1254cc.jpg"],
                ["examples/image_24d9705c165d4c818b9d10631d0ce48e_png.rf.5e504a35a21ec0f7adaba4a76a4edf09.jpg"],
                ["examples/image_2d402afa0296407d953e3fb2a46167a7_png.rf.f1aea2fc84dcf5428764241fe5843d53.jpg"],
                ["examples/image_4269233d29ec4a55941013d8660768db_png.rf.310026b583728ef0fc05a95e1fbffd42.jpg"],
                ["examples/image_536b176558764282b5dcfb33115db7bb_png.rf.0b1dfb32c8b26ed9520324d7e0123683.jpg"],
                ["examples/image_55dad25f64af4067be760720adfb3372_png.rf.5941414d221b13d9902e4005b5852a0c.jpg"],
                ["examples/image_572fd077c88b46bbbb3c6c5a74a93652_png.rf.f30b9b6d9e8abf00accb625882d0fdf9.jpg"],
                ["examples/image_674fd25133f64fd6bb6ddcfe36168583_png.rf.99e3cf8ca1ed9dc5b849b70394c6f545.jpg"],
                ["examples/image_6990bbbf052a4ae199b59e0151d1ce34_png.rf.c2d6327ad2ce2a66e2fdf6ab73882c91.jpg"],
                ["examples/image_91ff6fbe98c6465988897977f9a7a3ac_png.rf.e7925b19b659948901c256abc271b318.jpg"],
                ["examples/image_9c9a10f736f04d15a407c16e8eddd2b5_png.rf.89b86afb45331b0705a17f70369e0f3d.jpg"],
                ["examples/image_9c9d9969450e4db7ad86219f535c79c5_png.rf.23949fc7e161024b5d62c98a2279091e.jpg"],
                ["examples/image_9e67cb7ca8634c199296e5360aff9d52_png.rf.8331d74fd27b7369d7e9f7ad0d26caa4.jpg"],
                ["examples/image_a0e220e5d36b4253a0abf3db8e56c696_png.rf.1ba0ad185f497aca83d5c74087c181aa.jpg"],
                ["examples/image_b014c660bf2d439785cf1ffdfa9b5c55_png.rf.a306e8b69be0e029f98b52809649037e.jpg"],
                ["examples/image_b32be24ecb5f4af6ab4afb8f18f24f11_png.rf.2f605c0712858483d925480bda9c815c.jpg"],
                ["examples/image_b43bdbd062914dbdb513e2ef5f2b5d1d_png.rf.a7dfcd6f380c8ff12360459e06d67744.jpg"],
                ["examples/image_b4ec2525bfee4e8b8c154be463f7255e_png.rf.8b3986c3f6b2da8fa33f266734e57098.jpg"],
                ["examples/image_b7701fbd19444453a79356cae619bac7_png.rf.d0dd742a003c9ed23577eb367fd5ad92.jpg"],
                ["examples/image_de8817d6c699457bbee71252f69b83d2_png.rf.f00e820c7e9f340e16078d12722423c9.jpg"],
                ["examples/image_fed89c6e1d1c4f04a92ad4aca9f85f10_png.rf.799a7361fa4833d0b056c2e736c4048b.jpg"],
                ["examples/imagem_008_232_png.rf.7fc9f7b2b426747ebca6453e5e6ee2a6.jpg"],
                ["examples/imagem_032_png.rf.2611a927a635635b644206943646bc49.jpg"],
                ["examples/imagem_058-copia-_png.rf.22689b7c998b76dd1af82e218bb0ad7c.jpg"],
                ["examples/imagem_064-copia-_png.rf.2e109916a38a8cfe3b158303c3bfa95f.jpg"],
                ["examples/images18_jpg.rf.c6874bfb0609dc6d52defda4e161d25e.jpg"],
                ["examples/images219_jpg.rf.793784542da37f5a78a3837688314c97.jpg"],
                ["examples/ph_17939_63492_jpg.rf.bf0e962767adc644290645db26ab9e26.jpg"]
            ],
            inputs=image,
            label="示例图像"
        )

        gr.Markdown("## 视频目标检测")
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="上传视频", autoplay=True)
                video_conf_threshold = gr.Slider(label="视频置信度阈值", minimum=0.0, maximum=1.0, step=0.05, value=0.30)
                frame_skip = gr.Slider(label="帧跳过（越大越快）", minimum=1, maximum=60, step=1, value=15)
                max_frames = gr.Slider(label="最大处理帧数（0 = 全视频）", minimum=0, maximum=600, step=30, value=0)
                video_model = gr.Dropdown(
                    ["yolov11n", "yolov11s", "yolov11m", "yolov11l"],
                    label="选择视频检测模型",
                    info="选择要使用的 YOLOv11 模型版本。",
                    value="yolov11m",
                )
                video_btn = gr.Button("处理视频", variant="primary")
            with gr.Column():
                output_video = gr.Video(label="处理后视频", autoplay=True)
                output_predictions = gr.Textbox(label="预测结果", placeholder="预测结果会显示在这里……")

        video_btn.click(
            fn=video_detection,
            inputs=[video, video_conf_threshold, video_model, frame_skip, max_frames],
            outputs=[output_video, output_predictions],
        )

        gr.Examples(
            examples=[[str(STATIC_VIDEO_PATH)]],
            inputs=video,
            label="示例视频",
        )

    with gr.Tab("360°视频 / 三维重建"):
        gr.Markdown(
            """
## 360°视频 / 三维重建

上传视频后，系统会抽取关键帧、提取 ORB 特征、匹配相邻关键帧，并生成简化相机轨迹和 PLY 点云文件。当前为轻量本地重建预处理，不强依赖 COLMAP、OpenSfM 或 Open3D。
            """
        )

        with gr.Row():
            with gr.Column():
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
                reconstruction_btn = gr.Button("运行三维重建预处理", variant="primary")
            with gr.Column():
                reconstruction_output = gr.Image(label="关键帧预览")
                reconstruction_features = gr.Image(label="特征点预览")
                reconstruction_matches = gr.Image(label="相邻帧匹配预览")
                reconstruction_trajectory = gr.Image(label="相机轨迹图")
                reconstruction_ply = gr.File(label="点云 PLY 下载")
                reconstruction_status = gr.Textbox(label="重建状态", lines=5)
                reconstruction_json = gr.Code(label="重建结果 JSON", language="json", lines=12)
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
        gr.Markdown(
            """
## AI 灾情描述

该模块会汇总用户输入、目标检测与综合决策报告、正射影像日志、热红外分析结果和三维重建摘要。若本机 Ollama 可用，可调用本地模型补充描述；否则自动使用规则模板生成 Markdown 灾情描述。
            """
        )
        with gr.Row():
            with gr.Column():
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
                use_ollama = gr.Checkbox(label="尝试使用本地 Ollama", value=False)
                ollama_url = gr.Textbox(label="本地 Ollama 地址", value="http://127.0.0.1:11434")
                ollama_model = gr.Textbox(label="本地模型名称", value="llama3.2")
                scene_btn = gr.Button("生成 AI 灾情描述", variant="primary")
            with gr.Column():
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

    with gr.Tab("综合报告导出"):
        gr.Markdown(
            """
## 综合报告导出

点击按钮后，系统会汇总 `outputs/orthomosaic/`、`outputs/thermal/`、`outputs/reconstruction/` 和 `outputs/reports/` 中已有结果，生成 Markdown 与 HTML 综合报告。尚未执行的模块会在报告中标记为“该模块尚未执行”。
            """
        )
        with gr.Row():
            with gr.Column():
                final_report_btn = gr.Button("生成综合报告", variant="primary")
                final_report_status = gr.Textbox(label="状态提示", lines=4)
                final_report_md = gr.File(label="Markdown 报告下载")
                final_report_html = gr.File(label="HTML 报告下载")
            with gr.Column():
                final_report_preview = gr.Textbox(label="报告预览", lines=24)
        final_report_btn.click(
            fn=export_final_report,
            inputs=[],
            outputs=[final_report_status, final_report_md, final_report_html, final_report_preview],
        )

if __name__ == "__main__":
    app.launch(allowed_paths=[str(APP_DIR), str(ROOT_DIR / "static"), str(ROOT_DIR / "outputs")])
