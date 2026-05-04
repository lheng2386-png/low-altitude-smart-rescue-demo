import json
from pathlib import Path

import cv2
import numpy as np

from radiometric_thermal_service import analyze_radiometric_thermal


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "thermal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _as_path(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)


def analyze_simulated_thermal(image_file):
    """Analyze RGB/gray imagery as simulated hotspots, not real temperature."""
    path = _as_path(image_file)
    if not path:
        return None, None, "未上传图像。", "{}"
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, None, "图像读取失败。", "{}"

    is_simulated = True
    if image.ndim == 2:
        gray = image.astype(np.float32)
        base = cv2.cvtColor(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        base = image[:, :, :3]
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Simulated mode deliberately produces an intensity score, not Celsius.
    # Ordinary RGB/JPG/PNG data cannot be converted into real temperature.
    normalized_score = cv2.normalize(gray, None, 0.0, 1.0, cv2.NORM_MINMAX)
    normalized = (normalized_score * 255).clip(0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    threshold = float(np.percentile(normalized_score, 95))
    hotspot = normalized_score >= threshold
    hotspot_u8 = (hotspot.astype(np.uint8) * 255)
    overlay = cv2.addWeighted(base.astype(np.uint8), 0.45, heatmap, 0.55, 0)
    overlay[hotspot] = (0, 255, 255)

    contours, _ = cv2.findContours(hotspot_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hotspot_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 12:
            continue
        hotspot_count += 1
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)

    area_ratio = float(np.count_nonzero(hotspot)) / max(1, hotspot.size)
    max_score = float(np.max(normalized_score))
    mean_score = float(np.mean(normalized_score))
    if area_ratio > 0.08 or max_score > 0.86:
        risk = "High"
        explanation = "模拟热点范围较大，建议结合真实热红外或现场信息复核火源、被困人员或危险热源。"
    elif area_ratio > 0.025:
        risk = "Medium"
        explanation = "存在局部高温热点，建议结合可见光图像进行复核。"
    else:
        risk = "Low"
        explanation = "未发现大面积异常热点。"

    heatmap_path = OUTPUT_DIR / "thermal_heatmap.jpg"
    mask_path = OUTPUT_DIR / "hotspot_mask.jpg"
    overlay_path = OUTPUT_DIR / "thermal_overlay.jpg"
    result_path = OUTPUT_DIR / "thermal_result.json"
    cv2.imwrite(str(heatmap_path), heatmap)
    cv2.imwrite(str(mask_path), hotspot_u8)
    cv2.imwrite(str(overlay_path), overlay)

    result = {
        "thermal_mode": "simulated",
        "temperature_matrix": None,
        "temperature_matrix_path": None,
        "max_temperature": None,
        "mean_temperature": None,
        "unit": "none",
        "normalized_hotspot_score_max": round(max_score, 4),
        "normalized_hotspot_score_mean": round(mean_score, 4),
        "hotspot_score_threshold": round(threshold, 4),
        "hotspot_count": hotspot_count,
        "hotspot_area_ratio": round(area_ratio, 4),
        "risk_level": risk,
        "risk_explanation": explanation,
        "is_simulated_temperature": is_simulated,
        "is_real_temperature_measurement": False,
        "truthfulness_note": "This is a simulated thermal visualization based on grayscale/intensity analysis. It is not a real temperature measurement.",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    status = "模拟热红外分析完成。当前结果由图像灰度归一化生成，不代表真实热红外测温。"
    return str(heatmap_path), str(overlay_path), status, json.dumps(result, ensure_ascii=False, indent=2)


def analyze_real_radiometric_thermal(image_file, threshold_celsius=None):
    """Analyze a FLIR/DJI radiometric file without falling back to fake temperatures."""
    result = analyze_radiometric_thermal(
        _as_path(image_file),
        threshold_celsius=threshold_celsius,
        output_dir=OUTPUT_DIR,
    )
    result_path = OUTPUT_DIR / "thermal_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    status = (
        "真实 Radiometric Thermal 解析完成，已生成真实 temperature matrix。"
        if result.get("success") and result.get("is_real_temperature_measurement")
        else f"真实 Radiometric Thermal 解析失败：{result.get('error', 'unknown error')}。系统不会用灰度值伪造温度。"
    )
    return (
        result.get("heatmap_path") or None,
        result.get("overlay_path") or None,
        status,
        json.dumps(result, ensure_ascii=False, indent=2),
    )


def analyze_infrared_detection_placeholder(image_file):
    """Reserve an infrared object-detection path without claiming temperature measurement."""
    result = {
        "thermal_mode": "infrared_detection",
        "success": False,
        "is_real_temperature_measurement": False,
        "truthfulness_note": "红外目标检测不等于真实温度测量。本模式当前仅预留，后续可接 HIT-UAV 等红外目标检测数据与模型。",
        "error": "Infrared Detection is a placeholder in this version.",
    }
    status = "红外目标检测模式当前仅预留：它用于红外图像下的目标检测，不等于真实测温。"
    return None, None, status, json.dumps(result, ensure_ascii=False, indent=2)


def analyze_thermal(image_file, mode="Simulated Thermal / 模拟热红外", threshold_celsius=None):
    """Compatibility dispatcher for Gradio and existing tests."""
    mode_text = str(mode or "Simulated Thermal")
    if mode_text.startswith("Radiometric"):
        return analyze_real_radiometric_thermal(image_file, threshold_celsius=threshold_celsius)
    if mode_text.startswith("Infrared"):
        return analyze_infrared_detection_placeholder(image_file)
    return analyze_simulated_thermal(image_file)
