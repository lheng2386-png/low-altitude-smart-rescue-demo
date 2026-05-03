import json
from pathlib import Path

import cv2
import numpy as np


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


def analyze_thermal(image_file):
    """Analyze RGB/gray/thermal-like imagery and save thermal outputs."""
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

    temp = cv2.normalize(gray, None, 20.0, 80.0, cv2.NORM_MINMAX)
    normalized = cv2.normalize(temp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    threshold = np.percentile(temp, 95)
    hotspot = temp >= threshold
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
    max_temp = float(np.max(temp))
    mean_temp = float(np.mean(temp))
    if area_ratio > 0.08 or max_temp > 70:
        risk = "High"
        explanation = "疑似高温热点范围较大，建议优先复核火源、被困人员或危险热源。"
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
        "max_temperature": round(max_temp, 2),
        "mean_temperature": round(mean_temp, 2),
        "hotspot_count": hotspot_count,
        "hotspot_area_ratio": round(area_ratio, 4),
        "risk_level": risk,
        "risk_explanation": explanation,
        "is_simulated_temperature": is_simulated,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    status = "热红外/红外分析完成。当前为模拟热红外分析，温度矩阵由图像灰度归一化生成。"
    return str(heatmap_path), str(overlay_path), status, json.dumps(result, ensure_ascii=False, indent=2)

