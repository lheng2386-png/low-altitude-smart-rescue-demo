import base64
import json
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "static" / "images" / "showcase" / "argus_fusion"
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


def _pil_to_rgb_array(image):
    if isinstance(image, Image.Image):
        return np.array(ImageOps.exif_transpose(image).convert("RGB"))
    return np.array(Image.open(image).convert("RGB"))


def generate_orthomosaic(image_files):
    """Generate a lightweight orthomosaic preview from multiple UAV images."""
    paths = [_as_path(item) for item in (image_files or [])]
    paths = [path for path in paths if path]
    if not paths:
        return None, "未上传航测图像，无法生成正射影像预览。"

    images = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        max_side = max(image.shape[:2])
        if max_side > 1200:
            scale = 1200 / max_side
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        images.append(image)

    if not images:
        return None, "航测图像读取失败，请检查文件格式。"

    if len(images) >= 2 and hasattr(cv2, "Stitcher_create"):
        try:
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            status, stitched = stitcher.stitch(images)
            if status == cv2.Stitcher_OK and stitched is not None:
                out_path = OUTPUT_DIR / "orthomosaic_preview.png"
                cv2.imwrite(str(out_path), stitched)
                return cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB), (
                    f"正射影像预览生成成功：已基于 {len(images)} 张图像完成特征拼接。"
                    "这是轻量预览模式；完整高精度正射需接入 WebODM / OpenDroneMap。"
                )
        except Exception as exc:
            stitch_error = str(exc)
        else:
            stitch_error = f"OpenCV 拼接状态码：{status}"
    else:
        stitch_error = "图像数量不足或 OpenCV Stitcher 不可用。"

    thumbs = []
    target_h = min(360, min(img.shape[0] for img in images))
    for image in images:
        scale = target_h / image.shape[0]
        thumb = cv2.resize(image, (max(1, int(image.shape[1] * scale)), target_h))
        thumbs.append(thumb)
    mosaic = cv2.hconcat(thumbs) if len(thumbs) > 1 else thumbs[0]
    out_path = OUTPUT_DIR / "orthomosaic_contact_sheet.png"
    cv2.imwrite(str(out_path), mosaic)
    return cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB), (
        f"已生成航测接片预览，共 {len(images)} 张图像。"
        f"特征级正射拼接未完成：{stitch_error} "
        "当前预览用于任务评估；完整正射影像生成可通过 ARGUS/WebODM 风格接口接入。"
    )


def analyze_thermal_image(image_file, percentile=97.5):
    """Analyze a thermal/IR-like image and produce a hotspot overlay."""
    path = _as_path(image_file)
    if not path:
        return None, "未上传热红外/红外图像。"
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, "热红外/红外图像读取失败。"

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        base = image
    else:
        gray = image.astype(np.float32)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        base = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    normalized = cv2.normalize(gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    threshold = float(np.percentile(normalized, percentile))
    hotspot_mask = normalized >= threshold
    overlay = cv2.addWeighted(base.astype(np.uint8), 0.45, heatmap, 0.55, 0)
    overlay[hotspot_mask] = (0, 255, 255)

    contours, _ = cv2.findContours(hotspot_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 8:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h, area))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)

    out_path = OUTPUT_DIR / "thermal_hotspot_overlay.png"
    cv2.imwrite(str(out_path), overlay)
    summary = [
        "热红外/红外分析完成。",
        f"热点阈值：亮度百分位 {percentile}%，归一化阈值 {threshold:.1f}。",
        f"疑似热点区域数量：{len(boxes)}。",
        "当前模式基于像素强度识别热点；如需真实温度矩阵，需要接入相机热红外元数据解析。",
    ]
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), "\n".join(summary)


def reconstruct_360_video(video_file, frame_step=30):
    """Create a lightweight 360/3D reconstruction preview and report Stella/WebODM status."""
    path = _as_path(video_file)
    if not path:
        return None, "未上传 360° 视频或重建视频。"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, f"无法打开视频文件：{path}"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    keyframes = []
    idx = 0
    while len(keyframes) < 8:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(int(frame_step), 1) == 0:
            thumb = cv2.resize(frame, (240, 135))
            keyframes.append(thumb)
        idx += 1
    cap.release()

    if keyframes:
        rows = []
        for i in range(0, len(keyframes), 4):
            row = keyframes[i:i + 4]
            if len(row) < 4:
                row += [np.zeros_like(row[0]) for _ in range(4 - len(row))]
            rows.append(cv2.hconcat(row))
        contact = cv2.vconcat(rows)
    else:
        contact = np.zeros((270, 960, 3), dtype=np.uint8)

    out_path = OUTPUT_DIR / "reconstruction_keyframes.png"
    cv2.imwrite(str(out_path), contact)

    stella_available = shutil.which("stella-vslam") is not None or shutil.which("run_stella_slam") is not None
    ffmpeg_available = shutil.which("ffmpeg") is not None
    summary = [
        "360°视频重建 / 三维重建预处理完成。",
        f"视频尺寸：{width}x{height}，帧数：{frame_count}，FPS：{fps:.2f}。",
        f"已抽取关键帧预览：{len(keyframes)} 张。",
        f"ffmpeg 可用状态：{'可用' if ffmpeg_available else '不可用'}。",
        f"StellaVSLAM 可用状态：{'可用' if stella_available else '未检测到本地命令'}。",
        "当前已复刻 ARGUS 的 StellaVSLAM 工作流入口和预处理逻辑；若安装 StellaVSLAM/vocab/config，可扩展为真实 sparse/dense point cloud 输出。",
    ]
    return cv2.cvtColor(contact, cv2.COLOR_BGR2RGB), "\n".join(summary)


def describe_scene_with_local_llm(image_file, prompt_context=None, ollama_url="http://127.0.0.1:11434", model="llava:latest"):
    """Describe a UAV scene with a local Ollama vision model if available."""
    path = _as_path(image_file)
    if not path:
        return "未上传用于场景描述的图像。"

    context = prompt_context or "你是低空无人机应急救援图像分析助手，请描述画面中的灾情、目标、风险和救援关注点。"
    image = Image.open(path).convert("RGB")
    image.thumbnail((672, 672))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    payload = {
        "model": model,
        "prompt": context,
        "images": [image_b64],
        "stream": False,
    }
    try:
        response = requests.post(f"{ollama_url.rstrip('/')}/api/generate", json=payload, timeout=60)
        if response.status_code != 200:
            return f"本地大模型服务返回异常：HTTP {response.status_code}。请确认 Ollama 已启动并安装视觉模型。"
        data = response.json()
        text = data.get("response", "").strip()
        return text or "本地大模型未返回有效描述。"
    except Exception as exc:
        return (
            "未连接到本地大模型服务，已安全回退。\n"
            f"连接地址：{ollama_url}\n"
            f"模型：{model}\n"
            f"原因：{exc}\n"
            "安装并启动 Ollama 视觉模型后，可在此入口生成自动场景描述。"
        )

