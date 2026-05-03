import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "orthomosaic"
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


def _has_gps(path):
    try:
        image = Image.open(path)
        exif = image.getexif()
        gps_tag = next((key for key, value in ExifTags.TAGS.items() if value == "GPSInfo"), None)
        return bool(gps_tag and exif.get(gps_tag))
    except Exception:
        return False


def _read_image(path, max_side=1400):
    image = cv2.imread(str(path))
    if image is None:
        return None
    h, w = image.shape[:2]
    side = max(h, w)
    if side > max_side:
        scale = max_side / side
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def _grid_preview(images):
    if not images:
        return None
    thumb_h = 360
    thumbs = []
    for image in images:
        scale = thumb_h / image.shape[0]
        thumb = cv2.resize(image, (max(1, int(image.shape[1] * scale)), thumb_h))
        thumbs.append(thumb)
    cols = min(3, len(thumbs))
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i:i + cols]
        max_w = max(img.shape[1] for img in row)
        padded = []
        for img in row:
            if img.shape[1] < max_w:
                pad = np.zeros((img.shape[0], max_w - img.shape[1], 3), dtype=np.uint8)
                img = cv2.hconcat([img, pad])
            padded.append(img)
        while len(padded) < cols:
            padded.append(np.zeros_like(padded[0]))
        rows.append(cv2.hconcat(padded))
    return cv2.vconcat(rows)


def _feature_homography_stitch(images):
    if len(images) < 2:
        return None, "图像数量不足，无法进行特征匹配。"

    base = images[0]
    result = base.copy()
    orb = cv2.ORB_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for next_image in images[1:]:
        gray_a = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)
        kp_a, des_a = orb.detectAndCompute(gray_a, None)
        kp_b, des_b = orb.detectAndCompute(gray_b, None)
        if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
            return None, "ORB 特征点不足。"
        matches = sorted(matcher.match(des_a, des_b), key=lambda item: item.distance)[:300]
        if len(matches) < 8:
            return None, "有效匹配点不足。"
        src = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if homography is None:
            return None, "Homography 估计失败。"
        h1, w1 = result.shape[:2]
        h2, w2 = next_image.shape[:2]
        canvas_w = w1 + w2
        canvas_h = max(h1, h2)
        warped = cv2.warpPerspective(next_image, homography, (canvas_w, canvas_h))
        warped[0:h1, 0:w1] = result
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero((gray > 0).astype(np.uint8))
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            warped = warped[y:y + h, x:x + w]
        result = warped
    return result, ""


def process_orthomosaic(image_files):
    """Create an orthomosaic-style output and processing log."""
    paths = [_as_path(item) for item in (image_files or [])]
    paths = [Path(path) for path in paths if path]
    images = [_read_image(path) for path in paths]
    images = [image for image in images if image is not None]

    log = {
        "image_count": len(images),
        "has_gps": any(_has_gps(path) for path in paths),
        "stitcher_attempted": False,
        "feature_matching_attempted": False,
        "stitch_success": False,
        "fallback_reason": "",
        "output_path": "",
    }

    if not images:
        log["fallback_reason"] = "未读取到有效图像。"
        log_path = OUTPUT_DIR / "processing_log.json"
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        return None, "未上传或未读取到有效航测图像。", json.dumps(log, ensure_ascii=False, indent=2)

    if len(images) == 1:
        result = images[0]
        status = "已生成单幅航测图像预览；单张图像不能说明正射拼接已完成。"
        log["fallback_reason"] = "单张图像仅生成预览。"
    else:
        result = None
        status = ""
        if hasattr(cv2, "Stitcher_create"):
            log["stitcher_attempted"] = True
            try:
                stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
                code, stitched = stitcher.stitch(images)
                if code == cv2.Stitcher_OK and stitched is not None:
                    result = stitched
                    log["stitch_success"] = True
                    status = "OpenCV Stitcher 正射/全景拼接预览生成成功。"
                else:
                    log["fallback_reason"] = f"OpenCV Stitcher 状态码：{code}"
            except Exception as exc:
                log["fallback_reason"] = f"OpenCV Stitcher 异常：{exc}"

        if result is None:
            log["feature_matching_attempted"] = True
            result, reason = _feature_homography_stitch(images)
            if result is not None:
                log["stitch_success"] = True
                status = "ORB 特征匹配 + Homography 拼接预览生成成功。"
            else:
                log["fallback_reason"] = reason or log["fallback_reason"] or "特征匹配失败。"

        if result is None:
            result = _grid_preview(images)
            status = "正射拼接失败，已生成网格接片预览。"

    output_path = OUTPUT_DIR / "orthomosaic_result.jpg"
    cv2.imwrite(str(output_path), result)
    log["output_path"] = str(output_path)
    log_path = OUTPUT_DIR / "processing_log.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path), status, json.dumps(log, ensure_ascii=False, indent=2)

