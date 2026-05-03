"""Radiometric thermal image parsing helpers.

The FLIR path follows the same high-level idea as FlirImageExtractor: use
ExifTool to inspect/extract embedded raw thermal data, then convert valid raw
sensor values to Celsius only when the required radiometric metadata is present.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "thermal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


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


def check_radiometric_environment():
    """Check local tools required for radiometric parsing without installing them."""
    exiftool_available = shutil.which("exiftool") is not None
    numpy_available = True
    pil_available = True
    cv2_available = True
    can_parse_flir = bool(exiftool_available and numpy_available and pil_available and cv2_available)
    message = (
        "Radiometric 环境可用：已找到 exiftool，可尝试解析 FLIR radiometric JPG。"
        if can_parse_flir
        else "Radiometric 环境不完整：需要本机安装 exiftool，且 Python 环境具备 numpy/PIL/cv2。"
    )
    return {
        "exiftool_available": exiftool_available,
        "numpy_available": numpy_available,
        "pil_available": pil_available,
        "cv2_available": cv2_available,
        "flir_parser_available": can_parse_flir,
        "dji_sdk_available": False,
        "can_parse_flir_rjpeg": can_parse_flir,
        "can_parse_dji_rjpeg": False,
        "message": message,
    }


def run_exiftool(file_path, args=None):
    """Run exiftool safely and return text stdout/stderr."""
    path = _as_path(file_path)
    if not path:
        return 1, "", "No input file provided."
    if shutil.which("exiftool") is None:
        return 127, "", "exiftool is not available. Install exiftool before radiometric parsing."
    command = ["exiftool"] + list(args or []) + [str(path)]
    try:
        result = subprocess.run(command, check=False, capture_output=True, timeout=30)
        stdout = result.stdout.decode("utf-8", errors="ignore")
        stderr = result.stderr.decode("utf-8", errors="ignore")
        return result.returncode, stdout, stderr
    except Exception as exc:
        return 1, "", str(exc)


def _run_exiftool_binary(file_path, args=None):
    path = _as_path(file_path)
    if not path:
        return 1, b"", b"No input file provided."
    if shutil.which("exiftool") is None:
        return 127, b"", b"exiftool is not available."
    command = ["exiftool"] + list(args or []) + [str(path)]
    try:
        result = subprocess.run(command, check=False, capture_output=True, timeout=30)
        return result.returncode, result.stdout, result.stderr
    except Exception as exc:
        return 1, b"", str(exc).encode("utf-8", errors="ignore")


def _read_exif_json(file_path):
    code, stdout, stderr = run_exiftool(file_path, ["-json", "-n"])
    if code != 0 or not stdout.strip():
        return {}, stderr or "No exiftool metadata output."
    try:
        parsed = json.loads(stdout)
        if parsed and isinstance(parsed, list):
            return parsed[0], ""
        return {}, "ExifTool JSON output is empty."
    except Exception as exc:
        return {}, f"Failed to parse exiftool JSON: {exc}"


def _metadata_has_key(metadata, *needles):
    needles = [needle.lower() for needle in needles]
    for key, value in (metadata or {}).items():
        text = f"{key} {value}".lower()
        if all(needle in text for needle in needles):
            return True
    return False


def detect_radiometric_file_type(file_path):
    """Detect whether a file is FLIR radiometric JPG, DJI R-JPEG, plain JPG, or unsupported."""
    path = Path(_as_path(file_path) or "")
    if not path.exists():
        return {
            "file_type": "unsupported",
            "is_radiometric": False,
            "evidence": {"path": str(path), "exists": False},
            "message": "输入文件不存在。",
        }
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        return {
            "file_type": "unsupported",
            "is_radiometric": False,
            "evidence": {"suffix": suffix},
            "message": "当前仅支持 jpg/jpeg/png/tif/tiff 图像文件检测。",
        }

    metadata, metadata_error = _read_exif_json(path)
    evidence = {
        "suffix": suffix,
        "metadata_available": bool(metadata),
        "metadata_error": metadata_error,
        "make": metadata.get("Make") or metadata.get("CameraMake") or "",
        "model": metadata.get("Model") or metadata.get("CameraModel") or "",
        "has_raw_thermal_image_tag": _metadata_has_key(metadata, "raw", "thermal"),
        "has_thermal_data_tag": _metadata_has_key(metadata, "thermal", "data"),
    }
    combined = " ".join(str(value) for value in metadata.values()).lower() + " " + " ".join(metadata.keys()).lower()
    make_model = f"{evidence['make']} {evidence['model']}".lower()

    if "flir" in combined or "flir" in make_model or evidence["has_raw_thermal_image_tag"]:
        return {
            "file_type": "flir_radiometric_jpg",
            "is_radiometric": True,
            "evidence": evidence,
            "message": "检测到 FLIR / Raw Thermal 相关元数据，可尝试 radiometric 解析。",
        }

    if "dji" in combined and ("r-jpeg" in combined or "thermal" in combined or "dirp" in combined):
        return {
            "file_type": "dji_rjpeg",
            "is_radiometric": True,
            "evidence": evidence,
            "message": "检测到 DJI thermal/R-JPEG 相关元数据，但当前需要 DJI Thermal SDK / DIRP 才能解析。",
        }

    if suffix in {".jpg", ".jpeg"}:
        return {
            "file_type": "unknown_jpg",
            "is_radiometric": False,
            "evidence": evidence,
            "message": "这是普通 JPG 或未检测到 radiometric thermal data，不能进行真实测温。",
        }

    return {
        "file_type": "unsupported",
        "is_radiometric": False,
        "evidence": evidence,
        "message": "未检测到可解析的 radiometric thermal data。",
    }


def extract_flir_metadata(file_path):
    """Extract useful FLIR radiometric metadata via exiftool."""
    metadata, error = _read_exif_json(file_path)
    if not metadata:
        return {
            "success": False,
            "error": error or "无法读取 metadata。",
            "metadata": {},
        }

    fields = {
        "camera_make": metadata.get("Make") or metadata.get("CameraMake"),
        "camera_model": metadata.get("Model") or metadata.get("CameraModel"),
        "raw_thermal_image_exists": _metadata_has_key(metadata, "raw", "thermal"),
        "image_width": metadata.get("ImageWidth") or metadata.get("ExifImageWidth") or metadata.get("RawThermalImageWidth"),
        "image_height": metadata.get("ImageHeight") or metadata.get("ExifImageHeight") or metadata.get("RawThermalImageHeight"),
        "emissivity": metadata.get("Emissivity"),
        "reflected_apparent_temperature": metadata.get("ReflectedApparentTemperature"),
        "atmospheric_temperature": metadata.get("AtmosphericTemperature"),
        "relative_humidity": metadata.get("RelativeHumidity"),
        "object_distance": metadata.get("ObjectDistance"),
        "planck_r1": metadata.get("PlanckR1"),
        "planck_r2": metadata.get("PlanckR2"),
        "planck_b": metadata.get("PlanckB"),
        "planck_f": metadata.get("PlanckF"),
        "planck_o": metadata.get("PlanckO"),
        "raw_thermal_image_type": metadata.get("RawThermalImageType"),
    }
    return {"success": True, "error": "", "metadata": fields, "raw_metadata": metadata}


def _extract_raw_thermal_image(file_path, output_path):
    code, stdout, stderr = _run_exiftool_binary(file_path, ["-b", "-RawThermalImage"])
    if code != 0 or not stdout:
        return False, stderr.decode("utf-8", errors="ignore") or "RawThermalImage not found."
    Path(output_path).write_bytes(stdout)
    return True, ""


def _read_raw_thermal_array(raw_path):
    image = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        try:
            image = np.asarray(Image.open(raw_path))
        except Exception:
            return None
    if image.ndim == 3:
        image = image[:, :, 0]
    return image.astype(np.float64)


def _as_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _raw_to_celsius(raw, metadata):
    r1 = _as_float(metadata.get("planck_r1"))
    r2 = _as_float(metadata.get("planck_r2"))
    b = _as_float(metadata.get("planck_b"))
    f = _as_float(metadata.get("planck_f"))
    o = _as_float(metadata.get("planck_o"))
    if None in {r1, r2, b, f, o}:
        return None, "缺少 PlanckR1/PlanckR2/PlanckB/PlanckF/PlanckO，无法把 raw thermal data 换算为摄氏温度。"
    denominator = r2 * (raw + o)
    with np.errstate(divide="ignore", invalid="ignore"):
        kelvin = b / np.log((r1 / denominator) + f)
        celsius = kelvin - 273.15
    if not np.all(np.isfinite(celsius)):
        celsius = np.where(np.isfinite(celsius), celsius, np.nan)
    if np.all(np.isnan(celsius)):
        return None, "Planck 换算结果无有效温度值。"
    return celsius.astype(np.float32), ""


def parse_flir_radiometric_jpg(file_path):
    """Parse a FLIR radiometric JPG into a Celsius temperature matrix if possible."""
    metadata_result = extract_flir_metadata(file_path)
    metadata = metadata_result.get("metadata", {})
    if not metadata_result.get("success"):
        return {
            "success": False,
            "temperature_matrix": None,
            "unit": "celsius",
            "parser": "flir_exiftool",
            "metadata": metadata,
            "error": metadata_result.get("error", "无法读取 FLIR metadata。"),
        }

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_path = Path(tmp_dir) / "raw_thermal_image.tiff"
        extracted, error = _extract_raw_thermal_image(file_path, raw_path)
        if not extracted:
            return {
                "success": False,
                "temperature_matrix": None,
                "unit": "celsius",
                "parser": "flir_exiftool",
                "metadata": metadata,
                "error": f"未检测到 Raw Thermal Image，不能用普通图像灰度伪造真实温度。{error}",
            }
        raw = _read_raw_thermal_array(raw_path)
        if raw is None:
            return {
                "success": False,
                "temperature_matrix": None,
                "unit": "celsius",
                "parser": "flir_exiftool",
                "metadata": metadata,
                "error": "已提取 Raw Thermal Image，但无法读取为数值矩阵。",
            }
        temperature_matrix, error = _raw_to_celsius(raw, metadata)
        if temperature_matrix is None:
            return {
                "success": False,
                "temperature_matrix": None,
                "unit": "celsius",
                "parser": "flir_exiftool",
                "metadata": metadata,
                "error": error,
            }
    return {
        "success": True,
        "temperature_matrix": temperature_matrix,
        "unit": "celsius",
        "parser": "flir_exiftool",
        "metadata": metadata,
        "error": "",
    }


def parse_dji_rjpeg(file_path):
    """Placeholder for DJI R-JPEG parsing."""
    return {
        "success": False,
        "temperature_matrix": None,
        "unit": "celsius",
        "parser": "dji_dirp_sdk",
        "error": "DJI R-JPEG parsing requires DJI Thermal SDK / DIRP library. This project currently provides only a placeholder interface.",
    }


def compute_temperature_statistics(temperature_matrix, threshold_celsius=None):
    """Compute statistics from a real Celsius temperature matrix."""
    temp = np.asarray(temperature_matrix, dtype=np.float32)
    valid = temp[np.isfinite(temp)]
    if valid.size == 0:
        return {
            "error": "temperature_matrix 中没有有效温度值。",
            "min_temperature": None,
            "max_temperature": None,
            "mean_temperature": None,
            "median_temperature": None,
            "std_temperature": None,
            "hotspot_threshold": None,
            "hotspot_threshold_source": "none",
            "hotspot_pixel_count": 0,
            "hotspot_area_ratio": 0.0,
        }
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    if threshold_celsius is not None:
        threshold = float(threshold_celsius)
        threshold_source = "user_defined"
    elif std > 0:
        threshold = mean + 2.0 * std
        threshold_source = "mean_plus_2std"
    else:
        threshold = 37.5
        threshold_source = "default_37_5"
    hotspot = np.isfinite(temp) & (temp >= threshold)
    return {
        "min_temperature": round(float(np.min(valid)), 2),
        "max_temperature": round(float(np.max(valid)), 2),
        "mean_temperature": round(mean, 2),
        "median_temperature": round(float(np.median(valid)), 2),
        "std_temperature": round(std, 2),
        "hotspot_threshold": round(float(threshold), 2),
        "hotspot_threshold_source": threshold_source,
        "hotspot_pixel_count": int(np.count_nonzero(hotspot)),
        "hotspot_area_ratio": round(float(np.count_nonzero(hotspot)) / max(1, temp.size), 4),
        "note": "37.5°C 可作为人体/异常热点参考阈值之一，但不能直接作为医学判断。",
    }


def _normalize_temperature_for_display(temperature_matrix):
    temp = np.asarray(temperature_matrix, dtype=np.float32)
    valid = temp[np.isfinite(temp)]
    if valid.size == 0:
        return None
    min_v, max_v = float(np.min(valid)), float(np.max(valid))
    if max_v <= min_v:
        return np.zeros(temp.shape, dtype=np.uint8)
    normalized = (np.nan_to_num(temp, nan=min_v) - min_v) / (max_v - min_v)
    return (normalized * 255).clip(0, 255).astype(np.uint8)


def render_temperature_heatmap(temperature_matrix, output_path):
    """Render a real temperature matrix as a heatmap image."""
    normalized = _normalize_temperature_for_display(temperature_matrix)
    if normalized is None:
        return None, "temperature_matrix 无有效温度值，无法生成 heatmap。"
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    stats = compute_temperature_statistics(temperature_matrix)
    label = f"min {stats.get('min_temperature')}C  max {stats.get('max_temperature')}C"
    cv2.putText(heatmap, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), heatmap)
    return str(output_path), ""


def create_hotspot_overlay(base_image_path_or_array, temperature_matrix, output_path, threshold_celsius=None):
    """Create an overlay highlighting pixels above the temperature threshold."""
    temp = np.asarray(temperature_matrix, dtype=np.float32)
    stats = compute_temperature_statistics(temp, threshold_celsius=threshold_celsius)
    threshold = stats.get("hotspot_threshold")
    if threshold is None:
        return None, 0, 0.0, "temperature_matrix 无有效温度值。"

    normalized = _normalize_temperature_for_display(temp)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    if base_image_path_or_array is None:
        base = heatmap
    elif isinstance(base_image_path_or_array, (str, Path)):
        base = cv2.imread(str(base_image_path_or_array), cv2.IMREAD_COLOR)
        if base is None:
            base = heatmap
    else:
        base = np.asarray(base_image_path_or_array)
        if base.ndim == 2:
            base = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif base.shape[-1] == 3:
            base = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            base = base[:, :, :3].astype(np.uint8)
    if base.shape[:2] != temp.shape[:2]:
        base = cv2.resize(base, (temp.shape[1], temp.shape[0]), interpolation=cv2.INTER_AREA)

    hotspot = np.isfinite(temp) & (temp >= threshold)
    overlay = cv2.addWeighted(base.astype(np.uint8), 0.5, heatmap, 0.5, 0)
    overlay[hotspot] = (0, 255, 255)
    hotspot_u8 = (hotspot.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(hotspot_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 8:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    return str(output_path), stats["hotspot_pixel_count"], stats["hotspot_area_ratio"], ""


def analyze_radiometric_thermal(file_path, threshold_celsius=None, output_dir=None):
    """Analyze a real radiometric thermal image without falling back to fake temperatures."""
    path = Path(_as_path(file_path) or "")
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return {
            "success": False,
            "thermal_mode": "radiometric_thermal",
            "is_real_temperature_measurement": False,
            "file_type": "unsupported",
            "parser": "",
            "temperature_matrix_path": "",
            "heatmap_path": "",
            "overlay_path": "",
            "statistics": {},
            "metadata": {},
            "truthfulness_note": "输入文件不存在，系统不会生成伪温度。",
            "error": "输入文件不存在。",
        }

    detection = detect_radiometric_file_type(path)
    file_type = detection["file_type"]
    if file_type == "flir_radiometric_jpg":
        parsed = parse_flir_radiometric_jpg(path)
    elif file_type == "dji_rjpeg":
        parsed = parse_dji_rjpeg(path)
    else:
        return {
            "success": False,
            "thermal_mode": "radiometric_thermal",
            "is_real_temperature_measurement": False,
            "file_type": file_type,
            "parser": "",
            "temperature_matrix_path": "",
            "heatmap_path": "",
            "overlay_path": "",
            "statistics": {},
            "metadata": detection.get("evidence", {}),
            "truthfulness_note": "未检测到真实 radiometric thermal data，系统不会用灰度值伪造温度。",
            "error": detection.get("message", "不是可解析的 radiometric thermal 文件。"),
        }

    if not parsed.get("success"):
        return {
            "success": False,
            "thermal_mode": "radiometric_thermal",
            "is_real_temperature_measurement": False,
            "file_type": file_type,
            "parser": parsed.get("parser", ""),
            "temperature_matrix_path": "",
            "heatmap_path": "",
            "overlay_path": "",
            "statistics": {},
            "metadata": parsed.get("metadata", {}),
            "truthfulness_note": "未成功解析 temperature matrix，系统不会生成伪温度。",
            "error": parsed.get("error", "Radiometric parsing failed."),
        }

    temp = parsed["temperature_matrix"]
    matrix_path = output_dir / "temperature_matrix.npy"
    heatmap_path = output_dir / "true_temperature_heatmap.jpg"
    overlay_path = output_dir / "hotspot_overlay.jpg"
    result_path = output_dir / "radiometric_thermal_result.json"
    np.save(matrix_path, temp)
    heatmap_result, heatmap_error = render_temperature_heatmap(temp, heatmap_path)
    overlay_result, _, _, overlay_error = create_hotspot_overlay(path, temp, overlay_path, threshold_celsius=threshold_celsius)
    stats = compute_temperature_statistics(temp, threshold_celsius=threshold_celsius)
    result = {
        "success": True,
        "thermal_mode": "radiometric_thermal",
        "is_real_temperature_measurement": True,
        "file_type": file_type,
        "parser": parsed.get("parser", ""),
        "temperature_matrix_path": str(matrix_path),
        "heatmap_path": heatmap_result or "",
        "overlay_path": overlay_result or "",
        "statistics": stats,
        "metadata": parsed.get("metadata", {}),
        "truthfulness_note": "该结果来自 radiometric thermal 文件解析出的 temperature matrix。",
        "error": heatmap_error or overlay_error,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
