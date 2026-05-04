import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from radiometric_thermal_service import (  # noqa: E402
    analyze_radiometric_thermal,
    check_radiometric_environment,
    compute_temperature_statistics,
    create_hotspot_overlay,
    detect_radiometric_file_type,
    render_temperature_heatmap,
)


def _write_synthetic_jpg(path):
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 80
    image[:, :, 1] = np.linspace(20, 220, image.shape[1], dtype=np.uint8)
    image[12:28, 30:50, :] = (255, 230, 210)
    Image.fromarray(image).save(path)


def main():
    env = check_radiometric_environment()
    assert "exiftool_available" in env
    assert "dji_sdk_available" in env
    assert "dji_sdk_detected" in env
    assert "dji_parser_implemented" in env
    assert "dji_sdk_detected_but_parser_not_implemented" in env
    assert "supported_parsers" in env
    assert "warnings" in env
    assert "can_parse_flir_rjpeg" in env
    assert env["can_parse_dji_rjpeg"] is False
    assert "dji_dirp_sdk" not in env["supported_parsers"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        jpg_path = tmp_dir / "ordinary_rgb.jpg"
        _write_synthetic_jpg(jpg_path)

        detected = detect_radiometric_file_type(jpg_path)
        assert detected["file_type"] in {"ordinary_image", "unknown", "unsupported"}
        assert detected["is_radiometric_candidate"] is False
        assert detected["is_radiometric"] is False

        result = analyze_radiometric_thermal(jpg_path, output_dir=tmp_dir)
        assert result["success"] is False
        assert result["is_real_temperature_measurement"] is False
        assert result["thermal_mode"] == "radiometric"
        assert result["temperature_matrix_path"] is None
        assert result["temperature_matrix_shape"] is None
        assert result["statistics"] is None
        assert result["error_code"] == "NOT_RADIOMETRIC_DATA"
        assert "不会" in result["truthfulness_note"] or "not" in result["truthfulness_note"].lower()

        temp = np.array(
            [
                [20.0, 21.0, 22.0],
                [23.0, 40.0, 41.0],
                [24.0, 25.0, 42.0],
            ],
            dtype=np.float32,
        )
        stats = compute_temperature_statistics(temp, threshold_celsius=37.5)
        assert stats["max_temperature"] == 42.0
        assert stats["hotspot_threshold"] == 37.5
        assert stats["hotspot_pixel_count"] == 3

        heatmap_path, error = render_temperature_heatmap(temp, tmp_dir / "heatmap.jpg")
        assert not error
        assert heatmap_path and Path(heatmap_path).exists()

        overlay_path, hotspot_count, area_ratio, error = create_hotspot_overlay(
            jpg_path,
            temp,
            tmp_dir / "overlay.jpg",
            threshold_celsius=37.5,
        )
        assert not error
        assert overlay_path and Path(overlay_path).exists()
        assert hotspot_count == 3
        assert area_ratio > 0

        # Result JSON must remain serializable without numpy arrays.
        json.dumps(result, ensure_ascii=False)

    print("AeroRescue-AI radiometric thermal service smoke test passed.")


if __name__ == "__main__":
    main()
