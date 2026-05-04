import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from detection_runtime_service import (  # noqa: E402
    compare_detection_targets,
    format_detection_runtime_status,
    normalize_image_input,
    run_detection,
    run_dual_backend_compare_runtime,
    run_transformer_detection_runtime,
    run_yolo_detection_runtime,
)


def main():
    np_image = np.zeros((32, 48, 3), dtype=np.uint8)
    normalized = normalize_image_input(np_image)
    assert normalized["width"] == 48
    assert normalized["height"] == 32
    assert normalized["np_image"].shape == (32, 48, 3)
    assert isinstance(normalized["pil_image"], Image.Image)

    pil_image = Image.fromarray(np_image)
    normalized_pil = normalize_image_input(pil_image)
    assert normalized_pil["width"] == 48
    assert normalized_pil["height"] == 32

    unsupported = run_detection(np_image, detection_mode="qazi_disaster_management_reference")
    assert unsupported["success"] is False
    assert unsupported["error_code"] in {"UNSUPPORTED_DETECTION_MODE", "REFERENCE_BACKEND_NOT_EXECUTABLE"}
    assert unsupported["targets"] == []

    missing_yolo = run_yolo_detection_runtime(np_image, model_variant="missing_smoke_test_model")
    assert missing_yolo["success"] is False
    assert missing_yolo["error_code"] == "YOLO_WEIGHTS_MISSING"
    assert missing_yolo["targets"] == []

    transformer = run_transformer_detection_runtime(np_image)
    assert isinstance(transformer, dict)
    if not transformer["success"]:
        assert transformer["error_code"]
        assert transformer["message"]
        assert transformer["is_model_output"] is False
    assert transformer["can_enter_path_planning"] is False

    dual = run_dual_backend_compare_runtime(np_image, yolo_model_variant="missing_smoke_test_model")
    assert isinstance(dual, dict)
    assert "primary_result" in dual
    assert "auxiliary_result" in dual
    assert "consensus" in dual
    assert dual["can_enter_terp"] is False

    yolo_targets = [
        {
            "id": "T001",
            "class_name": "civilian",
            "bbox": [10, 10, 100, 100],
            "confidence": 0.8,
        }
    ]
    transformer_targets = [
        {
            "id": "TR001",
            "class_name": "human_candidate",
            "bbox": [20, 20, 110, 110],
            "confidence": 0.85,
        }
    ]
    consensus = compare_detection_targets(yolo_targets, transformer_targets)
    assert len(consensus["matched_pairs"]) >= 1
    assert consensus["truthfulness_note"]

    failed_status = format_detection_runtime_status(missing_yolo)
    assert "检测运行状态" in failed_status
    assert "真实性说明" in failed_status

    fake_success = {
        "success": True,
        "detection_mode": "yolo_rescue_targets",
        "backend_key": "yolo_rescue_targets",
        "model_variant": "yolov11m",
        "target_count": 1,
        "can_enter_terp": True,
        "can_enter_path_planning": True,
        "human_review_required": True,
        "truthfulness_note": "Smoke test fake success for formatting only.",
        "message": "OK",
    }
    success_status = format_detection_runtime_status(fake_success)
    assert "yolov11m" in success_status
    assert "Smoke test fake success" in success_status

    with tempfile.TemporaryDirectory() as tmpdir:
        saved = run_yolo_detection_runtime(np_image, model_variant="missing_smoke_test_model", output_dir=tmpdir)
        assert saved["success"] is False
        assert not list(Path(tmpdir).glob("*.pt"))

    print("AeroRescue-AI detection runtime service smoke test passed.")


if __name__ == "__main__":
    main()
