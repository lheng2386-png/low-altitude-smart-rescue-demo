import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from transformer_detection_service import (  # noqa: E402
    check_transformer_detection_environment,
    compare_yolo_and_transformer_detections,
    map_transformer_label_to_rescue_semantics,
    normalize_transformer_detections,
    normalize_transformer_label,
    run_transformer_detection,
)


def main():
    env = check_transformer_detection_environment()
    assert isinstance(env, dict)
    assert "transformers_available" in env
    assert "torch_available" in env
    assert "warnings" in env

    assert normalize_transformer_label("human") == "human_candidate"
    assert normalize_transformer_label("person") == "human_candidate"
    assert normalize_transformer_label("vehicle") == "vehicle"
    assert normalize_transformer_label("fire") == "fire"
    assert normalize_transformer_label("random_label") == "random_label"

    human_semantics = map_transformer_label_to_rescue_semantics("human")
    assert human_semantics["rescue_role"] == "human_candidate"
    assert human_semantics["human_review_required"] is True
    vehicle_semantics = map_transformer_label_to_rescue_semantics("vehicle")
    assert vehicle_semantics["rescue_role"] == "environment_risk"
    assert vehicle_semantics["can_enter_rescue_priority"] is False

    raw_results = [
        {"score": 0.9, "label": "human", "box": {"xmin": 10, "ymin": 20, "xmax": 110, "ymax": 220}},
        {"score": 0.8, "label": "vehicle", "box": {"xmin": 50, "ymin": 60, "xmax": 150, "ymax": 160}},
        {"score": 0.1, "label": "fire", "box": {"xmin": 0, "ymin": 0, "xmax": 30, "ymax": 30}},
    ]
    targets = normalize_transformer_detections(
        raw_results,
        image_width=200,
        image_height=240,
        model_key="rescuedet_deformable_detr",
        confidence_threshold=0.4,
    )
    assert len(targets) == 2
    assert targets[0]["class_name"] == "human_candidate"
    assert targets[0]["human_review_required"] is True
    assert targets[0]["bbox"] == [10.0, 20.0, 110.0, 220.0]
    assert targets[0]["center"] == [60.0, 120.0]
    assert targets[0]["area"] == 20000.0
    assert targets[1]["rescue_role"] == "environment_risk"

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
    comparison = compare_yolo_and_transformer_detections(yolo_targets, transformer_targets)
    assert comparison["success"] is True
    assert len(comparison["matched_pairs"]) >= 1
    assert "人工复核" in comparison["consensus_summary"] or "human candidate" in comparison["consensus_summary"].lower()
    assert "auxiliary" in comparison["truthfulness_note"].lower()

    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    result = run_transformer_detection(image, confidence_threshold=0.95)
    assert isinstance(result, dict)
    assert result.get("allow_download") is False
    if not result["success"]:
        assert result["error_code"] in {"DEPENDENCY_MISSING", "MODEL_UNAVAILABLE", "INFERENCE_FAILED", "INVALID_INPUT"}
        assert result["message"]
        assert result["is_model_output"] is False
        if result["error_code"] == "MODEL_UNAVAILABLE":
            assert "allow_download=True" in result["truthfulness_note"] or "Downloads are disabled" in result["message"]
    else:
        assert result["is_model_output"] is True

    print("灾情感知及影响评估 Transformer detection service smoke test passed.")


if __name__ == "__main__":
    main()
