import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from detection_backend_registry import (  # noqa: E402
    check_detection_backend_availability,
    get_active_detection_backends,
    list_detection_backends,
    summarize_detection_backend_capabilities,
)


def main():
    backends = list_detection_backends()
    assert isinstance(backends, list)
    keys = {item["backend_key"] for item in backends}
    assert "yolo_rescue_targets" in keys
    assert "transformer_rescuedet" in keys
    assert "post_disaster_survivor_yolo" in keys
    assert all(item.get("truthfulness_note") for item in backends)

    active = get_active_detection_backends()
    active_keys = {item["backend_key"] for item in active}
    assert "yolo_rescue_targets" in active_keys
    assert "transformer_rescuedet" in active_keys
    assert "post_disaster_survivor_yolo" not in active_keys
    assert "air_retinanet_sar_reference" not in active_keys

    yolo_availability = check_detection_backend_availability("yolo_rescue_targets", root_dir=ROOT_DIR)
    assert isinstance(yolo_availability, dict)
    assert "available" in yolo_availability
    assert "missing_requirements" in yolo_availability
    assert "available_variants" in yolo_availability
    assert yolo_availability["truthfulness_note"]

    transformer_availability = check_detection_backend_availability("transformer_rescuedet", root_dir=ROOT_DIR)
    assert isinstance(transformer_availability, dict)
    assert "available" in transformer_availability
    assert "missing_requirements" in transformer_availability
    assert transformer_availability["truthfulness_note"]

    planned = check_detection_backend_availability("post_disaster_survivor_yolo", root_dir=ROOT_DIR)
    assert planned["available"] is False
    reference = check_detection_backend_availability("air_retinanet_sar_reference", root_dir=ROOT_DIR)
    assert reference["available"] is False

    summary = summarize_detection_backend_capabilities()
    assert isinstance(summary, str)
    assert "YOLO Rescue Targets" in summary
    assert "Transformer RescueDet" in summary
    assert "human_candidate" in summary
    assert "confirmed civilian" in summary or "人工复核" in summary
    assert "参考" in summary or "未来训练" in summary

    print("AeroRescue-AI detection backend registry smoke test passed.")


if __name__ == "__main__":
    main()
