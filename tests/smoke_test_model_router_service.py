import sys
from pathlib import Path

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from detection.router.model_router_service import ModelRouterService  # noqa: E402
from detection.router.route_labels import (  # noqa: E402
    AIR_BACKEND,
    CLOSE_RANGE_CLEAR_RGB,
    DISASTER_AERIAL_SCENE,
    DISTANT_SMALL_HUMAN_CANDIDATE,
    NORMAL_DISASTER_RGB,
    QAZI_BACKEND,
    TRANSFORMER_BACKEND,
    YOLO_BACKEND,
)
from detection.router.router_schemas import RouterDecision  # noqa: E402


def _availability(air=False, qazi=False, yolo=True, transformer=True):
    return {
        YOLO_BACKEND: {"available": yolo, "status": "available" if yolo else "yolo_unavailable", "reason": "test"},
        TRANSFORMER_BACKEND: {"available": transformer, "status": "available" if transformer else "adapter_unavailable", "reason": "test"},
        AIR_BACKEND: {"available": air, "status": "available" if air else "adapter_unavailable", "reason": "missing_weights"},
        QAZI_BACKEND: {"available": qazi, "status": "available" if qazi else "adapter_unavailable", "reason": "missing_weights"},
    }


def main():
    router = ModelRouterService()
    image = Image.new("RGB", (640, 480), (190, 190, 190))

    decision = router.classify(image=image, route_override=NORMAL_DISASTER_RGB)
    assert isinstance(decision, dict)
    assert decision["route"] == NORMAL_DISASTER_RGB
    assert decision["display_mode_name"]
    assert decision["recommended_combo"] == [YOLO_BACKEND, TRANSFORMER_BACKEND]
    assert RouterDecision(**{key: decision[key] for key in RouterDecision.__dataclass_fields__}).to_dict()["route"] == NORMAL_DISASTER_RGB

    close_decision = router.classify(image=image, route_override=CLOSE_RANGE_CLEAR_RGB)
    close_plan = router.build_execution_plan(close_decision, _availability())
    assert close_plan["selected_backend_combo"] == [YOLO_BACKEND, TRANSFORMER_BACKEND]

    distant_decision = router.classify(image=image, route_override=DISTANT_SMALL_HUMAN_CANDIDATE)
    distant_plan = router.build_execution_plan(distant_decision, _availability(air=True))
    assert distant_plan["selected_backend_combo"] == [AIR_BACKEND, TRANSFORMER_BACKEND, YOLO_BACKEND]

    disaster_decision = router.classify(image=image, route_override=DISASTER_AERIAL_SCENE)
    disaster_plan = router.build_execution_plan(disaster_decision, _availability(qazi=True))
    assert disaster_plan["selected_backend_combo"] == [QAZI_BACKEND, YOLO_BACKEND, TRANSFORMER_BACKEND]

    air_fallback = router.build_execution_plan(distant_decision, _availability(air=False))
    assert air_fallback["fallback_applied"] is True
    assert air_fallback["selected_backend_combo"] == [YOLO_BACKEND, TRANSFORMER_BACKEND]
    assert any(item["backend"] == AIR_BACKEND for item in air_fallback["unavailable_backends"])

    qazi_fallback = router.build_execution_plan(disaster_decision, _availability(qazi=False))
    assert qazi_fallback["fallback_applied"] is True
    assert qazi_fallback["selected_backend_combo"] == [YOLO_BACKEND, TRANSFORMER_BACKEND]
    assert any(item["backend"] == QAZI_BACKEND for item in qazi_fallback["unavailable_backends"])

    low_confidence = dict(close_decision)
    low_confidence["router_confidence"] = 0.42
    low_conf_plan = router.build_execution_plan(low_confidence, _availability())
    assert low_conf_plan["fallback_applied"] is True
    assert "router_low_confidence_fallback" in low_conf_plan["fallback_reasons"]
    assert low_conf_plan["selected_backend_combo"] == [YOLO_BACKEND, TRANSFORMER_BACKEND]

    yolo_missing = router.build_execution_plan(close_decision, _availability(yolo=False))
    assert "yolo_unavailable" in yolo_missing["fallback_reasons"]
    assert any(item["backend"] == YOLO_BACKEND for item in yolo_missing["unavailable_backends"])

    print("S4 ModelRouterService smoke test passed.")


if __name__ == "__main__":
    main()
