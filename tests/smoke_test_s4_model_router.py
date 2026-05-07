import json
import sys
import tempfile
from pathlib import Path

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from detection.router.execution_plan_builder import (  # noqa: E402
    build_execution_plan,
    save_execution_plan,
    save_router_decision,
)
from detection.router.model_router_service import ModelRouterService  # noqa: E402
from detection.router.route_labels import (  # noqa: E402
    AIR_BACKEND,
    CLOSE_RANGE_CLEAR_RGB,
    DISASTER_AERIAL_SCENE,
    DISTANT_SMALL_HUMAN_CANDIDATE,
    QAZI_BACKEND,
    TRANSFORMER_BACKEND,
    YOLO_BACKEND,
)


def _availability(air=False, qazi=False, yolo=True, transformer=True):
    return {
        YOLO_BACKEND: {"backend": YOLO_BACKEND, "available": yolo, "reason": "test_yolo"},
        TRANSFORMER_BACKEND: {"backend": TRANSFORMER_BACKEND, "available": transformer, "reason": "test_transformer"},
        AIR_BACKEND: {"backend": AIR_BACKEND, "available": air, "reason": "adapter_not_configured"},
        QAZI_BACKEND: {"backend": QAZI_BACKEND, "available": qazi, "reason": "adapter_not_configured"},
    }


def _sample_image(tmpdir):
    path = Path(tmpdir) / "router_sample.png"
    Image.new("RGB", (640, 480), (180, 190, 200)).save(path)
    return path


def main():
    router = ModelRouterService()

    with tempfile.TemporaryDirectory() as tmp:
        image_path = _sample_image(tmp)

        close = router.decide_route(str(image_path), {"preferred_route": CLOSE_RANGE_CLEAR_RGB})
        assert close.route == CLOSE_RANGE_CLEAR_RGB
        assert close.display_mode_name == "高清通用目标检测"
        assert close.confidence_level in {"较高", "中等", "较低"}
        assert close.reason

        invalid = router.decide_route(str(image_path), {"preferred_route": "stale_route_id"})
        assert invalid.route == "normal_disaster_rgb"
        assert invalid.router_confidence <= 0.42
        assert "未知检测 route" in invalid.reason

        distant = router.decide_route(str(image_path), {"preferred_route": DISTANT_SMALL_HUMAN_CANDIDATE})
        assert distant.recommended_combo == [AIR_BACKEND, TRANSFORMER_BACKEND, YOLO_BACKEND]

        air_fallback = build_execution_plan(distant, _availability(air=False))
        assert air_fallback.fallback_applied is True
        assert air_fallback.selected_main_backend == YOLO_BACKEND
        assert air_fallback.selected_auxiliary_backends == [TRANSFORMER_BACKEND]
        assert AIR_BACKEND in [item["backend"] for item in air_fallback.unavailable_backends]

        qazi = router.decide_route(str(image_path), {"preferred_route": DISASTER_AERIAL_SCENE})
        qazi_fallback = build_execution_plan(qazi, _availability(qazi=False))
        assert qazi_fallback.fallback_applied is True
        assert qazi_fallback.selected_main_backend == YOLO_BACKEND
        assert qazi_fallback.selected_auxiliary_backends == [TRANSFORMER_BACKEND]
        assert QAZI_BACKEND in [item["backend"] for item in qazi_fallback.unavailable_backends]

        low_conf = router.decide_route(
            str(image_path),
            {"preferred_route": CLOSE_RANGE_CLEAR_RGB, "router_confidence": 0.42},
        )
        low_conf_plan = build_execution_plan(low_conf, _availability())
        assert low_conf_plan.fallback_applied is True
        assert low_conf_plan.selected_main_backend == YOLO_BACKEND
        assert low_conf_plan.selected_auxiliary_backends == [TRANSFORMER_BACKEND]

        yolo_missing = build_execution_plan(close, _availability(yolo=False, transformer=True))
        assert yolo_missing.selected_main_backend is None
        assert TRANSFORMER_BACKEND in yolo_missing.selected_auxiliary_backends
        assert "yolo_unavailable" in yolo_missing.reason

        plan_path = save_execution_plan(air_fallback, tmp)
        decision_path = save_router_decision(close, tmp)
        assert Path(plan_path).exists()
        assert Path(decision_path).exists()
        saved_plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        saved_decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
        assert saved_plan["display_mode_name"]
        assert saved_plan["confidence_level"]
        assert saved_plan["reason"]
        assert saved_decision["display_mode_name"]
        assert saved_decision["confidence_level"]
        assert saved_decision["reason"]

    print("S4 Model Router smoke test passed.")


if __name__ == "__main__":
    main()
