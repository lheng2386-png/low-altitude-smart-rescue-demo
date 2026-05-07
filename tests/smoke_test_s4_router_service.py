import json
import sys
import tempfile
from pathlib import Path

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from s4_router_service import (  # noqa: E402
    AIR_BACKEND,
    MODEL_EVIDENCE_NOTE,
    QAZI_BACKEND,
    TRANSFORMER_BACKEND,
    YOLO_BACKEND,
    build_s4_execution_plan,
    check_s4_backend_availability,
    classify_s4_route,
    run_s4_router_detection,
)


def _availability(**values):
    base = {
        YOLO_BACKEND: {"available": True, "status": "available", "reason": "test", "output_role": "main"},
        TRANSFORMER_BACKEND: {"available": True, "status": "available", "reason": "test", "output_role": "aux"},
        AIR_BACKEND: {"available": False, "status": "adapter_unavailable", "reason": "missing_weights", "output_role": "sar"},
        QAZI_BACKEND: {"available": False, "status": "adapter_unavailable", "reason": "missing_weights", "output_role": "disaster"},
    }
    for key, value in values.items():
        base[key].update(value)
    return base


def _mock_results():
    return {
        YOLO_BACKEND: [
            {"class_name": "civilian", "confidence": 0.86, "bbox": [20, 20, 80, 100], "center": [50, 60]},
        ],
        TRANSFORMER_BACKEND: [
            {"class_name": "human_candidate", "confidence": 0.78, "bbox": [24, 24, 82, 104], "center": [53, 64]},
        ],
        AIR_BACKEND: [
            {"class_name": "person", "confidence": 0.81, "bbox": [25, 25, 83, 105], "center": [54, 65]},
        ],
    }


def main():
    image = Image.new("RGB", (640, 480), (180, 190, 200))

    router = classify_s4_route(image, route_override="close_range_clear_rgb")
    assert router["route"] == "close_range_clear_rgb"
    plan = build_s4_execution_plan(router, _availability())
    assert plan["selected_backend_combo"] == [YOLO_BACKEND, TRANSFORMER_BACKEND]
    assert AIR_BACKEND in plan["skipped_backends"]

    distant = classify_s4_route(image, route_override="distant_small_human_candidate")
    distant_plan_available = build_s4_execution_plan(distant, _availability(**{AIR_BACKEND: {"available": True, "status": "available"}}))
    assert distant_plan_available["selected_backend_combo"] == [AIR_BACKEND, TRANSFORMER_BACKEND, YOLO_BACKEND]

    distant_plan_fallback = build_s4_execution_plan(distant, _availability())
    assert distant_plan_fallback["fallback_applied"] is True
    assert "air_adapter_unavailable" in distant_plan_fallback["fallback_reasons"]
    assert distant_plan_fallback["selected_backend_combo"] == [YOLO_BACKEND, TRANSFORMER_BACKEND]

    qazi = classify_s4_route(image, route_override="disaster_aerial_scene")
    qazi_plan = build_s4_execution_plan(qazi, _availability())
    assert qazi_plan["fallback_applied"] is True
    assert "qazi_adapter_unavailable" in qazi_plan["fallback_reasons"]
    assert any(item["backend"] == QAZI_BACKEND for item in qazi_plan["unavailable_backends"])

    low_conf = dict(router)
    low_conf["router_confidence"] = 0.42
    low_conf_plan = build_s4_execution_plan(low_conf, _availability())
    assert low_conf_plan["fallback_applied"] is True
    assert "router_low_confidence_fallback" in low_conf_plan["fallback_reasons"]

    availability = check_s4_backend_availability(
        model_variant="missing_smoke_test_model",
        availability_overrides={TRANSFORMER_BACKEND: {"available": True, "status": "available"}},
    )
    assert availability[YOLO_BACKEND]["available"] is False

    with tempfile.TemporaryDirectory() as tmp:
        result = run_s4_router_detection(
            image,
            model_variant="missing_smoke_test_model",
            route_override="distant_small_human_candidate",
            availability_overrides={
                YOLO_BACKEND: {"available": True, "status": "available", "reason": "mock"},
                TRANSFORMER_BACKEND: {"available": True, "status": "available", "reason": "mock"},
                AIR_BACKEND: {"available": True, "status": "available", "reason": "mock"},
            },
            mock_backend_results=_mock_results(),
            output_dir=tmp,
        )
        assert set(result["raw_results"]) == {AIR_BACKEND, TRANSFORMER_BACKEND, YOLO_BACKEND}
        assert result["rescue_candidates"]
        human_candidates = [item for item in result["rescue_candidates"] if item["class_name"] == "human_candidate"]
        assert all(item["human_review_required"] is True for item in human_candidates)
        assert any(item["cross_backend_agreement"] is True for item in result["rescue_candidates"])
        assert result["evidence_records"]
        assert all(item["human_review_required"] is True for item in result["evidence_records"])
        for key in ["s4_detection_overlay", "s4_fused_rescue_candidates", "s4_candidate_crops_sheet"]:
            assert Path(result["paths"][key]).exists()
        summary = json.loads(Path(result["paths"]["final_report_v2_s4_summary"]).read_text(encoding="utf-8"))
        assert "人工复核" in summary["truthfulness_boundary"]

    with tempfile.TemporaryDirectory() as tmp:
        qazi_result = run_s4_router_detection(
            image,
            model_variant="missing_smoke_test_model",
            route_override="disaster_aerial_scene",
            availability_overrides={YOLO_BACKEND: {"available": True, "status": "available", "reason": "mock"}, TRANSFORMER_BACKEND: {"available": True, "status": "available", "reason": "mock"}},
            mock_backend_results=_mock_results(),
            output_dir=tmp,
        )
        assert QAZI_BACKEND not in qazi_result["raw_results"] or qazi_result["raw_results"][QAZI_BACKEND]["success"] is False
        assert qazi_result["adapter_status"][QAZI_BACKEND]["status"] == "adapter_unavailable"
        assert qazi_result["raw_results"][QAZI_BACKEND]["detections"] == []

    with tempfile.TemporaryDirectory() as tmp:
        transformer_only = run_s4_router_detection(
            image,
            model_variant="missing_smoke_test_model",
            route_override="close_range_clear_rgb",
            availability_overrides={TRANSFORMER_BACKEND: {"available": True, "status": "available", "reason": "mock"}},
            mock_backend_results={TRANSFORMER_BACKEND: _mock_results()[TRANSFORMER_BACKEND]},
            output_dir=tmp,
        )
        assert transformer_only["execution_plan"]["selected_main_backend"] is None
        assert transformer_only["execution_plan"]["selected_backend_combo"] == [TRANSFORMER_BACKEND]
        assert transformer_only["rescue_candidates"]
        candidate = transformer_only["rescue_candidates"][0]
        assert candidate["source_backend"] == TRANSFORMER_BACKEND
        assert candidate["can_enter_terp"] is False
        assert candidate["can_enter_path_planning"] is False
        evidence = transformer_only["evidence_records"][0]
        assert "S7_terp_ranking" not in evidence["used_by"]
        assert "S8_path_planning" not in evidence["used_by"]

    forbidden = ["confirmed civilian", "confirmed survivor", "已确认幸存者"]
    assert not any(term in MODEL_EVIDENCE_NOTE for term in forbidden)
    print("S4 router service smoke test passed.")


if __name__ == "__main__":
    main()
