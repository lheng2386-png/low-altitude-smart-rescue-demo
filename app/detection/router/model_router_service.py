"""S4 Model Router service.

The router selects a detection mode and backend combo. It does not detect
bounding boxes and is intentionally replaceable by CLIP / ViT / DINOv2 later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .route_labels import (
    AIR_BACKEND,
    CLOSE_RANGE_CLEAR_RGB,
    DISASTER_AERIAL_SCENE,
    DISTANT_SMALL_HUMAN_CANDIDATE,
    NORMAL_DISASTER_RGB,
    QAZI_BACKEND,
    ROUTE_LABELS,
    TRANSFORMER_BACKEND,
    VIDEO_FRAME_SEQUENCE,
    YOLO_BACKEND,
)
from .router_schemas import RouterDecision


class ModelRouterService:
    """Rule-based S4 router with a classifier injection point."""

    def __init__(self, classifier=None, low_confidence_threshold=0.60):
        self.classifier = classifier
        self.low_confidence_threshold = float(low_confidence_threshold)

    def classify(self, image=None, video_frames=None, input_type="image", route_override=None):
        """Return RouterDecision as a dict.

        `classifier` may later be a CLIP / ViT / DINOv2 classifier that returns
        route and confidence. The rest of the service contract stays stable.
        """
        if route_override:
            return self._decision(route_override, 0.99).to_dict()
        if self.classifier is not None:
            classified = self.classifier.predict(image=image, video_frames=video_frames, input_type=input_type)
            route = classified.get("route", NORMAL_DISASTER_RGB)
            confidence = float(classified.get("router_confidence", classified.get("confidence", 0.0)))
            reason = classified.get("reason")
            return self._decision(route, confidence, reason=reason).to_dict()
        return self._rule_classify(image=image, video_frames=video_frames, input_type=input_type).to_dict()

    def build_execution_plan(self, router_decision, availability):
        """Apply backend availability and fallback to the router decision."""
        recommended = list(router_decision.get("recommended_combo", []))
        selected = list(recommended)
        fallback_applied = False
        fallback_reasons = []
        unavailable_backends = []

        if float(router_decision.get("router_confidence", 0.0)) < self.low_confidence_threshold:
            selected = [YOLO_BACKEND, TRANSFORMER_BACKEND]
            fallback_applied = True
            fallback_reasons.append("router_low_confidence_fallback")

        if AIR_BACKEND in selected and not availability.get(AIR_BACKEND, {}).get("available"):
            selected = [YOLO_BACKEND, TRANSFORMER_BACKEND]
            fallback_applied = True
            fallback_reasons.append("air_adapter_unavailable")
            unavailable_backends.append(
                {"backend": AIR_BACKEND, "reason": availability.get(AIR_BACKEND, {}).get("reason", "adapter_unavailable")}
            )

        if QAZI_BACKEND in selected and not availability.get(QAZI_BACKEND, {}).get("available"):
            selected = [YOLO_BACKEND, TRANSFORMER_BACKEND]
            fallback_applied = True
            fallback_reasons.append("qazi_adapter_unavailable")
            unavailable_backends.append(
                {"backend": QAZI_BACKEND, "reason": availability.get(QAZI_BACKEND, {}).get("reason", "adapter_unavailable")}
            )

        selected_backend_combo = []
        for backend in selected:
            if availability.get(backend, {}).get("available"):
                selected_backend_combo.append(backend)
            else:
                unavailable_backends.append({"backend": backend, "reason": availability.get(backend, {}).get("reason", "unavailable")})

        if not availability.get(YOLO_BACKEND, {}).get("available"):
            fallback_reasons.append("yolo_unavailable")

        skipped = [backend for backend in [YOLO_BACKEND, TRANSFORMER_BACKEND, AIR_BACKEND, QAZI_BACKEND] if backend not in selected]
        unavailable_backends = self._dedupe_unavailable(unavailable_backends)
        reason = router_decision.get("reason", "")
        if fallback_applied:
            reason = (
                f"Router 推荐 {router_decision.get('display_mode_name')}，但发生 fallback："
                f"{', '.join(fallback_reasons)}。当前只运行真实可用后端。"
            )

        return {
            "selection_mode": "router_auto_with_fallback" if fallback_applied else "router_auto",
            "route": router_decision.get("route"),
            "display_mode_name": router_decision.get("display_mode_name"),
            "router_confidence": round(float(router_decision.get("router_confidence", 0.0)), 4),
            "recommended_backend_combo": recommended,
            "requested_backend_combo": selected,
            "selected_backend_combo": selected_backend_combo,
            "skipped_backends": skipped,
            "unavailable_backends": unavailable_backends,
            "fallback_applied": bool(fallback_applied),
            "fallback_reasons": fallback_reasons,
            "reason": reason,
            "expected_outputs": [
                "s4_detection_overlay.png",
                "s4_fused_rescue_candidates.png",
                "s4_candidate_crops_sheet.png",
                "rescue_candidates.json",
                "backend_agreement.json",
                "evidence_records.json",
            ],
        }

    def _rule_classify(self, image=None, video_frames=None, input_type="image"):
        if input_type == "video" or video_frames:
            return self._decision(VIDEO_FRAME_SEQUENCE, 0.78)
        if image is None:
            return self._decision(NORMAL_DISASTER_RGB, 0.4, reason="未提供图像，Router 置信度较低，将回退到通用检测组合。")

        features = self._image_features(image)
        disaster_score = max(features["red_ratio"] * 4.0, features["blue_ratio"] * 3.0, features["dark_ratio"] * 1.4)
        distant_score = features["small_input_score"] + max(0.0, 25.0 - features["edge_strength"]) / 40.0
        clarity_score = min(1.0, (features["edge_strength"] + features["contrast"]) / 90.0)

        if disaster_score >= 0.45:
            return self._decision(
                DISASTER_AERIAL_SCENE,
                min(0.9, 0.68 + disaster_score / 4.0),
                reason="图像颜色与纹理呈现灾害航拍线索，系统启用灾害航拍场景检测流程。",
            )
        if distant_score >= 0.85:
            return self._decision(
                DISTANT_SMALL_HUMAN_CANDIDATE,
                min(0.9, 0.64 + distant_score / 5.0),
                reason="图像疑似为远距离或小尺寸输入，系统启用小目标人体检测强化流程。",
            )
        if clarity_score >= 0.45 and min(features["width"], features["height"]) >= 384:
            return self._decision(CLOSE_RANGE_CLEAR_RGB, min(0.88, 0.62 + clarity_score / 4.0))
        return self._decision(NORMAL_DISASTER_RGB, 0.72)

    def _decision(self, route, confidence, reason=None):
        return RouterDecision.from_route_config(route, ROUTE_LABELS[route], confidence, reason=reason)

    @staticmethod
    def _to_pil_rgb(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        array = np.asarray(image)
        if array.ndim == 2:
            return Image.fromarray(array.astype(np.uint8)).convert("RGB")
        if array.ndim == 3:
            if array.shape[-1] == 4:
                array = array[:, :, :3]
            return Image.fromarray(array.astype(np.uint8)).convert("RGB")
        raise ValueError("Unsupported image input for S4 ModelRouterService.")

    @classmethod
    def _image_features(cls, image):
        pil = cls._to_pil_rgb(image)
        arr = np.asarray(pil).astype(np.float32)
        gray = arr.mean(axis=2)
        width, height = pil.size
        gx = np.abs(np.diff(gray, axis=1)).mean() if width > 1 else 0.0
        gy = np.abs(np.diff(gray, axis=0)).mean() if height > 1 else 0.0
        edge_strength = float((gx + gy) / 2.0)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        red_ratio = float(np.mean((arr[:, :, 0] > 150) & (arr[:, :, 1] < 120) & (arr[:, :, 2] < 120)))
        blue_ratio = float(np.mean((arr[:, :, 2] > 135) & (arr[:, :, 0] < 130)))
        dark_ratio = float(np.mean(gray < 55))
        small_input_score = 1.0 - min(width, height) / 512.0
        return {
            "width": width,
            "height": height,
            "edge_strength": round(edge_strength, 4),
            "brightness": round(brightness, 4),
            "contrast": round(contrast, 4),
            "red_ratio": round(red_ratio, 4),
            "blue_ratio": round(blue_ratio, 4),
            "dark_ratio": round(dark_ratio, 4),
            "small_input_score": round(max(0.0, min(1.0, small_input_score)), 4),
        }

    @staticmethod
    def _dedupe_unavailable(items):
        seen = set()
        deduped = []
        for item in items:
            backend = item.get("backend")
            if backend in seen:
                continue
            seen.add(backend)
            deduped.append(item)
        return deduped
