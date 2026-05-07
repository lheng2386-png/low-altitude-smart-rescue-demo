"""S4 Model Router service.

The router selects a detection mode and backend combo. It does not detect
bounding boxes and is intentionally replaceable by CLIP / ViT / DINOv2 later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .route_labels import (
    CLOSE_RANGE_CLEAR_RGB,
    DISASTER_AERIAL_SCENE,
    DISTANT_SMALL_HUMAN_CANDIDATE,
    NORMAL_DISASTER_RGB,
    ROUTE_LABELS,
    VIDEO_FRAME_SEQUENCE,
)
from .execution_plan_builder import build_execution_plan
from .router_schemas import RouterDecision


def check_backend_status(yolo_weights_path=None, transformer_available=True):
    """Lightweight availability check for standalone router tests/integration."""
    yolo_available = bool(yolo_weights_path and Path(yolo_weights_path).exists())
    return {
        "yolo_main_detector": {
            "backend": "yolo_main_detector",
            "available": yolo_available,
            "reason": "local YOLO weights found" if yolo_available else "missing_weights",
        },
        "transformer_compare_detector": {
            "backend": "transformer_compare_detector",
            "available": bool(transformer_available),
            "reason": "available" if transformer_available else "adapter_unavailable",
        },
        "air_sar_detector": {
            "backend": "air_sar_detector",
            "available": False,
            "reason": "adapter_not_configured",
        },
        "qazi_disaster_detector": {
            "backend": "qazi_disaster_detector",
            "available": False,
            "reason": "adapter_not_configured",
        },
    }


class ModelRouterService:
    """Rule-based S4 router with a classifier injection point."""

    def __init__(self, classifier=None, low_confidence_threshold=0.60, strategy="rule_v0"):
        self.classifier = classifier
        self.low_confidence_threshold = float(low_confidence_threshold)
        self.strategy = strategy

    def decide_route(self, image_path: str, input_meta: dict | None = None) -> RouterDecision:
        """Select an S4 detection route for one image or video key frame.

        The current implementation is an explainable rule/mock router. The
        constructor-level classifier hook keeps the contract replaceable by
        CLIP / ViT / DINOv2 classifiers later.
        """
        input_meta = dict(input_meta or {})
        preferred_route = input_meta.get("preferred_route")
        if preferred_route:
            return self._decision(
                preferred_route,
                float(input_meta.get("router_confidence", 0.99)),
                reason=input_meta.get("reason") or "测试或上游流程指定检测模式，Router 按指定 route 生成执行建议。",
            )

        input_type = input_meta.get("input_type")
        if input_type in {"video", "video_frame"}:
            return self._decision(
                VIDEO_FRAME_SEQUENCE,
                float(input_meta.get("router_confidence", 0.78)),
                reason="输入为视频或视频关键帧，系统推荐进入视频关键帧目标检测流程。",
            )

        scene_hint = input_meta.get("scene_hint")
        if scene_hint in {"flood", "fire", "collapse", "disaster_aerial"}:
            return self._decision(
                DISASTER_AERIAL_SCENE,
                float(input_meta.get("router_confidence", 0.84)),
                reason="输入含灾害航拍场景提示，系统推荐航拍灾情场景检测；该判断不代表确认灾害类型。",
            )
        if scene_hint in {"small_human", "distant_person", "aerial_small_target"}:
            return self._decision(
                DISTANT_SMALL_HUMAN_CANDIDATE,
                float(input_meta.get("router_confidence", 0.87)),
                reason="输入含远距离小目标或疑似人员提示，系统推荐小目标人体检测强化流程；该判断不代表确认人员。",
            )

        try:
            image = self._to_pil_rgb(image_path)
        except Exception:
            return self._decision(
                NORMAL_DISASTER_RGB,
                float(input_meta.get("router_confidence", 0.42)),
                reason="图像路径不存在或无法读取，Router 保守回退到普通灾害 RGB 检测建议。",
            )

        if self.classifier is not None:
            classified = self.classifier.predict(image=image, input_meta=input_meta, strategy=self.strategy)
            return self._decision(
                classified.get("route", NORMAL_DISASTER_RGB),
                float(classified.get("router_confidence", classified.get("confidence", 0.0))),
                reason=classified.get("reason"),
            )

        return self._rule_classify(image=image, input_type="image")

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
        return build_execution_plan(router_decision, availability).to_dict()

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
        if route not in ROUTE_LABELS:
            reason = (
                f"未知检测 route '{route}'，Router 保守回退到普通灾害 RGB 检测建议；"
                "未执行未注册检测模式。"
            )
            route = NORMAL_DISASTER_RGB
            confidence = min(float(confidence), 0.42)
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
