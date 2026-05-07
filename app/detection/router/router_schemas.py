"""Schemas for S4 Router outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


def confidence_level_from_score(value):
    value = float(value or 0.0)
    if value >= 0.75:
        return "较高"
    if value >= 0.60:
        return "中等"
    return "较低"


@dataclass
class RouterDecision:
    """Router output for selecting an S4 detection mode, not bounding boxes."""

    route: str
    display_mode_name: str
    router_confidence: float
    confidence_level: str
    recommended_combo: list[str]
    reason: str
    fallback_applied: bool = False
    fallback_reason: str | None = None
    unavailable_backends: list[dict] = field(default_factory=list)

    def to_dict(self):
        return {
            "route": self.route,
            "display_mode_name": self.display_mode_name,
            "router_confidence": round(float(self.router_confidence), 4),
            "confidence_level": self.confidence_level or confidence_level_from_score(self.router_confidence),
            "recommended_combo": list(self.recommended_combo),
            "reason": self.reason,
            "fallback_applied": bool(self.fallback_applied),
            "fallback_reason": self.fallback_reason,
            "unavailable_backends": list(self.unavailable_backends),
        }

    @classmethod
    def from_route_config(
        cls,
        route,
        route_config,
        router_confidence,
        confidence_level=None,
        reason=None,
        fallback_applied=False,
        fallback_reason=None,
        unavailable_backends=None,
    ):
        return cls(
            route=route,
            display_mode_name=route_config["display_mode_name"],
            router_confidence=float(router_confidence),
            confidence_level=confidence_level or confidence_level_from_score(router_confidence),
            recommended_combo=list(route_config["recommended_combo"]),
            reason=reason or route_config["reason"],
            fallback_applied=bool(fallback_applied),
            fallback_reason=fallback_reason,
            unavailable_backends=list(unavailable_backends or []),
        )


@dataclass
class BackendStatus:
    """Availability state for one selectable S4 detection backend."""

    backend: str
    available: bool
    reason: str | None = None

    def to_dict(self):
        return {
            "backend": self.backend,
            "available": bool(self.available),
            "reason": self.reason,
        }


@dataclass
class ExecutionPlan:
    """Executable backend selection derived from a RouterDecision."""

    selection_mode: str
    route: str
    display_mode_name: str
    router_confidence: float
    confidence_level: str
    selected_main_backend: str | None
    selected_auxiliary_backends: list[str]
    skipped_backends: list[str]
    unavailable_backends: list[dict]
    fallback_applied: bool
    reason: str
    recommended_backend_combo: list[str] = field(default_factory=list)
    requested_backend_combo: list[str] = field(default_factory=list)
    selected_backend_combo: list[str] = field(default_factory=list)
    fallback_reasons: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)

    def to_dict(self):
        data = asdict(self)
        data["router_confidence"] = round(float(self.router_confidence), 4)
        data["fallback_applied"] = bool(self.fallback_applied)
        return data
