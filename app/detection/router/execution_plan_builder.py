"""Execution plan builder for the S4 Model Router.

The builder decides which real backend adapters should run for this S4 request.
It never fabricates detector output when a backend is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

from .route_labels import (
    AIR_BACKEND,
    FALLBACK_BACKENDS,
    QAZI_BACKEND,
    ROUTE_TO_BACKENDS,
    TRANSFORMER_BACKEND,
    YOLO_BACKEND,
)
from .router_schemas import BackendStatus, ExecutionPlan, RouterDecision, confidence_level_from_score


KNOWN_BACKENDS = [YOLO_BACKEND, TRANSFORMER_BACKEND, AIR_BACKEND, QAZI_BACKEND]
LOW_CONFIDENCE_THRESHOLD = 0.60
EXPECTED_OUTPUTS = [
    "s4_detection_overlay.png",
    "s4_fused_rescue_candidates.png",
    "s4_candidate_crops_sheet.png",
    "rescue_candidates.json",
    "backend_agreement.json",
    "evidence_records.json",
]


def _decision_to_dict(router_decision):
    if isinstance(router_decision, RouterDecision):
        return router_decision.to_dict()
    return dict(router_decision or {})


def _status_to_dict(status, backend):
    if isinstance(status, BackendStatus):
        return status.to_dict()
    if isinstance(status, dict):
        data = dict(status)
        data.setdefault("backend", backend)
        data.setdefault("available", False)
        return data
    return {"backend": backend, "available": False, "reason": "status_missing"}


def _normalize_backend_status(backend_status):
    normalized = {}
    for backend in KNOWN_BACKENDS:
        normalized[backend] = _status_to_dict((backend_status or {}).get(backend), backend)
    for backend, status in (backend_status or {}).items():
        if backend not in normalized:
            normalized[backend] = _status_to_dict(status, backend)
    return normalized


def _available(availability, backend):
    return bool(availability.get(backend, {}).get("available"))


def _reason(availability, backend, default="unavailable"):
    return availability.get(backend, {}).get("reason") or availability.get(backend, {}).get("status") or default


def _dedupe_unavailable(items):
    seen = set()
    deduped = []
    for item in items:
        backend = item.get("backend")
        if backend in seen:
            continue
        seen.add(backend)
        deduped.append({"backend": backend, "reason": item.get("reason")})
    return deduped


def _fallback_selection(availability):
    if not _available(availability, YOLO_BACKEND):
        selected = [backend for backend in FALLBACK_BACKENDS if _available(availability, backend)]
        if selected and selected[0] == TRANSFORMER_BACKEND:
            return None, selected
        return None, selected
    return YOLO_BACKEND, [backend for backend in FALLBACK_BACKENDS[1:] if _available(availability, backend)]


def build_execution_plan(router_decision, backend_status) -> ExecutionPlan:
    """Build a traceable S4 execution plan from router output and availability."""

    decision = _decision_to_dict(router_decision)
    availability = _normalize_backend_status(backend_status)
    route = decision.get("route")
    recommended = list(decision.get("recommended_combo") or ROUTE_TO_BACKENDS.get(route, FALLBACK_BACKENDS))
    selected_request = list(recommended)
    unavailable = []
    fallback_reasons = []

    confidence = float(decision.get("router_confidence", 0.0))
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        selected_request = list(FALLBACK_BACKENDS)
        fallback_reasons.append("router_low_confidence_fallback")

    if AIR_BACKEND in selected_request and not _available(availability, AIR_BACKEND):
        selected_request = list(FALLBACK_BACKENDS)
        fallback_reasons.append("air_adapter_unavailable")
        unavailable.append({"backend": AIR_BACKEND, "reason": _reason(availability, AIR_BACKEND, "adapter_not_configured")})

    if QAZI_BACKEND in selected_request and not _available(availability, QAZI_BACKEND):
        selected_request = list(FALLBACK_BACKENDS)
        fallback_reasons.append("qazi_adapter_unavailable")
        unavailable.append({"backend": QAZI_BACKEND, "reason": _reason(availability, QAZI_BACKEND, "adapter_not_configured")})

    selected_available = []
    for backend in selected_request:
        if _available(availability, backend):
            selected_available.append(backend)
        else:
            unavailable.append({"backend": backend, "reason": _reason(availability, backend)})

    if not _available(availability, YOLO_BACKEND):
        if "yolo_unavailable" not in fallback_reasons:
            fallback_reasons.append("yolo_unavailable")
        selected_main_backend = None
        selected_auxiliary_backends = [backend for backend in selected_available if backend != YOLO_BACKEND]
    else:
        selected_main_backend = selected_available[0] if selected_available else None
        selected_auxiliary_backends = selected_available[1:] if selected_available else []

    skipped = [backend for backend in KNOWN_BACKENDS if backend not in selected_request]
    fallback_applied = bool(fallback_reasons)
    unavailable = _dedupe_unavailable(unavailable)

    base_reason = decision.get("reason") or "Router 已生成检测模式建议。"
    if fallback_applied:
        reason = f"{base_reason} 已根据后端可用性应用回退：{', '.join(fallback_reasons)}。"
    else:
        reason = base_reason
    if "yolo_unavailable" in fallback_reasons:
        reason += " yolo_unavailable：YOLO 主检测权重不可用，不伪造主检测结果，也不将 Transformer 单独作为确认结论。"

    return ExecutionPlan(
        selection_mode="router_auto_with_fallback" if fallback_applied else "router_auto",
        route=route,
        display_mode_name=decision.get("display_mode_name", ""),
        router_confidence=confidence,
        confidence_level=decision.get("confidence_level") or confidence_level_from_score(confidence),
        selected_main_backend=selected_main_backend,
        selected_auxiliary_backends=selected_auxiliary_backends,
        skipped_backends=skipped,
        unavailable_backends=unavailable,
        fallback_applied=fallback_applied,
        reason=reason,
        recommended_backend_combo=recommended,
        requested_backend_combo=selected_request,
        selected_backend_combo=selected_available,
        fallback_reasons=fallback_reasons,
        expected_outputs=list(EXPECTED_OUTPUTS),
    )


def _json_payload(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def save_execution_plan(plan: ExecutionPlan, output_dir: str) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / "execution_plan.json"
    path.write_text(json.dumps(_json_payload(plan), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def save_router_decision(decision: RouterDecision, output_dir: str) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / "router_decision.json"
    path.write_text(json.dumps(_json_payload(decision), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
