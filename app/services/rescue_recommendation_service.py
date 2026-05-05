"""Route and rescue task recommendation helpers for S8.

Recommendations are decision-support references. Image-plane paths are never
GPS navigation, autonomous rescue routes, or field commands.
"""

from __future__ import annotations


RESCUE_RECOMMENDATION_TRUTHFULNESS_NOTE = (
    "Image-plane path is not GPS navigation. "
    "Route suggestion is a decision-support reference and not an autonomous rescue route. "
    "If no georeferenced map is connected, route coordinates are image pixels only. "
    "Risk-aware A* depends on segmentation quality; uploaded/demo masks are not automatic model segmentation. "
    "Missing map, segmentation, or start point context must be explicitly reported. "
    "Rescue task suggestion requires commander and field-team review."
)
NO_ROUTE_INVENTION_NOTE = (
    "The system must not invent reachable routes when no valid target or image plane is available."
)


def _score(candidate):
    try:
        return float(candidate.get("ec_terp_score", candidate.get("score", 0.0)) or 0.0)
    except Exception:
        return 0.0


def _rank(candidate):
    try:
        value = candidate.get("rank")
        return int(value) if value is not None else 10**9
    except Exception:
        return 10**9


def select_recommendation_targets(decision_candidates, max_targets=3):
    """Select top non-rejected decision candidates for rescue recommendations."""
    candidates = []
    for candidate in decision_candidates or []:
        if candidate.get("should_exclude_from_rescue_ranking"):
            continue
        if candidate.get("review_status") == "rejected_false_positive":
            continue
        candidates.append(dict(candidate))
    candidates.sort(key=lambda item: (_rank(item), -_score(item)))
    return candidates[: max(0, int(max_targets or 0))]


def build_route_context_notes(
    global_context_available=False,
    map_registration_available=False,
    segmentation_available=False,
    start_point_available=False,
):
    """Build explicit missing-context notes for route suggestions."""
    notes = []
    if not global_context_available:
        notes.append("No verified global map is connected for this route suggestion.")
    if not map_registration_available:
        notes.append("Route coordinates are image-plane pixels, not GPS waypoints.")
    if not segmentation_available:
        notes.append("No segmentation-derived risk map is connected; route risk awareness is limited.")
    if not start_point_available:
        notes.append("No verified rescue team start point is connected; default image-plane start may be used.")
    return notes


def summarize_route_result(route_result):
    """Return a compact image-plane route summary."""
    if not route_result:
        return {
            "route_found": False,
            "route_type": "image_plane_path",
            "path_length": 0,
            "total_cost": None,
            "start": [],
            "goal": [],
            "target_id": "",
            "message": "No route result is connected.",
        }
    return {
        "route_found": bool(route_result.get("found") or route_result.get("route_found")),
        "route_type": route_result.get("path_type") or route_result.get("route_type") or "image_plane_path",
        "path_length": route_result.get("path_length", len(route_result.get("path", []) or [])),
        "total_cost": route_result.get("total_cost", route_result.get("cost")),
        "start": route_result.get("start") or route_result.get("start_point") or [],
        "goal": route_result.get("goal") or route_result.get("target_point") or [],
        "target_id": route_result.get("target_id", ""),
        "message": route_result.get("message", ""),
    }


def _lookup(mapping, candidate):
    if not mapping:
        return None
    candidate_id = candidate.get("candidate_id")
    target_id = candidate.get("target_id")
    if isinstance(mapping, dict):
        return mapping.get(candidate_id) or mapping.get(target_id)
    return None


def build_task_recommendation_for_candidate(
    decision_candidate,
    route_result=None,
    route_overlay_path="",
    path_comparison=None,
    missing_context_notes=None,
):
    """Build a single route and task recommendation for one decision candidate."""
    candidate = dict(decision_candidate or {})
    route_summary = summarize_route_result(route_result)
    missing_context_notes = list(missing_context_notes or [])
    avoidance_notes = []
    if path_comparison:
        if path_comparison.get("message"):
            avoidance_notes.append(str(path_comparison.get("message")))
        if path_comparison.get("risk_reduction") is not None:
            avoidance_notes.append(f"Risk-aware comparison risk_reduction={path_comparison.get('risk_reduction')}.")
    if not route_summary["route_found"]:
        avoidance_notes.append("No valid image-plane route is connected for this candidate.")

    rank = candidate.get("rank", 1)
    return {
        "recommendation_id": f"R{int(rank or 1):03d}",
        "candidate_id": candidate.get("candidate_id", ""),
        "target_id": candidate.get("target_id", ""),
        "area_id": candidate.get("area_id", ""),
        "priority_rank": rank,
        "priority_level": candidate.get("priority_level") or candidate.get("ec_terp_level", ""),
        "ec_terp_score": candidate.get("ec_terp_score"),
        "recommended_action": candidate.get("recommended_action") or "prioritize_human_review_and_field_verification",
        "route_found": route_summary["route_found"],
        "route_type": "image_plane_path",
        "is_gps_navigation": False,
        "is_autonomous_rescue_route": False,
        "route_overlay_path": str(route_overlay_path or ""),
        "start": route_summary["start"],
        "goal": route_summary["goal"],
        "path_length": route_summary["path_length"],
        "total_cost": route_summary["total_cost"],
        "avoidance_notes": avoidance_notes,
        "task_suggestion": "建议优先复核该候选目标，并由指挥人员结合现场情况决定是否派遣救援队。",
        "missing_context_notes": missing_context_notes,
        "human_review_required": True,
        "truthfulness_note": RESCUE_RECOMMENDATION_TRUTHFULNESS_NOTE,
    }


def build_rescue_recommendations(
    decision_candidates,
    route_results=None,
    route_overlay_paths=None,
    path_comparisons=None,
    global_context_available=False,
    map_registration_available=False,
    segmentation_available=False,
    start_point_available=False,
    max_targets=3,
):
    """Build route and task recommendations for selected decision candidates."""
    selected = select_recommendation_targets(decision_candidates, max_targets=max_targets)
    missing_context_notes = build_route_context_notes(
        global_context_available=global_context_available,
        map_registration_available=map_registration_available,
        segmentation_available=segmentation_available,
        start_point_available=start_point_available,
    )
    recommendations = []
    for candidate in selected:
        recommendations.append(
            build_task_recommendation_for_candidate(
                candidate,
                route_result=_lookup(route_results, candidate),
                route_overlay_path=_lookup(route_overlay_paths, candidate) or "",
                path_comparison=_lookup(path_comparisons, candidate),
                missing_context_notes=missing_context_notes,
            )
        )
    return recommendations


def summarize_rescue_recommendations(recommendations):
    """Summarize S8 route/task recommendations."""
    recommendations = list(recommendations or [])
    return {
        "recommendation_count": len(recommendations),
        "route_found_count": sum(1 for item in recommendations if item.get("route_found")),
        "route_missing_count": sum(1 for item in recommendations if not item.get("route_found")),
        "high_priority_count": sum(1 for item in recommendations if item.get("priority_level") == "High"),
        "critical_priority_count": sum(1 for item in recommendations if item.get("priority_level") == "Critical"),
        "human_review_required_count": sum(1 for item in recommendations if item.get("human_review_required")),
        "gps_navigation_count": sum(1 for item in recommendations if item.get("is_gps_navigation")),
    }
