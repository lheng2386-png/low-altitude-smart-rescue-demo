from risk_engine import calculate_risk
from segmentation_engine import get_environment_context_for_target


def _rescue_reason(target, risk, environment_context):
    return risk["risk_reason"]


def rank_targets(targets, image_width, image_height, segmentation_mask=None):
    if not targets:
        return []

    ranked_targets = []

    for target in targets:
        environment_context = None
        if segmentation_mask is not None:
            environment_context = get_environment_context_for_target(target, segmentation_mask)

        risk = calculate_risk(target, image_width, image_height, environment_context)
        ranked_targets.append(
            {
                "target_id": target["id"],
                "class_name": target["class_name"],
                "confidence": target["confidence"],
                "bbox": target["bbox"],
                "risk_score": risk["risk_score"],
                "risk_level": risk["risk_level"],
                "environment_score": round(
                    float(environment_context.get("environment_risk_score", 0.0)),
                    2,
                )
                if environment_context
                else 0.0,
                "environment": environment_context.get("dominant_area_class", "not_available")
                if environment_context
                else "not_available",
                "reason": _rescue_reason(target, risk, environment_context),
            }
        )

    ranked_targets.sort(key=lambda item: item["risk_score"], reverse=True)

    for index, target in enumerate(ranked_targets, start=1):
        target["rank"] = index

    return ranked_targets
