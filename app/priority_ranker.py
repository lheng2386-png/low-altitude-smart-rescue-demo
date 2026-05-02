from risk_engine import calculate_risk
from segmentation_engine import get_environment_context_for_target


def _rescue_reason(target, risk, environment_context):
    class_name = target.get("class_name", "")
    confidence = float(target.get("confidence", 0.0))

    if class_name == "civilian":
        subject = "疑似被困平民"
    elif class_name == "rescuer":
        subject = "救援人员目标"
    elif class_name in {"dog", "cat", "horse", "cow"}:
        subject = "动物救援目标"
    else:
        subject = f"{class_name} 目标"

    confidence_text = "检测置信度较高" if confidence >= 0.6 else "检测置信度需要复核"

    if environment_context is None:
        return (
            f"{subject}，{confidence_text}。"
            "当前未接入灾区语义分割结果，风险评分仅基于目标类别、置信度和目标面积。"
        )

    return f"{subject}，{confidence_text}，{environment_context['environment_reason']}建议结合环境风险优先核查。"


def rank_targets(targets, image_width, image_height, segmentation_mask=None):
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
