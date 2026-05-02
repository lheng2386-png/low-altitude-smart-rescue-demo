CLASS_WEIGHTS = {
    "civilian": 1.0,
    "horse": 0.65,
    "cow": 0.65,
    "dog": 0.55,
    "cat": 0.55,
    "rescuer": 0.15,
}

CLASS_ALIASES = {
    "rescuers": "rescuer",
    "person": "civilian",
    "people": "civilian",
}

CLASS_REASONS = {
    "civilian": "检测到疑似被困平民，建议优先核查和救援。",
    "rescuer": "检测到救援人员，默认不作为被困目标优先处理。",
    "dog": "检测到小型动物，建议在人员救援后处理。",
    "cat": "检测到小型动物，建议在人员救援后处理。",
    "horse": "检测到大型动物，转移难度较高，建议标记并后续处理。",
    "cow": "检测到大型动物，转移难度较高，建议标记并后续处理。",
}


def normalize_class_name(class_name):
    normalized = str(class_name or "").strip().lower().replace(" ", "_").replace("-", "_")
    return CLASS_ALIASES.get(normalized, normalized)


def _clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def _risk_level(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def calculate_risk(target, image_width, image_height, environment_context=None):
    image_area = max(float(image_width * image_height), 1.0)
    area_weight = _clamp(float(target.get("area", 0.0)) / image_area)
    confidence = _clamp(float(target.get("confidence", 0.0)))
    raw_class_name = target.get("class_name", "")
    class_name = normalize_class_name(raw_class_name)
    class_weight = CLASS_WEIGHTS.get(class_name, 0.3)
    base_reason = CLASS_REASONS.get(
        class_name,
        "检测到未知救援相关目标，建议人工复核。",
    )

    if environment_context is None:
        risk_score = class_weight * 70 + confidence * 20 + area_weight * 10
        risk_reason = (
            f"{base_reason}"
            "当前未接入灾区语义分割结果，风险评分仅基于目标类别、置信度和目标面积。"
        )
    else:
        base_target_score = class_weight * 55 + confidence * 15 + area_weight * 10
        environment_score = _clamp(
            float(environment_context.get("environment_risk_score", 0.0)),
            0.0,
            30.0,
        )
        risk_score = base_target_score + environment_score
        risk_reason = (
            f"{base_reason}"
            f"{environment_context.get('environment_reason', '')}"
        )

    risk_score = round(_clamp(risk_score, 0.0, 100.0), 2)
    risk_level = _risk_level(risk_score)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
    }
