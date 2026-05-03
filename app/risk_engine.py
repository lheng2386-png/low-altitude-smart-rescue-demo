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
    "zh": {
        "civilian": "检测到疑似被困平民，建议优先核查和救援。",
        "rescuer": "检测到救援人员，默认不作为被困目标优先处理。",
        "dog": "检测到小型动物，建议在人员救援后处理。",
        "cat": "检测到小型动物，建议在人员救援后处理。",
        "horse": "检测到大型动物，转移难度较高，建议标记并后续处理。",
        "cow": "检测到大型动物，转移难度较高，建议标记并后续处理。",
    },
    "en": {
        "civilian": "Detected a possible trapped civilian. Prioritize verification and rescue.",
        "rescuer": "Detected a rescue worker. It is usually not treated as a trapped target.",
        "dog": "Detected a small animal. It can be handled after human rescue priorities.",
        "cat": "Detected a small animal. It can be handled after human rescue priorities.",
        "horse": "Detected a large animal. Transfer may be difficult, so keep it marked for later handling.",
        "cow": "Detected a large animal. Transfer may be difficult, so keep it marked for later handling.",
    },
}

UNKNOWN_REASONS = {
    "zh": "检测到未知救援相关目标，建议人工复核。",
    "en": "Detected an unknown rescue-related target. Manual review is recommended.",
}

NO_SEGMENTATION_REASONS = {
    "zh": "当前未接入灾区语义分割结果，风险评分仅基于目标类别、置信度和目标面积。",
    "en": "No disaster-scene segmentation is available yet, so the risk score is based only on target class, confidence, and target area.",
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


def calculate_risk(target, image_width, image_height, environment_context=None, language="zh"):
    image_area = max(float(image_width * image_height), 1.0)
    area_weight = _clamp(float(target.get("area", 0.0)) / image_area)
    confidence = _clamp(float(target.get("confidence", 0.0)))
    raw_class_name = target.get("class_name", "")
    class_name = normalize_class_name(raw_class_name)
    class_weight = CLASS_WEIGHTS.get(class_name, 0.3)
    base_reason = CLASS_REASONS.get(language, CLASS_REASONS["zh"]).get(
        class_name,
        UNKNOWN_REASONS.get(language, UNKNOWN_REASONS["zh"]),
    )

    if environment_context is None:
        risk_score = class_weight * 70 + confidence * 20 + area_weight * 10
        risk_reason = (
            f"{base_reason}"
            f"{NO_SEGMENTATION_REASONS.get(language, NO_SEGMENTATION_REASONS['zh'])}"
        )
    else:
        base_target_score = class_weight * 55 + confidence * 15 + area_weight * 10
        environment_score = _clamp(
            float(environment_context.get("environment_risk_score", 0.0)),
            0.0,
            30.0,
        )
        risk_score = base_target_score + environment_score
        env_reason = environment_context.get("environment_reason", "")
        risk_reason = (
            f"{base_reason}"
            f"{env_reason}"
        )

    risk_score = round(_clamp(risk_score, 0.0, 100.0), 2)
    risk_level = _risk_level(risk_score)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
    }
