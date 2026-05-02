CLASS_WEIGHTS = {
    "civilian": 1.0,
    "horse": 0.65,
    "cow": 0.65,
    "dog": 0.55,
    "cat": 0.55,
    "rescuer": 0.15,
}


def _clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def _risk_level(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def calculate_risk(target, image_width, image_height):
    image_area = max(float(image_width * image_height), 1.0)
    area_weight = _clamp(float(target.get("area", 0.0)) / image_area)
    confidence = _clamp(float(target.get("confidence", 0.0)))
    class_name = target.get("class_name", "")
    class_weight = CLASS_WEIGHTS.get(class_name, 0.3)

    risk_score = class_weight * 70 + confidence * 20 + area_weight * 10
    risk_score = round(_clamp(risk_score, 0.0, 100.0), 2)
    risk_level = _risk_level(risk_score)

    risk_reason = (
        f"{class_name} class weight={class_weight:.2f}, "
        f"confidence={confidence:.2f}, area_ratio={area_weight:.3f}"
    )

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
    }
