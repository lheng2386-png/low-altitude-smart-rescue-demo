from risk_engine import CLASS_WEIGHTS, normalize_class_name


def _clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(maximum, float(value)))


def _terp_level(score):
    if score >= 90:
        return "Critical"
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def _path_accessibility_score(path_result):
    if not path_result:
        return 0.0
    if not path_result.get("found"):
        return 20.0

    path_length = max(int(path_result.get("path_length", 0)), 1)
    if path_length <= 1:
        return 0.0

    avg_cost = float(path_result.get("total_cost", 0.0)) / max(path_length - 1, 1)
    return round(_clamp((avg_cost - 3.0) / 77.0 * 20.0, 0.0, 20.0), 2)


def _target_score(target, image_width, image_height):
    image_area = max(float(image_width * image_height), 1.0)
    class_name = normalize_class_name(target.get("class_name", ""))
    class_weight = CLASS_WEIGHTS.get(class_name, 0.3)
    confidence = _clamp(target.get("confidence", 0.0), 0.0, 1.0)
    area_weight = _clamp(float(target.get("area", 0.0)) / image_area, 0.0, 1.0)
    return round(class_weight * 45.0 + confidence * 15.0 + area_weight * 10.0, 2)


def calculate_terp(target, image_width, image_height, environment_context=None, path_result=None, language="zh"):
    """Calculate Target-Environment-Route Priority for one target."""
    target_score = _target_score(target, image_width, image_height)
    environment_score = 0.0
    if environment_context:
        environment_score = round(
            _clamp(environment_context.get("environment_risk_score", 0.0), 0.0, 20.0),
            2,
        )
    accessibility_score = _path_accessibility_score(path_result)
    terp_score = round(_clamp(target_score + environment_score + accessibility_score, 0.0, 100.0), 2)
    terp_level = _terp_level(terp_score)

    class_name = normalize_class_name(target.get("class_name", ""))
    if path_result and not path_result.get("found"):
        route_reason = (
            "Current target path is not reachable or no stable path was found, so accessibility risk is high."
            if language == "en"
            else "当前目标路径不可达或未找到稳定路径，可达性风险较高。"
        )
    elif accessibility_score > 10:
        route_reason = (
            "The reference route to this target is relatively costly, so field detour conditions should be checked."
            if language == "en"
            else "通往该目标的参考路径代价较高，建议现场复核绕行条件。"
        )
    elif path_result:
        route_reason = (
            "The reference route to this target is relatively accessible."
            if language == "en"
            else "通往该目标的参考路径可达性相对较好。"
        )
    else:
        route_reason = (
            "No path result is connected yet, so accessibility falls back to a default value."
            if language == "en"
            else "当前未接入路径结果，可达性暂按默认值处理。"
        )

    if environment_context:
        env_reason = environment_context.get("environment_reason", "")
    else:
        env_reason = "No segmentation-based environment result is connected yet." if language == "en" else "当前未接入语义分割环境结果。"

    intro = (
        "TERP combines target type, detection confidence, target scale, environmental risk, and route accessibility. "
        if language == "en"
        else "TERP 综合目标类型、检测置信度、目标尺度、环境风险和路径可达性。"
    )
    reason = f"{intro}{env_reason}{route_reason}"
    if class_name == "civilian" and terp_level in {"High", "Critical"}:
        reason += (
            "The target appears to be a trapped civilian and has a high overall priority. Prioritize verification."
            if language == "en"
            else "疑似被困平民且综合优先级较高，建议优先核查。"
        )

    return {
        "target_id": target.get("id") or target.get("target_id"),
        "class_name": target.get("class_name", "unknown"),
        "target_score": target_score,
        "environment_score": environment_score,
        "accessibility_score": accessibility_score,
        "terp_score": terp_score,
        "terp_level": terp_level,
        "reason": reason,
    }


def rank_targets_by_terp(targets, image_width, image_height, environment_contexts=None, path_results=None, language="zh"):
    """Rank targets with TERP while preserving original target ids."""
    if not targets:
        return []

    environment_contexts = environment_contexts or {}
    path_results = path_results or {}
    results = []
    for target in targets:
        target_id = target.get("id") or target.get("target_id")
        terp = calculate_terp(
            target,
            image_width,
            image_height,
            environment_context=environment_contexts.get(target_id),
            path_result=path_results.get(target_id),
            language=language,
        )
        results.append(terp)

    results.sort(key=lambda item: item["terp_score"], reverse=True)
    for index, item in enumerate(results, start=1):
        item["rank"] = index
    return results
