def evaluate_scene_applicability(targets, segmentation_mask=None, segmentation_source="none", auto_model_available=False):
    """Assess whether current inputs support target, environment, and route decisions."""
    target_count = len(targets or [])
    has_mask = segmentation_mask is not None
    source = str(segmentation_source or "none").lower()

    if target_count == 0:
        return {
            "applicable": False,
            "level": "No Target",
            "message": "当前未检测到明确救援目标，系统不会生成目标救援路径。",
            "allow_path_planning": False,
            "allow_environment_fusion": False,
        }

    if has_mask:
        return {
            "applicable": True,
            "level": "Full",
            "message": "当前图像包含检测目标和有效语义分割结果，可启用完整环境风险融合与风险感知路径规划。",
            "allow_path_planning": True,
            "allow_environment_fusion": True,
        }

    if "auto" in source and not auto_model_available:
        return {
            "applicable": True,
            "level": "Fallback",
            "message": "当前选择自动分割但未找到可用权重，系统已回退到目标检测与默认代价路径规划。",
            "allow_path_planning": True,
            "allow_environment_fusion": False,
        }

    return {
        "applicable": True,
        "level": "Target Only",
        "message": "当前未接入语义分割结果，系统仅基于目标检测和默认图像平面代价生成辅助建议。",
        "allow_path_planning": True,
        "allow_environment_fusion": False,
    }

