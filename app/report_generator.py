ANIMAL_CLASSES = {"dog", "cat", "horse", "cow"}


def _percent(summary, class_name):
    return float(summary.get(class_name, 0.0)) * 100


def _segmentation_section(segmentation_summary, ranked_targets):
    if not segmentation_summary:
        return (
            "三、语义分割环境风险\n"
            "当前未接入灾区语义分割结果，本报告仅基于目标检测结果生成。\n"
        )

    water_ratio = _percent(segmentation_summary, "water") + _percent(segmentation_summary, "pool")
    blocked_road_ratio = _percent(segmentation_summary, "road_blocked")
    damaged_building_ratio = (
        _percent(segmentation_summary, "minor_damage")
        + _percent(segmentation_summary, "major_damage")
        + _percent(segmentation_summary, "destroyed_building")
    )

    top_environment = "暂无"
    if ranked_targets:
        top_target = ranked_targets[0]
        top_environment = (
            f"{top_target['target_id']} 周边主导环境为 "
            f"{top_target.get('environment', 'unknown')}，"
            f"环境风险加分 {top_target.get('environment_score', 0.0):.2f}。"
        )

    return (
        "三、语义分割环境风险\n"
        f"水域/积水面积占比：{water_ratio:.1f}%。\n"
        f"道路阻断区域占比：{blocked_road_ratio:.1f}%。\n"
        f"建筑损毁区域占比：{damaged_building_ratio:.1f}%。\n"
        f"最高风险目标环境：{top_environment}\n"
    )


def _path_section(segmentation_summary, path_result, ranked_targets):
    if not ranked_targets:
        return (
            "四、路径规划建议\n"
            "当前图像未检测到明确救援目标，无法规划路径。\n"
        )

    if not path_result:
        return (
            "四、路径规划建议\n"
            "当前未能生成有效路径，原因：路径规划模块未返回结果。\n"
        )

    if not path_result.get("found"):
        return (
            "四、路径规划建议\n"
            f"当前未能生成有效路径，原因：{path_result.get('message', '未知原因')}。\n"
            "建议检查起点位置、目标检测结果或上传更准确的 segmentation mask 后重试。\n"
        )

    start = path_result.get("start", [0, 0])
    goal = path_result.get("goal", [0, 0])
    top_target = ranked_targets[0] if ranked_targets else None
    top_target_text = (
        f"{top_target['target_id']}（{top_target['class_name']}）"
        if top_target
        else "最高风险目标"
    )

    if segmentation_summary:
        env_text = "当前路径规划已结合 RescueNet-style mask 中的水域、道路阻断、建筑损毁等区域代价。"
    else:
        env_text = (
            "当前未上传 segmentation mask，路径规划仅基于图像平面默认代价地图，"
            "尚未考虑水域、道路阻断、建筑损毁等环境障碍。"
        )

    return (
        "四、路径规划建议\n"
        f"系统以起点 S({start[0]}, {start[1]}) 为救援出发点，以 {top_target_text} 为目标点，"
        "基于当前通行代价地图使用 A* 算法生成参考救援路径。\n"
        f"路径长度为 {path_result.get('path_length', 0)} 个像素点，累计路径代价为 {path_result.get('total_cost', 0.0):.2f}。\n"
        f"{env_text}\n"
        "建议救援人员优先沿该路径方向核查目标区域，并结合现场实际道路条件调整。\n"
    )


def generate_report(targets, ranked_targets, segmentation_summary=None, path_result=None):
    segmentation_summary = segmentation_summary or {}

    if not targets:
        segmentation_text = _segmentation_section(segmentation_summary, [])
        path_text = _path_section(segmentation_summary, path_result, [])
        return (
            "AeroRescue-AI 灾情识别与救援辅助报告\n\n"
            "当前图像未检测到明确救援目标。\n\n"
            f"{segmentation_text}\n"
            f"{path_text}\n"
            "初步建议：建议更换视角、提高图像清晰度或降低检测置信度阈值后复核。\n\n"
            "当前版本局限说明：如未上传语义分割 mask，系统无法自动判断水域、道路阻断、建筑损毁等环境风险。"
        )

    civilian_count = sum(1 for target in targets if target["class_name"] == "civilian")
    rescuer_count = sum(1 for target in targets if target["class_name"] == "rescuer")
    animal_count = sum(1 for target in targets if target["class_name"] in ANIMAL_CLASSES)
    top_target = ranked_targets[0] if ranked_targets else None

    if top_target:
        top_target_text = (
            f"{top_target['target_id']}（{top_target['class_name']}），"
            f"风险分数 {top_target['risk_score']:.2f}，"
            f"风险等级 {top_target['risk_level']}。"
        )
    else:
        top_target_text = "暂无。"

    high_risk_count = sum(1 for target in ranked_targets if target["risk_level"] == "High")
    medium_risk_count = sum(1 for target in ranked_targets if target["risk_level"] == "Medium")

    suggestions = [
        "优先核查排名靠前目标的位置和周边通行条件。",
        "对 civilian 目标优先进行人工复核和救援资源调度。",
        "对动物目标结合现场资源进行转移或标记。",
        "rescuer 目标默认风险权重较低，主要用于识别现场救援力量分布。",
    ]

    if segmentation_summary:
        suggestions.append("规划现场行动时避开水域、道路阻断区域和严重损毁建筑附近区域。")
    else:
        suggestions.append("当前未接入灾区语义分割结果，建议补充分割 mask 后复核环境风险。")

    segmentation_text = _segmentation_section(segmentation_summary, ranked_targets)
    path_text = _path_section(segmentation_summary, path_result, ranked_targets)

    return (
        "AeroRescue-AI 灾情识别与救援辅助报告\n\n"
        f"一、识别概况\n"
        f"本次识别目标总数：{len(targets)}。\n"
        f"civilian 数量：{civilian_count}；rescuer 数量：{rescuer_count}；"
        f"animal 数量：{animal_count}。\n\n"
        f"二、风险概况\n"
        f"高风险目标数量：{high_risk_count}；中风险目标数量：{medium_risk_count}。\n"
        f"最高风险目标：{top_target_text}\n\n"
        f"{segmentation_text}\n"
        f"{path_text}\n"
        f"五、初步救援建议\n"
        + "\n".join(f"- {item}" for item in suggestions)
        + "\n\n六、当前版本局限说明\n"
        "当前版本不会自动生成语义分割结果；如需环境风险融合，需要上传 RescueNet 风格 segmentation mask。"
    )
