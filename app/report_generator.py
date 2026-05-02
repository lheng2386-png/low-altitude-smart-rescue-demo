ANIMAL_CLASSES = {"dog", "cat", "horse", "cow"}


def generate_report(targets, ranked_targets):
    if not targets:
        return (
            "低空智援初版救援报告\n\n"
            "当前图像未检测到明确救援目标。\n\n"
            "初步建议：建议更换视角、提高图像清晰度或降低检测置信度阈值后复核。\n\n"
            "当前版本局限说明：目前仅基于目标类别、检测置信度和目标面积评估风险，"
            "后续将融合灾区语义分割结果，如水域、道路阻断、建筑损毁等环境风险因素。"
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

    return (
        "低空智援初版救援报告\n\n"
        f"一、识别概况\n"
        f"本次识别目标总数：{len(targets)}。\n"
        f"civilian 数量：{civilian_count}；rescuer 数量：{rescuer_count}；"
        f"animal 数量：{animal_count}。\n\n"
        f"二、风险概况\n"
        f"高风险目标数量：{high_risk_count}；中风险目标数量：{medium_risk_count}。\n"
        f"最高风险目标：{top_target_text}\n\n"
        f"三、初步救援建议\n"
        + "\n".join(f"- {item}" for item in suggestions)
        + "\n\n四、当前版本局限说明\n"
        "目前仅基于目标类别、检测置信度和目标面积评估风险，后续将融合灾区语义分割结果，"
        "如水域、道路阻断、建筑损毁等环境风险因素。"
    )
