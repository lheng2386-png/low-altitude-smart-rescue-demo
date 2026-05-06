from environment_risk import CLASS_DISPLAY_NAMES, CLASS_DISPLAY_NAMES_EN
from segmentation_source_metadata import build_segmentation_source_metadata


ANIMAL_CLASSES = {"dog", "cat", "horse", "cow"}
TARGET_CLASS_DISPLAY_NAMES = {
    "civilian": "平民",
    "rescuer": "救援人员",
    "dog": "犬",
    "cat": "猫",
    "horse": "马",
    "cow": "牛",
    "person": "平民",
    "people": "平民",
    "rescuers": "救援人员",
}

TEXT = {
    "zh": {
        "title": "灾情感知及影响评估 灾情识别与救援辅助报告",
        "no_target": "当前图像未检测到明确救援目标。",
        "overview": "一、识别概况",
        "terp": "二、TERP 联合优先级评估",
        "risk": "三、风险概况",
        "segmentation": "四、语义分割环境风险",
        "path": "五、路径规划建议",
        "comparison": "六、路径规划对比",
        "reliability": "七、路径规划可靠性说明",
        "damage": "八、灾损评估与救援入口",
        "suggestions": "九、初步救援建议",
        "limitations": "十、当前版本局限说明",
        "segmentation_empty": (
            "当前未接入语义分割结果，环境风险与路径代价主要基于默认图像平面假设生成。"
            "可通过上传语义分割掩码或提供训练好的语义分割权重文件启用环境风险融合。"
        ),
        "segmentation_enabled": "当前报告已结合语义分割掩码或自动分割结果进行环境风险融合。",
        "path_plane_limit": (
            "当前路径规划为图像平面参考路径，不等同于真实 GPS 路线，也未接入真实道路网络、无人机定位或飞控系统。"
        ),
        "path_missing": "当前未能生成有效路径，原因：路径规划模块未返回结果。",
        "path_unreachable": "当前未能生成有效路径，原因：{reason}。建议检查起点位置、目标检测结果或上传更准确的语义分割掩码后重试。",
        "path_no_target": "当前图像未检测到明确救援目标，无法规划路径。",
        "path_no_segmentation": "当前未接入语义分割结果，路径规划仅基于图像平面默认代价地图，尚未结合水域、道路阻断、建筑损毁等环境代价。",
        "path_with_segmentation": "当前路径规划已结合语义分割掩码中的水域、道路阻断、建筑损毁等区域代价。",
        "comparison_no_segmentation": "当前无法进行完整环境风险路径对比，因为未接入有效语义分割结果。",
        "comparison_with_segmentation": "当前已基于语义分割掩码或自动分割结果进行路径环境风险对比。",
        "top_target_none": "暂无。",
        "top_target_template": "{target_id}（{class_name}），风险分数 {risk_score:.2f}，风险等级 {risk_level}。",
        "top_terp_template": "{target_id}（{class_name}），TERP 分数 {terp_score:.2f}，等级 {terp_level}。",
        "counts": "本次识别目标总数：{total}。\n平民数量：{civilian}；救援人员数量：{rescuer}；动物数量：{animal}。",
        "risk_counts": "高风险目标数量：{high}；中风险目标数量：{medium}。",
        "top_risk_target": "最高风险目标：{text}",
        "top_terp_target": "最高 TERP 目标：{text}",
        "path_header": "系统以起点 S({sx}, {sy}) 为救援出发点，以 {target} 为目标点，基于当前通行代价地图使用 A* 算法生成参考救援路径。",
        "path_stats": "路径长度为 {length} 个像素点，累计路径代价为 {cost:.2f}。",
        "path_call_to_action": "建议救援人员优先沿该路径方向核查目标区域，并结合现场实际道路条件调整。",
        "unknown_target_reason": "检测到未知救援相关目标，建议人工复核。",
        "no_segmentation_path_report": "当前未接入语义分割结果，本报告仅基于目标检测结果生成。",
        "limitations_detail": (
            "当前自动语义分割为可选实验功能，需要本地训练权重文件；未提供权重文件时可上传语义分割掩码完成环境风险融合。\n"
            "当前路径规划为图像平面参考路径，不等同于真实 GPS 路线，也未接入真实道路网络、无人机定位或飞控系统。"
        ),
        "suggestions_items": [
            "优先核查排名靠前目标的位置和周边通行条件。",
            "对 civilian 目标优先进行人工复核和救援资源调度。",
            "对动物目标结合现场资源进行转移或标记。",
            "rescuer 目标默认风险权重较低，主要用于识别现场救援力量分布。",
            "规划现场行动时避开水域、道路阻断区域和严重损毁建筑附近区域。",
        ],
        "fallback_suggestions": [
            "建议更换视角、提高图像清晰度或降低检测置信度阈值后复核。",
        ],
    },
    "en": {
        "title": "灾情感知及影响评估 Disaster Recognition and Rescue Support Report",
        "no_target": "No clear rescue target was detected in the current image.",
        "overview": "1. Recognition Overview",
        "terp": "2. TERP Priority Assessment",
        "risk": "3. Risk Overview",
        "segmentation": "4. Segmentation-Based Environmental Risk",
        "path": "5. Path Planning Recommendation",
        "comparison": "6. Path Planning Comparison",
        "reliability": "7. Path Planning Reliability",
        "damage": "8. Damage Assessment and Rescue Entry",
        "suggestions": "9. Initial Rescue Suggestions",
        "limitations": "10. Current Limitations",
        "segmentation_empty": (
            "No segmentation result is connected yet, so environmental risk and path cost are mainly estimated from the default image-plane assumption. "
            "You can upload a segmentation mask or provide a trained segmentation checkpoint to enable environment-risk fusion."
        ),
        "segmentation_enabled": "This report has incorporated a segmentation mask or automatic segmentation result for environmental-risk fusion.",
        "path_plane_limit": (
            "Current path planning is a reference path on the image plane. It is not a real GPS route and does not connect to a road network, UAV localization, or flight control system."
        ),
        "path_missing": "No valid path could be generated because the path-planning module returned no result.",
        "path_unreachable": "No valid path could be generated because {reason}. Please check the start point, detection results, or upload a more accurate segmentation mask and try again.",
        "path_no_target": "No clear rescue target was detected, so no path can be planned.",
        "path_no_segmentation": "No segmentation result is connected yet, so path planning only uses the default image-plane cost map and does not account for water, blocked roads, or damaged buildings.",
        "path_with_segmentation": "The path planning has incorporated water, blocked-road, and damaged-building costs from the segmentation mask.",
        "comparison_no_segmentation": "A full environment-aware path comparison is not available because no valid segmentation result is connected.",
        "comparison_with_segmentation": "A path-risk comparison has been performed using the segmentation mask or automatic segmentation result.",
        "top_target_none": "None.",
        "top_target_template": "{target_id} ({class_name}), risk score {risk_score:.2f}, risk level {risk_level}.",
        "top_terp_template": "{target_id} ({class_name}), TERP score {terp_score:.2f}, level {terp_level}.",
        "counts": "Total detected targets: {total}.\ncivilian: {civilian}; rescuer: {rescuer}; animal: {animal}.",
        "risk_counts": "High-risk targets: {high}; medium-risk targets: {medium}.",
        "top_risk_target": "Highest-risk target: {text}",
        "top_terp_target": "Highest TERP target: {text}",
        "path_header": "The system uses S({sx}, {sy}) as the rescue start point and {target} as the goal, then generates a reference rescue path with A* on the current traversability cost map.",
        "path_stats": "Path length: {length} pixels. Total path cost: {cost:.2f}.",
        "path_call_to_action": "Rescuers should verify the target region along this route first and adjust it according to real field conditions.",
        "unknown_target_reason": "Detected an unknown rescue-related target. Manual review is recommended.",
        "no_segmentation_path_report": "No segmentation result is connected, so this report is generated only from detection results.",
        "limitations_detail": (
            "Automatic segmentation is optional and experimental. It requires a local trained checkpoint; if no checkpoint is available, you can upload a segmentation mask to enable environment-risk fusion.\n"
            "Current path planning is a reference path on the image plane. It is not a real GPS route and does not connect to a road network, UAV localization, or a flight-control system."
        ),
        "suggestions_items": [
            "Check the highest-ranked targets first and review the surrounding traversability.",
            "Prioritize manual verification and rescue allocation for civilian targets.",
            "Handle animal targets with field resources after human rescue priorities.",
            "Rescuer targets usually carry lower priority and mainly help map rescue-force distribution.",
            "Keep rescue actions away from water, blocked roads, and severely damaged buildings.",
        ],
        "fallback_suggestions": [
            "Try a different viewpoint, clearer image, or a lower detection confidence threshold and review again.",
        ],
    },
}

LEVEL_TEXT = {
    "zh": {"Low": "低", "Medium": "中", "High": "高", "Critical": "极高"},
    "en": {"Low": "Low", "Medium": "Medium", "High": "High", "Critical": "Critical"},
}


def _lang(language):
    return "en" if str(language).lower().startswith("en") else "zh"


def _t(language, key):
    return TEXT[_lang(language)][key]


def _level_text(level, language):
    return LEVEL_TEXT[_lang(language)].get(level, level)


def _class_display_name(class_name, language):
    if _lang(language) == "en":
        return class_name
    return TARGET_CLASS_DISPLAY_NAMES.get(class_name, CLASS_DISPLAY_NAMES.get(class_name, class_name))


def _format_suggestions(language, items):
    return "\n".join(f"- {item}" for item in items)


def _percent(summary, class_name):
    return float((summary or {}).get(class_name, 0.0)) * 100.0


def _zh_internal_message(message):
    if not message:
        return ""
    replacements = {
        "A* path planning succeeded.": "A* 路径规划成功。",
        "No target is available for path planning.": "当前没有可用于路径规划的目标。",
        "Path comparison is limited because one or both paths were not found.": "由于一条或两条路径未能生成，路径对比结果有限。",
        "No segmentation mask, path comparison is limited; baseline and risk-aware routes use equivalent assumptions.": "当前没有语义分割掩码，路径对比仅作演示，普通路径和风险感知路径使用近似假设。",
        "Risk-aware path reduces high-risk exposure compared with baseline.": "与普通 A* 相比，风险感知 A* 降低了高风险区域暴露。",
        "Risk-aware path does not reduce high-risk exposure in this case.": "本案例中风险感知 A* 未明显降低高风险区域暴露。",
    }
    text = str(message)
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("baseline", "普通 A*")
    text = text.replace("Baseline", "普通 A*")
    text = text.replace("Risk-Aware", "风险感知")
    text = text.replace("risk-aware", "风险感知")
    text = text.replace("segmentation mask", "语义分割掩码")
    text = text.replace("automatic segmentation", "自动语义分割")
    return text


def _segmentation_source_section(segmentation_source_metadata, language):
    if not segmentation_source_metadata:
        return ""
    metadata = segmentation_source_metadata
    if isinstance(segmentation_source_metadata, str):
        metadata = build_segmentation_source_metadata(segmentation_source_metadata)
    lines = [
        "分割来源：{}".format(metadata.get("source_label", "Unknown")) if _lang(language) == "zh" else "Segmentation source: {}".format(metadata.get("source_label", "Unknown")),
        "是否为模型自动预测：{}".format("是" if metadata.get("is_model_prediction") else "否") if _lang(language) == "zh" else "Is model prediction: {}".format("Yes" if metadata.get("is_model_prediction") else "No"),
        "Checkpoint 路径：{}".format(metadata.get("checkpoint_path") or "未使用") if _lang(language) == "zh" else "Checkpoint path: {}".format(metadata.get("checkpoint_path") or "not used"),
        "真实性说明：{}".format(metadata.get("truthfulness_note", "")) if _lang(language) == "zh" else "Truthfulness note: {}".format(metadata.get("truthfulness_note", "")),
    ]
    return "\n".join(lines)


def _segmentation_section(segmentation_summary, ranked_targets, language, segmentation_source_metadata=None):
    if not segmentation_summary:
        section = f"{_t(language, 'segmentation')}\n{_t(language, 'segmentation_empty')}\n"
        source_text = _segmentation_source_section(segmentation_source_metadata, language)
        return section + (source_text + "\n" if source_text else "")

    water_ratio = _percent(segmentation_summary, "water") + _percent(segmentation_summary, "pool")
    blocked_road_ratio = _percent(segmentation_summary, "road_blocked")
    damaged_building_ratio = (
        _percent(segmentation_summary, "minor_damage")
        + _percent(segmentation_summary, "major_damage")
        + _percent(segmentation_summary, "destroyed_building")
    )

    top_environment = "None"
    if ranked_targets:
        top_target = ranked_targets[0]
        dominant_class = top_target.get("environment", "unknown")
        dominant_display = (
            CLASS_DISPLAY_NAMES_EN.get(dominant_class, dominant_class)
            if _lang(language) == "en"
            else CLASS_DISPLAY_NAMES.get(dominant_class, dominant_class)
        )
        top_environment = (
            f"{top_target['target_id']} {dominant_display}, "
            f"environment risk bonus {top_target.get('environment_score', 0.0):.2f}."
            if _lang(language) == "en"
            else (
                f"{top_target['target_id']} 周边主导环境为 "
                f"{dominant_display}，"
                f"环境风险加分 {top_target.get('environment_score', 0.0):.2f}。"
            )
        )

    section = (
        f"{_t(language, 'segmentation')}\n"
        f"{_t(language, 'segmentation_enabled')}\n"
        + (
            f"Water / pool ratio: {water_ratio:.1f}%.\n"
            f"Blocked-road ratio: {blocked_road_ratio:.1f}%.\n"
            f"Damaged-building ratio: {damaged_building_ratio:.1f}%.\n"
            f"Top target environment: {top_environment}\n"
            if _lang(language) == "en"
            else (
                f"水域/积水面积占比：{water_ratio:.1f}%。\n"
                f"道路阻断区域占比：{blocked_road_ratio:.1f}%。\n"
                f"建筑损毁区域占比：{damaged_building_ratio:.1f}%。\n"
                f"最高风险目标环境：{top_environment}\n"
            )
        )
    )
    source_text = _segmentation_source_section(segmentation_source_metadata, language)
    if source_text:
        section += "\n" + source_text + "\n"
    return section


def _terp_section(terp_rankings, language):
    if not terp_rankings:
        return (
            f"{_t(language, 'terp')}\n"
            + (
                "No TERP ranking was generated. This may happen when no clear target is detected or the path inputs are insufficient.\n"
                if _lang(language) == "en"
                else "当前未生成 TERP 排名，可能是未检测到明确目标或路径评估输入不足。\n"
            )
        )

    top = terp_rankings[0]
    top_text = _t(language, "top_terp_template").format(
        target_id=top["target_id"],
        class_name=CLASS_DISPLAY_NAMES_EN.get(top["class_name"], top["class_name"])
        if _lang(language) == "en"
        else _class_display_name(top["class_name"], language),
        terp_score=top["terp_score"],
        terp_level=_level_text(top["terp_level"], language),
    )
    if _lang(language) == "en":
        return (
            f"{_t(language, 'terp')}\n"
            "TERP combines target type, detection confidence, target scale, environmental risk, and route accessibility.\n"
            f"Highest TERP target: {top_text}\n"
            f"Target risk component: {top['target_score']:.2f}; environment component: {top['environment_score']:.2f}; "
            f"accessibility component: {top['accessibility_score']:.2f}.\n"
            f"Reason: {top['reason']}\n"
        )
    return (
        f"{_t(language, 'terp')}\n"
        "TERP 将目标类别、检测置信度、目标尺度、环境风险和路径可达性融合为综合救援优先级。\n"
        f"最高 TERP 目标：{top_text}\n"
        f"目标风险项：{top['target_score']:.2f}；环境风险项：{top['environment_score']:.2f}；"
        f"路径可达性风险项：{top['accessibility_score']:.2f}。\n"
        f"排序原因：{top['reason']}\n"
    )


def _path_section(segmentation_summary, path_result, ranked_targets, language):
    if not ranked_targets:
        return f"{_t(language, 'path')}\n{_t(language, 'path_no_target')}\n"

    if not path_result:
        return f"{_t(language, 'path')}\n{_t(language, 'path_missing')}\n"

    if not path_result.get("found"):
        reason = path_result.get("message", "Unknown")
        if _lang(language) == "zh":
            reason = _zh_internal_message(reason)
        return (
            f"{_t(language, 'path')}\n"
            + (
                f"No valid path could be generated because {reason}.\n"
                "Please check the start point, detection results, or upload a more accurate segmentation mask and try again.\n"
                if _lang(language) == "en"
                else f"当前未能生成有效路径，原因：{reason}。\n建议检查起点位置、目标检测结果或上传更准确的语义分割掩码后重试。\n"
            )
        )

    start = path_result.get("start", [0, 0])
    top_target = ranked_targets[0] if ranked_targets else None
    top_target_text = (
        f"{top_target['target_id']} ({top_target['class_name']})"
        if _lang(language) == "en"
        else f"{top_target['target_id']}（{_class_display_name(top_target['class_name'], language)}）"
    )

    env_text = _t(language, "path_with_segmentation") if segmentation_summary else _t(language, "path_no_segmentation")

    return (
        f"{_t(language, 'path')}\n"
        + _t(language, "path_header").format(
            sx=start[0],
            sy=start[1],
            target=top_target_text,
        )
        + "\n"
        + _t(language, "path_stats").format(
            length=path_result.get("path_length", 0),
            cost=path_result.get("total_cost", 0.0),
        )
        + "\n"
        + f"{env_text}\n"
        + f"{_t(language, 'path_call_to_action')}\n"
    )


def _path_comparison_section(path_comparison, segmentation_summary, language):
    if not path_comparison:
        return (
            f"{_t(language, 'comparison')}\n"
            + (
                "No baseline vs. Risk-Aware A* comparison is available yet.\n"
                if _lang(language) == "en"
                else "当前未生成普通 A* 与风险感知 A* 对比结果。\n"
            )
        )

    prefix = _t(language, "comparison_with_segmentation") if segmentation_summary else _t(language, "comparison_no_segmentation")
    return (
        f"{_t(language, 'comparison')}\n"
        f"{prefix}\n"
        + (
            f"Baseline A*: length {path_comparison.get('baseline_length', 0)}, "
            f"cost {path_comparison.get('baseline_cost', 0.0):.2f}, "
            f"high-risk ratio {path_comparison.get('baseline_environment_risk', 0.0) * 100:.2f}%.\n"
            f"Risk-Aware A*: length {path_comparison.get('risk_aware_length', 0)}, "
            f"cost {path_comparison.get('risk_aware_cost', 0.0):.2f}, "
            f"high-risk ratio {path_comparison.get('risk_aware_environment_risk', 0.0) * 100:.2f}%.\n"
            f"Path risk reduction: {path_comparison.get('risk_reduction', 0.0) * 100:.2f}%.\n"
            f"Explanation: {path_comparison.get('message', '')}\n"
            if _lang(language) == "en"
            else (
                f"普通 A*：长度 {path_comparison.get('baseline_length', 0)}，"
                f"累计代价 {path_comparison.get('baseline_cost', 0.0):.2f}，"
                f"高风险区域比例 {path_comparison.get('baseline_environment_risk', 0.0) * 100:.2f}%。\n"
                f"风险感知 A*：长度 {path_comparison.get('risk_aware_length', 0)}，"
                f"累计代价 {path_comparison.get('risk_aware_cost', 0.0):.2f}，"
                f"高风险区域比例 {path_comparison.get('risk_aware_environment_risk', 0.0) * 100:.2f}%。\n"
                f"路径环境风险降低：{path_comparison.get('risk_reduction', 0.0) * 100:.2f}%。\n"
                f"说明：{_zh_internal_message(path_comparison.get('message', ''))}\n"
            )
        )
    )


def _path_reliability_section(damage_assessment, language):
    reliability = (damage_assessment or {}).get("path_planning_reliability", {})
    if not reliability:
        return (
            f"{_t(language, 'reliability')}\n"
            + (
                "No path-planning reliability metadata is available.\n"
                if _lang(language) == "en"
                else "当前没有路径规划可靠性元数据。\n"
            )
        )

    if _lang(language) == "en":
        return (
            f"{_t(language, 'reliability')}\n"
            "Path type: image-plane reference path, not real GPS navigation.\n"
            "Scene Mode method: rule-based assessment, not a trained scene-classification model.\n"
            f"Reliability level: {reliability.get('reliability_level', 'unknown')}.\n"
            f"Mask source: {reliability.get('mask_source', 'unknown')}.\n"
            f"Mask dependency: {'yes' if reliability.get('mask_dependency') else 'no'}.\n"
            f"Mask risk note: {reliability.get('mask_risk_note', '')}\n"
            f"Human review required: {'yes' if reliability.get('human_review_required') else 'no'}.\n"
            f"Reliability note: {reliability.get('reliability_note', '')}\n"
        )

    return (
        f"{_t(language, 'reliability')}\n"
        "当前路径为图像平面参考路径，不是真实 GPS 导航。\n"
        "Scene Mode 为基于目标框面积、segmentation mask、Road-Clear 面积比例和边界道路连通性的规则判断，不是训练出的场景分类模型。\n"
        f"当前可靠性等级：{reliability.get('reliability_level', 'unknown')}。\n"
        f"Mask 来源：{reliability.get('mask_source', 'unknown')}。\n"
        f"Mask 依赖：{'是' if reliability.get('mask_dependency') else '否'}。\n"
        f"Mask 风险说明：{reliability.get('mask_risk_note', '')}\n"
        f"是否建议人工复核：{'是' if reliability.get('human_review_required') else '否'}。\n"
        f"可靠性说明：{reliability.get('reliability_note', '')}\n"
    )


def _damage_assessment_section(damage_assessment, language, segmentation_source_metadata=None):
    if not damage_assessment:
        return (
            f"{_t(language, 'damage')}\n"
            + (
                "No damage-assessment result is available. Path planning is kept conservative when scene context is insufficient.\n"
                if _lang(language) == "en"
                else "当前没有灾损评估结果。当场景上下文不足时，系统会保守关闭路径规划。\n"
            )
        )

    building = damage_assessment.get("building_damage", {})
    road = damage_assessment.get("road_stats", {})
    entry = damage_assessment.get("entry", {})
    scene_mode = damage_assessment.get("scene_mode", "Unknown")
    scene_reason = damage_assessment.get("scene_mode_reason", "")
    path_enabled = bool(damage_assessment.get("path_planning_enabled"))
    path_reason = damage_assessment.get("path_planning_reason", "")
    path_gate = damage_assessment.get("path_planning_gate", {})
    force_path = bool(damage_assessment.get("force_path_planning") or path_gate.get("force_path_planning"))
    start_source = damage_assessment.get("path_start_source") or path_gate.get("start_source", "")
    source_text = _segmentation_source_section(segmentation_source_metadata, language)

    if _lang(language) == "en":
        entry_text = (
            f"Rescue entry found at ({entry.get('entry_point_x')}, {entry.get('entry_point_y')}). "
            "The entry is selected from a Road-Clear connected component near the image boundary, not from an arbitrary corner."
            if entry.get("entry_found")
            else f"No reliable rescue entry was generated. Reason: {entry.get('entry_reason') or path_reason}"
        )
        section = (
            f"{_t(language, 'damage')}\n"
            f"Overall damage level: {damage_assessment.get('overall_damage_level', 'Unknown')}.\n"
            f"Building damage areas: no damage {building.get('no_damage_area', 0)} px; "
            f"medium damage {building.get('medium_damage_area', 0)} px; "
            f"major damage {building.get('major_damage_area', 0)} px; "
            f"total destruction {building.get('total_destruction_area', 0)} px.\n"
            f"Road state: clear-road ratio {road.get('road_clear_ratio', 0.0) * 100:.2f}%; "
            f"blocked-road ratio {road.get('road_blocked_ratio', 0.0) * 100:.2f}%.\n"
            f"Scene mode: {scene_mode}. Reason: {scene_reason}\n"
            f"Path planning enabled: {'yes' if path_enabled else 'no'}. {path_reason}\n"
            f"Path start source: {start_source or 'not enabled'}.\n"
            f"{entry_text}\n"
        )
        if force_path:
            section += "Warning: this route was generated in force-debug mode and has limited reliability.\n"
        if source_text:
            section += f"{source_text}\n"
        return section

    entry_text = (
        f"已找到救援入口：({entry.get('entry_point_x')}, {entry.get('entry_point_y')})。"
        "入口来自靠近图像边界的 Road-Clear 可通行道路连通区域，不是手动任意角落起点。"
        if entry.get("entry_found")
        else f"未生成可靠救援入口。原因：{entry.get('entry_reason') or path_reason}"
    )
    section = (
        f"{_t(language, 'damage')}\n"
        f"整体灾损等级：{damage_assessment.get('overall_damage_level', 'Unknown')}。\n"
        f"建筑损毁统计：无损 {building.get('no_damage_area', 0)} 像素；"
        f"中度损毁 {building.get('medium_damage_area', 0)} 像素；"
        f"严重损毁 {building.get('major_damage_area', 0)} 像素；"
        f"完全毁坏 {building.get('total_destruction_area', 0)} 像素。\n"
        f"道路状态：可通行道路比例 {road.get('road_clear_ratio', 0.0) * 100:.2f}%；"
        f"阻断道路比例 {road.get('road_blocked_ratio', 0.0) * 100:.2f}%。\n"
        f"场景模式：{scene_mode}。原因：{scene_reason}\n"
        f"路径规划启用：{'是' if path_enabled else '否'}。{path_reason}\n"
        f"路径起点来源：{start_source or '未启用'}。\n"
        f"{entry_text}\n"
    )
    if force_path:
        section += "警告：该路径为强制调试模式生成，可靠性有限。\n"
    if source_text:
        section += f"{source_text}\n"
    return section


def generate_report(targets, ranked_targets, segmentation_summary=None, path_result=None, terp_rankings=None, path_comparison=None, damage_assessment=None, segmentation_source_metadata=None, language="zh"):
    segmentation_summary = segmentation_summary or {}
    terp_rankings = terp_rankings or []
    language = _lang(language)

    if not targets:
        segmentation_text = _segmentation_section(segmentation_summary, [], language, segmentation_source_metadata)
        terp_text = _terp_section([], language)
        path_text = _path_section(segmentation_summary, path_result, [], language)
        path_comparison_text = _path_comparison_section(path_comparison, segmentation_summary, language)
        reliability_text = _path_reliability_section(damage_assessment, language)
        damage_text = _damage_assessment_section(damage_assessment, language, segmentation_source_metadata)
        return (
            f"{_t(language, 'title')}\n\n"
            + f"{_t(language, 'no_target')}\n\n"
            + f"{segmentation_text}\n"
            + f"{terp_text}\n"
            + f"{path_text}\n"
            + f"{path_comparison_text}\n"
            + f"{reliability_text}\n"
            + f"{damage_text}\n"
            + (
                f"Initial recommendation: {'; '.join(_t(language, 'fallback_suggestions'))}\n\n"
                if language == "en"
                else f"初步建议：{_format_suggestions(language, _t(language, 'fallback_suggestions'))}\n\n"
            )
            + f"{_t(language, 'limitations')}\n{_t(language, 'limitations_detail')}"
        )

    civilian_count = sum(1 for target in targets if target["class_name"] == "civilian")
    rescuer_count = sum(1 for target in targets if target["class_name"] == "rescuer")
    animal_count = sum(1 for target in targets if target["class_name"] in ANIMAL_CLASSES)
    top_target = ranked_targets[0] if ranked_targets else None

    if top_target:
        class_display = (
            CLASS_DISPLAY_NAMES_EN.get(top_target["class_name"], top_target["class_name"])
            if _lang(language) == "en"
            else _class_display_name(top_target["class_name"], language)
        )
        top_target_text = _t(language, "top_target_template").format(
            target_id=top_target["target_id"],
            class_name=class_display,
            risk_score=top_target["risk_score"],
            risk_level=_level_text(top_target["risk_level"], language),
        )
    else:
        top_target_text = _t(language, "top_target_none")

    high_risk_count = sum(1 for target in ranked_targets if target["risk_level"] == "High")
    medium_risk_count = sum(1 for target in ranked_targets if target["risk_level"] == "Medium")

    if language == "en":
        suggestions = [
            _t(language, "suggestions_items")[0],
            _t(language, "suggestions_items")[1],
            _t(language, "suggestions_items")[2],
            _t(language, "suggestions_items")[3],
        ]
        if segmentation_summary:
            suggestions.append(_t(language, "suggestions_items")[4])
        else:
            suggestions.append(
                "No segmentation result is connected yet. Consider adding a mask or training checkpoint to review environment risk."
            )
    else:
        suggestions = list(_t(language, "suggestions_items")[:4])
        if segmentation_summary:
            suggestions.append(_t(language, "suggestions_items")[4])
        else:
            suggestions.append("当前未接入语义分割结果，建议补充分割掩码或训练权重文件后复核环境风险。")

    segmentation_text = _segmentation_section(segmentation_summary, ranked_targets, language, segmentation_source_metadata)
    terp_text = _terp_section(terp_rankings, language)
    path_text = _path_section(segmentation_summary, path_result, ranked_targets, language)
    path_comparison_text = _path_comparison_section(path_comparison, segmentation_summary, language)
    reliability_text = _path_reliability_section(damage_assessment, language)
    damage_text = _damage_assessment_section(damage_assessment, language, segmentation_source_metadata)

    return (
        f"{_t(language, 'title')}\n\n"
        f"{_t(language, 'overview')}\n"
        + _t(language, "counts").format(total=len(targets), civilian=civilian_count, rescuer=rescuer_count, animal=animal_count)
        + "\n\n"
        + f"{terp_text}\n"
        + f"{_t(language, 'risk')}\n"
        + _t(language, "risk_counts").format(high=high_risk_count, medium=medium_risk_count)
        + "\n"
        + f"{_t(language, 'top_risk_target').format(text=top_target_text)}\n\n"
        + f"{segmentation_text}\n"
        + f"{path_text}\n"
        + f"{path_comparison_text}\n"
        + f"{reliability_text}\n"
        + f"{damage_text}\n"
        + f"{_t(language, 'suggestions')}\n"
        + _format_suggestions(language, suggestions)
        + "\n\n"
        + f"{_t(language, 'limitations')}\n"
        + _t(language, "limitations_detail")
    )
