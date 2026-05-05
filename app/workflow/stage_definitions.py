"""Nine-stage rescue workflow definitions for AeroRescue-AI.

These definitions describe what each stage can truthfully represent in a
low-altitude disaster response workflow. They intentionally keep preview,
simulated, uploaded, image-plane, and AI-assisted outputs separate from real
surveying, real temperature measurement, field confirmation, GPS navigation,
and final rescue conclusions.
"""

from __future__ import annotations

from copy import deepcopy


RESCUE_WORKFLOW_STAGES = [
    {
        "stage_id": "S1",
        "stage_key": "global_mapping",
        "stage_name_zh": "高空建图",
        "stage_name_en": "Global Mapping",
        "real_action": "高空大范围巡航",
        "uav_layer": "high_altitude_mapping",
        "input_data": ["重叠 RGB 图", "可选 GPS/EXIF", "可选航测图像集"],
        "system_modules": ["orthomosaic_preview", "real_odm_optional", "reconstruction_preview"],
        "outputs": ["灾区总地图", "正射预览", "ODM 输出记录", "三维预览记录"],
        "truthfulness_boundary": "Fast Preview / OpenCV Stitch / ORB Homography is not a real ODM georeferenced orthomosaic.",
        "required_human_review": True,
    },
    {
        "stage_id": "S2",
        "stage_key": "macro_analysis",
        "stage_name_zh": "宏观灾情分析",
        "stage_name_en": "Macro Disaster Analysis",
        "real_action": "在总地图上看整体灾情",
        "uav_layer": "map_level_analysis",
        "input_data": ["地图/正射图/正射预览图", "可选语义分割 mask"],
        "system_modules": ["macro_segmentation", "environment_risk_analysis", "damage_assessment"],
        "outputs": ["积水区", "废墟区", "可通行区", "道路阻断区", "宏观风险摘要"],
        "truthfulness_boundary": "Uploaded/Demo Mask is not automatic model segmentation.",
        "required_human_review": True,
    },
    {
        "stage_id": "S3",
        "stage_key": "area_tasking",
        "stage_name_zh": "重点区域划分",
        "stage_name_en": "Area Tasking",
        "real_action": "选定重点小区域",
        "uav_layer": "task_area_selection",
        "input_data": ["地图分区", "宏观风险区域", "热点区域"],
        "system_modules": ["area_tasking", "hotspot_marking"],
        "outputs": ["A区", "B区", "C区", "重点巡查区域列表"],
        "truthfulness_boundary": "Area tasking is an auxiliary recommendation and requires commander review.",
        "required_human_review": True,
    },
    {
        "stage_id": "S4",
        "stage_key": "local_recon",
        "stage_name_zh": "中高空局部精查",
        "stage_name_en": "Local Reconnaissance",
        "real_action": "中高空精查某个区域",
        "uav_layer": "mid_altitude_rgb_recon",
        "input_data": ["局部 RGB 图", "局部视频", "重点区域 ID"],
        "system_modules": ["object_detection", "transformer_detection_optional", "local_segmentation"],
        "outputs": ["疑似人员", "车辆", "道路", "建筑", "废墟", "局部环境结果"],
        "truthfulness_boundary": "AI detections are candidates and not confirmed civilians or confirmed field findings.",
        "required_human_review": True,
    },
    {
        "stage_id": "S5",
        "stage_key": "target_verification",
        "stage_name_zh": "低空目标复核",
        "stage_name_en": "Target Verification",
        "real_action": "低空靠近疑似目标",
        "uav_layer": "low_altitude_close_inspection",
        "input_data": ["近距离 RGB 图", "候选目标框", "目标区域截图"],
        "system_modules": ["target_review", "evidence_crop", "candidate_status"],
        "outputs": ["候选目标截图", "目标周边截图", "复核状态"],
        "truthfulness_boundary": "Target verification still requires human review; visual evidence alone is not a final rescue conclusion.",
        "required_human_review": True,
    },
    {
        "stage_id": "S6",
        "stage_key": "thermal_check",
        "stage_name_zh": "热红外辅助复查",
        "stage_name_en": "Thermal Check",
        "real_action": "热成像复查",
        "uav_layer": "low_or_mid_altitude_thermal",
        "input_data": ["热红外图", "双光图", "候选目标 ID"],
        "system_modules": ["simulated_thermal", "radiometric_thermal_optional", "rgb_thermal_support"],
        "outputs": ["热异常区域", "热源支持证据", "temperature_matrix 状态"],
        "truthfulness_boundary": "Simulated Thermal is not real temperature measurement; RGB images cannot provide real temperature_matrix.",
        "required_human_review": True,
    },
    {
        "stage_id": "S7",
        "stage_key": "decision_fusion",
        "stage_name_zh": "决策融合",
        "stage_name_en": "Decision Fusion",
        "real_action": "综合判断",
        "uav_layer": "command_center_fusion",
        "input_data": ["多模态结果", "目标检测结果", "环境风险", "热红外结果", "路径可达性"],
        "system_modules": ["EC-TERP", "priority_ranking", "risk_fusion"],
        "outputs": ["救援优先级排序", "候选目标优先级", "解释原因"],
        "truthfulness_boundary": "EC-TERP provides decision-support priority ranking and does not replace rescue command judgment.",
        "required_human_review": True,
    },
    {
        "stage_id": "S8",
        "stage_key": "rescue_recommendation",
        "stage_name_zh": "路径与任务建议",
        "stage_name_en": "Rescue Recommendation",
        "real_action": "给救援队建议",
        "uav_layer": "command_to_rescue_team",
        "input_data": ["地图", "风险区", "候选目标", "起点", "可通行区域"],
        "system_modules": ["image_plane_path_planning", "risk_aware_astar", "task_suggestion"],
        "outputs": ["推荐路线", "危险区", "绕行建议", "救援队任务建议"],
        "truthfulness_boundary": "Image-plane path is not GPS navigation or an autonomous rescue route.",
        "required_human_review": True,
    },
    {
        "stage_id": "S9",
        "stage_key": "evidence_report",
        "stage_name_zh": "证据链与报告",
        "stage_name_en": "Evidence and Report",
        "real_action": "留痕复盘",
        "uav_layer": "post_mission_review",
        "input_data": ["所有阶段中间结果", "人工复核记录", "证据链"],
        "system_modules": ["Evidence Ledger", "Final Report", "LLM Report Assistant optional"],
        "outputs": ["报告", "证据链", "任务摘要", "真实性边界摘要"],
        "truthfulness_boundary": "Final Report is an AI-assisted decision-support report and requires human review.",
        "required_human_review": True,
    },
]


def get_stage_definition(stage_key):
    """Return a copy of the stage definition for stage_key."""
    for stage in RESCUE_WORKFLOW_STAGES:
        if stage["stage_key"] == stage_key:
            return deepcopy(stage)
    raise KeyError(f"Unknown rescue workflow stage: {stage_key}")


def list_stage_keys():
    """Return workflow stage keys in mission order."""
    return [stage["stage_key"] for stage in RESCUE_WORKFLOW_STAGES]


def build_default_stage_state():
    """Build default per-stage runtime state from the static definitions."""
    stages = {}
    for index, definition in enumerate(RESCUE_WORKFLOW_STAGES):
        stage_key = definition["stage_key"]
        stages[stage_key] = {
            "stage_id": definition["stage_id"],
            "stage_key": stage_key,
            "status": "ready" if index == 0 else "pending",
            "started_at": "",
            "completed_at": "",
            "output_ref": "",
            "result_type": "",
            "evidence_ids": [],
            "human_review_required": bool(definition["required_human_review"]),
            "truthfulness_boundary": definition["truthfulness_boundary"],
        }
    return stages

