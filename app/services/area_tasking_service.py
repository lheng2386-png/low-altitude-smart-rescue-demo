"""Build commander-review area tasking recommendations from macro zones."""

from __future__ import annotations


RISK_PRIORITY = {
    "Critical": 4,
    "High": 3,
    "Medium": 2,
    "Low": 1,
    "Unknown": 0,
}

AREA_IDS = ["A", "B", "C", "D", "E"]

ZONE_REASONS = {
    "flood_or_water_zone": "该区域存在较高比例积水，建议派出中高空 RGB 无人机进行局部精查。",
    "blocked_road_zone": "该区域存在道路阻断迹象，建议重点核查通行条件和绕行可能性。",
    "damaged_building_zone": "该区域存在建筑损毁迹象，建议对疑似被困点和结构风险进行局部精查。",
    "accessible_road_zone": "该区域可能包含可通行道路，可作为救援接近路线的候选区域进行复核。",
    "vehicle_zone": "该区域出现车辆相关线索，建议复核是否存在人员聚集或道路占用。",
    "vegetation_or_obstacle_zone": "该区域存在植被或障碍物线索，建议复核对通行和视线的影响。",
}


def build_manual_area_task(
    area_id="A",
    reason="User provided local RGB imagery without macro map context.",
):
    """Build a manual local-recon area task without claiming confirmed disaster severity."""
    area_id = str(area_id or "A").strip() or "A"
    return {
        "area_id": area_id,
        "area_name": f"人工指定重点巡查区 {area_id}",
        "source_zone_id": "",
        "zone_type": "manual_local_recon_area",
        "risk_level": "Unknown",
        "reason": "用户已提供局部 RGB 图像，本区域作为人工指定局部精查区域；缺少全局地图和宏观风险上下文。",
        "recommended_next_action": "run_local_recon_detection",
        "required_followup_stage": "local_recon",
        "human_review_required": True,
        "truthfulness_note": "Manual area tasking does not imply confirmed disaster severity.",
        "source_reason": str(reason or ""),
    }


def _risk_rank(zone):
    risk_level = str(zone.get("risk_level") or "Unknown")
    area_percent = float(zone.get("area_percent") or 0.0)
    return (RISK_PRIORITY.get(risk_level, 0), area_percent)


def build_area_tasks_from_macro_zones(macro_zones, max_areas=3):
    """Return up to max_areas tasking recommendations from macro risk zones."""
    zones = list(macro_zones or [])
    if not zones:
        return [
            {
                "area_id": "A",
                "area_name": "重点巡查区 A",
                "source_zone_id": "",
                "zone_type": "manual_selection_required",
                "risk_level": "Unknown",
                "reason": "未获得可靠宏观分割区域，建议由指挥人员手动选择重点巡查区域。",
                "recommended_next_action": "manual_area_selection_required",
                "required_followup_stage": "local_recon",
                "human_review_required": True,
            }
        ]

    max_areas = max(1, min(int(max_areas or 3), len(AREA_IDS)))
    ranked_zones = sorted(zones, key=_risk_rank, reverse=True)
    tasks = []
    for index, zone in enumerate(ranked_zones[:max_areas]):
        area_id = AREA_IDS[index]
        zone_type = str(zone.get("zone_type") or "unknown_zone")
        tasks.append(
            {
                "area_id": area_id,
                "area_name": f"重点巡查区 {area_id}",
                "source_zone_id": str(zone.get("zone_id") or ""),
                "zone_type": zone_type,
                "risk_level": str(zone.get("risk_level") or "Unknown"),
                "reason": ZONE_REASONS.get(
                    zone_type,
                    "该区域由宏观分析识别为需要进一步关注的候选区域，建议人工复核后安排局部精查。",
                ),
                "recommended_next_action": "dispatch_mid_altitude_rgb_recon",
                "required_followup_stage": "local_recon",
                "human_review_required": True,
            }
        )
    return tasks
