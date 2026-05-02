HIGH_RISK_CLASSES = {
    "water",
    "pool",
    "road_blocked",
    "major_damage",
    "destroyed_building",
}

MEDIUM_RISK_CLASSES = {
    "minor_damage",
    "tree",
    "vehicle",
}

LOW_RISK_CLASSES = {
    "background",
    "no_damage_building",
    "road_clear",
}

ENVIRONMENT_RISK_SCORES = {
    "water": 28.0,
    "pool": 22.0,
    "road_blocked": 24.0,
    "major_damage": 26.0,
    "destroyed_building": 30.0,
    "minor_damage": 14.0,
    "tree": 12.0,
    "vehicle": 10.0,
    "road_clear": 3.0,
    "no_damage_building": 4.0,
    "background": 0.0,
}

CLASS_DISPLAY_NAMES = {
    "background": "背景",
    "water": "水域",
    "no_damage_building": "无损建筑",
    "minor_damage": "轻微损毁建筑",
    "major_damage": "严重损毁建筑",
    "destroyed_building": "完全毁坏建筑",
    "vehicle": "车辆",
    "road_clear": "可通行道路",
    "road_blocked": "道路阻断",
    "tree": "树木",
    "pool": "水池/积水",
}


def get_environment_risk_score(class_name):
    return ENVIRONMENT_RISK_SCORES.get(class_name, 0.0)


def get_environment_risk_level(class_name):
    if class_name in HIGH_RISK_CLASSES:
        return "High"
    if class_name in MEDIUM_RISK_CLASSES:
        return "Medium"
    return "Low"


def describe_environment_classes(class_names):
    if not class_names:
        return "未发现明显环境风险"

    labels = [CLASS_DISPLAY_NAMES.get(class_name, class_name) for class_name in class_names]
    return "、".join(labels)
