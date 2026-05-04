"""Registry of external decision-layer reference projects.

This module only records reference value and adaptation boundaries. It does not
claim to integrate the full external systems.
"""


class DecisionReferenceRegistryError(Exception):
    pass


DECISION_REFERENCE_PROJECTS = {
    "sarenv_search_planning": {
        "display_name": "SAREnv UAV Search and Rescue Evaluation Framework",
        "source": "namurproject/SAREnv",
        "repository": "https://github.com/namurproject/SAREnv",
        "reference_type": "search_probability_and_path_evaluation",
        "status": "reference_design_not_integrated",
        "related_decision_layer": [
            "search probability heatmap",
            "coverage path evaluation",
            "probabilistic search",
            "multi-UAV search planning",
        ],
        "possible_local_adaptation": [
            "image-plane probability heatmap",
            "search priority map",
            "coverage score",
            "missed-risk area score",
        ],
        "active_runtime": False,
        "truthfulness_note": "SAREnv is used as a reference for UAV search-and-rescue probability maps and search path evaluation. AeroRescue-AI does not currently run the full SAREnv geospatial framework.",
    },
    "skai_building_damage": {
        "display_name": "Google Research SKAI Building Damage Assessment",
        "source": "google-research/skai",
        "repository": "https://github.com/google-research/skai",
        "reference_type": "building_damage_assessment",
        "status": "reference_design_not_integrated",
        "related_decision_layer": [
            "building damage risk",
            "collapsed building priority",
            "shelter/trapped-person risk",
            "damage-aware report generation",
        ],
        "possible_local_adaptation": [
            "segmentation-based building damage score",
            "major_damage / destroyed_building risk weighting",
            "damage severity summary",
        ],
        "active_runtime": False,
        "truthfulness_note": "SKAI is used as a reference for building damage assessment from aerial disaster imagery. AeroRescue-AI does not claim to run SKAI models unless a real SKAI pipeline and outputs are provided.",
    },
    "inasafe_impact_modeling": {
        "display_name": "InaSAFE Disaster Impact Modeling Reference",
        "source": "inasafe/inasafe",
        "repository": "https://github.com/inasafe/inasafe",
        "reference_type": "gis_impact_modeling",
        "status": "reference_design_not_integrated",
        "related_decision_layer": [
            "hazard impact scenario",
            "exposure and vulnerability",
            "risk calculation",
            "preparedness and response planning",
        ],
        "possible_local_adaptation": [
            "image-plane impact score",
            "exposure-like target count",
            "hazard area ratio",
            "decision explanation template",
        ],
        "active_runtime": False,
        "truthfulness_note": "InaSAFE is a QGIS disaster impact assessment reference. AeroRescue-AI currently uses lightweight image-plane risk scoring, not full GIS/QGIS impact modeling.",
    },
    "fields2cover_coverage_planning": {
        "display_name": "Fields2Cover Coverage Path Planning Reference",
        "source": "Fields2Cover/Fields2Cover",
        "repository": "https://github.com/Fields2Cover/Fields2Cover",
        "reference_type": "coverage_path_planning",
        "status": "reference_design_not_integrated",
        "related_decision_layer": [
            "coverage path planning",
            "area scanning",
            "coverage route ordering",
            "obstacle-aware coverage",
        ],
        "possible_local_adaptation": [
            "image-plane coverage path preview",
            "grid-based sweep coverage",
            "unsearched area ratio",
            "coverage completeness score",
        ],
        "active_runtime": False,
        "truthfulness_note": "Fields2Cover is a robust coverage path planning library, mainly for offline vehicle coverage. AeroRescue-AI only references its coverage planning idea and does not currently link the C++ library.",
    },
    "pythonrobotics_path_algorithms": {
        "display_name": "PythonRobotics Path Planning Algorithm Reference",
        "source": "AtsushiSakai/PythonRobotics",
        "repository": "https://github.com/AtsushiSakai/PythonRobotics",
        "reference_type": "path_planning_algorithm_reference",
        "status": "lightweight_algorithm_reference",
        "related_decision_layer": [
            "A*",
            "Dijkstra",
            "D*",
            "D* Lite",
            "RRT",
            "RRT*",
            "potential field",
            "coverage path planning",
        ],
        "possible_local_adaptation": [
            "A* comparison",
            "Dijkstra baseline",
            "D* / D* Lite future dynamic replanning",
            "RRT future alternative path planning",
            "coverage path preview",
        ],
        "active_runtime": "partial_reference_only",
        "truthfulness_note": "PythonRobotics is used as an algorithm reference. AeroRescue-AI currently runs its own lightweight image-plane path planning and does not claim full PythonRobotics integration.",
    },
}


def list_decision_reference_projects(include_inactive=True):
    records = []
    for reference_key, config in DECISION_REFERENCE_PROJECTS.items():
        if not include_inactive and not config.get("active_runtime"):
            continue
        record = dict(config)
        record["reference_key"] = reference_key
        records.append(record)
    return records


def get_decision_reference_config(reference_key):
    if reference_key not in DECISION_REFERENCE_PROJECTS:
        raise DecisionReferenceRegistryError(f"Unknown decision reference project: {reference_key}")
    config = dict(DECISION_REFERENCE_PROJECTS[reference_key])
    config["reference_key"] = reference_key
    return config


def get_active_or_lightweight_decision_references():
    keys = [
        "sarenv_search_planning",
        "skai_building_damage",
        "inasafe_impact_modeling",
        "fields2cover_coverage_planning",
        "pythonrobotics_path_algorithms",
    ]
    return [get_decision_reference_config(key) for key in keys]


def summarize_decision_reference_capabilities():
    return """## 决策层参考项目能力说明

### SAREnv：搜救概率热力图与搜索路径评估参考
- 可借鉴：search probability heatmap、coverage path evaluation、probabilistic search
- 本项目落地：image-plane search priority map / coverage score
- 边界：不运行完整 SAREnv 地理空间框架

### SKAI：建筑损毁评估参考
- 可借鉴：灾后建筑损毁自动评估
- 本项目落地：基于 segmentation 的 major_damage / destroyed_building 风险权重
- 边界：不声明运行 SKAI 模型

### InaSAFE：灾害影响评估与 GIS 风险建模参考
- 可借鉴：hazard exposure vulnerability impact
- 本项目落地：image-plane impact score 和报告解释模板
- 边界：不声明完整 QGIS / GIS 分析

### Fields2Cover：覆盖路径规划参考
- 可借鉴：coverage path planning、area scanning、obstacle-aware coverage
- 本项目落地：grid-based sweep coverage preview / unsearched area ratio
- 边界：不直接链接 C++ Fields2Cover

### PythonRobotics：路径规划算法参考
- 可借鉴：A*、Dijkstra、D*、D* Lite、RRT、coverage path planning
- 本项目落地：路径规划算法对比和可视化设计
- 边界：不声明完整集成 PythonRobotics

当前决策层仍是图像平面辅助决策，**不是 GPS 导航**，**不是完整 GIS 灾害影响评估**，也**不替代人工救援判断**。
"""


def format_decision_reference_summary_for_report():
    return (
        "当前决策层参考了 SAREnv 的搜索优先级与覆盖评估思想、SKAI 的建筑损毁评估思想、InaSAFE 的灾害影响建模思想、"
        "Fields2Cover 的覆盖路径规划思想以及 PythonRobotics 的路径算法参考；其中只做了轻量 image-plane 落地，"
        "没有声明完整 GIS、GPS 或外部库的全量集成。"
    )
