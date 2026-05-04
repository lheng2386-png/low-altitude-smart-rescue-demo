# Decision Reference and Fusion Notes

## Why This Layer Exists

Detection and segmentation are perception layers. A competition-stage rescue assistant also needs an auxiliary decision layer that explains why a region, target, or route should be prioritized.

This layer keeps the system truthful:

- it records where each idea comes from;
- it distinguishes design references from executable code;
- it keeps image-plane adaptation separate from real GIS or GPS navigation;
- it still requires human review.

## SAREnv Reference

SAREnv contributes ideas for:

- search probability heatmap;
- coverage path evaluation;
- probabilistic search;
- multi-UAV search planning.

Lightweight AeroRescue-AI adaptation:

- image-plane search priority map;
- search priority score;
- missed-risk area score.

Boundary:

- not a full geospatial SAREnv runtime;
- not GPS navigation;
- not a true UAV search simulator.

## SKAI Reference

SKAI contributes ideas for:

- aerial building damage assessment;
- damage-aware prioritization.

Lightweight AeroRescue-AI adaptation:

- segmentation-based damage impact score;
- major_damage / destroyed_building weighting;
- damage severity summary in reports.

Boundary:

- not a SKAI model output;
- not a claim of reproduced benchmark numbers.

## InaSAFE Reference

InaSAFE contributes ideas for:

- hazard exposure and vulnerability;
- impact scenario explanation;
- preparedness and response planning.

Lightweight AeroRescue-AI adaptation:

- image-plane impact score;
- hazard area ratio reasoning;
- report explanation templates.

Boundary:

- not QGIS / full GIS impact modeling;
- not a fabricated disaster-analysis system.

## Fields2Cover Reference

Fields2Cover contributes ideas for:

- coverage path planning;
- area scanning;
- obstacle-aware coverage.

Lightweight AeroRescue-AI adaptation:

- coverage score;
- high-priority area coverage;
- unsearched high-priority ratio.

Boundary:

- not the C++ Fields2Cover library;
- not a real UAV flight route;
- not a GPS planner.

## PythonRobotics Reference

PythonRobotics contributes ideas for:

- A*;
- Dijkstra;
- D*;
- D* Lite;
- RRT;
- coverage path planning.

Lightweight AeroRescue-AI adaptation:

- path algorithm comparison ideas;
- future dynamic replanning direction;
- coverage path preview concepts.

Boundary:

- not a full PythonRobotics integration;
- not a claim of benchmark reproduction.

## New Modules

- `app/decision_reference_registry.py`
- `app/decision_fusion_adapter.py`

## Truthfulness Boundary

- image-plane auxiliary decision only;
- not real GPS navigation;
- not full GIS impact assessment;
- not a real UAV route planner;
- not automatic rescue decision;
- human review required.

## Competition Wording

本系统在 TERP 和 Risk-Aware A* 的基础上，引入决策层参考融合框架。SAREnv 启发搜索优先级热力图，SKAI 和 InaSAFE 启发灾损影响评分，Fields2Cover 启发覆盖路径评估，PythonRobotics 提供路径规划算法参考。当前实现为轻量级图像平面辅助决策，不声称完整 GIS 分析、真实 GPS 导航或自动救援决策。
