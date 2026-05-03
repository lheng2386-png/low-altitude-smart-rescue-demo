# TERP And Risk-Aware A* Path Planning

This document describes the current AeroRescue-AI decision model used to connect target detection, environmental understanding, route accessibility, and rescue reporting.

## Why TERP

Object detection alone can answer what is visible in a UAV image, but rescue decisions need more context:

- Is the target a likely trapped civilian, a rescuer, or an animal?
- Is the target near water, blocked roads, or damaged buildings?
- Can rescue teams reach the target through a lower-risk path?
- Does the image contain multiple targets competing for attention?

TERP gives the project its own decision model instead of only listing YOLO detections.

## TERP Definition

**TERP** means **Target-Environment-Route Priority Model**.

Chinese name:

**目标—环境—可达性联合救援优先级评估模型**

The model fuses target risk, environmental risk, and route accessibility risk:

```text
target_score = class_weight * 45 + confidence * 15 + area_weight * 10
environment_score = 0-20
accessibility_score = 0-20
terp_score = target_score + environment_score + accessibility_score
```

## Score Terms

| Term | Meaning |
| --- | --- |
| `class_weight` | Rescue-oriented class importance, such as civilian > animal > rescuer |
| `confidence` | YOLO detection confidence |
| `area_weight` | Normalized bbox area in the image |
| `environment_score` | Local environmental risk around the target |
| `accessibility_score` | Route difficulty estimated from A* path cost |

TERP levels:

| Score Range | Level |
| --- | --- |
| 0-40 | Low |
| 40-70 | Medium |
| 70-90 | High |
| 90+ | Critical |

## Risk-Aware A*

Baseline A* uses a uniform cost map, so it mainly follows geometric distance on the image plane.

Risk-Aware A* uses the segmentation class-id mask to convert disaster regions into traversal costs:

| Class | Path Meaning |
| --- | --- |
| `road_clear` | Low cost |
| `background` | Default cost |
| `tree`, `vehicle`, `minor_damage` | Medium cost |
| `major_damage`, `road_blocked` | High cost |
| `water`, `pool`, `destroyed_building` | Very high cost |

This makes the planner prefer clearer and safer regions when segmentation information is available.

## Baseline vs Risk-Aware A* Metrics

The comparison module reports:

- baseline path length
- baseline accumulated path cost
- risk-aware path length
- risk-aware accumulated path cost
- baseline path high-risk area ratio
- risk-aware path high-risk area ratio
- estimated risk reduction

If no segmentation mask is available, the comparison is limited because both routes are based on default image-plane assumptions.

## TERP In Demo Cases

The five competition demo cases use TERP in different ways:

| Case | TERP Focus |
| --- | --- |
| Flood Civilian Rescue | Civilian class priority plus nearby water risk |
| Building Collapse | Environment score from major or destroyed building regions |
| Road Blocked | Accessibility score from high-cost blocked-road routing |
| Multi-target Priority | Ranking across civilians, animals, and rescuers |
| No Target / Fallback | Safe reporting when no target is detected |

## Explaining Path Comparison Figures

Demo case outputs may include `risk_aware_path_overlay.png` and `dual_path_overlay.png`.

- Baseline A* is the uniform-cost image-plane route. It is useful as a simple comparison, but it does not represent a real safest route.
- Risk-Aware A* uses segmentation-derived costs to prefer lower-risk regions when a mask is available.
- `path_risk_reduction` means the reduction in the ratio of path pixels crossing high-risk classes such as water, road blocked, major damage, destroyed building, or pool.
- A longer Risk-Aware path can still be better in the prototype if it reduces high-risk area exposure.

These overlays are competition explanation tools. They are not GPS routes and should not be described as autonomous UAV navigation outputs.

## Current Limitations

- Current path planning is image-plane only.
- It is not a real GPS route.
- It does not use a real road network.
- It does not connect to UAV localization or flight control.
- Segmentation quality directly affects path cost quality.
- High-risk areas are high cost, not fully forbidden obstacles.

## Future Work

- Multi-target route planning.
- GIS / GPS integration.
- UAV live stream processing.
- Edge deployment.
- Federated multi-UAV training.
