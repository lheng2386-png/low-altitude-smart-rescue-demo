# 灾情感知及影响评估 Competition Presentation Outline

## 1. Project Background

- Low-altitude UAVs can quickly collect post-disaster visual information.
- Rescue teams need fast target recognition, environmental risk understanding, priority ranking, and route suggestions.
- 灾情感知及影响评估 turns UAV imagery into a decision-support loop for emergency rescue.

## 2. Pain Points

- Raw UAV images require manual interpretation.
- Detection confidence alone cannot decide rescue priority.
- Environmental hazards such as water, blocked roads, and damaged buildings affect rescue risk.
- Rescue teams need explainable reports for fast communication.

## 3. Overall Solution

```text
UAV Image / Video
→ Target Detection
→ Segmentation Source
→ Environment Risk Fusion
→ TERP Priority Ranking
→ Risk-Aware A* Path Planning
→ Chinese Rescue Report
```

## 4. System Architecture

- Gradio local prototype
- Image Tab for full decision workflow
- Video Tab for lightweight detection preview
- Demo Gallery for competition case presentation
- Offline demo case generator for reproducible showcase outputs

## 5. Technical Route

1. YOLOv11 disaster target detection.
2. Structured target extraction.
3. Uploaded Mask / Auto Segmentation Model / None segmentation source.
4. Segmentation mask validation and summary.
5. Environment risk fusion.
6. TERP target-environment-route priority model.
7. Baseline A* and Risk-Aware A* path comparison.
8. Template-based Chinese rescue report.
9. Demo case output generation.

## 6. Core Innovations

### Innovation 1: TERP Priority Model

Target-Environment-Route Priority Model combines:

- target class importance
- confidence
- bbox area
- environmental risk
- route accessibility

### Innovation 2: Risk-Aware A*

- Baseline A* uses uniform cost.
- Risk-Aware A* uses segmentation-derived class costs.
- The comparison reports length, cost, high-risk exposure, and risk reduction.

### Innovation 3: Detection-Segmentation-Decision-Report Closed Loop

The prototype connects perception outputs to ranking, route planning, and rescue reporting instead of stopping at object detection.

### Innovation 4: Competition Showcase Pipeline

The project includes generated demo cases, structured outputs, and reusable report artifacts for PPT and demo video production.

## 7. Function Demonstration

- Image upload and YOLO detection.
- Optional segmentation mask upload.
- Auto segmentation fallback when no checkpoint exists.
- Detection details table.
- Segmentation overlay and summary.
- Risk ranking and TERP ranking.
- Baseline A* vs Risk-Aware A* comparison.
- Path overlay.
- Chinese rescue report.
- Demo Gallery and showcase cases.

## 8. Experiments And Cases

Five prepared cases:

1. Flood Civilian Rescue.
2. Building Collapse.
3. Road Blocked.
4. Multi-target Priority.
5. No Target / Fallback.

Generated artifacts per case:

- input image
- manually prepared demo mask
- detection overlay
- segmentation overlay
- path overlay
- target table
- TERP ranking
- path comparison JSON
- rescue report
- case summary

Important note:

Manually prepared demo masks are used for decision-layer demonstration and are not automatic segmentation predictions.

## 9. Current Limitations

- Current system is a local Gradio prototype.
- Image-plane route planning is not GPS navigation.
- No real GIS road network is connected.
- No UAV localization or flight control is connected.
- Auto segmentation needs a trained local checkpoint.
- Model comparison scaffold exists, but formal metrics must come from real experiments.

## 10. Future Plan

- Polish 3-5 real showcase cases.
- Produce a 3-minute demo video.
- Build PPT and project report.
- Run formal model comparison experiments.
- Train and evaluate lightweight segmentation.
- Improve multi-target route planning.
- Prepare platform-style UI mockup.
- Complete final NOTICE / Attribution / License cleanup before public release or submission.
