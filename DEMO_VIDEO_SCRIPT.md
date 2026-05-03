# AeroRescue-AI Demo Video Script

## Work Title

**AeroRescue-AI：面向低空应急救援的无人机多模态灾情识别与辅助决策系统**

## 30-Second Background Introduction

Disaster rescue teams often need quick situational awareness before entering flooded, collapsed, or blocked areas. Low-altitude UAV imagery can provide a fast overhead view, but raw images alone do not directly answer who should be rescued first, which areas are risky, or what route should be checked. AeroRescue-AI is a competition-stage prototype that converts UAV images into target detection, environment understanding, rescue priority ranking, image-plane path planning, and a Chinese rescue assistance report.

## System Workflow

```text
UAV Image / Video
→ YOLOv11 Target Detection
→ Segmentation Source
→ TERP Priority Model
→ Risk-Aware A* Path Planning
→ Chinese Rescue Report
```

## Image Tab Demo Script

1. Open the local Gradio app.
2. Select the Image Tab.
3. Upload a disaster-scene image from `app/examples/`.
4. Select `Uploaded Mask`, `Auto Segmentation Model`, or `None`.
5. For Uploaded Mask, upload `app/examples/masks/demo_rescuenet_mask.png` or a manually prepared demo mask from `static/images/showcase/<case_id>/demo_mask.png`.
6. Set rescue start point, for example `start_x=20`, `start_y=-1`.
7. Click the detection button.
8. Show the processed detection image.
9. Show segmentation overlay and segmentation summary.
10. Show risk ranking and TERP ranking.
11. Show Risk-Aware A* path overlay and baseline comparison.
12. Show the generated Chinese rescue report.

Important narration:

> The uploaded or generated demo mask is used to demonstrate the decision layer. Unless a trained checkpoint is provided, it should not be described as automatic segmentation prediction.

## Demo Gallery Script

1. Switch to the Demo Gallery Tab.
2. Explain the complete workflow.
3. Show core innovations:
   - TERP Target-Environment-Route Priority Model
   - Risk-Aware A* image-plane rescue path planning
   - Detection-Segmentation-Decision-Report closed loop
4. Scroll through generated demo case outputs.
5. Mention that each generated case contains input image, detection overlay, segmentation overlay, path overlay, TERP table, path comparison, report, and case summary.

## TERP Innovation Explanation

TERP means **Target-Environment-Route Priority Model**. It combines:

- target class importance
- YOLO confidence
- target bbox scale
- nearby environmental risk
- route accessibility cost

This lets the system rank targets based on rescue meaning rather than detection confidence alone.

## Risk-Aware A* Innovation Explanation

Baseline A* uses a uniform image-plane cost map and mainly follows distance. Risk-Aware A* uses segmentation class costs, so water, blocked roads, major damage, destroyed buildings, and pools receive higher traversal costs. The system then compares baseline and risk-aware paths and reports whether high-risk path exposure is reduced.

## Limitations And Future Work

Current limitations:

- The system is a local Gradio prototype.
- Path planning is image-plane only, not GPS navigation.
- It does not use real road networks.
- It is not connected to UAV localization or flight control.
- Automatic segmentation requires a local trained checkpoint; otherwise the system uses uploaded masks or no-segmentation fallback.
- Manually prepared demo masks are not automatic segmentation predictions.

Future work:

- Prepare more real demo cases.
- Train and evaluate a lightweight segmentation model.
- Add formal model comparison experiments.
- Improve multi-target route planning.
- Build a platform-style mission dashboard.

## Three-Minute Voiceover Draft

Hello everyone, this project is AeroRescue-AI, a low-altitude UAV emergency rescue decision-support prototype.

In flood, collapse, and blocked-road disaster scenes, UAV images can quickly provide overhead information, but raw images are not enough for rescue decisions. Rescue teams also need to know which targets are present, which target should be checked first, what environmental risks surround the target, and what route is safer for initial inspection.

AeroRescue-AI connects these steps into one workflow. First, the system receives a UAV-style image or video. It runs YOLOv11 disaster target detection and outputs target boxes, categories, confidence, and bounding-box coordinates. Then the Image Tab can use a segmentation source: an uploaded mask, an optional local segmentation checkpoint, or no segmentation fallback.

With segmentation information, the system estimates environmental risk, such as water, blocked road, major damage, destroyed building, vehicle, tree, and clear road regions. These environment factors are then fused into the TERP model.

TERP means Target-Environment-Route Priority Model. It combines target class, detection confidence, target size, environment risk, and route accessibility into a single rescue priority score. For example, a suspected civilian near water or damaged buildings can receive higher priority than a lower-risk animal or a rescuer already participating in rescue work.

Next, AeroRescue-AI performs path planning. The baseline A* path uses a uniform cost map, while Risk-Aware A* uses segmentation-derived costs. Water, blocked roads, major damage, destroyed buildings, and pools have higher path costs, so the planner tries to avoid risky regions when possible. The system also compares baseline and Risk-Aware A* and reports path length, accumulated cost, and high-risk exposure reduction.

Finally, the system generates a Chinese rescue assistance report. The report includes recognition overview, TERP ranking, environmental risk summary, path planning suggestions, path comparison, rescue advice, and current limitations.

In the Demo Gallery, we prepare five competition cases: flood civilian rescue, building collapse, road blocked, multi-target priority, and no-target fallback. These cases show both successful decision outputs and safe fallback behavior.

At the current stage, AeroRescue-AI is still a local prototype. The route is an image-plane reference path, not GPS navigation, and it is not connected to real road networks, UAV localization, or flight control. Automatic segmentation is experimental and requires a local checkpoint; manually prepared demo masks are used only for decision-layer demonstration.

Next, we will prepare stronger real demo cases, presentation slides, a demo video, and formal model comparison experiments.
