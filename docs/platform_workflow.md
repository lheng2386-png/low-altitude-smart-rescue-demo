# AeroRescue-AI Platform Workflow

AeroRescue-AI is currently a local Gradio prototype. The future competition story is a UAV rescue decision-support platform where mission data, UAV imagery, detection outputs, environment layers, TERP ranking, route planning, and rescue reports are organized around one rescue task.

This document describes the future platform workflow. It does not mean the current repository already includes user accounts, a database, GIS maps, GPS routing, flight control, or cloud deployment.

## Why Platformize

The current prototype already connects detection, segmentation-source selection, TERP priority modeling, Risk-Aware A* path planning, and Chinese report generation. A platform layer would make these outputs easier to review, archive, compare, and explain during a rescue mission or competition demo.

## Current Local Prototype

Implemented now:

- Gradio local prototype
- UAV-style image upload
- Basic video detection preview
- YOLOv11 target detection
- Uploaded Mask / Auto Segmentation Model / None source selection
- Segmentation mask validation and overlay
- Environment risk fusion
- TERP priority model
- Baseline A* and Risk-Aware A* comparison
- Chinese rescue assistance report
- Demo case and showcase-output scaffolding

## Future Platform Modules

| Module | Role |
| --- | --- |
| Mission Management | Create and track disaster rescue missions |
| UAV Image/Video Upload | Store incoming UAV images, video clips, and metadata |
| Detection Result Review | Review target boxes, class labels, confidence, and manual corrections |
| Segmentation/Environment Layer | Manage uploaded masks or trained segmentation outputs |
| TERP Priority Dashboard | Compare target priority by target, environment, and route factors |
| Risk-Aware Route Planner | Visualize baseline and risk-aware image-plane routes |
| Report Center | Generate, export, and archive rescue reports |
| Case Archive | Save representative demo cases for review, PPT, and video production |

## Future Workflow

```text
Mission Created
→ UAV Image / Video Uploaded
→ Target Detection
→ Detection Review
→ Segmentation Source Selected
→ Environment Risk Layer Generated
→ TERP Priority Dashboard
→ Risk-Aware Route Planner
→ Rescue Report Center
→ Case Archive
```

## Not Implemented Yet

- User login
- Database
- GIS map
- GPS route planning
- UAV flight control
- Cloud deployment
- Real-time multi-UAV mission synchronization

## Competition Positioning

For the current stage, AeroRescue-AI should be presented as a local AI rescue decision-support prototype. The platform workflow is a roadmap and design direction for turning the current Gradio demo into a structured rescue application.
