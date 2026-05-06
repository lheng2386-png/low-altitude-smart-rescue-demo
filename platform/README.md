# 灾情感知及影响评估 Platform Prototype

灾情感知及影响评估 currently runs as a local Gradio prototype, but the release direction is a platform-style rescue workflow rather than a single loose demo.

## Platform Modules

| Platform Area | Purpose | Current Status |
| --- | --- | --- |
| Mission Dashboard | Create and review rescue cases | Mockup documented |
| UAV Upload Center | Import image / video from low-altitude UAVs | Implemented in Gradio prototype |
| Detection Review | Review YOLOv11 target detections | Implemented |
| Environment Layer | Upload / infer segmentation masks | Implemented with fallback |
| TERP Dashboard | Rank targets by target, environment, and route factors | Implemented |
| Scene Gate | Decide whether current image supports rescue decisions | Prototype logic documented |
| Risk-Aware Access Planner | Compare ordinary A* and risk-aware A* | Implemented |
| Report Center | Generate Chinese rescue reports | Implemented |
| Case Archive | Store demo case outputs | Implemented as local files |

## Not Implemented Yet

- User login.
- Database-backed case management.
- Cloud deployment.
- Real GIS map.
- GPS route planning.
- UAV flight-control integration.

See `dashboard_mockup.md` and `workflow_mockup.md`.

