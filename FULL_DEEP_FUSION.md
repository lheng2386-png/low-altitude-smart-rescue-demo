# AeroRescue-AI Full Deep Fusion Version

AeroRescue-AI Full Deep Fusion Version upgrades the project from a lightweight Gradio demo into a unified low-altitude UAV emergency rescue AI prototype.

## Fusion Strategy

| Source Area | Deep Fusion Result |
| --- | --- |
| ARGUS-style platform workflow | Copied frontend pages, API pattern, report/image routers, schemas, and platform assets into `integrated_modules/argus/`; added `platform/` mockup documents |
| Urban disaster YOLO detection | Copied original Gradio app, requirements, data config, sample images, detector gallery assets into `integrated_modules/urban_disaster_monitor/` and `static/images/reference/urban_disaster_monitor/` |
| Post-disaster detection models | Copied Detection-Models / DINO / Faster R-CNN structures and benchmark assets into `integrated_modules/detection_models/`, `model_comparison/`, and `static/images/reference/detection_models/` |
| RescueNet-style segmentation | Copied dataset loader, train/evaluate structure, segmentation models, class reference image into `integrated_modules/rescuenet/`, `segmentation_reference/`, and `static/images/reference/rescuenet/` |

## AeroRescue-AI Native Innovations Kept

- TERP Target-Environment-Route Priority Model.
- Scene Applicability Gate.
- Risk-Aware Access Planning.
- Chinese rescue report generation.
- Demo case generator.
- Smoke tests.
- Segmentation source selection.
- Uploaded mask and missing-checkpoint fallback.

## What Is Reference vs Generated

- Reference assets are stored under `static/images/reference/`.
- AeroRescue-AI generated outputs are stored under `static/images/showcase/`.
- Reference benchmark figures are not claimed as reproduced AeroRescue-AI metrics.
- Manually prepared demo masks are not claimed as automatic segmentation predictions.

