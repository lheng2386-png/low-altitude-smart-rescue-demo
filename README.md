# 灾情感知及影响评估

**灾情感知及影响评估：面向低空应急救援的无人机多模态灾情识别与辅助决策系统**

灾情感知及影响评估 is a competition-stage low-altitude UAV emergency rescue AI system. It fuses mature open-source platform workflow ideas, YOLOv11 disaster target detection, post-disaster model-comparison structures, RescueNet-style semantic segmentation, TERP rescue priority modeling, risk-aware image-plane access planning, and Chinese rescue report generation into one unified decision-support prototype.

<div align="center">

<img src="static/images/showcase/aerorescue_gradio_interface.png" alt="灾情感知及影响评估 interface" width="860"/>

</div>

```text
UAV Image / Video
→ Disaster Target Detection
→ Disaster-Scene Segmentation
→ TERP Priority Decision
→ Scene Applicability Gate
→ Risk-Aware Access Planning
→ Rescue Report
→ Platform-style Case Archive
```

## What Is 灾情感知及影响评估

灾情感知及影响评估 is not just a loose Gradio demo. It is a competition-stage prototype for low-altitude UAV emergency rescue. The current local application demonstrates the complete closed loop:

1. Upload UAV-style disaster imagery.
2. Detect civilians, rescuers, and animals.
3. Add disaster-scene segmentation through uploaded masks, experimental checkpoints, or fallback mode.
4. Evaluate rescue priority with TERP.
5. Compare ordinary A* and risk-aware A* image-plane access planning.
6. Generate a Chinese rescue assistance report.
7. Archive generated case outputs for presentation and review.

The current system is still local and prototype-oriented. It is not a deployed cloud platform, not real GPS navigation, and not connected to UAV flight control.

## Full Deep Fusion

灾情感知及影响评估 now integrates mature reference material as concrete project files instead of leaving the four repositories as light citations.

| Fusion Source | Integrated Role | Concrete Project Integration |
| --- | --- | --- |
| ARGUS | Platform-style UAV rescue workflow | `integrated_modules/argus/`, `platform/`, `docs/platform_design_from_argus.md`, platform reference assets |
| urban-disaster-monitor | YOLOv11 disaster target detection | `integrated_modules/urban_disaster_monitor/`, copied Gradio app, copied sample images, detection gallery assets |
| Post-Disaster-Dataset / Detection-Models | Survivor detection and model comparison | `integrated_modules/detection_models/`, `model_comparison/`, copied DINO/Faster R-CNN structures, reference figures |
| RescueNet-style segmentation | Disaster-scene semantic segmentation | `integrated_modules/rescuenet/`, `segmentation_reference/`, copied segmentation class figure and model structures |

See:

- `FULL_DEEP_FUSION.md`
- `MODULE_FUSION_SUMMARY.md`
- `REFERENCE_RESULTS.md`

## Core Mature Modules

| Module | Status | Output |
| --- | --- | --- |
| Detection Module | Implemented | Annotated image/video, class, confidence, bounding box |
| Segmentation Module | Implemented + experimental auto checkpoint | Overlay, area summary, environment context |
| Decision Module | Implemented | Risk score, TERP score, rescue priority ranking |
| Scene Applicability Gate | Implemented as fallback policy | Blocks over-claiming when targets, masks, or checkpoints are missing |
| Risk-Aware Access Planning | Implemented | Ordinary A* vs risk-aware A* comparison and path overlay |
| Report Module | Implemented | Chinese rescue assistance report |
| Platform Mockup Module | Integrated | Mission dashboard, report center, case archive design |
| Model Comparison Module | Integrated | Reference benchmark assets and reproducible evaluation scaffold |
| 3D Reconstruction Module | Added external-tool wrappers | Video frame extraction, quality filtering, COLMAP/ODM command preparation, truthfulness report |

## Detection Gallery

The detection module uses the six disaster-response classes `civilian`, `rescuer`, `dog`, `cat`, `horse`, and `cow`. The active 灾情感知及影响评估 app keeps local YOLOv11 weights under `models/<variant>/best.pt` and does not download models at runtime.

<div align="center">

<img src="static/images/reference/urban_disaster_monitor/flood_input.jpg" alt="Reference flood input" width="390"/>
<img src="static/images/reference/urban_disaster_monitor/flood_annotated.webp" alt="Reference annotated detection" width="390"/>

</div>

<div align="center">

<img src="static/images/reference/urban_disaster_monitor/custom_detector_output.png" alt="Reference custom detector output" width="390"/>
<img src="static/images/reference/urban_disaster_monitor/class_metrics.png" alt="Reference class metrics" width="390"/>

</div>

Reference figures above are used as detection-module presentation assets. 灾情感知及影响评估 generated outputs are stored separately under `static/images/showcase/`.

## Segmentation Gallery

The segmentation module follows an 11-class disaster-scene mask system:

`background`, `water`, `no_damage_building`, `minor_damage`, `major_damage`, `destroyed_building`, `vehicle`, `road_clear`, `road_blocked`, `tree`, `pool`.

<div align="center">

<img src="static/images/reference/rescuenet/rescuenet_all_classes.png" alt="Reference segmentation classes" width="780"/>

</div>

Supported modes:

- Uploaded class-id or RGB mask.
- Manually prepared demo mask for decision-layer demonstration.
- Optional automatic segmentation checkpoint if a trained local checkpoint exists.
- No-segmentation fallback.

Manual demo masks are not automatic segmentation predictions. Auto segmentation remains experimental until a real checkpoint and evaluation results are provided.

## Model Comparison Gallery

The model comparison module now contains copied reference structures and reference figures. It does not fabricate mAP, FPS, or latency.

<div align="center">

<img src="static/images/reference/detection_models/faster_rcnn_architecture.png" alt="Faster R-CNN reference architecture" width="390"/>
<img src="static/images/reference/detection_models/dino_framework.png" alt="DINO reference framework" width="390"/>

</div>

<div align="center">

<img src="static/images/reference/detection_models/faster_rcnn_map.png" alt="Faster R-CNN reference mAP" width="390"/>
<img src="static/images/reference/detection_models/dino_sota_table.png" alt="DINO reference table" width="390"/>

</div>

See `model_comparison/reference_results.md` and `model_comparison/results_template.csv`.

## Platform Workflow Gallery

The project now contains a platform-style rescue workflow package under `platform/` and copied platform structures under `integrated_modules/argus/`.

```text
Mission Dashboard
→ UAV Image / Video Upload
→ Detection Review
→ Segmentation Environment Layer
→ TERP Dashboard
→ Scene Applicability Gate
→ Risk-Aware Access Planner
→ Report Center
→ Case Archive
```

The current product is a platform-inspired prototype. It does not claim to run a full cloud platform.

## 灾情感知及影响评估 Innovations

| Innovation | Description |
| --- | --- |
| TERP | Target-Environment-Route Priority Model combines target type, confidence, scale, environment risk, and route accessibility |
| Scene Applicability Gate | Prevents unsupported decisions when targets, masks, or checkpoints are missing |
| Risk-Aware Access Planning | Compares ordinary A* and segmentation-cost A* on the image plane |
| Detection-Segmentation-Decision-Report Closed Loop | Turns UAV imagery into detection, environment understanding, priority ranking, path suggestion, and Chinese report |
| Multi-source Repository Fusion | Integrates mature platform, detection, benchmark, and segmentation assets into one rescue prototype |

## Demo Cases

Generated outputs are available under `static/images/showcase/`.

| Case | Focus |
| --- | --- |
| Case 01 Flood Civilian Rescue | Water risk, TERP priority, risk-aware route |
| Case 02 Building Collapse | Major / destroyed building risk |
| Case 03 Road Blocked | Road-block cost and route detour |
| Case 04 Multi-target Priority | Multiple target ranking |
| Case 05 No Target / Fallback | Safe fallback behavior |

<div align="center">

<img src="static/images/showcase/case_01_flood/detection_overlay.png" alt="灾情感知及影响评估 generated detection output" width="390"/>
<img src="static/images/showcase/case_01_flood/dual_path_overlay.png" alt="灾情感知及影响评估 generated path comparison" width="390"/>

</div>

Demo masks generated by the case script are manually prepared for decision-layer demonstration. They are not automatic segmentation predictions.

## Run Locally

```bash
cd app
python app.py
```

Open:

```text
http://127.0.0.1:7860/
```

Run smoke tests:

```bash
python tests/smoke_test_core.py
```

Generate demo cases:

```bash
python scripts/generate_demo_cases.py
```

Run model comparison help:

```bash
python model_comparison/evaluate_detection_models.py --help
```

Run 3D reconstruction frame extraction:

```bash
python -m modules.reconstruction_3d.video_to_frames --video data/demo/input.mp4 --output outputs/reconstruction_3d/frames --fps 1
```

## 3D Reconstruction Module

The `modules/reconstruction_3d/` package adds a real reconstruction workflow boundary for normal UAV imagery, extracted UAV video frames, and 360 panorama sequences without fabricating geometry:

- `video_to_frames.py` extracts sampled frames from a video and writes `frames_metadata.json`.
- `frame_quality_filter.py` removes blurry, too-dark, and optionally near-duplicate frames before reconstruction.
- `colmap_standard_pipeline.py` runs the standard COLMAP SfM chain for normal UAV images or extracted video frames: `feature_extractor`, `sequential_matcher` or `exhaustive_matcher`, and `mapper`, with optional dense stereo and meshing.
- `colmap_360_pipeline.py` runs a COLMAP `panorama_sfm.py` workflow for true 360 equirectangular panorama frame sequences. If COLMAP or the panorama script is unavailable, it writes a transparent missing-dependency status in `reconstruction_status.json`.
- `odm_pipeline.py` prepares or runs an OpenDroneMap Docker workflow for UAV image folders and supports `auto`, `perspective`, `fisheye`, `spherical`, and `equirectangular` camera lens settings.
- `reconstruction_workflow.py` orchestrates extraction, quality filtering, COLMAP/ODM execution, and report generation into one transparent workflow status.
- `reconstruction_report.py` writes `reconstruction_report.json` and `reconstruction_report.md` with explicit truthfulness boundaries.

Normal UAV reconstruction and 360 panorama reconstruction are not interchangeable. Standard UAV reconstruction expects perspective images with sufficient overlap and parallax. The 360 pipeline expects multiple equirectangular panorama frames from real camera motion; a single panorama image is not 3D reconstruction.

Required external dependencies for real COLMAP execution:

- COLMAP executable available on `PATH`.
- For 360 panorama reconstruction, a real `panorama_sfm.py` script path.
- Optional dense reconstruction requires enough compute for COLMAP PatchMatch stereo.

Example standard COLMAP command:

```bash
python -m modules.reconstruction_3d.colmap_standard_pipeline \
  --image-dir outputs/reconstruction_3d/selected_frames \
  --output-dir outputs/reconstruction_3d/colmap_standard \
  --matcher sequential \
  --run-dense false
```

Example 360 panorama COLMAP command:

```bash
python -m modules.reconstruction_3d.colmap_360_pipeline \
  --panorama-image-dir outputs/reconstruction_3d/selected_frames \
  --output-dir outputs/reconstruction_3d/colmap_360 \
  --panorama-sfm-script third_party/colmap/python/examples/panorama_sfm.py
```

Example unified workflow command:

```bash
python -m modules.reconstruction_3d.reconstruction_workflow \
  --mode standard_uav \
  --image-dir outputs/reconstruction_3d/selected_frames \
  --output-dir outputs/reconstruction_3d/workflow \
  --run-quality-filter false \
  --matcher sequential \
  --run-dense false
```

Supported workflow modes:

- `standard_uav`: normal perspective UAV images or extracted video frames through COLMAP standard SfM.
- `360_panorama`: true equirectangular panorama frame sequences through COLMAP panorama SfM.
- `odm`: UAV image folders through OpenDroneMap.
- `report_only`: generate a truthfulness report without running reconstruction.

The workflow writes `workflow_status.json` and a report folder. Each step is explicitly marked as `success`, `skipped`, `dependency_missing`, `script_missing`, `invalid_input`, `command_failed`, or `output_missing`.

The Gradio app exposes this workflow in the “360°视频 / 三维重建预处理” tab as the Real Workflow section. The older ORB/PLY preview remains available under a collapsed lightweight-preview section and is labeled as non-SfM/non-ODM.

This module can output extracted frames, selected reconstruction frames, prepared COLMAP/ODM command templates, verified external-tool output paths, and a reconstruction report. It reports real output paths only when files exist after an actual external-tool run.

Expected COLMAP workspace outputs include `database.db`, `sparse/0/` with COLMAP cameras/images/points3D files, optional `dense/fused.ply`, optional mesh outputs, `logs/`, and `reconstruction_status.json`.

Limitations are intentional and competition-critical: Fast Preview is not a real ODM orthophoto, 360 panorama viewing is not true 3D reconstruction, reconstruction is relative-scale unless GPS/GCP constraints are available, RGB frames do not contain true temperature matrices, no GPS/GCP means no absolute georeferenced rescue route, and all outputs remain human-reviewed auxiliary decision support rather than autonomous rescue commands or GPS navigation routes.

## OpenDroneMap / ODM Photogrammetry Pipeline

The ODM pipeline under `modules/reconstruction_3d/odm_pipeline.py` is the project path for real UAV photogrammetry products such as orthophoto, DSM/DTM, point cloud, textured model, and ODM report files. It uses Docker and the local OpenDroneMap image; it does not generate placeholder orthophotos, DEMs, meshes, GPS coordinates, or georeferenced outputs.

Fast Preview and real ODM output are different. Fast Preview is only a quick visual aid. A real ODM orthophoto is reported only when the Docker ODM command runs successfully and `odm_orthophoto/odm_orthophoto.tif` actually exists in the ODM project folder.

COLMAP and ODM also serve different roles:

- COLMAP standard/360 pipelines focus on SfM reconstruction, sparse models, optional dense point clouds, and mesh outputs.
- ODM is an end-to-end UAV photogrammetry workflow that can produce orthophotos, DSM/DTM, point clouds, textured models, and reports when the input imagery and metadata support it.

Required dependencies:

- Docker executable available on `PATH`.
- Local ODM Docker image, usually `opendronemap/odm`.
- The module does not auto-pull the image unless `auto_pull` is explicitly enabled.

Example ODM command:

```bash
python -m modules.reconstruction_3d.odm_pipeline \
  --image-dir outputs/reconstruction_3d/selected_frames \
  --output-dir outputs/reconstruction_3d/odm \
  --project-name aerorescue_odm \
  --camera-lens auto \
  --feature-quality medium \
  --pc-quality medium \
  --dsm true \
  --dtm false
```

Expected ODM output files are detected only when they exist:

- `odm_orthophoto/odm_orthophoto.tif`
- `odm_dem/dsm.tif`
- `odm_dem/dtm.tif`
- `odm_georeferencing/odm_georeferenced_model.laz`
- `odm_georeferencing/odm_georeferenced_model.las`
- `odm_texturing/odm_textured_model.obj`
- `odm_texturing/odm_textured_model_geo.obj`
- `odm_report/report.pdf`

ODM reconstruction requires sufficient image overlap, parallax, texture, lighting, and sharpness. Without GPS/GCP/RTK or reliable EXIF geotags, outputs should not be treated as survey-grade georeferenced products or rescue navigation routes. All ODM outputs are auxiliary spatial evidence for human-reviewed disaster assessment.

## Repository Structure

```text
app/                         Gradio app and decision modules
modules/reconstruction_3d/   Real 3D reconstruction wrappers and reports
integrated_modules/          Copied / migrated reference code and README material
static/images/reference/     Reference assets separated by source
static/images/showcase/      灾情感知及影响评估 generated outputs
platform/                    Platform-inspired dashboard and workflow mockups
model_comparison/            Detection benchmark and reference-result module
segmentation_reference/      Segmentation classes, palette, and sample assets
demo_cases/                  Case configs and expected outputs
docs/                        Static site and design documents
```

## Current Limitations

- Current system is a competition-stage local prototype, not a complete cloud platform.
- Path planning is image-plane reference planning, not real GPS navigation.
- No real road network, GIS engine, UAV localization, or flight-control system is connected.
- 3D reconstruction wrappers do not invent point clouds, camera poses, ODM outputs, GPS routes, or reconstruction success.
- Automatic segmentation requires a trained local checkpoint; without one, the system falls back to uploaded masks or no segmentation.
- Reference benchmark figures are not 灾情感知及影响评估 reproduced results.
- Manual demo masks are not automatic segmentation predictions.

## LLM Safety Regression Tests

The optional LLM report assistant, mission copilot, tool-orchestrated mission planner, and evidence auditor are tested with mock providers by default. These tests do not require `OPENAI_API_KEY` and must not call external LLM APIs.

Run the one-click LLM demo:

```bash
app/venv/bin/python scripts/run_llm_demo.py
```

With an already activated compatible environment, `python scripts/run_llm_demo.py` is equivalent.

See [docs/llm_demo.md](docs/llm_demo.md) for the demo dataset pack, MockProvider flow, real API configuration, generated files, and authenticity boundaries.

Run the LLM safety suite:

```bash
app/venv/bin/python tests/test_llm_safety.py
```

Useful related smoke tests:

```bash
app/venv/bin/python tests/smoke_test_llm_mission_report_assistant.py
app/venv/bin/python tests/smoke_test_llm_report_panel.py
app/venv/bin/python tests/smoke_test_mission_copilot.py
app/venv/bin/python tests/smoke_test_mission_copilot_panel.py
app/venv/bin/python tests/test_llm_mission_planner.py
app/venv/bin/python tests/smoke_test_mission_planner_panel.py
app/venv/bin/python tests/test_llm_evidence_auditor.py
app/venv/bin/python tests/smoke_test_evidence_audit_panel.py
```

The suite checks that LLM outputs do not confirm civilians, survivors, casualties, real temperatures, GPS navigation routes, survey-grade orthomosaics, model-generated segmentation, or real rescue conclusions. The planner can only execute white-listed backend tools. The auditor reports consistency issues and suggestions without overwriting source evidence. All LLM outputs remain auxiliary explanations that require human review.

## Roadmap

| Step | Status |
| --- | --- |
| Step 1 Detection Demo | Done |
| Step 2 Decision Layer | Done |
| Step 3 Segmentation Integration | Done |
| Step 4 Path Planning | Done |
| Step 5 TERP + Risk-Aware A* | Done |
| Step 6 Demo Cases + Showcase Outputs | Done |
| Step 7 Full Deep Fusion | Current |
| Step 8 Formal Model Comparison | Planned |
| Step 9 Platform UI / Dashboard Prototype | Planned |
| Step 10 Presentation Video + PPT | Planned |
| Step 11 NOTICE / Attribution / License cleanup | Final stage |

## Final Compliance TODO

NOTICE / Attribution / License cleanup will be completed before final public release or competition submission.
