# ARGUS Five-Capability Replication In 灾情感知及影响评估

This document explains how the five ARGUS-style mature capabilities are replicated and fused into 灾情感知及影响评估.

## 1. UAV Orthomosaic Generation

Copied source structure:

- `integrated_modules/argus/code/odm_manager.py`
- `integrated_modules/argus/code/image_processing_service.py`

Active 灾情感知及影响评估 implementation:

- `app/argus_fusion_engine.py::generate_orthomosaic`
- Gradio tab: `ARGUS 融合功能`

Current behavior:

- Accepts multiple UAV images.
- Attempts OpenCV feature stitching.
- Falls back to a contact-sheet orthomosaic preview if feature stitching fails.
- Clearly states that high-precision orthophoto generation requires WebODM/OpenDroneMap.

## 2. Thermal / Infrared Analysis

Copied source structure:

- `integrated_modules/argus/code/image_processing_service.py`

Active 灾情感知及影响评估 implementation:

- `app/argus_fusion_engine.py::analyze_thermal_image`

Current behavior:

- Reads grayscale or RGB thermal/IR-like images.
- Produces a heatmap overlay.
- Extracts high-intensity hotspot regions.
- Draws hotspot boxes.
- States that true temperature values require thermal matrix / camera metadata.

## 3. Target Detection

Copied source structure:

- `integrated_modules/argus/code/yolo_sliding_window_inference.py`
- `integrated_modules/urban_disaster_monitor/code/original_gradio_app.py`

Active 灾情感知及影响评估 implementation:

- `app/app.py`
- `models/yolov11*/best.pt`

Current behavior:

- Image detection.
- Video detection.
- Local model cache.
- Chinese bounding boxes.
- Structured targets.
- TERP, path planning, and report integration.

## 4. 360° Video Reconstruction / 3D Reconstruction

Copied source structure:

- `integrated_modules/argus/code/tasks_reconstruction_stella.py`
- `integrated_modules/argus/code/reconstruction_preprocess.py`
- `integrated_modules/argus/code/stella_sparse.yaml`
- `integrated_modules/argus/code/stella_dense_fast.yaml`

Active 灾情感知及影响评估 implementation:

- `app/argus_fusion_engine.py::reconstruct_360_video`

Current behavior:

- Reads uploaded video.
- Extracts keyframes.
- Reports FPS, resolution, frame count, ffmpeg availability, and StellaVSLAM availability.
- Provides the StellaVSLAM integration point without pretending that a point cloud exists when Stella is not installed.

## 5. Local LLM Automatic Scene Description

Copied source structure:

- `integrated_modules/argus/code/image_describer_ollama.py`

Active 灾情感知及影响评估 implementation:

- `app/argus_fusion_engine.py::describe_scene_with_local_llm`

Current behavior:

- Accepts an image.
- Sends it to local Ollama vision model through `/api/generate`.
- Uses a rescue-scene prompt.
- If Ollama or the model is missing, returns a clear fallback message instead of failing.

## Product Boundary

This is a source-level replication and adaptive fusion for 灾情感知及影响评估. Lightweight features run directly. Heavy services are represented by copied source structures and callable integration points:

- WebODM / OpenDroneMap for high-precision orthophoto.
- StellaVSLAM for sparse/dense point cloud.
- Ollama + local vision model for scene description.

No output is described as a real GPS route, full 3D model, or deployed UAV control capability unless the required external engine is installed and executed.

