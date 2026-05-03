# Agent Guardrails for AeroRescue-AI

This repository is a rescue decision-support prototype. Agents must keep the system honest, local, and testable.

## Capability boundaries

- Fast Preview means OpenCV stitching preview, not true orthomosaic.
- Real ODM Orthomosaic means OpenDroneMap / ODM actually ran and produced `odm_orthophoto.tif`.
- Simulated Thermal means grayscale-normalized hotspot analysis, not real thermal measurement.
- Radiometric Thermal is only valid when the input is a real radiometric thermal file and temperature is parsed.
- 3D Reconstruction Preview means keyframes, ORB features, tracks, trajectory, and PLY preview. It is not full reconstruction.
- Full 3D Reconstruction is only valid when a real pipeline such as ODM or COLMAP produces actual point cloud or model outputs.
- Path Planning is an image-plane reference path, not real GPS navigation.
- Uploaded Mask / Demo Mask is not automatic segmentation output.
- Auto Segmentation is only valid when a real local checkpoint is loaded and inference succeeds.

## Do not exaggerate

- Do not fake ODM outputs.
- Do not fake thermal temperatures.
- Do not fake 3D reconstruction results.
- Do not fake segmentation checkpoints.
- Do not fake model metrics.
- Do not present reference figures as reproduced results.
- Do not commit `outputs/`, checkpoints, large videos, or other generated artifacts.
- Do not majorly rewrite README unless the user explicitly asks.

## Required completion report

Every task completion must report:

- Which files changed
- What was implemented
- What was not implemented
- Which tests were run
- Whether tests passed
- Known limitations
- Whether the change is recommended to merge

## Testing discipline

Before and after meaningful changes, try to run:

```bash
python tests/smoke_test_core.py
```

If a segmentation visualization test was added or changed, also try:

```bash
python tests/smoke_test_damage_segmentation_visualizer.py
```

Always keep the result honest. If something is preview, fallback, or simulated, say so clearly.
