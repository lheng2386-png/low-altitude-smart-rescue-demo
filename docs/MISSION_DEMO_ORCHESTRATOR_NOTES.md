# One-Click Mission Demo Orchestrator Notes

## Why This Module Exists

AeroRescue-AI now contains many independent modules: detection backends, segmentation source tracking, TERP, image-plane path planning, decision fusion, module status scanning, mission evidence ledger, and Final Report 2.0.

The One-Click Mission Demo Orchestrator connects these modules into a single competition-stage mission flow so a demo can be run from one image without manually clicking every module one by one.

## One-Click Flow

The orchestrator follows this local mission chain:

1. UAV image input
2. Target detection
3. Segmentation / environment layer
4. TERP and risk-aware decision stage
5. Image-plane path planning when the gate allows it
6. Lightweight decision fusion
7. Optional thermal analysis
8. Module Execution Status Scanner
9. Mission Evidence Ledger
10. Final Report 2.0

## Stage Status

Each stage returns a structured status:

- `success`: the stage produced a valid result.
- `partial_success`: the stage produced partial useful output while some substeps failed or were unavailable.
- `failed`: the stage failed with a structured error.
- `skipped`: the user selected not to run the stage.
- `not_requested`: the stage was not requested in the current mission.

One failed optional stage does not crash the whole mission. The mission summary and Final Report 2.0 record what actually happened.

## Truthfulness Boundaries

- The orchestrator does not fabricate missing detections.
- Missing YOLO weights produce a structured failure, not fake targets.
- Transformer RescueDet failures produce structured errors, not fake `human_candidate` detections.
- Uploaded masks are not automatic model predictions.
- Auto segmentation is only valid when a real checkpoint is loaded and inference succeeds.
- Simulated Thermal is grayscale/intensity-based hotspot visualization, not real temperature measurement.
- Radiometric Thermal is only a real measurement when a radiometric file is parsed into a `temperature_matrix`.
- Fast Preview is not real ODM orthomosaic generation.
- Risk-aware path planning remains an image-plane reference path, not GPS navigation.
- Decision Fusion is a lightweight image-plane adaptation, not full GIS, SAREnv, SKAI, InaSAFE, or Fields2Cover output.
- Final Report 2.0 is driven by the scanner and mission evidence ledger.

## Output Layout

The default local output layout is:

```text
outputs/mission_demo/<mission_id>/
  outputs/
    detection/
    segmentation_inference/
    decision_fusion/
    thermal/
    reports/
```

The module writes local artifacts under `outputs/`. These runtime outputs are not intended for GitHub commits.

## Competition Demo Wording

Suggested wording:

“AeroRescue-AI provides a one-click mission demo orchestrator that chains UAV image input, rescue target detection, segmentation-based risk context, TERP priority ranking, image-plane path planning, decision fusion, module status scanning, mission evidence ledger, and Final Report 2.0 into a single traceable workflow. The system does not force unavailable modules to appear successful; skipped, failed, simulated, preview, and real model outputs are explicitly marked in the mission evidence chain.”
