# Full Actual Integration Roadmap

## 1. Project Goal

AeroRescue-AI aims to become a low-altitude UAV disaster rescue decision-support system with real engineering integration across perception, path planning, search evaluation, damage assessment, and impact modeling repositories.

This roadmap does not claim that every external repository is already executable inside AeroRescue-AI. It defines the path from reference-only knowledge to reproducible demos, adapters, unified schemas, evaluation, Evidence Ledger integration, and Final Report integration.

## 2. Integration Principles

- Do not fake checkpoints.
- Do not fake datasets.
- Do not fake mAP, mIoU, precision, recall, FPS, or SOTA claims.
- Do not fake GPS routes.
- Do not describe image-plane paths as GPS navigation.
- Do not describe synthetic demo cases as real rescue benchmarks.
- Do not describe unreproduced external repositories as executable integrations.
- Do not describe `human_candidate` as confirmed civilian.
- Do not describe EC-TERP as an automatic rescue decision system.
- Every external output must carry source metadata, dependency status, checkpoint status, dataset status, and truthfulness boundaries.

## 3. Repository Family Clarification

The practical engineering scope is often described as "five detection plus five planning/decision repositories." In this codebase, the detailed list contains more URLs because some belong to the same engineering family:

- ARGUS, `transformer_pipeline`, `rescuedet-deformable-detr`, and `rescuedet-yolos-small` are treated as one Transformer auxiliary detection family.
- qazi0/real-time-disaster-management, Accenture/AIR, and VTSaR have separate roles: disaster detection reference, SAR/APD detection reference, and dataset validation support.
- Planning and decision references include PythonRobotics, Fields2Cover, SAREnv, SKAI, and InaSAFE.

## 4. Final Landing Targets

| Target | Final Goal | Current Truthful Position |
| --- | --- | --- |
| YOLO Rescue Targets | Executable local rescue-target detection if `best.pt` exists | Blocked by checkpoint when weights are absent |
| ARGUS / Transformer RescueDet family | Optional auxiliary detector for `human_candidate`, `vehicle`, `fire` | Adapter scaffold exists; model availability must be verified locally |
| qazi0 disaster management | Disaster-context detector or classifier adapter | Planned; official demo not reproduced yet |
| Accenture/AIR | SAR person detection adapter or comparison reference | Planned; official demo not reproduced yet |
| VTSaR | Dataset validation support for aerial SAR person detection | Planned dataset support |
| PythonRobotics | Path algorithm comparison and image-plane planner reference | Lightweight adapter exists; not GPS navigation |
| Fields2Cover | Coverage planning adapter if dependency is available | Planned; C++ dependency not integrated |
| SAREnv | Search probability and coverage evaluation adapter | Planned; full geospatial framework not integrated |
| SKAI | Building damage assessment adapter if model/data available | Planned; current segmentation score is not SKAI output |
| InaSAFE | GIS impact modeling adapter if QGIS stack available | Planned; current image-plane score is not full GIS analysis |

## 5. Dependency / Checkpoint / Dataset Requirements

- Detection backends generally require model weights or locally cached public models.
- Transformer RescueDet must not auto-download during import or app startup.
- VTSaR and other datasets require license/storage review before creating manifests.
- Fields2Cover and InaSAFE may require heavy native/GIS dependencies and should be reproduced in isolated environments first.
- SKAI requires compatible imagery and real model artifacts before any SKAI output can be claimed.

## 6. Phased Roadmap

### Phase 1: Official Demo Reproduction

Reproduce each external repository's official demo outside the main application. Save commands, dependency versions, input samples, and output artifacts. Do not claim integration before this phase succeeds.

### Phase 2: Adapter Wrapping

Create small adapters that normalize inputs and outputs into AeroRescue-AI schemas. Adapters must preserve repository/source metadata and truthfulness notes.

### Phase 3: Unified Demo Dataset

Prepare a small, licensed, reproducible demo dataset for detection, planning, damage, and report flows. Synthetic cases must remain labeled as synthetic/demo.

### Phase 4: Unified Output Schema

Map all repository outputs into consistent JSON schemas for Detection Runtime, Decision Fusion, Module Status Scanner, Mission Evidence Ledger, Final Report V2, and UI.

### Phase 5: Evaluation

Only after real data and real outputs exist, calculate metrics such as mAP, mIoU, precision, recall, ranking agreement, or path cost. Do not report placeholder metrics.

### Phase 6: Evidence Ledger / Final Report / UI Integration

Feed successful adapter outputs into Scanner, Evidence Ledger, Final Report V2, and UI. Failed, missing, simulated, preview-only, or reference-only results must remain clearly labeled.

## 7. Status Dashboard

The machine-readable dashboard is generated at:

```text
outputs/full_actual_integration/status_dashboard.json
```

It tracks:

- `current_state`
- dependency status
- checkpoint status
- dataset status
- expected input schema
- expected output schema
- truthfulness limitations
- next actions

The dashboard is a planning/runtime artifact, not a claim that all repositories are integrated.

## 8. Risks

- Heavy native/GIS dependencies may not fit the Gradio demo runtime.
- Public model checkpoints may require network/cache/license review.
- Dataset access may be restricted.
- External repositories may have incompatible licenses or stale dependencies.
- Image-plane decisions must not be presented as GPS/GIS rescue routing.

## 9. Truthfulness Boundary

This roadmap is a long-term engineering plan. `planned`, `blocked_by_dependency`, `blocked_by_checkpoint`, `blocked_by_dataset`, and `adapter_created` do not mean executable integration or evaluation success. Every upgrade of state must be backed by real local artifacts, reproducible commands, and evidence records.
