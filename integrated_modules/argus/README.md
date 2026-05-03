# ARGUS Fusion Package

This folder stores ARGUS-derived platform workflow material integrated into AeroRescue-AI.

## Copied Material

- `ARGUS_README_REFERENCE.md`: original platform README reference.
- `code/home.tsx`: copied frontend home page structure.
- `code/overview.tsx`: copied overview / mission-style page structure.
- `code/report.tsx`: copied report page structure.
- `code/api.ts`: copied frontend API access pattern.
- `code/reports_router.py`: copied report router structure.
- `code/images_router.py`: copied image upload / image management router structure.
- `code/report_schema.py`: copied report schema structure.
- `code/image_schema.py`: copied image schema structure.
- `code/odm_manager.py`: copied WebODM manager for orthophoto task creation/download.
- `code/image_processing_service.py`: copied image upload, thumbnail, and metadata-processing service.
- `code/image_describer_ollama.py`: copied Ollama-based scene description worker.
- `code/tasks_reconstruction_stella.py`: copied StellaVSLAM 360°/3D reconstruction worker.
- `code/reconstruction_preprocess.py`: copied video codec / flip preprocessing helpers.
- `code/stella_sparse.yaml`, `code/stella_dense_fast.yaml`: copied reconstruction presets.
- `code/yolo_sliding_window_inference.py`: copied YOLO sliding-window inference structure.

## AeroRescue-AI Integration

AeroRescue-AI does not run the full ARGUS stack. Instead, this material is used to upgrade the project from a single demo screen into a platform-inspired rescue workflow:

1. Mission / case creation.
2. UAV image and video upload.
3. Detection and segmentation result review.
4. TERP priority dashboard.
5. Risk-aware access planning.
6. Report center.
7. Case archive.

The implementation target is documented in `platform/` and `docs/platform_design_from_argus.md`.

The five replicated feature entries are documented in root `ARGUS_FULL_REPLICATION.md`.
