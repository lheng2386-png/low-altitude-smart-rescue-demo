# Integration Plan

## Reproducibility Plan
- Reproduce the official repository demo in an isolated environment.
- Record exact dependency versions, command lines, and required local files.
- Do not claim success until official outputs are generated locally.

## Adapter Plan
- Expected adapter file: `app/detection_runtime_service.py`
- Wrap inputs and outputs into 灾情感知及影响评估 schemas only after real outputs exist.
- Preserve source metadata and truthfulness notes in every output.

## Roadmap
- Verify local best.pt variants
- Run YOLO smoke inference
- Record validation metrics only from a real validation split
