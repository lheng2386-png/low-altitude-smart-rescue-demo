# Integration Plan

## Reproducibility Plan
- Reproduce the official repository demo in an isolated environment.
- Record exact dependency versions, command lines, and required local files.
- Do not claim success until official outputs are generated locally.

## Adapter Plan
- Expected adapter file: `app/transformer_detection_service.py`
- Wrap inputs and outputs into AeroRescue-AI schemas only after real outputs exist.
- Preserve source metadata and truthfulness notes in every output.

## Roadmap
- Verify local cache or explicit model path
- Run official/demo inference without startup auto-download
- Compare with YOLO output
