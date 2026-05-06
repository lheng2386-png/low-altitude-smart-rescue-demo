# Integration Plan

## Reproducibility Plan
- Reproduce the official repository demo in an isolated environment.
- Record exact dependency versions, command lines, and required local files.
- Do not claim success until official outputs are generated locally.

## Adapter Plan
- Expected adapter file: `to be defined after reproduction`
- Wrap inputs and outputs into 灾情感知及影响评估 schemas only after real outputs exist.
- Preserve source metadata and truthfulness notes in every output.

## Roadmap
- Keep InaSAFE isolated due to QGIS/GIS dependency weight
- Define schema bridge for impact report
- Do not call image-plane score a GIS result
