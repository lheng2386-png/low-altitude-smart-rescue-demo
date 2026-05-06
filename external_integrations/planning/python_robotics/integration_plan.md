# Integration Plan

## Reproducibility Plan
- Reproduce the official repository demo in an isolated environment.
- Record exact dependency versions, command lines, and required local files.
- Do not claim success until official outputs are generated locally.

## Adapter Plan
- Expected adapter file: `app/path_planner.py`
- Wrap inputs and outputs into 灾情感知及影响评估 schemas only after real outputs exist.
- Preserve source metadata and truthfulness notes in every output.

## Roadmap
- Add Dijkstra/RRT comparison adapters
- Record path cost comparisons
- Keep GPS=false metadata
