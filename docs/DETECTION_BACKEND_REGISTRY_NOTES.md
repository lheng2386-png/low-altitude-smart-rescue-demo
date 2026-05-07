# Detection Backend Registry Notes

## Why This Registry Exists

灾情感知及影响评估 uses a detection backend registry to keep rescue detection capabilities separated and truthful.

The registry helps distinguish:

- current runnable backends;
- optional auxiliary backends;
- future training targets;
- reference-only projects.

This prevents reference projects or planned models from being presented as current 灾情感知及影响评估 outputs.

## Current Main Backend

### YOLO Rescue Targets

Classes:

- civilian
- rescuer
- dog
- cat
- horse
- cow

Role:

- primary rescue target detection;
- TERP priority ranking input;
- path planning target input.

Truthfulness boundary:

- YOLO results are real local model outputs only when local `best.pt` weights exist.
- YOLO detections support decision assistance but still require human review.

## Optional Auxiliary Backend

### Transformer RescueDet

Supported model candidates:

- `RoblabWhGe/rescuedet-deformable-detr`
- `RoblabWhGe/rescuedet-yolos-small`

Classes:

- human_candidate
- vehicle
- fire

Role:

- auxiliary scene detection;
- YOLO + Transformer bbox consistency analysis;
- human-like target review support;
- environment-risk supplement.

Truthfulness boundary:

- `human_candidate` is not confirmed civilian.
- `fire` and `vehicle` are scene-risk supplements, not trapped-person targets.
- Dependency availability does not guarantee that the Hugging Face model is cached or inference will succeed.

## Future Training Backend

### Post-Disaster Survivor YOLO

Reference data direction:

- https://github.com/HaoqianSong/Post-Disaster-Dataset

Current status:

- planned training target;
- not active;
- no checkpoint is claimed;
- no validation result is claimed.

This backend can only become active after a real checkpoint and validation results are provided.

## Reference Backends

### qazi0 / AIR / Bahmanyar-Merkle / VTSaR

These projects and papers are disaster-management or search-and-rescue person detection references.

Current status:

- reference only;
- not integrated as 灾情感知及影响评估 runtime outputs;
- not allowed to enter TERP or path planning as current model results.

Reference scope:

- qazi0 / real-time-disaster-management: disaster-management detection workflow reference.
- Accenture / AIR: search-and-rescue person detection repository reference.
- Bahmanyar and Merkle (2023), "Saving Lives from Above: Person Detection in Disaster Response Using Deep Neural Networks": aerial / UAV person detection literature reference.
- VTSaR: aerial search-and-rescue person detection dataset or future backend reference.

## Truthfulness Boundaries

- Do not fake checkpoints.
- Do not fake mAP, precision, recall, FPS, or latency.
- Do not treat `human_candidate` as confirmed `civilian`.
- Do not present reference figures as reproduced 灾情感知及影响评估 results.
- Do not present planned backends as current capabilities.

## Competition Wording

Suggested wording:

> 系统采用 Detection Backend Registry 管理多种救援目标检测后端。当前主后端为 YOLO Rescue Targets，用于平民、救援人员和动物目标检测；可选辅助后端为 Transformer RescueDet，用于 human_candidate、vehicle、fire 等场景目标补充识别；qazi0 real-time-disaster-management、Accenture/AIR、Bahmanyar 和 Merkle 2023 航拍人员检测文献、Post-Disaster Survivor YOLO、VTSaR 等作为未来训练或相关工作参考，不被包装成当前已完成能力。
