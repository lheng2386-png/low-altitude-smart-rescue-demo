# Detection Runtime Service Notes

## Registry vs Runtime

Detection Backend Registry is the capability table. It records what each backend is, what it can produce, and what truthfulness boundaries apply.

Detection Runtime Service is the execution layer. It only runs current executable backends and returns a unified result structure.

## Current Executable Modes

### YOLO Rescue Targets

Runtime key:

- `yolo_rescue_targets`

Classes:

- civilian
- rescuer
- dog
- cat
- horse
- cow

Role:

- primary rescue-target detection;
- TERP input;
- path-planning target input.

Boundary:

- local `models/<variant>/best.pt` weights must exist;
- missing weights produce a structured failure;
- no weights are downloaded or fabricated.

### Transformer RescueDet from ARGUS

Runtime key:

- `transformer_rescuedet_argus`

Classes:

- human_candidate
- vehicle
- fire

Role:

- auxiliary scene detection;
- human-like target review evidence;
- vehicle/fire scene-risk supplement.

Boundary:

- `human_candidate` is not confirmed civilian;
- fire and vehicle are not trapped-person targets;
- models are not loaded at import time;
- downloads are disabled unless explicitly requested by a future caller.

### YOLO + Transformer Compare

Runtime key:

- `dual_backend_compare`

Role:

- YOLO remains the primary detection result;
- Transformer provides auxiliary bbox consistency evidence;
- overlapping YOLO civilian/rescuer and Transformer human_candidate boxes increase review priority, not final rescue certainty.

## Current Non-Executable Resources

The following remain planned or reference-only and do not enter runtime:

- qazi0 / real-time-disaster-management reference;
- Accenture / AIR reference;
- zxq309 / VTSaR reference;
- Post-Disaster Survivor YOLO future training backend.

These resources may inform future work, but they are not current 灾情感知及影响评估 model outputs.

## Output Files

When an `output_dir` is provided, the runtime may save:

- `detection_overlay.png`
- `detection_result.json`
- `detection_metadata.json`
- `transformer_detection_result.json`
- `transformer_detection_metadata.json`
- `dual_detection_result.json`
- `dual_detection_consensus.json`

Runtime outputs belong under `outputs/detection/` or a temporary directory and should not be committed to GitHub.

## Truthfulness Boundaries

- Do not fake model weights.
- Do not download models silently.
- Do not train models in runtime.
- Do not report mAP, precision, recall, FPS, or latency unless measured on a real validation setup.
- Do not treat `human_candidate` as confirmed civilian.
- Do not treat reference backends as current capabilities.

## Competition Wording

系统采用 Detection Runtime Service 对多种检测后端进行统一调度。YOLO Rescue Targets 是当前主检测后端，用于救援目标检测、TERP 排序与路径规划目标点生成；ARGUS Transformer RescueDet 是可选辅助后端，用于 human_candidate、vehicle、fire 等场景目标补充识别；双后端一致性分析用于提示人工复核优先级，但不替代人工确认。
