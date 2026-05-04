# Transformer Detection Notes

## Purpose

AeroRescue-AI keeps YOLO as the primary rescue-target detector and adds Transformer RescueDet as an optional auxiliary backend.

The Transformer backend is useful for cross-checking human-like targets and adding scene-risk context such as vehicles or fire. It does not replace YOLO and does not replace human rescue judgment.

## Supported Candidate Models

- `RoblabWhGe/rescuedet-deformable-detr`
- `RoblabWhGe/rescuedet-yolos-small`

These models are loaded lazily through `transformers.pipeline("object-detection")`. They are not loaded at application import time.

By default, AeroRescue-AI disables Hugging Face downloads during detection. The backend first tries a local model path or an existing local Hugging Face cache. A future caller may pass `allow_download=True`, but that must be an explicit user choice.

## Label Boundaries

- `human`, `person`, `people` are mapped to `human_candidate`.
- `human_candidate` is not a confirmed civilian.
- `human_candidate` requires manual review before any rescue decision.
- `vehicle` and `fire` are treated as environment-risk context, not trapped-person targets.

## YOLO vs Transformer Consistency

If a YOLO human-class target such as `civilian` overlaps with a Transformer `human_candidate`, the system can report auxiliary bbox consistency.

This does not create a final rescue judgment. Transformer-only `human_candidate` detections are review prompts, not confirmed civilians.

## Limitations

- Requires optional `transformers`, `torch`, and `huggingface_hub` dependencies.
- Default inference uses local cache/local paths only. If the model is not cached, the backend returns `MODEL_UNAVAILABLE` instead of downloading silently.
- The project does not fabricate detections if dependencies or models are unavailable.
- No mAP, precision, recall, or FPS is reported without a real evaluation dataset.

## Competition Wording

系统支持 YOLO 与 Transformer 双后端检测。YOLO 用于平民、救援人员及动物等救援目标识别；Transformer RescueDet 用于 human、vehicle、fire 等救援场景目标补充识别。系统通过双后端 bbox 一致性分析辅助提高人员目标复核优先级，但所有检测结果仍需人工确认。
