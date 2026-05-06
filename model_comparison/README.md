# Model Comparison Module

This module is no longer an empty scaffold. It contains copied reference assets, copied detector code structures, a local registry, and a clear separation between reference results and 灾情感知及影响评估 reproduced results.

## What Is Included

| Item | Location | Status |
| --- | --- | --- |
| YOLOv11 local variants | `models/yolov11*/best.pt` | available if weights exist |
| Local evaluator | `evaluate_detection_models.py` | implemented for inference summaries |
| Model registry | `model_registry.json` | implemented |
| Results table | `results_template.csv` | includes pending and reference rows |
| Reference result notes | `reference_results.md` | implemented |
| Faster R-CNN reference figures | `static/images/reference/detection_models/` | copied |
| DINO reference figures | `static/images/reference/detection_models/` | copied |
| Copied reference code | `integrated_modules/detection_models/code/` | copied |

## Result Policy

Reference rows are marked `reference_not_reproduced`. They are useful for report structure and benchmark planning, but they are not presented as 灾情感知及影响评估 reproduced metrics.

## Candidate Models

- YOLOv11n
- YOLOv11s
- YOLOv11m
- YOLOv11l
- Faster R-CNN
- DINO
- DETR-family models

## Metrics For Future Reproduction

- Precision
- Recall
- mAP@0.5
- FPS
- CPU latency
- Model size

## Run Local Inference Summary

```bash
python model_comparison/evaluate_detection_models.py --image-folder app/examples --max-images 5
```

If labels are unavailable, the script reports inference summaries only and does not compute mAP.

