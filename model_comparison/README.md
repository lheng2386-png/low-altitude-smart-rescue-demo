# Model Comparison Scaffold

This directory prepares the detection-model comparison stage for AeroRescue-AI. It follows the idea of comparing multiple post-disaster UAV detection models, but it does not include fabricated metrics.

## Why Compare Models

The current AeroRescue-AI prototype uses YOLOv11 weights stored under `models/`. For a competition report, later experiments can compare accuracy, speed, model size, and CPU latency across lightweight and larger detectors.

## Candidate Models

- YOLOv11n
- YOLOv11s
- YOLOv11m
- YOLOv11l
- Optional Faster R-CNN
- Optional RetinaNet
- Optional DETR

Only models with local weights should be evaluated. This scaffold does not download weights or datasets.

## Metrics

- `mAP@0.5`
- Precision
- Recall
- FPS
- Model size
- CPU inference time

`results_template.csv` is only a template. It is not a final result table.

## Run A Lightweight Inference Summary

```bash
python model_comparison/evaluate_detection_models.py --image-folder app/examples --max-images 5
```

If labels are unavailable, the script only reports inference summaries and does not compute mAP.
