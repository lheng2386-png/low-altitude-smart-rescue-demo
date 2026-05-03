# Segmentation Dataset Setup

This document describes how to prepare local image and mask data for the AeroRescue-AI semantic segmentation module.

The segmentation module is used for disaster-scene environmental understanding. It helps the system identify water, blocked roads, damaged buildings, vehicles, trees, and passable road areas so that risk scoring and A* path planning can consider environmental context.

## Data Policy

This repository does not include a full semantic segmentation dataset.

Large datasets should stay on your local machine or external storage. Do not commit full segmentation datasets to GitHub.

The following paths are ignored by Git:

```text
data/
datasets/
checkpoints/
```

## Recommended Directory Structure

Prepare your local data like this:

```text
data/segmentation/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Each image and mask pair should share the same filename stem:

```text
data/segmentation/train/images/sample_001.jpg
data/segmentation/train/masks/sample_001.png
```

## Mask Format

Class-id masks are recommended.

- Use PNG for class-id masks.
- Each pixel value should be the class id.
- Do not use JPG for class-id masks because compression can change pixel values.
- RGB color masks can also be used if their colors match the project color mapping in `app/segmentation_engine.py`.
- Masks are validated before environment risk fusion. Unknown class ids fall back to no segmentation instead of crashing the app.

## Class IDs

| ID | Class |
| ---: | --- |
| 0 | background |
| 1 | water |
| 2 | no_damage_building |
| 3 | minor_damage |
| 4 | major_damage |
| 5 | destroyed_building |
| 6 | vehicle |
| 7 | road_clear |
| 8 | road_blocked |
| 9 | tree |
| 10 | pool |

## Train A Lightweight Model

After preparing local data, run:

```bash
python training/train_segmentation.py \
  --data-root data/segmentation \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --input-size 512 \
  --model unet \
  --num-classes 11 \
  --output checkpoints/segmentation_model.pth
```

The checkpoint is saved under `checkpoints/`, which is ignored by Git.

## Evaluate A Checkpoint

```bash
python training/evaluate_segmentation.py \
  --data-root data/segmentation \
  --split val \
  --checkpoint checkpoints/segmentation_model.pth \
  --save-overlays
```

Evaluation prints pixel accuracy, mean IoU, and per-class IoU. Optional overlays are saved to `static/images/demo_outputs/segmentation_eval/` and should only come from your local data or this project output.
