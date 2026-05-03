# AeroRescue-AI Segmentation Training

This document describes the local training flow for the 11-class post-disaster scene segmentation model. It does not claim that a trained checkpoint is included in this repository.

## Dataset Structure

Prepare local data as class-id masks:

```text
data/segmentation/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

Image and mask stems should match, for example `image_001.jpg` and `image_001.png`.

## Classes

| ID | Class |
| --- | --- |
| 0 | Background |
| 1 | Water |
| 2 | Building-No-Damage |
| 3 | Building-Medium-Damage |
| 4 | Building-Major-Damage |
| 5 | Building-Total-Destruction |
| 6 | Vehicle |
| 7 | Road-Clear |
| 8 | Road-Blocked |
| 9 | Tree |
| 10 | Pool |

Masks should preferably be PNG class-id masks where each pixel stores an integer from 0 to 10. RGB masks are tolerated through the existing color-map conversion, but class-id PNG is the recommended training format.

## Train

```bash
python training/train_segmentation.py \
  --data_root data/segmentation \
  --epochs 30 \
  --batch_size 4 \
  --lr 1e-3 \
  --img_size 512 \
  --num_workers 2 \
  --output_dir outputs/segmentation_training
```

The script selects CUDA, MPS, or CPU automatically.

Outputs:

```text
outputs/segmentation_training/
  checkpoints/
    latest.pth
    best.pth
  history.json
  train_log.txt
  loss_curve.png
  miou_curve.png
```

`best.pth` is selected by validation mIoU.

## Evaluate

```bash
python training/evaluate_segmentation.py \
  --data_root data/segmentation \
  --checkpoint outputs/segmentation_training/checkpoints/best.pth \
  --split val \
  --img_size 512 \
  --output_dir outputs/segmentation_eval
```

Outputs:

```text
outputs/segmentation_eval/
  eval_metrics.json
  preview/
```

The metrics file includes pixel accuracy, mean IoU, and per-class IoU. Preview images combine original image, ground-truth color mask, and predicted color mask.

## Enable Automatic Segmentation in Gradio

After training, keep the checkpoint at:

```text
outputs/segmentation_training/checkpoints/best.pth
```

or copy it to:

```text
app/segmentation_weights/segmentation_model.pth
```

Then use the Gradio segmentation module or the existing Auto Segmentation Model mode. If no checkpoint exists, the app reports that automatic segmentation is unavailable and does not fabricate a mask.

## Missing Dataset Behavior

If `data/segmentation/` does not exist or train/val pairs are missing, training exits with a clear message. This means the code is ready, but the model is not trained until real local data is provided.
