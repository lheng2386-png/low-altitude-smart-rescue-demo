# RescueNet-Style Segmentation Fusion Package

This folder stores migrated semantic segmentation references for 灾情感知及影响评估's disaster-scene environment layer.

## Copied Material

- `RESCUENET_README_REFERENCE.md`: original segmentation README reference.
- `code/rescuenet_dataset.py`: copied dataset loader structure.
- `code/train.py`, `code/evaluate.py`: copied training and evaluation structure.
- `code/unet.py`, `code/pspnet.py`, `code/deeplabv3_plus.py`: copied mature segmentation model structures.

## Integrated Function

灾情感知及影响评估 keeps its lightweight runtime path:

1. Uploaded class-id / RGB mask.
2. Optional automatic segmentation checkpoint.
3. No-segmentation fallback.

The migrated RescueNet-style code and gallery assets make the segmentation layer a visible, mature subsystem while avoiding false claims about an existing trained 灾情感知及影响评估 segmentation checkpoint.

