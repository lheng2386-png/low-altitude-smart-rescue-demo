# Segmentation Reference Module

This module turns the segmentation layer into a visible mature subsystem instead of a hidden mask-upload utility.

## Supported Modes

| Mode | Status | Meaning |
| --- | --- | --- |
| Uploaded mask | Implemented | User uploads a class-id or RGB segmentation mask |
| Reference mask | Implemented for demos | Manually prepared mask used for decision-layer demonstration |
| Automatic checkpoint | Experimental | Runs only if a local trained checkpoint exists |
| No segmentation | Implemented fallback | Risk and path use target-only/default assumptions |

## Important Warning

Manually prepared demo masks are not automatic segmentation predictions. They exist to demonstrate how environment risk affects TERP and risk-aware path planning.

## Gallery Assets

- `sample_images/rescuenet_all_classes.png`: copied segmentation class reference example.
- `sample_masks/manual_demo_rescuenet_mask.png`: manually prepared demo mask from the local 灾情感知及影响评估 examples.

See `class_palette.md` and `sample_assets.md`.

