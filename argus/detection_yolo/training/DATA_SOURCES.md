# Detection Training Data Sources

This file tracks sources intended for improving the Argus YOLO detector. Do not
train on images with unknown permission or unsupported annotations.

## Current Local Dataset

- Source: `urban-disaster-monitor/dataset`
- Format: YOLO
- License metadata: CC BY 4.0 in `data.yaml`
- Current useful classes: `civilian`, `rescuer`, plus animal classes
- Status: already local, directly usable

## VisDrone DET

- Homepage: https://github.com/VisDrone/VisDrone-Dataset
- Format: VisDrone DET annotations
- Useful Argus classes: `civilian`, `vehicle`
- Converter: `convert_visdrone_to_yolo.py`
- Status: recommended public UAV source for small people and vehicles

Convert after downloading the VisDrone DET folders:

```bash
python3 convert_visdrone_to_yolo.py \
  --source /path/to/VisDrone \
  --output ../datasets/visdrone_argus_yolo
```

Then merge it into the fused dataset:

```bash
python3 prepare_fused_yolo_dataset.py \
  --argus-yolo-dataset ../datasets/visdrone_argus_yolo \
  --image-mode symlink
```

## DOTA

- Homepage: https://captain-whu.github.io/DOTA/
- Format: oriented bounding boxes in `labelTxt`
- Useful Argus classes: `vehicle`, `boat`, `aircraft`, `bridge`, `road`
- Converter: `convert_dota_to_argus_yolo.py`
- Status: recommended source for aerial/remote-sensing target variety

Convert after downloading the DOTA split folders:

```bash
python3 convert_dota_to_argus_yolo.py \
  --source /path/to/DOTA \
  --output ../datasets/dota_argus_yolo
```

## SeaDronesSee

- Homepage: https://github.com/Ben93kie/SeaDronesSee
- Useful Argus classes: `civilian`, `boat`
- Status: recommended source for water rescue scenes
- Converter: use `convert_coco_to_argus_yolo.py` when using COCO-format
  releases/exports.

Example after downloading a COCO-format SeaDronesSee export:

```bash
python3 convert_coco_to_argus_yolo.py \
  --source-prefix seadronessee \
  --split train=/path/to/seadronessee/train.json=/path/to/seadronessee/train/images \
  --split valid=/path/to/seadronessee/valid.json=/path/to/seadronessee/valid/images \
  --split test=/path/to/seadronessee/test.json=/path/to/seadronessee/test/images \
  --output ../datasets/seadronessee_argus_yolo
```

## Fire / Smoke UAV Data

- Recommended source: D-Fire / fire-smoke detection datasets with YOLO labels
- Useful Argus classes: `fire`, `smoke`
- Status: add only releases with clear training permission and labels
- Converter: if already YOLO, pass the dataset root to
  `prepare_fused_yolo_dataset.py --argus-yolo-dataset`.

Example:

```bash
python3 prepare_fused_yolo_dataset.py \
  --argus-yolo-dataset /path/to/dfire_yolo \
  --image-mode symlink
```

## Flood / Building Data

- Useful Argus classes: `flood_water`, `building`
- Status: candidate source
- Recommended source: FloodNet-style UAV flood datasets
- Converter: use `convert_coco_to_argus_yolo.py` for COCO-style detection
  exports. If only segmentation masks are available, convert mask extents to
  boxes and keep a derived-annotation note in the summary.

Example after downloading a COCO-format flood dataset:

```bash
python3 convert_coco_to_argus_yolo.py \
  --source-prefix floodnet \
  --split train=/path/to/floodnet/train.json=/path/to/floodnet/train/images \
  --split valid=/path/to/floodnet/valid.json=/path/to/floodnet/valid/images \
  --output ../datasets/floodnet_argus_yolo
```

## Merge More Public Sources

After converting public sources into YOLO format, merge them with the local
AeroRescue dataset:

```bash
python3 prepare_fused_yolo_dataset.py \
  --argus-yolo-dataset ../datasets/visdrone_argus_yolo \
  --argus-yolo-dataset ../datasets/seadronessee_argus_yolo \
  --argus-yolo-dataset ../datasets/floodnet_argus_yolo \
  --argus-yolo-dataset /path/to/dfire_yolo \
  --image-mode symlink
```
