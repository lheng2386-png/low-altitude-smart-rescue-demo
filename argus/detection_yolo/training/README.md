# Argus Fused YOLO Training

This folder prepares and trains one YOLO model for the Argus YOLO worker.

## Class Space

The fused model uses these classes:

```text
cat, civilian, cow, dog, horse, rescuer, vehicle, fire, smoke
boat, flood_water, building, road, tree, vegetation, bridge, aircraft
```

Similar Argus/VisDrone labels are remapped before training:

- `person`, `pedestrian`, `people`, `human` -> `civilian`
- `car`, `van`, `truck`, `bus`, `motorcycle`, `bicycle`, `tricycle` -> `vehicle`
- `fire`, `flame` -> `fire`
- `smoke` -> `smoke`
- `boat`, `ship`, `vessel` -> `boat`
- `water`, `flood`, `standing_water`, `inundation` -> `flood_water`
- `building`, `house`, `roof`, `structure` -> `building`
- `road`, `street`, `highway` -> `road`
- `tree`, `trees` -> `tree`
- `vegetation`, `grass`, `forest` -> `vegetation`
- `bridge` -> `bridge`
- `aircraft`, `plane`, `airplane`, `helicopter` -> `aircraft`

## Export Your Argus Images for Review

Use this when you want to improve detection with your own drone images. The
exported labels are preannotations from the current detector, not final ground
truth. Review and correct them in a labeling tool before training.

From `argus/detection_yolo/training/` on the machine that can reach the API:

```bash
python3 export_argus_review_dataset.py \
  --api-base http://127.0.0.1:8008 \
  --output ../datasets/argus_review_yolo
```

To export one report:

```bash
python3 export_argus_review_dataset.py \
  --api-base http://127.0.0.1:8008 \
  --report-id 1 \
  --output ../datasets/argus_review_yolo
```

The output is a standard YOLO dataset:

```text
argus/detection_yolo/datasets/argus_review_yolo/
  data.yaml
  train/images + train/labels
  valid/images + valid/labels
  test/images + test/labels
  review_manifest.json
  export_summary.json
```

After review, train directly on that dataset:

```bash
python3 train_fused_yolo.py \
  --data ../datasets/argus_review_yolo/data.yaml \
  --base-model yolo11x.pt \
  --epochs 80 \
  --imgsz 1280 \
  --batch 4 \
  --device 0 \
  --name argus_review_yolo11x \
  --export ../models/argus-fused-yolov11-best.pt
```

## Build Data

From `argus/detection_yolo/training/`:

```bash
python3 prepare_fused_yolo_dataset.py
```

To add VisDrone after downloading the DET dataset:

```bash
python3 convert_visdrone_to_yolo.py \
  --source /path/to/VisDrone \
  --output ../datasets/visdrone_argus_yolo

python3 prepare_fused_yolo_dataset.py \
  --argus-yolo-dataset ../datasets/visdrone_argus_yolo \
  --image-mode symlink
```

To add DOTA after downloading the dataset:

```bash
python3 convert_dota_to_argus_yolo.py \
  --source /path/to/DOTA \
  --output ../datasets/dota_argus_yolo

python3 prepare_fused_yolo_dataset.py \
  --argus-yolo-dataset ../datasets/visdrone_argus_yolo \
  --argus-yolo-dataset ../datasets/dota_argus_yolo \
  --image-mode symlink
```

To add a converted Argus/VisDrone YOLO dataset:

```bash
python3 prepare_fused_yolo_dataset.py \
  --argus-yolo-dataset /path/to/argus_or_visdrone_yolo_dataset
```

The output is written to:

```text
argus/detection_yolo/datasets/argus_fused_yolo/
```

## Train

CPU smoke training:

```bash
python3 train_fused_yolo.py --epochs 1 --batch 2 --device cpu
```

Normal training, when GPU is available:

```bash
python3 train_fused_yolo.py --epochs 50 --batch 8 --device 0
```

The exported worker model is:

```text
argus/detection_yolo/models/argus-fused-yolov11-best.pt
```

Argus `docker-compose.yml` mounts that file into the YOLO worker at:

```text
/models/argus-fused-yolov11-best.pt
```
