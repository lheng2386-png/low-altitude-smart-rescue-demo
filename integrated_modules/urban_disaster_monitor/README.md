# Urban Disaster Monitor Fusion Package

This folder stores the migrated detection-demo material used to strengthen AeroRescue-AI's YOLOv11 disaster target detection module.

## Copied Material

- `URBAN_DISASTER_MONITOR_README_REFERENCE.md`: original detection README reference.
- `code/original_gradio_app.py`: copied Gradio detection app.
- `code/original_requirements.txt`: copied app dependency list.
- `code/data.yaml`: copied six-class YOLO data configuration.
- `sample_flood_25.jpg`: copied flood image sample.
- `sample_flood_animals.jpg`: copied flood animal image sample.

## Integrated Function

AeroRescue-AI already uses the six disaster-response classes:

- `civilian`
- `rescuer`
- `dog`
- `cat`
- `horse`
- `cow`

The active implementation keeps local model loading from `models/<variant>/best.pt`, adds model caching, Chinese UI labels, structured detection outputs, TERP scoring, segmentation fusion, and path planning. The copied app is preserved here as the mature detection-demo reference.

