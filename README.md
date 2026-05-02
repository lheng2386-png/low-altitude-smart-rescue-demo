# Low-Altitude Smart Rescue Demo

Current project name:

**低空智援：面向灾害救援的无人机智能感知与辅助决策系统**

This repository is the current competition demo for a low-altitude UAV rescue assistance system. The present version focuses on a Gradio-based YOLO disaster target detection workflow plus an initial decision layer for risk scoring, rescue priority ranking, and template-based Chinese rescue report generation.

The project is intentionally kept lightweight at this stage. It does not integrate ARGUS, Detection-Models, RescueNet, FastAPI, React, Docker, or any large language model API.

## Current Status

Completed:

- Gradio image/video demo can be started locally.
- YOLO weights are loaded from local files under `models/<variant>/best.pt`.
- Image upload returns an annotated detection result image.
- Detection details include class, confidence, bounding box, center point, and area.
- Initial risk scoring is added for detected rescue targets.
- Rescue targets are ranked by priority.
- A Chinese rescue report is generated from detection and ranking results.

Supported target classes:

| Class | Meaning |
| --- | --- |
| `civilian` | Civilian / possible rescue target |
| `rescuer` | Rescue worker |
| `dog` | Dog |
| `cat` | Cat |
| `horse` | Horse |
| `cow` | Cow |

## Repository Structure

```text
.
├── app
│   ├── app.py
│   ├── risk_engine.py
│   ├── priority_ranker.py
│   ├── report_generator.py
│   ├── requirements.txt
│   └── examples
├── models
│   ├── yolov11n
│   ├── yolov11s
│   ├── yolov11m
│   └── yolov11l
├── dataset
├── notebooks
├── static
├── FIRST_STEP_RUN.md
└── SECOND_STEP_DECISION_LAYER.md
```

## Environment

Recommended:

- Python 3.10 to 3.12
- macOS, Linux, or Windows
- GPU is optional

CPU inference works for image detection. Video detection can run on CPU, but it may be slow.

The current local verification used Python 3.12.

## Model Files

The app expects local YOLO weight files at:

```text
models/yolov11n/best.pt
models/yolov11s/best.pt
models/yolov11m/best.pt
models/yolov11l/best.pt
```

At least one model variant must exist for the corresponding dropdown option to run. If a selected model file is missing, the app will show a clear error message instead of generating fake detections.

## Run Locally

```bash
git clone https://github.com/lheng2386-png/low-altitude-smart-rescue-demo.git
cd low-altitude-smart-rescue-demo/app

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python app.py
```

Open the local Gradio address:

```text
http://127.0.0.1:7860
```

If port `7860` is already occupied, stop the old process or change the Gradio port in `app/app.py`.

## App Outputs

After uploading an image, the page returns:

- Annotated image with detection boxes
- Detection details table
- Risk ranking table
- Generated Chinese rescue report

If no target is detected, the app keeps the tables empty and reports:

```text
当前图像未检测到明确救援目标
```

## Decision Layer

The initial risk formula is:

```text
risk_score = class_weight * 70 + confidence * 20 + area_weight * 10
```

Class weights:

| Class | Weight |
| --- | ---: |
| `civilian` | 1.00 |
| `horse` | 0.65 |
| `cow` | 0.65 |
| `dog` | 0.55 |
| `cat` | 0.55 |
| `rescuer` | 0.15 |

Risk levels:

| Score Range | Level |
| --- | --- |
| 0-40 | Low |
| 40-70 | Medium |
| 70-100 | High |

See [SECOND_STEP_DECISION_LAYER.md](SECOND_STEP_DECISION_LAYER.md) for details.

## Deployment Note

The previous Hugging Face Space deployment workflow has been removed because it still pointed to an old external Space and required a missing or unrelated `HF_TOKEN` secret. This repository is currently maintained as a local Gradio demo.

Before enabling automatic deployment again, configure a new deployment target, repository secret, and workflow owned by this project.

## Next Step

The next planned stage is to connect RescueNet semantic segmentation so the system can include environmental risk factors such as flood water, blocked roads, damaged buildings, vehicles, and passable areas.

## License And Attribution

This repository contains adapted open-source code, dataset structure, model artifacts, and documentation assets. Keep the license and citation files with the repository when redistributing or publishing the project.
