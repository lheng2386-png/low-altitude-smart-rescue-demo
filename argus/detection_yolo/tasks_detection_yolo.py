from celery import Celery
import os
import logging
import redis
import requests
from collections import defaultdict

from yolo_inference import YOLOInferencer   # NEW MODULE (see next step)

from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

REDIS_HOST = os.getenv("HOST_REDIS", "redis")
REDIS_PORT = int(os.getenv("PORT_REDIS", 6379))
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8008")
DEVICE = os.getenv("DEVICE", "cuda:0")
DEFAULT_AERORESCUE_MODEL_PATH = "/models/aerorescue-yolov11s-best.pt"
DEFAULT_FUSED_MODEL_PATH = "/models/argus-fused-yolov11-best.pt"
DEFAULT_ARGUS_MODEL_PATH = "./yolo11l-p2-visdrone-argus-1280-best.pt"
DEFAULT_ULTRALYTICS_MODEL = "yolo11x.pt"
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# Celery app for *this* pipeline
celery_app = Celery(
    "detector_yolo",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
)


def _env_flag(name, default=True):
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_yolo_model_configs():
    """Select YOLO models used by the Argus YOLO worker.

    Preference order:
    1. ARGUS_YOLO_MODEL_PATH, or the trained fused model path;
    2. the mounted AeroRescue model path;
    3. the original Argus local model, or the previous VisDrone fallback.
    """
    configs = []
    preset = os.getenv("ARGUS_YOLO_PRESET", "balanced").strip().lower()
    max_mode = preset in {"max", "ultra", "strongest"}
    quality_mode = preset in {"quality", "accurate", "high_accuracy", "max", "ultra", "strongest"}
    preferred_model_path = os.getenv("ARGUS_YOLO_MODEL_PATH", DEFAULT_FUSED_MODEL_PATH)
    preferred_imgsz = int(os.getenv("ARGUS_YOLO_IMGSZ", "1536" if max_mode else "1280" if quality_mode else "640"))
    aerorescue_model_path = os.getenv("ARGUS_AERORESCUE_YOLO_MODEL_PATH", DEFAULT_AERORESCUE_MODEL_PATH)
    argus_model_path = os.getenv("ARGUS_FALLBACK_YOLO_MODEL_PATH", DEFAULT_ARGUS_MODEL_PATH)
    argus_imgsz = int(os.getenv("ARGUS_FALLBACK_YOLO_IMGSZ", "2048" if max_mode else "1536" if quality_mode else "1280"))
    ultralytics_model = os.getenv("ARGUS_ULTRALYTICS_YOLO_MODEL", DEFAULT_ULTRALYTICS_MODEL)
    ultralytics_imgsz = int(os.getenv("ARGUS_ULTRALYTICS_YOLO_IMGSZ", "1536" if max_mode else "1280"))
    ensemble_enabled = _env_flag("ARGUS_YOLO_ENSEMBLE", default=quality_mode)
    ultralytics_enabled = _env_flag("ARGUS_ULTRALYTICS_YOLO_ENABLED", default=quality_mode)

    if preferred_model_path and os.path.isfile(preferred_model_path):
        configs.append(
            {
                "model_path": preferred_model_path,
                "imgsz": preferred_imgsz,
                "source": "argus_fused_yolov11",
            }
        )

    if not configs and aerorescue_model_path and os.path.isfile(aerorescue_model_path):
        configs.append(
            {
                "model_path": aerorescue_model_path,
                "imgsz": preferred_imgsz,
                "source": "aerorescue_local_yolov11",
            }
        )

    if (ensemble_enabled or not configs) and argus_model_path and os.path.isfile(argus_model_path):
        configs.append(
            {
                "model_path": argus_model_path,
                "imgsz": argus_imgsz,
                "source": "argus_local_yolov11",
            }
        )
    elif ensemble_enabled or not configs:
        model_path = hf_hub_download(
            repo_id=os.getenv("ARGUS_HF_YOLO_REPO", "erbayat/yolov11n-visdrone"),
            filename=os.getenv("ARGUS_HF_YOLO_FILENAME", "best.pt"),
        )
        configs.append(
            {
                "model_path": model_path,
                "imgsz": int(os.getenv("ARGUS_HF_YOLO_IMGSZ", "1280")),
                "source": "huggingface_yolov11_fallback",
            }
        )

    if ultralytics_enabled:
        configs.append(
            {
                "model_path": ultralytics_model,
                "imgsz": ultralytics_imgsz,
                "source": "ultralytics_yolo11x_coco",
            }
        )

    return configs if ensemble_enabled else configs[:1]


def _canonical_merge_label(label):
    label = str(label or "").strip().lower().replace(" ", "_")
    human_labels = {"civilian", "rescuer", "person", "people", "pedestrian", "human", "human_candidate"}
    vehicle_labels = {"car", "bus", "truck", "van", "vehicle", "tractor", "trailer", "awning-tricycle", "tricycle"}
    small_vehicle_labels = {"motor", "motorcycle", "bicycle"}
    animal_labels = {"dog", "cat", "horse", "cow"}
    if label in human_labels:
        return "human_target"
    if label in vehicle_labels:
        return "land_vehicle"
    if label in small_vehicle_labels:
        return "small_vehicle"
    if label in animal_labels:
        return label
    return label


def _model_source_priority(source, label):
    """Rank model sources for duplicate detections.

    The AeroRescue YOLOv11 weights have local validation metrics in the
    upstream project, so they win overlapping rescue-target detections by
    default. Argus detections still fill gaps when they find targets the
    AeroRescue model does not cover.
    """
    source = str(source or "")
    label_group = _canonical_merge_label(label)
    priority_by_source = {
        "argus_fused_yolov11": 4,
        "aerorescue_local_yolov11": 3,
        "argus_local_yolov11": 2,
        "huggingface_yolov11_fallback": 1,
        "ultralytics_yolo11x_coco": 1,
    }
    priority = priority_by_source.get(source, 0)
    if source == "argus_local_yolov11" and label_group in {"land_vehicle", "small_vehicle"}:
        priority += 1
    return priority


def _choose_better_annotation(current, candidate):
    current_rank = (
        _model_source_priority(current.get("model_source"), current.get("category_name")),
        float(current.get("score", 0.0)),
    )
    candidate_rank = (
        _model_source_priority(candidate.get("model_source"), candidate.get("category_name")),
        float(candidate.get("score", 0.0)),
    )
    return candidate if candidate_rank > current_rank else current


def _iou_xywh(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0 else 0.0


def merge_ensemble_annotations(annotations, iou_thresh=0.50):
    """Merge overlapping annotations from multiple YOLO models.

    Similar overlapping targets are unified into one annotation. The retained
    annotation comes from the higher-priority model source, while detections
    found by only one model are preserved as supplements.
    """
    grouped = defaultdict(list)
    for ann in annotations:
        grouped[ann.get("image_id")].append(dict(ann))

    merged = []
    for image_id, image_annotations in grouped.items():
        ordered = sorted(image_annotations, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        kept = []
        for ann in ordered:
            label = _canonical_merge_label(ann.get("category_name"))
            duplicate = False
            for kept_ann in kept:
                same_group = label == _canonical_merge_label(kept_ann.get("category_name"))
                if same_group and _iou_xywh(ann.get("bbox", [0, 0, 0, 0]), kept_ann.get("bbox", [0, 0, 0, 0])) >= iou_thresh:
                    duplicate = True
                    sources = set(kept_ann.get("model_sources", []))
                    sources.update(ann.get("model_sources", []))
                    source = ann.get("model_source")
                    if source:
                        sources.add(source)
                    better = _choose_better_annotation(kept_ann, ann)
                    if better is ann:
                        kept_ann.clear()
                        kept_ann.update(dict(ann))
                    kept_ann["semantic_group"] = label
                    kept_ann["model_sources"] = sorted(sources)
                    kept_ann["merged_duplicate_count"] = int(kept_ann.get("merged_duplicate_count", 1))
                    kept_ann["fusion_note"] = "Overlapping similar detections were unified; this annotation keeps the higher-priority model output."
                    break
            if not duplicate:
                source = ann.get("model_source")
                ann["model_sources"] = [source] if source else []
                ann["semantic_group"] = label
                ann["merged_duplicate_count"] = 1
                kept.append(ann)
        merged.extend(kept)
    return merged


@celery_app.task(name="detection_yolo.run")
def run_detection_yolo(report_id: int, images: list[dict]):
    logger.info(f"[YOLO] Starting detection for report {report_id}")

    r.set(f"detection:{report_id}:status", "running")
    r.set(f"detection:{report_id}:progress", 0)
    r.set(f"detection:{report_id}:message", "Initializing YOLOv11 inference…")

    def set_progress(step: int, total_steps: int, message: str):
        progress = int((step / total_steps) * 100)
        r.set(f"detection:{report_id}:progress", progress)
        r.set(f"detection:{report_id}:message", message)

    try:
        # model_path = hf_hub_download(
        #     repo_id="StephanST/WALDO30",
        #     filename="WALDO30_yolov8m_640x640.pt"
        # )
        # model_path = hf_hub_download(
        #     repo_id="erbayat/yolov11n-visdrone",
        #     filename="best.pt"
        # )
        # model_path = "yolo11l.pt"
        # model_path = hf_hub_download(
        #     repo_id="mshamrai/yolov8l-visdrone",
        #     filename="best.pt"
        # )
        #infer = YOLOInferencer(model_name=model_path, progress_callback=set_progress, device=DEVICE)
        model_configs = resolve_yolo_model_configs()
        inferencers = []
        for model_config in model_configs:
            logger.warning(
                "[YOLO] Using model source=%s path=%s imgsz=%s",
                model_config["source"],
                model_config["model_path"],
                model_config["imgsz"],
            )
            inferencers.append(
                (
                    model_config,
                    YOLOInferencer(
                        model_name=model_config["model_path"],
                        imgsz=model_config["imgsz"],
                        progress_callback=set_progress,
                        device=DEVICE,
                    ),
                )
            )
        r.set(f"detection:{report_id}:message", "Running YOLOv11 inference…")

        # create 4 image long batches for progress tracking
        batch_size = 4
        total_batches = (len(images) + batch_size - 1) // batch_size
        for i in range(total_batches):
            batch_images = images[i*batch_size:(i+1)*batch_size]
            batch_annotations = []
            for model_config, infer in inferencers:
                annotations = infer.run(batch_images)
                for ann in annotations:
                    ann["model_source"] = model_config["source"]
                batch_annotations.extend(annotations)
            annotations = merge_ensemble_annotations(batch_annotations)
            url = f"{BACKEND_URL}/detections/r/{report_id}"
            resp = requests.put(url, json={"detections": annotations}, timeout=30)
            resp.raise_for_status()
            set_progress(i + 1, total_batches, f"Processed batch {i + 1} of {total_batches}")
        # annotations = infer.run(images)
        # logger.info(f"[YOLO] Inference completed for report {report_id}")
        # logger.info(f"[YOLO] Annotations: {annotations[:2]}")  # Log first 2 annotations for brevity

        # Save result as your backend expects:
        # url = f"{BACKEND_URL}/detections/r/{report_id}"
        # resp = requests.put(url, json={"detections": annotations}, timeout=30)
        # resp.raise_for_status()

        r.set(f"detection:{report_id}:status", "finished")
        r.set(f"detection:{report_id}:progress", 100)
        r.set(f"detection:{report_id}:message", "Detection with YOLO completed successfully")

    except Exception as e:
        logger.error(e)
        r.set(f"detection:{report_id}:status", "error")
        r.set(f"detection:{report_id}:message", str(e))
        r.set(f"detection:{report_id}:progress", 0)
        raise
