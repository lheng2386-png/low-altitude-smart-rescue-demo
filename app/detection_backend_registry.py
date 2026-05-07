"""Detection backend registry for 灾情感知及影响评估.

The registry separates runnable model backends from planned or reference-only
backends. It does not load models, download weights, or report metrics.
"""

from pathlib import Path


class DetectionBackendRegistryError(Exception):
    """Raised when an unknown detection backend is requested."""


DETECTION_BACKENDS = {
    "yolo_rescue_targets": {
        "display_name": "YOLO Rescue Targets",
        "backend_type": "ultralytics_yolo",
        "status": "active_if_weights_exist",
        "model_variants": ["yolov11n", "yolov11s", "yolov11m", "yolov11l"],
        "weights_pattern": "models/<variant>/best.pt",
        "primary_classes": ["civilian", "rescuer", "dog", "cat", "horse", "cow"],
        "output_role": "primary_rescue_target_detection",
        "can_enter_terp": True,
        "can_enter_path_planning": True,
        "requires_human_review": True,
        "truthfulness_note": "YOLO detections are local model outputs if weights exist. Results support rescue-priority ranking but still require human review.",
    },
    "transformer_rescuedet": {
        "display_name": "Transformer RescueDet",
        "backend_type": "huggingface_transformer",
        "status": "optional_if_dependencies_and_model_available",
        "model_variants": ["rescuedet_deformable_detr", "rescuedet_yolos_small"],
        "model_ids": [
            "RoblabWhGe/rescuedet-deformable-detr",
            "RoblabWhGe/rescuedet-yolos-small",
        ],
        "primary_classes": ["human_candidate", "vehicle", "fire"],
        "output_role": "auxiliary_scene_detection",
        "can_enter_terp": "human_candidate_only_with_review",
        "can_enter_path_planning": False,
        "requires_human_review": True,
        "truthfulness_note": "Transformer RescueDet is an auxiliary detection backend. human_candidate is not confirmed civilian. fire and vehicle are used as scene-risk supplements.",
    },
    "post_disaster_survivor_yolo": {
        "display_name": "Post-Disaster Survivor YOLO",
        "backend_type": "future_training_target",
        "status": "planned_dataset_training",
        "reference_repo": "https://github.com/HaoqianSong/Post-Disaster-Dataset",
        "primary_classes": ["survivor", "civilian_candidate"],
        "output_role": "future_survivor_detection",
        "can_enter_terp": False,
        "can_enter_path_planning": False,
        "requires_human_review": True,
        "truthfulness_note": "This backend is planned for future training on post-disaster survivor data. It is not active until a real checkpoint and validation results are provided.",
    },
    "air_retinanet_sar_reference": {
        "display_name": "AIR RetinaNet SAR Reference",
        "backend_type": "reference_only",
        "status": "reference_not_integrated",
        "reference_repo": "https://github.com/Accenture/AIR",
        "primary_classes": ["person"],
        "output_role": "search_and_rescue_reference",
        "can_enter_terp": False,
        "can_enter_path_planning": False,
        "requires_human_review": True,
        "truthfulness_note": "AIR is a land search-and-rescue person detection reference project. It is not integrated as an active 灾情感知及影响评估 backend.",
    },
    "qazi_disaster_management_reference": {
        "display_name": "Real-Time Disaster Management Reference",
        "backend_type": "reference_only",
        "status": "reference_not_integrated",
        "reference_repo": "https://github.com/qazi0/real-time-disaster-management",
        "primary_classes": ["disaster_context_reference"],
        "output_role": "disaster_management_reference",
        "can_enter_terp": False,
        "can_enter_path_planning": False,
        "requires_human_review": True,
        "truthfulness_note": "This disaster-management detector is a reference item only. It is not integrated as an active 灾情感知及影响评估 backend.",
    },
    "bahmanyar_merkle_person_detection_reference": {
        "display_name": "Saving Lives from Above Person Detection Reference",
        "backend_type": "literature_reference",
        "status": "reference_not_integrated",
        "reference_paper": "Bahmanyar, R. and Merkle, N. (2023), Saving Lives from Above: Person Detection in Disaster Response Using Deep Neural Networks",
        "reference_url": "https://doi.org/10.5194/isprs-annals-X-1-W1-2023-343-2023",
        "primary_classes": ["person", "human_candidate"],
        "output_role": "future_aerial_person_detection_reference",
        "can_enter_terp": False,
        "can_enter_path_planning": False,
        "requires_human_review": True,
        "truthfulness_note": "This paper is a literature reference for aerial and UAV person detection using deep neural networks. It is not an active 灾情感知及影响评估 model backend.",
    },
    "vtsar_dataset_reference": {
        "display_name": "VTSaR Aerial SAR Dataset Reference",
        "backend_type": "reference_dataset_or_future_backend",
        "status": "reference_not_integrated",
        "reference_repo": "https://github.com/zxq309/VTSaR",
        "primary_classes": ["person", "human_candidate"],
        "output_role": "future_aerial_person_detection_reference",
        "can_enter_terp": False,
        "can_enter_path_planning": False,
        "requires_human_review": True,
        "truthfulness_note": "VTSaR is a reference for future aerial search-and-rescue person detection. It is not an active model backend yet.",
    },
    "sardet_or_vtsar_reference": {
        "display_name": "Aerial SAR Person Detection Reference",
        "backend_type": "reference_dataset_or_future_backend",
        "status": "reference_not_integrated",
        "reference_repos": ["https://github.com/zxq309/VTSaR"],
        "primary_classes": ["person", "human_candidate"],
        "output_role": "future_aerial_person_detection_reference",
        "can_enter_terp": False,
        "can_enter_path_planning": False,
        "requires_human_review": True,
        "truthfulness_note": "This is a reference for future aerial search-and-rescue person detection. It is not an active model backend yet.",
    },
}


ACTIVE_BACKEND_KEYS = ("yolo_rescue_targets", "transformer_rescuedet")
REFERENCE_STATUSES = {"reference_not_integrated", "planned_dataset_training"}


def _public_backend_record(backend_key, config):
    return {
        "backend_key": backend_key,
        "display_name": config["display_name"],
        "backend_type": config["backend_type"],
        "status": config["status"],
        "primary_classes": list(config.get("primary_classes", [])),
        "output_role": config.get("output_role"),
        "can_enter_terp": config.get("can_enter_terp"),
        "can_enter_path_planning": config.get("can_enter_path_planning"),
        "requires_human_review": config.get("requires_human_review", True),
        "truthfulness_note": config.get("truthfulness_note", ""),
    }


def list_detection_backends(include_reference=True):
    """List detection backends without loading any model."""
    records = []
    for backend_key, config in DETECTION_BACKENDS.items():
        if not include_reference and backend_key not in ACTIVE_BACKEND_KEYS:
            continue
        records.append(_public_backend_record(backend_key, config))
    return records


def get_detection_backend_config(backend_key):
    """Return full backend config for a known backend key."""
    if backend_key not in DETECTION_BACKENDS:
        raise DetectionBackendRegistryError(f"Unknown detection backend: {backend_key}")
    config = dict(DETECTION_BACKENDS[backend_key])
    config["backend_key"] = backend_key
    return config


def get_active_detection_backends():
    """Return runnable or optional backend records. Planned/reference items are excluded."""
    return [_public_backend_record(key, DETECTION_BACKENDS[key]) for key in ACTIVE_BACKEND_KEYS]


def _root_path(root_dir=None):
    if root_dir is not None:
        return Path(root_dir).resolve()
    return Path(__file__).resolve().parents[1]


def _check_yolo_availability(root_dir=None):
    root = _root_path(root_dir)
    config = get_detection_backend_config("yolo_rescue_targets")
    available_variants = []
    missing_requirements = []
    for variant in config["model_variants"]:
        weights_path = root / "models" / variant / "best.pt"
        if weights_path.exists():
            available_variants.append(variant)
    available = bool(available_variants)
    if not available:
        missing_requirements.append("missing local YOLO weights under models/yolov11*/best.pt")
    return {
        "backend_key": "yolo_rescue_targets",
        "available": available,
        "status": config["status"],
        "missing_requirements": missing_requirements,
        "available_variants": available_variants,
        "message": "YOLO Rescue Targets is available because at least one local best.pt exists."
        if available
        else "YOLO Rescue Targets is not available because no local YOLO best.pt weights were found.",
        "truthfulness_note": config["truthfulness_note"],
    }


def _check_transformer_availability():
    config = get_detection_backend_config("transformer_rescuedet")
    try:
        from transformer_detection_service import check_transformer_detection_environment

        env = check_transformer_detection_environment()
        missing = []
        if not env.get("transformers_available"):
            missing.append("transformers")
        if not env.get("torch_available"):
            missing.append("torch")
        if not env.get("huggingface_hub_available"):
            missing.append("huggingface_hub")
        available = bool(env.get("success"))
        message = (
            "Transformer dependencies are importable. This does not guarantee that the model is cached or inference will succeed."
            if available
            else "Transformer RescueDet is not available because optional dependencies are missing."
        )
    except Exception as exc:
        available = False
        missing = [f"environment check failed: {exc}"]
        message = "Transformer RescueDet availability check failed before model loading."
    return {
        "backend_key": "transformer_rescuedet",
        "available": available,
        "status": config["status"],
        "missing_requirements": missing,
        "available_variants": list(config.get("model_variants", [])) if available else [],
        "message": message,
        "truthfulness_note": config["truthfulness_note"],
    }


def check_detection_backend_availability(backend_key, root_dir=None):
    """Check backend availability without downloading or loading large models."""
    config = get_detection_backend_config(backend_key)
    if backend_key == "yolo_rescue_targets":
        return _check_yolo_availability(root_dir=root_dir)
    if backend_key == "transformer_rescuedet":
        return _check_transformer_availability()
    return {
        "backend_key": backend_key,
        "available": False,
        "status": config["status"],
        "missing_requirements": ["backend is planned or reference-only"],
        "available_variants": [],
        "message": f"{config['display_name']} is {config['status']} and is not an active 灾情感知及影响评估 runtime backend.",
        "truthfulness_note": config["truthfulness_note"],
    }


def summarize_detection_backend_capabilities():
    """Return a Chinese Markdown summary for UI/report use."""
    yolo = DETECTION_BACKENDS["yolo_rescue_targets"]
    transformer = DETECTION_BACKENDS["transformer_rescuedet"]
    survivor = DETECTION_BACKENDS["post_disaster_survivor_yolo"]
    air = DETECTION_BACKENDS["air_retinanet_sar_reference"]
    qazi = DETECTION_BACKENDS["qazi_disaster_management_reference"]
    bahmanyar = DETECTION_BACKENDS["bahmanyar_merkle_person_detection_reference"]
    vtsar = DETECTION_BACKENDS["sardet_or_vtsar_reference"]
    return f"""## 目标检测后端能力说明

### 当前主后端：{yolo['display_name']}
- 类别：{', '.join(yolo['primary_classes'])}
- 用途：救援目标检测、TERP 排序、路径规划目标点。
- 状态：本地 `models/<variant>/best.pt` 存在时才是可执行模型输出。
- 真实性说明：{yolo['truthfulness_note']}

### 可选辅助后端：{transformer['display_name']}
- 类别：{', '.join(transformer['primary_classes'])}
- 用途：辅助场景检测和双后端一致性分析。
- 边界：`human_candidate` 不等于 confirmed civilian，需要人工复核；`fire` / `vehicle` 只作为场景风险补充。
- 真实性说明：{transformer['truthfulness_note']}

### 未来训练后端：{survivor['display_name']}
- 状态：计划训练，不是当前能力。
- 类别规划：{', '.join(survivor['primary_classes'])}
- 说明：只有训练出真实 checkpoint 并完成验证后，才可启用。

### 已纳入 S4 工程记录的参考工作
- {qazi['display_name']}：来自 {qazi['reference_repo']}，用于说明灾害管理检测流程参考；当前不是可执行 S4 后端。
- {air['display_name']}：来自 {air['reference_repo']}，用于说明搜救人员检测方向参考；当前不是可执行 S4 后端。
- {bahmanyar['display_name']}：Bahmanyar 和 Merkle 2023 的航拍 / 无人机人员检测深度神经网络文献参考，DOI：{bahmanyar['reference_url']}。
- {vtsar['display_name']}：空中搜救人员检测数据 / 未来后端参考。
- 边界：以上参考工作不会进入当前 TERP 或路径规划流程，也不代表本项目已复现实验指标。
"""


def format_detection_backend_summary_for_report():
    """Return a compact Chinese report paragraph for backend provenance."""
    return (
        "当前目标检测主后端为 YOLO Rescue Targets，用于 civilian、rescuer 及动物目标检测；"
        "可选辅助后端为 Transformer RescueDet，用于 human_candidate、vehicle、fire 等场景补充识别。"
        "qazi0 real-time-disaster-management、AIR、Bahmanyar 和 Merkle 2023 航拍人员检测文献、"
        "Post-Disaster Survivor YOLO、VTSaR 等目前属于未来训练或参考后端，不作为当前系统真实输出，"
        "也不报告未验证的 mAP、precision、recall 或 FPS。"
    )
