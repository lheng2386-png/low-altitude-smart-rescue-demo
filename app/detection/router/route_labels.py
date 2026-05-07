"""Route and backend labels for the S4 Model Router."""

YOLO_BACKEND = "yolo_main_detector"
TRANSFORMER_BACKEND = "transformer_compare_detector"
AIR_BACKEND = "air_sar_detector"
QAZI_BACKEND = "qazi_disaster_detector"

CLOSE_RANGE_CLEAR_RGB = "close_range_clear_rgb"
NORMAL_DISASTER_RGB = "normal_disaster_rgb"
DISTANT_SMALL_HUMAN_CANDIDATE = "distant_small_human_candidate"
DISASTER_AERIAL_SCENE = "disaster_aerial_scene"
VIDEO_FRAME_SEQUENCE = "video_frame_sequence"

ROUTE_ORDER = [
    CLOSE_RANGE_CLEAR_RGB,
    NORMAL_DISASTER_RGB,
    DISTANT_SMALL_HUMAN_CANDIDATE,
    DISASTER_AERIAL_SCENE,
    VIDEO_FRAME_SEQUENCE,
]

ROUTE_LABELS = {
    CLOSE_RANGE_CLEAR_RGB: {
        "display_mode_name": "高清通用目标检测",
        "recommended_combo": [YOLO_BACKEND, TRANSFORMER_BACKEND],
        "reason": "图像清晰度较高且适合通用救援目标检测，Router 选择 YOLO 与 Transformer 组合。",
    },
    NORMAL_DISASTER_RGB: {
        "display_mode_name": "灾害图像目标检测",
        "recommended_combo": [YOLO_BACKEND, TRANSFORMER_BACKEND],
        "reason": "图像为普通灾害 RGB 场景，Router 选择 YOLO 与 Transformer 组合。",
    },
    DISTANT_SMALL_HUMAN_CANDIDATE: {
        "display_mode_name": "远距离疑似人员强化检测",
        "recommended_combo": [AIR_BACKEND, TRANSFORMER_BACKEND, YOLO_BACKEND],
        "reason": "图像疑似为远距离航拍场景，目标尺度较小，推荐 AIR、Transformer 与 YOLO 组合。",
    },
    DISASTER_AERIAL_SCENE: {
        "display_mode_name": "航拍灾情场景检测",
        "recommended_combo": [QAZI_BACKEND, YOLO_BACKEND, TRANSFORMER_BACKEND],
        "reason": "图像具有明显灾害航拍特征，推荐 qazi0、YOLO 与 Transformer 组合。",
    },
    VIDEO_FRAME_SEQUENCE: {
        "display_mode_name": "视频关键帧目标检测",
        "recommended_combo": [YOLO_BACKEND, TRANSFORMER_BACKEND],
        "reason": "输入为视频关键帧序列，先抽帧后按关键帧路由并合并候选目标。",
    },
}

ROUTE_TO_BACKENDS = {
    route: list(config["recommended_combo"])
    for route, config in ROUTE_LABELS.items()
}

FALLBACK_BACKENDS = [YOLO_BACKEND, TRANSFORMER_BACKEND]

BACKEND_DISPLAY = {
    YOLO_BACKEND: "YOLO Detector",
    TRANSFORMER_BACKEND: "Transformer Detector",
    AIR_BACKEND: "Accenture/AIR SAR Detector",
    QAZI_BACKEND: "qazi0 Disaster Detector",
}

BACKEND_OUTPUT_ROLE = {
    YOLO_BACKEND: "rescue_target_detection",
    TRANSFORMER_BACKEND: "compare_detection",
    AIR_BACKEND: "sar_human_detection",
    QAZI_BACKEND: "disaster_aerial_detection",
}
