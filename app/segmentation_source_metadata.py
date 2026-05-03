"""Truthful source metadata for segmentation masks and visualizations."""


SOURCE_LABELS = {
    "auto_model": "Auto Segmentation Model / 自动分割模型",
    "uploaded_mask": "Uploaded Mask / 用户上传掩码",
    "demo_fallback": "Demo / Fallback / 演示兜底",
    "none": "None / 未接入分割",
}


TRUTHFULNESS_NOTES = {
    "auto_model": {
        True: "该分割结果来自真实本地 checkpoint 的自动模型推理。",
        False: "当前未检测到可用的语义分割模型 checkpoint，无法执行自动分割。请上传 mask 或先训练模型；系统不会生成假分割图。",
    },
    "uploaded_mask": {
        True: "该分割结果来自用户上传 mask，不代表自动模型预测。",
        False: "Uploaded Mask 模式未获得有效 mask，因此没有可用分割结果。",
    },
    "demo_fallback": {
        True: "该分割结果仅用于流程演示或兜底，不代表真实模型输出。",
        False: "Demo / Fallback 模式不会伪造模型预测结果。",
    },
    "none": {
        True: "当前未接入语义分割结果。",
        False: "当前未接入语义分割结果。",
    },
}


def _normalize_source_type(source_type):
    text = str(source_type or "none").strip().lower()
    aliases = {
        "auto": "auto_model",
        "auto segmentation model": "auto_model",
        "自动分割模型": "auto_model",
        "uploaded": "uploaded_mask",
        "uploaded mask": "uploaded_mask",
        "上传掩码": "uploaded_mask",
        "demo": "demo_fallback",
        "fallback": "demo_fallback",
        "demo / fallback": "demo_fallback",
        "无分割": "none",
        "no segmentation": "none",
    }
    text = aliases.get(text, text)
    return text if text in SOURCE_LABELS else "none"


def build_segmentation_source_metadata(
    source_type,
    checkpoint_path=None,
    model_available=False,
    prediction_success=False,
    mask_path=None,
    fallback_reason=None,
):
    """Build consistent metadata describing where a segmentation mask came from."""
    source_type = _normalize_source_type(source_type)

    if source_type == "auto_model":
        is_model_prediction = bool(model_available and prediction_success)
        truthfulness_note = TRUTHFULNESS_NOTES["auto_model"][is_model_prediction]
    elif source_type == "uploaded_mask":
        is_model_prediction = False
        truthfulness_note = TRUTHFULNESS_NOTES["uploaded_mask"][True]
    elif source_type == "demo_fallback":
        is_model_prediction = False
        truthfulness_note = TRUTHFULNESS_NOTES["demo_fallback"][True]
    else:
        is_model_prediction = False
        truthfulness_note = TRUTHFULNESS_NOTES["none"][True]

    if fallback_reason:
        truthfulness_note = f"{truthfulness_note} 原因：{fallback_reason}"

    return {
        "source_type": source_type,
        "source_label": SOURCE_LABELS[source_type],
        "is_model_prediction": bool(is_model_prediction),
        "checkpoint_path": str(checkpoint_path or ""),
        "model_available": bool(model_available),
        "prediction_success": bool(prediction_success),
        "mask_path": str(mask_path or ""),
        "fallback_reason": str(fallback_reason or ""),
        "truthfulness_note": truthfulness_note,
    }


def format_segmentation_source_status(metadata, language="zh"):
    """Format segmentation source metadata for UI and reports."""
    metadata = metadata or build_segmentation_source_metadata("none")
    lines = [
        f"当前分割来源：{metadata.get('source_label', '')}",
        f"是否为模型自动预测：{'是' if metadata.get('is_model_prediction') else '否'}",
        f"Checkpoint 路径：{metadata.get('checkpoint_path') or '未使用'}",
        f"Mask 路径：{metadata.get('mask_path') or '未使用'}",
        f"回退原因：{metadata.get('fallback_reason') or '无'}",
        f"真实性说明：{metadata.get('truthfulness_note', '')}",
    ]
    return "\n".join(lines)


def segmentation_visualization_note(metadata):
    """Return the correct text for black-background color mask provenance."""
    metadata = metadata or build_segmentation_source_metadata("none")
    source_type = metadata.get("source_type")
    if metadata.get("is_model_prediction"):
        return "该黑底彩色图由本地语义分割模型自动预测 mask 后渲染生成。"
    if source_type == "uploaded_mask":
        return "该黑底彩色图由用户上传 mask 渲染生成，不代表自动模型预测。"
    if source_type == "demo_fallback":
        return "该黑底彩色图仅用于流程演示，不代表真实模型输出。"
    return "当前没有可用分割 mask，因此不会生成黑底彩色分割图。"
