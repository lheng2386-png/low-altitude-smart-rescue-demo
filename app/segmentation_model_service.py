"""Inference service for locally trained 灾情感知及影响评估 segmentation checkpoints."""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.segmentation.lightweight_unet import NUM_CLASSES, LightweightUNet  # noqa: E402


DEFAULT_CHECKPOINT_CANDIDATES = [
    ROOT_DIR / "models" / "segmentation" / "best.pt",
    ROOT_DIR / "outputs" / "segmentation_training" / "checkpoints" / "best.pth",
    ROOT_DIR / "outputs" / "segmentation_training" / "checkpoints" / "latest.pth",
    ROOT_DIR / "checkpoints" / "segmentation_model.pth",
    ROOT_DIR / "app" / "segmentation_weights" / "segmentation_model.pth",
]
ALLOWED_CHECKPOINT_ROOTS = [
    ROOT_DIR / "models" / "segmentation",
    ROOT_DIR / "outputs" / "segmentation_training" / "checkpoints",
    ROOT_DIR / "checkpoints",
    ROOT_DIR / "app" / "segmentation_weights",
]
ALLOWED_CHECKPOINT_SUFFIXES = {".pth", ".pt"}


def select_device(device=None):
    """Select CUDA, MPS, or CPU."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_default_checkpoint():
    """Return the first existing checkpoint, or the preferred default path."""
    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_CHECKPOINT_CANDIDATES[0]


def _is_relative_to(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_allowed_checkpoint(checkpoint_path):
    """Resolve a checkpoint path and require it to live under project-owned roots."""
    checkpoint = Path(checkpoint_path) if checkpoint_path else get_default_checkpoint()
    if not checkpoint.is_absolute():
        checkpoint = ROOT_DIR / checkpoint

    try:
        resolved = checkpoint.resolve(strict=False)
    except Exception as exc:
        return None, f"Segmentation checkpoint path could not be resolved: {exc}"

    if resolved.suffix.lower() not in ALLOWED_CHECKPOINT_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_CHECKPOINT_SUFFIXES))
        return None, f"Segmentation checkpoint must use one of these suffixes: {allowed}."

    allowed_roots = [root.resolve(strict=False) for root in ALLOWED_CHECKPOINT_ROOTS]
    if not any(_is_relative_to(resolved, root) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        return None, f"Segmentation checkpoint is outside allowed project directories: {roots}."

    return resolved, ""


def _status(message, checkpoint_path=None, device=None, ok=False, img_size=None):
    return {
        "ok": bool(ok),
        "message": message,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "device": str(device) if device else "",
        "img_size": img_size,
        "num_classes": NUM_CLASSES,
    }


def load_segmentation_model(checkpoint_path=None, device=None):
    """Load a trained local checkpoint. Missing or invalid checkpoints return a clear status."""
    checkpoint, path_error = resolve_allowed_checkpoint(checkpoint_path)
    if checkpoint is None:
        return None, _status(path_error, checkpoint_path, ok=False)
    run_device = select_device(device)
    if not checkpoint.exists():
        return None, _status(
            "Segmentation checkpoint not found. Train a model first or provide a valid checkpoint path.",
            checkpoint,
            run_device,
            ok=False,
        )

    try:
        payload = torch.load(checkpoint, map_location=run_device, weights_only=True)
        num_classes = int(payload.get("num_classes", NUM_CLASSES)) if isinstance(payload, dict) else NUM_CLASSES
        model = LightweightUNet(num_classes=num_classes)
        state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state_dict)
        model.to(run_device)
        model.eval()
        return model, _status(
            f"Segmentation model loaded successfully from {checkpoint}.",
            checkpoint,
            run_device,
            ok=True,
        )
    except Exception as exc:
        return None, _status(
            f"Segmentation checkpoint load failed: {exc}",
            checkpoint,
            run_device,
            ok=False,
        )


def _to_rgb(image):
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return array.astype(np.uint8)


def _preprocess(image, img_size):
    image_rgb = _to_rgb(image)
    height, width = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (int(img_size), int(img_size)), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0), width, height


@torch.no_grad()
def predict_segmentation(image, checkpoint_path=None, img_size=512, device=None):
    """Predict a 2D class-id mask from a local trained checkpoint."""
    model, status = load_segmentation_model(checkpoint_path=checkpoint_path, device=device)
    status["img_size"] = int(img_size)
    if model is None:
        return None, status["message"], status

    try:
        run_device = select_device(status.get("device") or device)
        tensor, width, height = _preprocess(image, img_size)
        tensor = tensor.to(run_device)
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
        status["ok"] = True
        status["message"] = "Automatic segmentation prediction completed with a trained local checkpoint."
        return pred, status["message"], status
    except Exception as exc:
        status["ok"] = False
        status["message"] = f"Automatic segmentation prediction failed: {exc}"
        return None, status["message"], status
