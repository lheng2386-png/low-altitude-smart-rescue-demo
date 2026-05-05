from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch  # type: ignore
    from torch import nn  # type: ignore

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

    class _MissingModule:
        pass

    class _MissingNN:
        Module = _MissingModule

    nn = _MissingNN()

try:
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False


NUM_SEGMENTATION_CLASSES = 11
SEGMENTATION_CLASS_NAMES = [
    "background",
    "water",
    "no_damage_building",
    "minor_damage",
    "major_damage",
    "destroyed_building",
    "vehicle",
    "road_clear",
    "road_blocked",
    "tree",
    "pool",
]

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DEFAULT_SEGMENTATION_WEIGHT_CANDIDATES = [
    ROOT_DIR / "models" / "segmentation" / "best.pt",
    ROOT_DIR / "outputs" / "segmentation_training" / "checkpoints" / "best.pth",
    ROOT_DIR / "outputs" / "segmentation_training" / "checkpoints" / "latest.pth",
    ROOT_DIR / "checkpoints" / "segmentation_model.pth",
    APP_DIR / "segmentation_weights" / "segmentation_model.pth",
]
ALLOWED_SEGMENTATION_CHECKPOINT_ROOTS = [
    ROOT_DIR / "models" / "segmentation",
    ROOT_DIR / "outputs" / "segmentation_training" / "checkpoints",
    ROOT_DIR / "checkpoints",
    APP_DIR / "segmentation_weights",
]
ALLOWED_SEGMENTATION_CHECKPOINT_SUFFIXES = {".pth", ".pt"}


class DoubleConv(nn.Module):
    """Small convolution block used by the lightweight UNet."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LightweightUNet(nn.Module):
    """Readable UNet-style segmentation model for 11-class disaster masks."""

    def __init__(self, num_classes=NUM_SEGMENTATION_CLASSES, base_channels=32):
        super().__init__()
        self.enc1 = DoubleConv(3, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def _device(device=None):
    if not TORCH_AVAILABLE:
        return None
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_default_segmentation_weights():
    """Return the preferred local segmentation checkpoint path."""
    for weights_path in DEFAULT_SEGMENTATION_WEIGHT_CANDIDATES:
        if weights_path.exists():
            return weights_path
    return DEFAULT_SEGMENTATION_WEIGHT_CANDIDATES[0]


def _is_relative_to(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_allowed_segmentation_checkpoint(weights_path):
    """Resolve a checkpoint path and require it to live under project-owned roots."""
    if weights_path is None:
        return None, "Automatic segmentation model weights not found. Please upload a segmentation mask or train segmentation weights."

    path = Path(weights_path)
    if not path.is_absolute():
        path = ROOT_DIR / path

    try:
        resolved = path.resolve(strict=False)
    except Exception as exc:
        return None, f"Automatic segmentation checkpoint path could not be resolved: {exc}"

    if resolved.suffix.lower() not in ALLOWED_SEGMENTATION_CHECKPOINT_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SEGMENTATION_CHECKPOINT_SUFFIXES))
        return None, f"Automatic segmentation checkpoint must use one of these suffixes: {allowed}."

    allowed_roots = [root.resolve(strict=False) for root in ALLOWED_SEGMENTATION_CHECKPOINT_ROOTS]
    if not any(_is_relative_to(resolved, root) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        return None, f"Automatic segmentation checkpoint path is outside allowed project directories: {roots}."

    return resolved, ""


def get_segmentation_model_status(weights_path):
    """Return whether an automatic segmentation checkpoint is available."""
    path, error = resolve_allowed_segmentation_checkpoint(weights_path)
    if path is None:
        return {
            "available": False,
            "path": str(weights_path) if weights_path else None,
            "message": error,
        }

    if not path.exists():
        return {
            "available": False,
            "path": str(path),
            "message": "Automatic segmentation model weights not found. Please upload a segmentation mask or train segmentation weights.",
        }

    return {
        "available": True,
        "path": str(path),
        "message": f"Automatic segmentation model weights found: {path}",
    }


def load_segmentation_model(weights_path=None, device=None):
    """Load a lightweight 11-class segmentation model if weights exist."""
    if not TORCH_AVAILABLE:
        return None, {
            "available": False,
            "path": str(weights_path) if weights_path else None,
            "message": "Automatic segmentation requires torch, but torch is not installed in this environment.",
            "dependency_missing": "torch",
        }

    if weights_path is None:
        weights_path = get_default_segmentation_weights()

    status = get_segmentation_model_status(weights_path)
    if not status["available"]:
        return None, status

    try:
        run_device = _device(device)
        model = LightweightUNet(num_classes=NUM_SEGMENTATION_CLASSES)
        checkpoint = torch.load(status["path"], map_location=run_device, weights_only=True)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(run_device)
        model.eval()
        status["device"] = str(run_device)
        status["message"] = f"Auto segmentation model loaded successfully from {status['path']} on {run_device}."
        return model, status
    except Exception as exc:
        status["available"] = False
        status["message"] = f"Automatic segmentation checkpoint load failed: {exc}"
        return None, status


def _to_rgb_array(image):
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return array.astype(np.uint8)


def _preprocess(image, input_size):
    image_rgb = _to_rgb_array(image)
    original_height, original_width = image_rgb.shape[:2]
    if CV2_AVAILABLE:
        resized = cv2.resize(image_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    else:
        resized = np.asarray(Image.fromarray(image_rgb).resize((input_size, input_size), Image.BILINEAR))
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required for automatic segmentation preprocessing.")
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0), original_width, original_height


def _no_grad_decorator():
    if TORCH_AVAILABLE:
        return torch.no_grad()

    def _decorator(func):
        return func

    return _decorator


@_no_grad_decorator()
def predict_segmentation_mask(image, model, device=None, input_size=512):
    """Predict an 11-class class-id mask with a loaded segmentation model."""
    if model is None:
        return None

    try:
        run_device = _device(device)
        model.to(run_device)
        model.eval()

        tensor, original_width, original_height = _preprocess(image, input_size)
        tensor = tensor.to(run_device)
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        if CV2_AVAILABLE:
            return cv2.resize(pred, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        return np.asarray(Image.fromarray(pred).resize((original_width, original_height), Image.NEAREST)).astype(np.uint8)
    except Exception:
        return None
