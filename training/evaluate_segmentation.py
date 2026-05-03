"""Evaluate AeroRescue-AI segmentation checkpoints."""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/aerorescue_matplotlib")

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from damage_segmentation_visualizer import render_segmentation_mask  # noqa: E402
from models.segmentation.lightweight_unet import CLASS_NAMES, NUM_CLASSES, LightweightUNet  # noqa: E402
from segmentation_engine import load_segmentation_mask  # noqa: E402
from training.train_segmentation import find_pairs, metrics_from_confusion, update_confusion_matrix  # noqa: E402


class EvalSegmentationDataset(Dataset):
    """Evaluation dataset with original image path retained for previews."""

    def __init__(self, pairs, img_size=512):
        self.pairs = list(pairs)
        self.img_size = int(img_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = load_segmentation_mask(mask_path)
        image_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, torch.from_numpy(mask_resized.astype(np.int64)).long(), str(image_path), str(mask_path)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = int(payload.get("num_classes", NUM_CLASSES)) if isinstance(payload, dict) else NUM_CLASSES
    base_channels = int(payload.get("base_channels", 32)) if isinstance(payload, dict) else 32
    model = LightweightUNet(num_classes=num_classes, base_channels=base_channels)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, num_classes


def save_preview(image_path, gt_mask_path, pred_mask, output_path):
    image = np.array(Image.open(image_path).convert("RGB"))
    gt = load_segmentation_mask(gt_mask_path)
    pred = cv2.resize(pred_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_color = render_segmentation_mask(gt)
    pred_color = render_segmentation_mask(pred)
    gt_overlay = cv2.addWeighted(image, 0.55, gt_color, 0.45, 0)
    pred_overlay = cv2.addWeighted(image, 0.55, pred_color, 0.45, 0)
    top_row = np.hstack([image, gt_color])
    bottom_row = np.hstack([pred_color, pred_overlay])
    divider = np.full((10, top_row.shape[1], 3), 245, dtype=np.uint8)
    combined = np.vstack([top_row, divider, bottom_row])
    Image.fromarray(combined).save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a local AeroRescue-AI segmentation checkpoint.")
    parser.add_argument("--data_root", "--data-root", default="data/segmentation")
    parser.add_argument("--checkpoint", default="outputs/segmentation_training/checkpoints/best.pth")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--img_size", "--img-size", "--input-size", type=int, default=512)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=1)
    parser.add_argument("--output_dir", "--output-dir", default="outputs/segmentation_eval")
    parser.add_argument("--max_previews", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "preview"

    if not data_root.exists():
        print(f"Data root not found: {data_root}")
        return 1
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        print("Train a model first with training/train_segmentation.py. No metrics were fabricated.")
        return 1
    pairs = find_pairs(data_root / args.split)
    if not pairs:
        print(f"No image/mask pairs found under {data_root / args.split}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    device = select_device()
    model, num_classes = load_checkpoint(checkpoint, device)
    loader = DataLoader(EvalSegmentationDataset(pairs, args.img_size), batch_size=args.batch_size, shuffle=False)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    saved = 0

    with torch.no_grad():
        for images, masks, image_paths, mask_paths in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labels = masks.detach().cpu().numpy()
            update_confusion_matrix(confusion, preds.reshape(-1), labels.reshape(-1), num_classes)

            if saved < args.max_previews:
                out_path = preview_dir / f"{Path(image_paths[0]).stem}_segmentation_preview.png"
                save_preview(image_paths[0], mask_paths[0], preds[0], out_path)
                saved += 1

    pixel_accuracy, mean_iou, per_class_iou = metrics_from_confusion(confusion)
    per_class = {}
    for index in range(num_classes):
        per_class[str(index)] = {
            "name": CLASS_NAMES.get(index, f"class_{index}"),
            "iou": float(per_class_iou[index]) if index < len(per_class_iou) else 0.0,
        }
    metrics = {
        "checkpoint": str(checkpoint),
        "split": args.split,
        "sample_count": len(pairs),
        "device": str(device),
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "per_class_iou": per_class,
        "preview_dir": str(preview_dir),
    }
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Metrics saved to: {output_dir / 'eval_metrics.json'}")
    print(f"Preview images saved to: {preview_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
