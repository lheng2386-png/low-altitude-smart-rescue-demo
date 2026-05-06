"""Train 灾情感知及影响评估's 11-class post-disaster segmentation model."""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/aerorescue_matplotlib")

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch import nn
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from models.segmentation.lightweight_unet import CLASS_NAMES, NUM_CLASSES, LightweightUNet  # noqa: E402
from segmentation_engine import load_segmentation_mask  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


def select_device():
    """Choose CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_pairs(split_dir):
    """Match image and mask files by filename stem."""
    image_dir = Path(split_dir) / "images"
    mask_dir = Path(split_dir) / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        return []
    masks = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS
    }
    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS and image_path.stem in masks:
            pairs.append((image_path, masks[image_path.stem]))
    return pairs


def check_dataset(data_root):
    """Return train/val/test pairs and a clear validation message."""
    data_root = Path(data_root)
    if not data_root.exists():
        return None, f"Data root not found: {data_root}. Expected data/segmentation/train/images and masks."
    pairs = {
        "train": find_pairs(data_root / "train"),
        "val": find_pairs(data_root / "val"),
        "test": find_pairs(data_root / "test"),
    }
    if not pairs["train"]:
        return None, f"No training image/mask pairs found under {data_root / 'train'}."
    if not pairs["val"]:
        return None, f"No validation image/mask pairs found under {data_root / 'val'}."
    return pairs, "Dataset structure is valid."


def _normalize(image):
    tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


class SegmentationDataset(Dataset):
    """Image/class-id-mask dataset for local disaster-scene segmentation."""

    def __init__(self, pairs, img_size=512, augment=False):
        self.pairs = list(pairs)
        self.img_size = int(img_size)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.pairs)

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if random.random() < 0.25:
            angle = random.choice([-10, -5, 5, 10])
            h, w = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        if random.random() < 0.25:
            pil = Image.fromarray(image)
            pil = ImageEnhance.Brightness(pil).enhance(random.uniform(0.85, 1.15))
            pil = ImageEnhance.Contrast(pil).enhance(random.uniform(0.85, 1.15))
            image = np.array(pil)
        return image, mask

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = load_segmentation_mask(mask_path)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            image, mask = self._augment(image, mask)
        return _normalize(image), torch.from_numpy(mask.astype(np.int64)).long()


def update_confusion_matrix(confusion, preds, labels, num_classes):
    """Update confusion matrix for semantic segmentation metrics."""
    mask = (labels >= 0) & (labels < num_classes)
    indices = num_classes * labels[mask].astype(np.int64) + preds[mask].astype(np.int64)
    confusion += np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion):
    """Compute pixel accuracy, mean IoU, and per-class IoU."""
    correct = np.diag(confusion)
    total = confusion.sum()
    pixel_accuracy = float(correct.sum() / total) if total else 0.0
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - correct
    per_class_iou = np.divide(correct, union, out=np.zeros_like(correct, dtype=np.float64), where=union > 0)
    valid = union > 0
    mean_iou = float(per_class_iou[valid].mean()) if np.any(valid) else 0.0
    return pixel_accuracy, mean_iou, per_class_iou.tolist()


def run_epoch(model, loader, criterion, optimizer, device, num_classes, train=True):
    """Run one train or validation epoch."""
    model.train(train)
    total_loss = 0.0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels = masks.detach().cpu().numpy()
        update_confusion_matrix(confusion, preds.reshape(-1), labels.reshape(-1), num_classes)
    pixel_acc, miou, per_class_iou = metrics_from_confusion(confusion)
    return total_loss / max(len(loader.dataset), 1), pixel_acc, miou, per_class_iou


def save_curves(history, output_dir):
    """Save loss and mIoU curves."""
    output_dir = Path(output_dir)
    epochs = [item["epoch"] for item in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [item["train_loss"] for item in history], label="train_loss")
    plt.plot(epochs, [item["val_loss"] for item in history], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [item["val_mIoU"] for item in history], label="val_mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "miou_curve.png", dpi=160)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train 灾情感知及影响评估 11-class segmentation model.")
    parser.add_argument("--data_root", "--data-root", default="data/segmentation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", "--img-size", "--input-size", type=int, default=512)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=2)
    parser.add_argument("--num_classes", "--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--output_dir", "--output-dir", default="outputs/segmentation_training")
    parser.add_argument("--model_name", "--model", default="lightweight_unet", choices=["lightweight_unet", "unet"])
    parser.add_argument("--base_channels", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    pairs, message = check_dataset(args.data_root)
    if pairs is None:
        print(message)
        print("No training was started. Prepare class-id masks first; do not treat this as a trained result.")
        return 1

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.txt"
    history_path = output_dir / "history.json"

    device = select_device()
    train_loader = DataLoader(
        SegmentationDataset(pairs["train"], args.img_size, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        SegmentationDataset(pairs["val"], args.img_size, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = LightweightUNet(num_classes=args.num_classes, base_channels=args.base_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    best_miou = -1.0
    best_path = checkpoint_dir / "best.pth"
    latest_path = checkpoint_dir / "latest.pth"

    with log_path.open("w", encoding="utf-8") as log_file:
        header = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "device": str(device),
            "num_classes": args.num_classes,
            "class_names": dict(CLASS_NAMES),
            "train_samples": len(pairs["train"]),
            "val_samples": len(pairs["val"]),
            "test_samples": len(pairs["test"]),
            "args": vars(args),
        }
        log_file.write(json.dumps(header, ensure_ascii=False, indent=2) + "\n")
        print(f"Training samples: {len(pairs['train'])}; validation samples: {len(pairs['val'])}; device: {device}")

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, train_miou, _ = run_epoch(
                model, train_loader, criterion, optimizer, device, args.num_classes, train=True
            )
            val_loss, val_acc, val_miou, per_class_iou = run_epoch(
                model, val_loader, criterion, optimizer, device, args.num_classes, train=False
            )
            row = {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 6),
                "train_pixel_acc": round(float(train_acc), 6),
                "train_mIoU": round(float(train_miou), 6),
                "val_loss": round(float(val_loss), 6),
                "val_pixel_acc": round(float(val_acc), 6),
                "val_mIoU": round(float(val_miou), 6),
                "val_per_class_iou": per_class_iou,
            }
            history.append(row)
            line = (
                f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_mIoU={val_miou:.4f}"
            )
            print(line)
            log_file.write(line + "\n")
            log_file.flush()

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "num_classes": args.num_classes,
                "model_name": args.model_name,
                "base_channels": args.base_channels,
                "img_size": args.img_size,
                "epoch": epoch,
                "val_mIoU": float(val_miou),
                "class_names": dict(CLASS_NAMES),
            }
            torch.save(checkpoint, latest_path)
            if val_miou > best_miou:
                best_miou = float(val_miou)
                torch.save(checkpoint, best_path)
                print(f"Saved best checkpoint: {best_path}")
                log_file.write(f"Saved best checkpoint: {best_path}\n")

    history_payload = {
        "best_val_mIoU": best_miou,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "device": str(device),
        "num_classes": args.num_classes,
        "class_names": dict(CLASS_NAMES),
        "train_samples": len(pairs["train"]),
        "val_samples": len(pairs["val"]),
        "history": history,
    }
    history_path.write_text(json.dumps(history_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_curves(history, output_dir)

    print(f"Best val mIoU: {best_miou:.4f}")
    print(f"Best checkpoint path: {best_path}")
    print(f"Device: {device}")
    print(f"Num classes: {args.num_classes}")
    print(f"Dataset samples: train={len(pairs['train'])}, val={len(pairs['val'])}, test={len(pairs['test'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
