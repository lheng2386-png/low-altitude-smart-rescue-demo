import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from segmentation_engine import load_segmentation_mask  # noqa: E402
from segmentation_model import LightweightUNet  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def find_pairs(split_dir):
    """Match image and mask files by stem under one split directory."""
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        return []

    masks = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS
    }
    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = masks.get(image_path.stem)
        if mask_path:
            pairs.append((image_path, mask_path))
    return pairs


class SegmentationDataset(Dataset):
    """Local image/mask dataset for 11-class disaster segmentation."""

    def __init__(self, pairs, input_size):
        self.pairs = pairs
        self.input_size = int(input_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = load_segmentation_mask(mask_path)

        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        mask_tensor = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor


def update_confusion_matrix(confusion, preds, labels, num_classes):
    mask = (labels >= 0) & (labels < num_classes)
    indices = num_classes * labels[mask].astype(np.int64) + preds[mask].astype(np.int64)
    values = np.bincount(indices, minlength=num_classes ** 2)
    confusion += values.reshape(num_classes, num_classes)


def metrics_from_confusion(confusion):
    correct = np.diag(confusion)
    total = confusion.sum()
    pixel_accuracy = float(correct.sum() / total) if total else 0.0
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - correct
    iou = np.divide(correct, union, out=np.zeros_like(correct, dtype=np.float64), where=union > 0)
    valid = union > 0
    mean_iou = float(iou[valid].mean()) if np.any(valid) else 0.0
    return pixel_accuracy, mean_iou


def run_epoch(model, loader, criterion, optimizer, device, num_classes, train=True):
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

    avg_loss = total_loss / max(len(loader.dataset), 1)
    pixel_accuracy, mean_iou = metrics_from_confusion(confusion)
    return avg_loss, pixel_accuracy, mean_iou


def parse_args():
    parser = argparse.ArgumentParser(description="Train the AeroRescue-AI lightweight segmentation model.")
    parser.add_argument("--data-root", default="data/segmentation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--model", default="unet", choices=["unet"])
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--output", default="checkpoints/segmentation_model.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Data root not found: {data_root}")
        print("Prepare local image/mask folders first. See SEGMENTATION_DATASET_SETUP.md.")
        return 1

    train_pairs = find_pairs(data_root / "train")
    val_pairs = find_pairs(data_root / "val")
    if not train_pairs:
        print(f"No training pairs found under {data_root / 'train'}")
        print("Expected train/images and train/masks with matching file stems.")
        return 1
    if not val_pairs:
        print(f"No validation pairs found under {data_root / 'val'}")
        print("Expected val/images and val/masks with matching file stems.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")

    train_loader = DataLoader(
        SegmentationDataset(train_pairs, args.input_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        SegmentationDataset(val_pairs, args.input_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = LightweightUNet(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_miou = -1.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_miou = run_epoch(
            model, train_loader, criterion, optimizer, device, args.num_classes, train=True
        )
        val_loss, val_acc, val_miou = run_epoch(
            model, val_loader, criterion, optimizer, device, args.num_classes, train=False
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_mIoU={train_miou:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_mIoU={val_miou:.4f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": args.num_classes,
                    "model": args.model,
                    "input_size": args.input_size,
                    "best_miou": best_miou,
                },
                output_path,
            )
            print(f"Saved checkpoint: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
