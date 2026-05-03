import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from segmentation_engine import create_segmentation_overlay, load_segmentation_mask  # noqa: E402
from segmentation_model import LightweightUNet  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
CLASS_NAMES = [
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


def find_pairs(split_dir):
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        return []
    masks = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS
    }
    return [
        (path, masks[path.stem])
        for path in sorted(image_dir.iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.stem in masks
    ]


class EvalSegmentationDataset(Dataset):
    def __init__(self, pairs, input_size):
        self.pairs = pairs
        self.input_size = input_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = load_segmentation_mask(mask_path)

        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask.astype(np.uint8), (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, torch.from_numpy(resized_mask).long(), str(image_path)


def update_confusion(confusion, preds, labels, num_classes):
    mask = (labels >= 0) & (labels < num_classes)
    indices = num_classes * labels[mask].astype(np.int64) + preds[mask].astype(np.int64)
    values = np.bincount(indices, minlength=num_classes ** 2)
    confusion += values.reshape(num_classes, num_classes)


def compute_iou(confusion):
    correct = np.diag(confusion)
    total = confusion.sum()
    pixel_accuracy = float(correct.sum() / total) if total else 0.0
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - correct
    per_class_iou = np.divide(correct, union, out=np.zeros_like(correct, dtype=np.float64), where=union > 0)
    valid = union > 0
    mean_iou = float(per_class_iou[valid].mean()) if np.any(valid) else 0.0
    return pixel_accuracy, mean_iou, per_class_iou, valid


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an AeroRescue-AI segmentation checkpoint.")
    parser.add_argument("--data-root", default="data/segmentation")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--checkpoint", default="checkpoints/segmentation_model.pth")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--overlay-dir", default="static/images/demo_outputs/segmentation_eval")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    checkpoint_path = Path(args.checkpoint)
    if not data_root.exists():
        print(f"Data root not found: {data_root}")
        print("Prepare local image/mask folders first. See SEGMENTATION_DATASET_SETUP.md.")
        return 1
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Train a checkpoint first with training/train_segmentation.py.")
        return 1

    pairs = find_pairs(data_root / args.split)
    if not pairs:
        print(f"No image/mask pairs found under {data_root / args.split}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")

    model = LightweightUNet(num_classes=args.num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    loader = DataLoader(EvalSegmentationDataset(pairs, args.input_size), batch_size=args.batch_size, shuffle=False)
    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    overlay_dir = Path(args.overlay_dir)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for images, masks, image_paths in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labels = masks.numpy()
            update_confusion(confusion, preds.reshape(-1), labels.reshape(-1), args.num_classes)

            if args.save_overlays and saved < 5:
                pred = preds[0].astype(np.uint8)
                original_path = image_paths[0]
                original = np.array(Image.open(original_path).convert("RGB"))
                pred = cv2.resize(pred, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
                overlay = create_segmentation_overlay(original, pred)
                out_path = overlay_dir / f"{Path(original_path).stem}_overlay.png"
                Image.fromarray(overlay).save(out_path)
                saved += 1

    pixel_accuracy, mean_iou, per_class_iou, valid = compute_iou(confusion)
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print("Per-class IoU:")
    for index, class_name in enumerate(CLASS_NAMES[: args.num_classes]):
        suffix = "" if valid[index] else " (not present)"
        print(f"  {index:02d} {class_name}: {per_class_iou[index]:.4f}{suffix}")
    if args.save_overlays:
        print(f"Saved overlays: {overlay_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
