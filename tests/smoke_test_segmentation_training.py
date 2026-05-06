"""Smoke test for segmentation training utilities."""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.segmentation.lightweight_unet import LightweightUNet  # noqa: E402
from training.train_segmentation import (  # noqa: E402
    SegmentationDataset,
    check_dataset,
    find_pairs,
    run_epoch,
)


def write_sample(root, split, stem, image, mask):
    image_dir = root / split / "images"
    mask_dir = root / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(image_dir / f"{stem}.png")
    Image.fromarray(mask).save(mask_dir / f"{stem}.png")


def build_tiny_dataset(root):
    for split in ["train", "val"]:
        for index in range(2):
            image = np.full((32, 32, 3), 100 + index * 20, dtype=np.uint8)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[:, 4:10] = 7
            mask[8:16, 12:20] = 2 + (index % 3)
            mask[16:24, 20:28] = 4
            write_sample(root, split, f"sample_{index}", image, mask)


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = Path(tmpdir) / "data" / "segmentation"
        build_tiny_dataset(data_root)

        pairs, message = check_dataset(data_root)
        assert pairs is not None, message
        assert len(pairs["train"]) == 2
        assert len(pairs["val"]) == 2

        dataset = SegmentationDataset(pairs["train"], img_size=32, augment=True)
        image, mask = dataset[0]
        assert image.shape == (3, 32, 32)
        assert mask.shape == (32, 32)

        model = LightweightUNet(num_classes=11, base_channels=4)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        train_loss, train_acc, train_miou, _ = run_epoch(model, loader, criterion, optimizer, torch.device("cpu"), 11, train=True)
        val_loss, val_acc, val_miou, _ = run_epoch(model, loader, criterion, optimizer, torch.device("cpu"), 11, train=False)
        assert train_loss >= 0
        assert val_loss >= 0
        assert 0 <= train_acc <= 1
        assert 0 <= val_acc <= 1
        assert 0 <= train_miou <= 1
        assert 0 <= val_miou <= 1

        print("灾情感知及影响评估 segmentation training smoke test passed.")


if __name__ == "__main__":
    main()
