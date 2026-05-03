"""Lightweight UNet for 11-class post-disaster scene segmentation."""

from collections import OrderedDict

import torch
from torch import nn


NUM_CLASSES = 11

CLASS_NAMES = OrderedDict(
    [
        (0, "Background"),
        (1, "Water"),
        (2, "Building-No-Damage"),
        (3, "Building-Medium-Damage"),
        (4, "Building-Major-Damage"),
        (5, "Building-Total-Destruction"),
        (6, "Vehicle"),
        (7, "Road-Clear"),
        (8, "Road-Blocked"),
        (9, "Tree"),
        (10, "Pool"),
    ]
)

COLOR_MAP = OrderedDict(
    [
        (0, (0, 0, 0)),
        (1, (61, 230, 250)),
        (2, (180, 120, 120)),
        (3, (235, 255, 7)),
        (4, (255, 184, 6)),
        (5, (255, 0, 0)),
        (6, (255, 0, 245)),
        (7, (140, 140, 140)),
        (8, (160, 150, 20)),
        (9, (4, 250, 7)),
        (10, (255, 235, 0)),
    ]
)


class DoubleConv(nn.Module):
    """Two convolution layers with batch normalization and ReLU."""

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
    """Small UNet-style model suitable for local undergraduate project training."""

    def __init__(self, num_classes=NUM_CLASSES, base_channels=32):
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

    def _align(self, decoder_feature, encoder_feature):
        if decoder_feature.shape[-2:] == encoder_feature.shape[-2:]:
            return decoder_feature
        return torch.nn.functional.interpolate(
            decoder_feature,
            size=encoder_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self._align(self.up3(b), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self._align(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._align(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)
