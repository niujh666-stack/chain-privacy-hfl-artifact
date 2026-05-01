from __future__ import annotations

import torch
from torch import nn


class SplitCNN(nn.Module):
    """CNN split into a low-sensitive trunk and high-sensitive head."""

    def __init__(self, channels: int, num_classes: int, feature_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(feature_dim, num_classes))
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, return_features: bool = False):
        z = self.trunk(x)
        logits = self.head(z)
        if return_features:
            return logits, z
        return logits


def make_model(num_classes: int, channels: int, image_size: int | None = None, feature_dim: int = 128) -> SplitCNN:
    del image_size
    return SplitCNN(channels=channels, num_classes=num_classes, feature_dim=feature_dim)
