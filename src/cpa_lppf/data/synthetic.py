from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticSpec:
    size: int
    num_classes: int = 10
    channels: int = 3
    image_size: int = 32
    seed: int = 0
    num_writers: int | None = None


class SyntheticVisionDataset(Dataset):
    """Class-structured image-like data used for fast reviewer checks."""

    def __init__(self, spec: SyntheticSpec):
        self.spec = spec
        rng = np.random.default_rng(spec.seed)
        labels = np.arange(spec.size) % spec.num_classes
        rng.shuffle(labels)
        self.targets = labels.astype(np.int64).tolist()

        prototypes = rng.normal(0.45, 0.20, size=(spec.num_classes, spec.channels, spec.image_size, spec.image_size))
        # Add simple class-specific bars so gradient and property attacks have signal.
        for c in range(spec.num_classes):
            row = (c * 3) % spec.image_size
            col = (c * 5) % spec.image_size
            prototypes[c, :, row : row + 2, :] += 0.35
            prototypes[c, :, :, col : col + 2] += 0.25
        prototypes = np.clip(prototypes, 0.0, 1.0)

        images = []
        for y in self.targets:
            x = prototypes[y] + rng.normal(0.0, 0.12, size=prototypes[y].shape)
            images.append(np.clip(x, 0.0, 1.0).astype(np.float32))
        self.data = torch.tensor(np.stack(images), dtype=torch.float32)

        writers = spec.num_writers or min(100, max(1, spec.size // 20))
        # Each writer is biased toward a subset of labels.
        self.client_ids = [int((i * 997 + self.targets[i] * 13) % writers) for i in range(spec.size)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.data[idx], int(self.targets[idx])
