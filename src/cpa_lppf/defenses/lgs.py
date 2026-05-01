from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch


def state_norm(update: Mapping[str, torch.Tensor]) -> torch.Tensor:
    total = torch.tensor(0.0)
    for v in update.values():
        total = total + v.detach().float().pow(2).sum().cpu()
    return total.sqrt()


def clip_state_dict(update: Mapping[str, torch.Tensor], clip_norm: float) -> OrderedDict[str, torch.Tensor]:
    norm = state_norm(update)
    scale = min(1.0, float(clip_norm) / (float(norm) + 1e-12))
    return OrderedDict((k, v * scale) for k, v in update.items())


class LowSensitiveGradientSanitizer:
    """LGS: norm clipping followed by Gaussian perturbation."""

    def __init__(self, clip_norm: float = 1.0, noise_multiplier: float = 0.05, seed: int = 0):
        self.clip_norm = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.seed = int(seed)

    def __call__(self, update: Mapping[str, torch.Tensor], round_idx: int = 0, client_id: int = 0):
        clipped = clip_state_dict(update, self.clip_norm)
        out = OrderedDict()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + 1000003 * int(round_idx) + 9176 * int(client_id))
        for k, v in clipped.items():
            sigma = self.noise_multiplier * self.clip_norm
            noise = torch.randn(v.shape, generator=gen, dtype=v.dtype) * sigma
            out[k] = v + noise.to(v.device)
        return out
