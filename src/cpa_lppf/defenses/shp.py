from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch


class SelectiveHighSensitiveProtector:
    """SHP: sensitivity-aware masking plus structured dropout for high-sensitive heads."""

    def __init__(self, mask_ratio: float = 0.15, structured_dropout: float = 0.10, seed: int = 0):
        self.mask_ratio = float(mask_ratio)
        self.structured_dropout = float(structured_dropout)
        self.seed = int(seed)

    def __call__(self, update: Mapping[str, torch.Tensor], round_idx: int = 0, client_id: int = 0):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + 7127 * int(round_idx) + 104729 * int(client_id))
        out = OrderedDict()
        for k, v in update.items():
            x = v.detach().clone()
            if x.numel() == 0:
                out[k] = x
                continue
            # Mask the largest-magnitude elements because they are most stable for reconstruction.
            flat = x.abs().reshape(-1)
            num_mask = int(self.mask_ratio * flat.numel())
            mask = torch.ones_like(flat, dtype=torch.bool)
            if num_mask > 0:
                _, idx = torch.topk(flat, k=min(num_mask, flat.numel()), largest=True)
                mask[idx] = False
            x = (x.reshape(-1) * mask.to(x.dtype)).reshape_as(x)

            # Structured dropout on rows for matrix-like classifier weights.
            if x.ndim >= 2 and self.structured_dropout > 0:
                row_keep = torch.rand(x.shape[0], generator=gen) > self.structured_dropout
                shape = [x.shape[0]] + [1] * (x.ndim - 1)
                x = x * row_keep.to(x.dtype).reshape(shape)
            elif self.structured_dropout > 0:
                keep = torch.rand(x.shape, generator=gen) > self.structured_dropout
                x = x * keep.to(x.dtype)
            out[k] = x
        return out
