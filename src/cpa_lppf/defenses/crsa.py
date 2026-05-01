from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch


class CollusionResistantSecureAggregator:
    """CRSA simulator.

    The training server still receives the aggregate update, but the adversarial
    view of individual high-sensitive shares is restricted and noised.
    """

    def __init__(self, noise_multiplier: float = 0.05, seed: int = 0):
        self.noise_multiplier = float(noise_multiplier)
        self.seed = int(seed)

    def protect_observed_shares(self, shares, round_idx: int, client_id: int, colluding_edges: int):
        # Return fewer useful shares than the nominal collusion set when possible.
        visible = max(1, int(colluding_edges) - 1)
        selected = shares[:visible]
        protected = []
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + 7919 * int(round_idx) + 104729 * int(client_id))
        for x_coord, share_dict in selected:
            noisy = OrderedDict()
            for k, v in share_dict.items():
                if torch.is_floating_point(v):
                    noise = torch.randn(v.shape, generator=gen, dtype=v.dtype) * self.noise_multiplier
                    noisy[k] = v + noise
                else:
                    noisy[k] = v.clone()
            protected.append((x_coord, noisy))
        return protected
