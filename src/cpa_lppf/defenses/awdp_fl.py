from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np
import torch

from cpa_lppf.defenses.lgs import LowSensitiveGradientSanitizer, state_norm


class AWDPFLDefense:
    """Adaptive weighted DP-FL baseline.

    The clipping norm follows a running quantile of observed update norms.
    """

    name = "AWDP-FL"
    use_par = False
    privacy_discriminator = None
    par_lambda = 0.0

    def __init__(self, cfg: Any):
        self.base_noise = float(cfg.baselines.awdp_base_noise)
        self.quantile = float(cfg.baselines.awdp_target_clip_quantile)
        self.seed = int(cfg.seed)
        self.norms = deque(maxlen=200)

    def _sanitizer(self, update: Mapping[str, torch.Tensor]) -> LowSensitiveGradientSanitizer:
        n = float(state_norm(update))
        self.norms.append(n)
        clip = float(np.quantile(np.asarray(self.norms), self.quantile)) if self.norms else max(n, 1.0)
        clip = max(clip, 1e-3)
        return LowSensitiveGradientSanitizer(clip_norm=clip, noise_multiplier=self.base_noise, seed=self.seed)

    def to(self, device: torch.device | str):
        return self

    def extra_parameters(self):
        return []

    def sanitize_low_update(self, update: Mapping[str, torch.Tensor], round_idx: int, client_id: int):
        return self._sanitizer(update)(update, round_idx, client_id)

    def protect_high_update(self, update: Mapping[str, torch.Tensor], round_idx: int, client_id: int):
        return self._sanitizer(update)(update, round_idx, client_id + 17)

    def protect_observed_shares(self, shares, round_idx: int, client_id: int, colluding_edges: int):
        return shares[:colluding_edges]
