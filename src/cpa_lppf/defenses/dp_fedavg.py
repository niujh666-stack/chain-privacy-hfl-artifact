from __future__ import annotations

from typing import Any, Mapping

import torch

from cpa_lppf.defenses.lgs import LowSensitiveGradientSanitizer


class DPFedAvgDefense:
    name = "DP-FedAvg"
    use_par = False
    privacy_discriminator = None
    par_lambda = 0.0

    def __init__(self, cfg: Any):
        self.sanitizer = LowSensitiveGradientSanitizer(
            clip_norm=float(cfg.baselines.dp_clip_norm),
            noise_multiplier=float(cfg.baselines.dp_noise_multiplier),
            seed=int(cfg.seed),
        )

    def to(self, device: torch.device | str):
        return self

    def extra_parameters(self):
        return []

    def sanitize_low_update(self, update: Mapping[str, torch.Tensor], round_idx: int, client_id: int):
        return self.sanitizer(update, round_idx, client_id)

    def protect_high_update(self, update: Mapping[str, torch.Tensor], round_idx: int, client_id: int):
        return self.sanitizer(update, round_idx, client_id + 99991)

    def protect_observed_shares(self, shares, round_idx: int, client_id: int, colluding_edges: int):
        return shares[:colluding_edges]
