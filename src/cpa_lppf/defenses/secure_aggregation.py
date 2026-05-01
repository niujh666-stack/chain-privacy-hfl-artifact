from __future__ import annotations

from typing import Any, Mapping

import torch


class SecureAggregationDefense:
    """Secure aggregation baseline.

    It hides individual observed shares from the attacker but does not apply
    semantic representation regularization.
    """

    name = "Secure Aggregation"
    use_par = False
    privacy_discriminator = None
    par_lambda = 0.0

    def __init__(self, cfg: Any):
        self.seed = int(cfg.seed)

    def to(self, device: torch.device | str):
        return self

    def extra_parameters(self):
        return []

    def sanitize_low_update(self, update: Mapping[str, torch.Tensor], round_idx: int, client_id: int):
        return update

    def protect_high_update(self, update: Mapping[str, torch.Tensor], round_idx: int, client_id: int):
        return update

    def protect_observed_shares(self, shares, round_idx: int, client_id: int, colluding_edges: int):
        del round_idx, client_id, colluding_edges
        # A colluding party sees at most one unusable share in this baseline.
        return shares[:1]
