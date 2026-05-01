from __future__ import annotations

from typing import Any, Mapping

import torch


class NoDefense:
    name = "No Defense"
    use_par = False
    privacy_discriminator = None
    par_lambda = 0.0
    private_classes = 10

    def to(self, device: torch.device | str):
        return self

    def extra_parameters(self):
        return []

    def sanitize_low_update(self, update: Mapping[str, torch.Tensor], round_idx: int, client_id: int):
        return update

    def protect_high_update(self, update: Mapping[str, torch.Tensor], round_idx: int, client_id: int):
        return update

    def protect_observed_shares(self, shares, round_idx: int, client_id: int, colluding_edges: int):
        return shares[:colluding_edges]


def make_defense(name: str, cfg: Any, feature_dim: int = 128):
    normalized = name.lower().replace("_", "-")
    if normalized in {"none", "no-defense", "nodefense"}:
        return NoDefense()
    if normalized in {"dp", "dp-fedavg", "user-level-dp-fl"}:
        from cpa_lppf.defenses.dp_fedavg import DPFedAvgDefense

        return DPFedAvgDefense(cfg)
    if normalized in {"secure-aggregation", "secagg"}:
        from cpa_lppf.defenses.secure_aggregation import SecureAggregationDefense

        return SecureAggregationDefense(cfg)
    if normalized in {"awdp", "awdp-fl"}:
        from cpa_lppf.defenses.awdp_fl import AWDPFLDefense

        return AWDPFLDefense(cfg)
    if normalized in {"lppf", "layered"}:
        from cpa_lppf.defenses.lppf import LPPFDefense

        return LPPFDefense(cfg, feature_dim=feature_dim)
    raise ValueError(f"unknown defense: {name}")
