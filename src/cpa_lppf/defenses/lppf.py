from __future__ import annotations

from typing import Any, Mapping

import torch

from cpa_lppf.defenses.crsa import CollusionResistantSecureAggregator
from cpa_lppf.defenses.lgs import LowSensitiveGradientSanitizer
from cpa_lppf.defenses.par import PrivacyAdversarialRegularizer
from cpa_lppf.defenses.shp import SelectiveHighSensitiveProtector


class LPPFDefense:
    """Layered Privacy Protection Framework.

    LGS interrupts Stage I, CRSA interrupts Stage II, SHP interrupts Stage III,
    and PAR weakens Stage IV.
    """

    name = "LPPF"
    use_par = True

    def __init__(self, cfg: Any, feature_dim: int = 128):
        self.lgs = LowSensitiveGradientSanitizer(
            clip_norm=float(cfg.lppf.lgs_clip_norm),
            noise_multiplier=float(cfg.lppf.lgs_noise_multiplier),
            seed=int(cfg.seed),
        )
        self.crsa = CollusionResistantSecureAggregator(
            noise_multiplier=float(cfg.lppf.crsa_noise_multiplier),
            seed=int(cfg.seed) + 888,
        )
        self.shp = SelectiveHighSensitiveProtector(
            mask_ratio=float(cfg.lppf.shp_mask_ratio),
            structured_dropout=float(cfg.lppf.shp_structured_dropout),
            seed=int(cfg.seed) + 999,
        )
        self.par = PrivacyAdversarialRegularizer(
            feature_dim=feature_dim,
            private_classes=int(cfg.lppf.private_classes),
            lambd=float(cfg.lppf.par_lambda),
        )
        self.privacy_discriminator = self.par.discriminator
        self.par_lambda = float(cfg.lppf.par_lambda)
        self.private_classes = int(cfg.lppf.private_classes)

    def to(self, device: torch.device | str):
        self.par.to(device)
        return self

    def extra_parameters(self):
        return list(self.par.parameters())

    def par_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.par(features, labels)

    def sanitize_low_update(self, update: Mapping[str, torch.Tensor], round_idx: int, client_id: int):
        return self.lgs(update, round_idx, client_id)

    def protect_high_update(self, update: Mapping[str, torch.Tensor], round_idx: int, client_id: int):
        return self.shp(update, round_idx, client_id)

    def protect_observed_shares(self, shares, round_idx: int, client_id: int, colluding_edges: int):
        return self.crsa.protect_observed_shares(shares, round_idx, client_id, colluding_edges)
