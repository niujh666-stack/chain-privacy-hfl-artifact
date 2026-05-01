from __future__ import annotations

import torch
from torch import nn

from cpa_lppf.models.privacy_discriminator import PrivacyDiscriminator


class PrivacyAdversarialRegularizer(nn.Module):
    """PAR: lightweight discriminator trained through gradient reversal."""

    def __init__(self, feature_dim: int, private_classes: int, lambd: float = 0.05):
        super().__init__()
        self.discriminator = PrivacyDiscriminator(feature_dim=feature_dim, private_classes=private_classes)
        self.lambd = float(lambd)
        self.private_classes = int(private_classes)

    def forward(self, features: torch.Tensor, private_labels: torch.Tensor) -> torch.Tensor:
        logits = self.discriminator(features, grl_lambda=self.lambd)
        return torch.nn.functional.cross_entropy(logits, private_labels % self.private_classes)
