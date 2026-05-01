from __future__ import annotations

import torch
from torch import nn


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReverse.apply(x, lambd)


class PrivacyDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, private_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, private_classes),
        )

    def forward(self, features: torch.Tensor, grl_lambda: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(features, grl_lambda))
