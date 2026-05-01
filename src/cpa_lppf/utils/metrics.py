from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def topk_accuracy_from_probs(probs: np.ndarray, y_true: Iterable[int], ks: tuple[int, ...] = (1, 3)) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=np.int64)
    order = np.argsort(-probs, axis=1)
    out: dict[str, float] = {}
    for k in ks:
        hit = (order[:, :k] == y[:, None]).any(axis=1).mean() if len(y) else 0.0
        out[f"top{k}"] = float(hit)
    return out


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == target).float().mean().item())


def topk_accuracy_torch(logits: torch.Tensor, target: torch.Tensor, ks: tuple[int, ...] = (1, 3)) -> dict[str, float]:
    max_k = min(max(ks), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = {}
    for k in ks:
        k = min(k, logits.shape[1])
        res[f"top{k}"] = float(correct[:k].reshape(-1).float().sum(0).item() / max(1, target.numel()))
    return res


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    mse = torch.mean((x.detach().float() - y.detach().float()) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def ssim_simple(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    """A compact global SSIM approximation for artifact evaluation."""
    x = x.detach().float().reshape(x.shape[0], -1)
    y = y.detach().float().reshape(y.shape[0], -1)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = x.mean(dim=1)
    mu_y = y.mean(dim=1)
    var_x = x.var(dim=1, unbiased=False)
    var_y = y.var(dim=1, unbiased=False)
    cov = ((x - mu_x[:, None]) * (y - mu_y[:, None])).mean(dim=1)
    score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(score.mean().clamp(-1, 1).item())
