from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cpa_lppf.attacks.base import AttackResult, labels_from_records, metrics_from_probs, vectorize_update
from cpa_lppf.utils.metrics import psnr, ssim_simple


class DLGAttack:
    """Deep Leakage from Gradients style baseline.

    The record-level ASR uses low-sensitive update features.  The static
    `invert_batch` method performs actual gradient matching for image recovery.
    """

    name = "DLG"

    def __init__(self, feature_dim: int = 4096, steps: int = 300, lr: float = 0.1):
        self.feature_dim = int(feature_dim)
        self.steps = int(steps)
        self.lr = float(lr)
        self.clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("lr", LogisticRegression(max_iter=1000, multi_class="auto")),
            ]
        )
        self.classes_ = None

    def _features(self, records: list) -> np.ndarray:
        return np.stack([vectorize_update(r.low_update, self.feature_dim) for r in records])

    def fit(self, records: list):
        x = self._features(records)
        y = labels_from_records(records)
        if len(np.unique(y)) < 2:
            y = y.copy(); y[0] = (y[0] + 1) % max(2, int(y.max()) + 2)
        self.clf.fit(x, y)
        self.classes_ = self.clf.named_steps["lr"].classes_
        return self

    def predict_proba(self, records: list, num_classes: int | None = None):
        local = self.clf.predict_proba(self._features(records))
        if num_classes is None:
            return local
        probs = np.zeros((len(records), num_classes), dtype=np.float64)
        for j, c in enumerate(self.classes_):
            if 0 <= int(c) < num_classes:
                probs[:, int(c)] = local[:, j]
        row_sum = probs.sum(axis=1, keepdims=True)
        return np.where(row_sum > 0, probs / row_sum, 1.0 / num_classes)

    def evaluate(self, records: list, num_classes: int | None = None) -> AttackResult:
        return metrics_from_probs(self.name, self.predict_proba(records, num_classes), labels_from_records(records))

    def invert_batch(self, model, images: torch.Tensor, labels: torch.Tensor, device: torch.device):
        model = model.to(device).eval()
        images = images.to(device)
        labels = labels.to(device).long()
        criterion = torch.nn.CrossEntropyLoss()
        logits = model(images)
        loss = criterion(logits, labels)
        true_grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=False)
        true_grads = [g.detach() for g in true_grads]

        dummy_x = torch.rand_like(images, requires_grad=True)
        dummy_logits = torch.randn(images.shape[0], model.num_classes, device=device, requires_grad=True)
        opt = torch.optim.Adam([dummy_x, dummy_logits], lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            pred = model(dummy_x.clamp(0, 1))
            soft_y = torch.softmax(dummy_logits, dim=1)
            dummy_loss = -(soft_y * torch.log_softmax(pred, dim=1)).sum(dim=1).mean()
            dummy_grads = torch.autograd.grad(dummy_loss, list(model.parameters()), create_graph=True)
            grad_loss = sum(torch.mean((dg - tg) ** 2) for dg, tg in zip(dummy_grads, true_grads))
            grad_loss.backward()
            opt.step()
        recon = dummy_x.detach().clamp(0, 1)
        return {"reconstruction": recon.cpu(), "psnr": psnr(recon.cpu(), images.cpu()), "ssim": ssim_simple(recon.cpu(), images.cpu())}
