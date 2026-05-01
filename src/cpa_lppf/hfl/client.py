from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from cpa_lppf.models.split_model import delta_split
from cpa_lppf.utils.serialization import clone_state_dict


@dataclass
class ClientResult:
    client_id: int
    n_samples: int
    low_update: dict[str, torch.Tensor]
    high_update: dict[str, torch.Tensor]
    private_label: int
    label_hist: list[int]
    train_loss: float


class HFLClient:
    def __init__(self, client_id: int, loader, cfg: Any):
        self.client_id = int(client_id)
        self.loader = loader
        self.cfg = cfg

    def _label_hist(self, num_classes: int) -> list[int]:
        hist = [0 for _ in range(num_classes)]
        dataset = self.loader.dataset
        indices = getattr(dataset, "indices", None)
        base = getattr(dataset, "dataset", dataset)
        targets = getattr(base, "targets", None)
        if indices is not None and targets is not None:
            for idx in indices:
                hist[int(targets[idx])] += 1
        else:
            for _, y in self.loader:
                for val in y.view(-1).tolist():
                    hist[int(val)] += 1
        return hist

    def train(self, global_model: nn.Module, defense, device: torch.device, round_idx: int) -> ClientResult:
        local_model = type(global_model)(
            channels=global_model.trunk[0].in_channels,
            num_classes=global_model.num_classes,
            feature_dim=global_model.feature_dim,
        )
        local_model.load_state_dict(clone_state_dict(global_model.state_dict(), cpu=False))
        local_model.to(device)
        defense.to(device)
        local_model.train()

        params = list(local_model.parameters()) + list(defense.extra_parameters())
        optimizer = torch.optim.AdamW(params, lr=float(self.cfg.training.lr), weight_decay=float(self.cfg.training.weight_decay))
        criterion = nn.CrossEntropyLoss()
        losses = []
        max_batches = self.cfg.training.max_batches_per_client
        for epoch in range(int(self.cfg.hfl.local_epochs)):
            for batch_idx, (x, y) in enumerate(self.loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                x = x.to(device)
                y = y.to(device).long()
                optimizer.zero_grad(set_to_none=True)
                if getattr(defense, "use_par", False):
                    logits, features = local_model(x, return_features=True)
                    task_loss = criterion(logits, y)
                    privacy_loss = defense.par_loss(features, y)
                    loss = task_loss + privacy_loss
                else:
                    logits = local_model(x)
                    loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))

        global_state = clone_state_dict(global_model.state_dict())
        local_state = clone_state_dict(local_model.state_dict())
        low, high = delta_split(local_state, global_state)
        low = defense.sanitize_low_update(low, round_idx, self.client_id)
        high = defense.protect_high_update(high, round_idx, self.client_id)
        hist = self._label_hist(int(self.cfg.dataset.num_classes))
        private_label = int(max(range(len(hist)), key=lambda i: hist[i])) if hist else 0
        return ClientResult(
            client_id=self.client_id,
            n_samples=len(self.loader.dataset),
            low_update=low,
            high_update=high,
            private_label=private_label,
            label_hist=hist,
            train_loss=float(sum(losses) / max(1, len(losses))),
        )
