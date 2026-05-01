from __future__ import annotations

import torch

from cpa_lppf.hfl.aggregation import combine_updates, fedavg
from cpa_lppf.utils.metrics import accuracy
from cpa_lppf.utils.serialization import apply_delta_to_model


class CloudServer:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def aggregate_and_update(self, low_items, high_items) -> dict[str, torch.Tensor]:
        low_avg = fedavg(low_items)
        high_avg = fedavg(high_items)
        delta = combine_updates(low_avg, high_avg)
        apply_delta_to_model(self.model, delta)
        return delta

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device).long()
            logits = self.model(x)
            loss_sum += float(criterion(logits, y).cpu().item())
            correct += int((logits.argmax(dim=1) == y).sum().cpu().item())
            total += int(y.numel())
        return {"loss": loss_sum / max(1, total), "acc": correct / max(1, total)}
