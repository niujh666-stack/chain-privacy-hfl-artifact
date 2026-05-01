from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm import trange

from cpa_lppf.hfl.client import HFLClient
from cpa_lppf.hfl.edge import EdgeNode
from cpa_lppf.hfl.secret_sharing import ShamirTensorSecretSharing
from cpa_lppf.hfl.server import CloudServer
from cpa_lppf.utils.serialization import clone_state_dict


@dataclass
class TraceRecord:
    round_idx: int
    client_id: int
    private_label: int
    label_hist: list[int]
    low_update: dict[str, torch.Tensor]
    high_update: dict[str, torch.Tensor]
    observed_shares: list
    global_state: dict[str, torch.Tensor]
    train_loss: float


class HierarchicalFLTrainer:
    def __init__(self, model: torch.nn.Module, client_loaders, test_loader, cfg: Any, defense, device: torch.device):
        self.cfg = cfg
        self.server = CloudServer(model, device)
        self.clients = [HFLClient(i, loader, cfg) for i, loader in enumerate(client_loaders)]
        self.test_loader = test_loader
        self.device = device
        self.defense = defense
        self.rng = np.random.default_rng(int(cfg.seed))
        malicious = set(range(int(cfg.hfl.colluding_edges)))
        self.edges = [EdgeNode(i, malicious=i in malicious) for i in range(int(cfg.hfl.num_edges))]
        self.sharing = ShamirTensorSecretSharing(
            num_shares=int(cfg.hfl.share_number),
            threshold=int(cfg.hfl.reconstruction_threshold),
            seed=int(cfg.seed),
        )

    def _select_clients(self, round_idx: int) -> list[int]:
        n = len(self.clients)
        m = min(int(self.cfg.hfl.clients_per_round), n)
        # Random but deterministic selection per round.
        return self.rng.choice(n, size=m, replace=False).astype(int).tolist()

    def run(self, rounds: int | None = None, target_clients: int | None = None, show_progress: bool = True):
        rounds = int(rounds if rounds is not None else self.cfg.hfl.rounds)
        target_clients = int(target_clients if target_clients is not None else self.cfg.hfl.target_clients)
        target_set = set(range(min(target_clients, len(self.clients))))
        trace: list[TraceRecord] = []
        history: list[dict[str, float]] = []
        iterator = trange(rounds, disable=not show_progress, desc="HFL rounds")
        for r in iterator:
            selected = self._select_clients(r)
            low_items = []
            high_items = []
            round_losses = []
            global_before = clone_state_dict(self.server.model.state_dict())
            for cid in selected:
                result = self.clients[cid].train(self.server.model, self.defense, self.device, r)
                round_losses.append(result.train_loss)
                low_items.append((result.low_update, result.n_samples))
                high_items.append((result.high_update, result.n_samples))

                shares = self.sharing.share_state_dict(result.high_update)
                for edge_idx, share in enumerate(shares):
                    self.edges[edge_idx % len(self.edges)].receive_share(r, cid, share)
                observed = self.defense.protect_observed_shares(
                    shares,
                    round_idx=r,
                    client_id=cid,
                    colluding_edges=int(self.cfg.hfl.colluding_edges),
                )
                if cid in target_set:
                    trace.append(
                        TraceRecord(
                            round_idx=r,
                            client_id=cid,
                            private_label=result.private_label,
                            label_hist=result.label_hist,
                            low_update=result.low_update,
                            high_update=result.high_update,
                            observed_shares=observed,
                            global_state=global_before,
                            train_loss=result.train_loss,
                        )
                    )
            self.server.aggregate_and_update(low_items, high_items)
            if (r + 1) % int(self.cfg.training.evaluate_every) == 0 or r == rounds - 1:
                metrics = self.server.evaluate(self.test_loader)
                metrics.update({"round": r, "train_loss": float(np.mean(round_losses)) if round_losses else 0.0})
                history.append(metrics)
                iterator.set_postfix(acc=f"{metrics['acc']:.3f}", loss=f"{metrics['loss']:.3f}")
        return trace, history, self.server.model
