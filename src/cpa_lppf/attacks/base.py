from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from cpa_lppf.utils.metrics import topk_accuracy_from_probs
from cpa_lppf.utils.serialization import state_dict_to_vector


@dataclass
class AttackResult:
    attack: str
    top1_asr: float
    top3_asr: float
    n: int
    extra: dict


def vectorize_update(update: dict[str, torch.Tensor], max_dim: int) -> np.ndarray:
    vec = state_dict_to_vector(update, max_dim=max_dim).numpy().astype(np.float32)
    if vec.size < max_dim:
        vec = np.pad(vec, (0, max_dim - vec.size))
    return vec[:max_dim]


def labels_from_records(records: Iterable) -> np.ndarray:
    return np.asarray([int(r.private_label) for r in records], dtype=np.int64)


def train_test_split_records(records: list, test_ratio: float = 0.4, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    cut = max(1, int(len(records) * (1.0 - test_ratio)))
    train_idx = idx[:cut]
    test_idx = idx[cut:] if cut < len(records) else idx[:cut]
    return [records[i] for i in train_idx], [records[i] for i in test_idx]


def metrics_from_probs(name: str, probs: np.ndarray, y: np.ndarray) -> AttackResult:
    if probs.shape[1] == 1:
        probs = np.concatenate([1.0 - probs, probs], axis=1)
    acc = topk_accuracy_from_probs(probs, y, ks=(1, min(3, probs.shape[1])))
    top3 = acc.get("top3", acc.get(f"top{min(3, probs.shape[1])}", acc["top1"]))
    return AttackResult(name, float(acc["top1"]), float(top3), int(len(y)), {})
