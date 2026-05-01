from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def get_targets(dataset: Dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if isinstance(targets, torch.Tensor):
            return targets.cpu().numpy().astype(int).tolist()
        return [int(x) for x in targets]
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(int(y))
    return labels


def get_client_ids(dataset: Dataset) -> list[int] | None:
    ids = getattr(dataset, "client_ids", None)
    if ids is None:
        return None
    return [int(x) for x in ids]


def iid_partition(num_items: int, num_clients: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(num_items)
    rng.shuffle(idx)
    return [part.astype(int).tolist() for part in np.array_split(idx, num_clients)]


def dirichlet_partition(labels: Sequence[int], num_clients: int, alpha: float, seed: int, min_size: int = 1) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels)
    for _ in range(100):
        parts: list[list[int]] = [[] for _ in range(num_clients)]
        for c in classes:
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            cut = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
            for client_id, split in enumerate(np.split(idx, cut)):
                parts[client_id].extend(split.astype(int).tolist())
        if min(len(p) for p in parts) >= min_size:
            for p in parts:
                rng.shuffle(p)
            return parts
    return iid_partition(len(labels), num_clients, seed)


def writer_partition(client_ids: Sequence[int], num_clients: int, seed: int) -> list[list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, uid in enumerate(client_ids):
        groups[int(uid)].append(idx)
    rng = np.random.default_rng(seed)
    writers = list(groups)
    rng.shuffle(writers)
    parts = [[] for _ in range(num_clients)]
    for pos, writer in enumerate(writers):
        parts[pos % num_clients].extend(groups[writer])
    return parts


def build_client_loaders(train_dataset: Dataset, cfg: Any) -> tuple[list[DataLoader], list[list[int]]]:
    labels = get_targets(train_dataset)
    num_clients = int(cfg.hfl.num_clients)
    partition = str(cfg.dataset.partition).lower()
    if partition == "writer" and get_client_ids(train_dataset) is not None:
        indices = writer_partition(get_client_ids(train_dataset) or [], num_clients, int(cfg.seed))
    elif partition == "iid":
        indices = iid_partition(len(train_dataset), num_clients, int(cfg.seed))
    else:
        indices = dirichlet_partition(labels, num_clients, float(cfg.hfl.dirichlet_alpha), int(cfg.seed))

    loaders = []
    for part in indices:
        subset = Subset(train_dataset, part)
        loaders.append(
            DataLoader(
                subset,
                batch_size=int(cfg.training.batch_size),
                shuffle=True,
                num_workers=int(cfg.dataset.num_workers),
                drop_last=False,
            )
        )
    return loaders, indices


def build_test_loader(test_dataset: Dataset, cfg: Any) -> DataLoader:
    return DataLoader(
        test_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.dataset.num_workers),
        drop_last=False,
    )
