from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cpa_lppf.data.synthetic import SyntheticSpec, SyntheticVisionDataset


class LEAFFEMNISTDataset(Dataset):
    """Loader for the LEAF/FEMNIST JSON format.

    Expected layout:
        data/raw/femnist/train/*.json
        data/raw/femnist/test/*.json
    Each JSON contains users, user_data, x, and y fields.
    """

    def __init__(self, root: str | Path, train: bool = True):
        split = "train" if train else "test"
        base = Path(root) / "raw" / "femnist" / split
        files = sorted(base.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No LEAF FEMNIST JSON files found in {base}")
        xs: list[np.ndarray] = []
        ys: list[int] = []
        client_ids: list[int] = []
        user_to_id: dict[str, int] = {}
        for file in files:
            with file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            for user in payload.get("users", []):
                user_to_id.setdefault(user, len(user_to_id))
                uid = user_to_id[user]
                data = payload["user_data"][user]
                for x, y in zip(data["x"], data["y"]):
                    arr = np.asarray(x, dtype=np.float32).reshape(1, 28, 28)
                    xs.append(arr)
                    ys.append(int(y))
                    client_ids.append(uid)
        self.data = torch.tensor(np.stack(xs), dtype=torch.float32)
        self.targets = ys
        self.client_ids = client_ids

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.data[idx], int(self.targets[idx])


def _torchvision_available() -> bool:
    try:
        import torchvision  # noqa: F401

        return True
    except Exception:
        return False


def get_dataset(cfg: Any, train: bool = True) -> Dataset:
    name = str(cfg.dataset.name).lower()
    size = int(cfg.dataset.synthetic_size if train else cfg.dataset.test_size)
    root = Path(cfg.dataset.root)
    if bool(cfg.dataset.synthetic) or name.startswith("synthetic"):
        return SyntheticVisionDataset(
            SyntheticSpec(
                size=size,
                num_classes=int(cfg.dataset.num_classes),
                channels=int(cfg.dataset.channels),
                image_size=int(cfg.dataset.image_size),
                seed=int(cfg.seed) + (0 if train else 10000),
                num_writers=int(cfg.hfl.num_clients),
            )
        )

    if name == "cifar10" and _torchvision_available():
        try:
            from torchvision import datasets, transforms

            transform = transforms.Compose([transforms.ToTensor()])
            ds = datasets.CIFAR10(root=str(root), train=train, download=bool(cfg.dataset.download), transform=transform)
            ds.client_ids = None
            return ds
        except Exception:
            if not bool(getattr(cfg.dataset, "synthetic_fallback", True)):
                raise

    if name in {"femnist", "emnist"}:
        try:
            return LEAFFEMNISTDataset(root, train=train)
        except Exception:
            pass
        if _torchvision_available():
            try:
                from torchvision import datasets, transforms

                transform = transforms.Compose([transforms.ToTensor()])
                # EMNIST ByClass has 62 classes and is a practical public proxy when LEAF FEMNIST is absent.
                ds = datasets.EMNIST(
                    root=str(root), split="byclass", train=train, download=bool(cfg.dataset.download), transform=transform
                )
                ds.client_ids = None
                return ds
            except Exception:
                if not bool(getattr(cfg.dataset, "synthetic_fallback", True)):
                    raise

    # Deterministic fallback keeps the artifact runnable without network access.
    return SyntheticVisionDataset(
        SyntheticSpec(
            size=size,
            num_classes=int(cfg.dataset.num_classes),
            channels=int(cfg.dataset.channels),
            image_size=int(cfg.dataset.image_size),
            seed=int(cfg.seed) + (0 if train else 10000),
            num_writers=int(cfg.hfl.num_clients),
        )
    )


def dataset_metadata(cfg: Any) -> dict[str, int]:
    return {
        "num_classes": int(cfg.dataset.num_classes),
        "channels": int(cfg.dataset.channels),
        "image_size": int(cfg.dataset.image_size),
    }
