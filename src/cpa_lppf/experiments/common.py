from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from cpa_lppf.config import load_config, save_config
from cpa_lppf.data.datasets import dataset_metadata, get_dataset
from cpa_lppf.data.partition import build_client_loaders, build_test_loader
from cpa_lppf.defenses import make_defense
from cpa_lppf.hfl.trainer import HierarchicalFLTrainer
from cpa_lppf.models import make_model
from cpa_lppf.utils.logging import get_logger, setup_experiment_dir
from cpa_lppf.utils.seed import set_seed


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_trainer(cfg: Any, defense_name: str, output_dir: str | Path | None = None):
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    meta = dataset_metadata(cfg)
    model = make_model(num_classes=meta["num_classes"], channels=meta["channels"], image_size=meta["image_size"])
    defense = make_defense(defense_name, cfg, feature_dim=model.feature_dim)
    train_ds = get_dataset(cfg, train=True)
    test_ds = get_dataset(cfg, train=False)
    client_loaders, _ = build_client_loaders(train_ds, cfg)
    test_loader = build_test_loader(test_ds, cfg)
    trainer = HierarchicalFLTrainer(model, client_loaders, test_loader, cfg, defense, device)
    if output_dir:
        setup_experiment_dir(output_dir)
        save_config(cfg, Path(output_dir) / "resolved_config.yaml")
    return trainer, cfg, device


def save_results(rows: list[dict], path: str | Path) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df
