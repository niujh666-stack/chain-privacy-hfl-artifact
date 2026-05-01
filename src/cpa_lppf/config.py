from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "cpa_lppf",
    "seed": 2026,
    "device": "auto",
    "output_dir": "runs/default",
    "dataset": {
        "name": "synthetic-cifar10",
        "root": "data",
        "download": False,
        "synthetic": True,
        "synthetic_fallback": True,
        "synthetic_size": 2000,
        "test_size": 500,
        "num_classes": 10,
        "channels": 3,
        "image_size": 32,
        "num_workers": 0,
        "partition": "dirichlet",
    },
    "hfl": {
        "num_clients": 20,
        "num_edges": 5,
        "clients_per_edge": 4,
        "clients_per_round": 10,
        "rounds": 2,
        "local_epochs": 1,
        "dirichlet_alpha": 0.5,
        "share_number": 3,
        "reconstruction_threshold": 2,
        "colluding_edges": 2,
        "target_clients": 10,
    },
    "training": {
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
        "max_batches_per_client": None,
        "evaluate_every": 1,
    },
    "attack": {
        "feature_dim": 4096,
        "trials_per_target": 5,
        "dlg_steps": 300,
        "gismn_steps": 500,
        "private_classes": 10,
    },
    "lppf": {
        "lgs_clip_norm": 1.0,
        "lgs_noise_multiplier": 0.05,
        "crsa_noise_multiplier": 0.05,
        "shp_mask_ratio": 0.15,
        "shp_structured_dropout": 0.10,
        "par_lambda": 0.05,
        "private_classes": 10,
    },
    "baselines": {
        "dp_clip_norm": 1.0,
        "dp_noise_multiplier": 0.20,
        "awdp_base_noise": 0.12,
        "awdp_target_clip_quantile": 0.75,
    },
}


class Config(dict):
    """Dictionary with recursive attribute access."""

    def __init__(self, data: Mapping[str, Any]):
        super().__init__()
        for key, value in data.items():
            self[key] = self._wrap(value)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, Mapping):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def to_dict(self) -> dict[str, Any]:
        def unwrap(x: Any) -> Any:
            if isinstance(x, Config):
                return {k: unwrap(v) for k, v in x.items()}
            if isinstance(x, list):
                return [unwrap(v) for v in x]
            return x

        return unwrap(self)


def deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> Config:
    data = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        data = deep_update(data, loaded)
    if overrides:
        data = deep_update(data, overrides)
    return Config(data)


def save_config(cfg: Config | Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict() if isinstance(cfg, Config) else dict(cfg)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
