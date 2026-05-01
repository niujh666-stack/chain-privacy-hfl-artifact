from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str = "cpa_lppf", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    return logger


def setup_experiment_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)
    return path
