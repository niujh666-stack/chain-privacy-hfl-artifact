from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cpa_lppf.experiments.run_defense_matrix import run as run_matrix
from cpa_lppf.utils.logging import setup_experiment_dir


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    out = setup_experiment_dir(args.output)
    frames = []
    for cfg in args.configs:
        name = Path(cfg).stem
        df = run_matrix(cfg, str(out / name))
        df.insert(0, "dataset_config", name)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(out / "cross_dataset.csv", index=False)


if __name__ == "__main__":
    main()
