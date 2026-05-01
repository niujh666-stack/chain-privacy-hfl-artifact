from __future__ import annotations

import argparse
from copy import deepcopy

from cpa_lppf.config import load_config
from cpa_lppf.experiments.run_defense_matrix import run as run_matrix


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="LPPF ablation helper")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    # This script is intentionally minimal: ablations are represented by copied YAML files
    # where one LPPF coefficient at a time is set to zero.
    cfg = load_config(args.config)
    del cfg
    run_matrix(args.config, args.output)


if __name__ == "__main__":
    main()
