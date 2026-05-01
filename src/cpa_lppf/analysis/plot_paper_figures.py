from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cpa_lppf.utils.plotting import plot_attack_bars, plot_defense_matrix


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    df = pd.read_csv(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if "defense" in df.columns:
        plot_defense_matrix(df, out / "defense_matrix.png")
        no_def = df[df["defense"].astype(str).str.lower().isin(["no defense", "none", "nodefense"])]
        if not no_def.empty:
            plot_attack_bars(no_def, out / "attack_success.png")
    else:
        plot_attack_bars(df, out / "attack_success.png")


if __name__ == "__main__":
    main()
