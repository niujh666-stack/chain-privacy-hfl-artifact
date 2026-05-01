from __future__ import annotations

import argparse

import pandas as pd


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    df = pd.read_csv(args.input)
    cols = [c for c in ["defense", "attack", "top1_asr", "top3_asr", "n"] if c in df.columns]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
