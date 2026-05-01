from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_attack_bars(df: pd.DataFrame, output: str | Path, title: str = "Attack success rate") -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pivot = df.pivot_table(index="attack", values="top1_asr", aggfunc="mean")
    ax = pivot.plot(kind="bar", legend=False)
    ax.set_ylabel("Top-1 ASR")
    ax.set_title(title)
    ax.set_ylim(0, max(1.0, float(pivot.max().iloc[0]) * 1.15))
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def plot_defense_matrix(df: pd.DataFrame, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pivot = df.pivot_table(index="defense", columns="attack", values="top1_asr", aggfunc="mean")
    ax = pivot.plot(kind="bar")
    ax.set_ylabel("Top-1 ASR")
    ax.set_title("Defense by attack matrix")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return output
