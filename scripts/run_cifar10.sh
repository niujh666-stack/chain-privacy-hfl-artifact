#!/usr/bin/env bash
set -euo pipefail
python -m cpa_lppf.experiments.run_attack --config configs/cifar10.yaml --output runs/cifar10_attack
python -m cpa_lppf.experiments.run_defense_matrix --config configs/cifar10.yaml --output runs/cifar10_defense_matrix
python -m cpa_lppf.analysis.plot_paper_figures --input runs/cifar10_defense_matrix/defense_matrix.csv --output runs/cifar10_figures
