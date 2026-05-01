#!/usr/bin/env bash
set -euo pipefail
python -m cpa_lppf.experiments.run_attack --config configs/femnist.yaml --output runs/femnist_attack
python -m cpa_lppf.experiments.run_defense_matrix --config configs/femnist.yaml --output runs/femnist_defense_matrix
