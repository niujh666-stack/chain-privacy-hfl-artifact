#!/usr/bin/env bash
set -euo pipefail
bash scripts/run_cifar10.sh
bash scripts/run_femnist.sh
python -m cpa_lppf.experiments.run_cross_dataset --configs configs/cifar10.yaml configs/femnist.yaml --output runs/cross_dataset
