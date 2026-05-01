#!/usr/bin/env bash
set -euo pipefail
python -m cpa_lppf.analysis.plot_paper_figures --input "$1" --output "${2:-runs/figures}"
