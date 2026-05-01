# Artifact notes

## What to verify

1. The HFL simulator separates low-sensitive and high-sensitive model components.
2. High-sensitive updates are split into shares and distributed to edge nodes.
3. CPA consumes low-sensitive updates, collusive high-sensitive shares, reconstructed high-sensitive parameters, and semantic priors.
4. LPPF deploys LGS, CRSA, SHP, and PAR at different stages.

## Quick review command

```bash
pip install -e .[dev]
pytest -q
python -m cpa_lppf.experiments.run_defense_matrix --config configs/synthetic.yaml --output runs/reviewer_quick
```

## Full review command

```bash
bash scripts/reproduce_all.sh
```

The full run downloads or loads public datasets and can take several hours on CPU. Use a CUDA device when available.
