# Reviewer guide

The artifact is organized so reviewers can inspect each paper component independently.

- `src/cpa_lppf/attacks/cpa.py` implements Algorithm 1 from the paper.
- `src/cpa_lppf/defenses/lppf.py` composes the four LPPF modules.
- `src/cpa_lppf/hfl/trainer.py` creates the hierarchical FL trace consumed by the attacks.
- `src/cpa_lppf/hfl/secret_sharing.py` implements threshold share generation and reconstruction.
- `src/cpa_lppf/experiments/run_defense_matrix.py` generates the defense-by-attack matrix used for Table 2 style analysis.

The synthetic configuration is not meant to reproduce the exact paper numbers; it validates the pipeline quickly.  Use `configs/cifar10.yaml` and `configs/femnist.yaml` for paper-scale evaluation.
