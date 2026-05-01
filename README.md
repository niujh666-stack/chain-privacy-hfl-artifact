# CPA-LPPF-HFL

This repository is a research artifact for **Chain-based Privacy Leakage in Hierarchical Federated Learning: Attacks and Defenses**.  It implements a reproducible simulation framework for hierarchical federated learning (HFL), the four-stage Chain-based Privacy Attack (CPA), and the Layered Privacy Protection Framework (LPPF).

The code follows the paper setting:

- a client-side model is split into a low-sensitive convolutional trunk and a high-sensitive fully connected head;
- low-sensitive updates are visible to the cloud server;
- high-sensitive components are transmitted as threshold shares to edge nodes;
- a curious cloud server can collude with malicious edge nodes;
- CPA is implemented as intermediate attribute inference, high-sensitive share acquisition, attribute-guided parameter reconstruction, and final privacy inference;
- LPPF is implemented with LGS, CRSA, SHP, and PAR.

## Repository map

```text
configs/                    Experiment configurations for CIFAR-10, FEMNIST, and a quick synthetic run.
scripts/                    Shell commands used by reviewers to reproduce the artifact.
src/cpa_lppf/data/           Dataset loading and federated client partitioning.
src/cpa_lppf/models/         Split CNN backbone and privacy discriminator.
src/cpa_lppf/hfl/            Hierarchical FL clients, edge nodes, server, aggregation, and secret sharing.
src/cpa_lppf/attacks/        DLG, GI-SMN-style inversion, property inference, and CPA.
src/cpa_lppf/defenses/       DP-FedAvg, secure aggregation, AWDP-FL, and LPPF modules.
src/cpa_lppf/experiments/    End-to-end experiment entry points.
src/cpa_lppf/analysis/       Result summarization and figure generation.
tests/                       Lightweight unit tests.
docs/                        Artifact instructions for reviewers.
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

For a CPU-only quick check, install the CPU build of PyTorch according to the official PyTorch instructions, then run:

```bash
pip install -e .
pytest -q
python -m cpa_lppf.experiments.run_attack --config configs/synthetic.yaml --output runs/synthetic_attack
python -m cpa_lppf.experiments.run_defense_matrix --config configs/synthetic.yaml --output runs/synthetic_matrix
```

## Full runs

CIFAR-10 and FEMNIST/EMNIST runs are controlled by `configs/cifar10.yaml` and `configs/femnist.yaml`.

```bash
bash scripts/run_cifar10.sh
bash scripts/run_femnist.sh
```

The default full configuration follows the paper-level setup: 100 clients, 5 edge nodes, 3 shares, threshold 2, local epoch 1, AdamW, cosine learning-rate schedule, and a CNN split into a convolutional trunk and a fully connected head.  The synthetic configuration is intentionally smaller so reviewers can quickly verify the pipeline without downloading external datasets.

## Ethical use

This artifact is for controlled privacy research and review only.  Do not use it to attack real federated-learning deployments or infer information about users without explicit authorization.
