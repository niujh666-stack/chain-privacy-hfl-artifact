from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cpa_lppf.attacks import ChainPrivacyAttack, DLGAttack, GISMNAttack, PropertyInferenceAttack
from cpa_lppf.attacks.base import train_test_split_records
from cpa_lppf.config import load_config
from cpa_lppf.experiments.common import build_trainer, save_results
from cpa_lppf.utils.logging import get_logger, setup_experiment_dir


def build_attacks(cfg):
    return [
        DLGAttack(feature_dim=int(cfg.attack.feature_dim), steps=int(cfg.attack.dlg_steps)),
        GISMNAttack(feature_dim=int(cfg.attack.feature_dim), steps=int(cfg.attack.gismn_steps)),
        PropertyInferenceAttack(feature_dim=int(cfg.attack.feature_dim), source="low"),
        ChainPrivacyAttack(
            feature_dim=int(cfg.attack.feature_dim),
            num_classes=int(cfg.dataset.num_classes),
            share_number=int(cfg.hfl.share_number),
            threshold=int(cfg.hfl.reconstruction_threshold),
            seed=int(cfg.seed),
        ),
    ]


def run(config_path: str, output: str):
    logger = get_logger()
    out = setup_experiment_dir(output)
    cfg = load_config(config_path, overrides={"output_dir": output})
    defenses = ["none", "dp-fedavg", "secure-aggregation", "awdp-fl", "lppf"]
    rows = []
    for defense_name in defenses:
        logger.info("running defense: %s", defense_name)
        trainer, cfg, _ = build_trainer(cfg, defense_name, out / defense_name)
        trace, history, model = trainer.run(show_progress=True)
        torch.save({"trace": trace, "history": history, "model": model.state_dict()}, out / defense_name / "trace.pt")
        train_records, test_records = train_test_split_records(trace, seed=int(cfg.seed))
        for attack in build_attacks(cfg):
            attack.fit(train_records)
            result = attack.evaluate(test_records, num_classes=int(cfg.dataset.num_classes)) if attack.name != "CPA" else attack.evaluate(test_records)
            row = {
                "defense": getattr(trainer.defense, "name", defense_name),
                "attack": result.attack,
                "top1_asr": result.top1_asr,
                "top3_asr": result.top3_asr,
                "n": result.n,
            }
            row.update(result.extra)
            rows.append(row)
            logger.info("%s vs %s top1=%.4f", row["defense"], row["attack"], row["top1_asr"])
    return save_results(rows, out / "defense_matrix.csv")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run(args.config, args.output)


if __name__ == "__main__":
    main()
