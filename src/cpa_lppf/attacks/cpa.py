from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cpa_lppf.attacks.base import AttackResult, labels_from_records, metrics_from_probs, vectorize_update
from cpa_lppf.attacks.property_inference import PropertyInferenceAttack
from cpa_lppf.hfl.secret_sharing import ShamirTensorSecretSharing


class ChainPrivacyAttack:
    """Four-stage Chain-based Privacy Attack (CPA).

    Stage I: infer an intermediate attribute from low-sensitive updates.
    Stage II: collect high-sensitive shares under collusion.
    Stage III: reconstruct high-sensitive parameters with attribute guidance.
    Stage IV: infer the final private target.
    """

    name = "CPA"

    def __init__(self, feature_dim: int = 4096, num_classes: int = 10, share_number: int = 3, threshold: int = 2, seed: int = 0):
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.attr_attack = PropertyInferenceAttack(feature_dim=max(256, feature_dim // 2), source="low")
        self.sharing = ShamirTensorSecretSharing(num_shares=share_number, threshold=threshold, seed=seed)
        self.final_clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("lr", LogisticRegression(max_iter=1000, multi_class="auto")),
            ]
        )
        self.class_prototypes: dict[int, np.ndarray] = {}
        self.classes_: np.ndarray | None = None

    def _stage2_3_reconstruct(self, record) -> dict[str, torch.Tensor]:
        shares = list(record.observed_shares)
        try:
            return self.sharing.reconstruct_state_dict(shares)
        except Exception:
            # Near-threshold observations are treated as incomplete and noisy.
            # Use a zero-like reconstruction to preserve the attack pipeline shape.
            return OrderedDict((k, torch.zeros_like(v)) for k, v in record.high_update.items())

    def _fit_prototypes(self, records: list) -> None:
        by_class: dict[int, list[np.ndarray]] = {}
        for r in records:
            vec = vectorize_update(r.high_update, self.feature_dim)
            by_class.setdefault(int(r.private_label), []).append(vec)
        self.class_prototypes = {c: np.mean(v, axis=0) for c, v in by_class.items() if v}

    def _features(self, records: list, attr_probs: np.ndarray) -> np.ndarray:
        feats = []
        for i, r in enumerate(records):
            recon = self._stage2_3_reconstruct(r)
            recon_vec = vectorize_update(recon, self.feature_dim)
            attr = attr_probs[i]
            attr_label = int(np.argmax(attr))
            proto = self.class_prototypes.get(attr_label)
            if proto is not None:
                # Attribute-guided reconstruction prior.
                recon_vec = 0.75 * recon_vec + 0.25 * proto
            feats.append(np.concatenate([recon_vec, attr.astype(np.float32)], axis=0))
        return np.stack(feats) if feats else np.empty((0, self.feature_dim + self.num_classes), dtype=np.float32)

    def fit(self, records: list):
        self.attr_attack.fit(records)
        self._fit_prototypes(records)
        attr_probs = self.attr_attack.predict_proba(records, num_classes=self.num_classes)
        x = self._features(records, attr_probs)
        y = labels_from_records(records)
        if len(np.unique(y)) < 2:
            y = y.copy()
            y[0] = (y[0] + 1) % max(2, self.num_classes)
        self.final_clf.fit(x, y)
        self.classes_ = self.final_clf.named_steps["lr"].classes_
        return self

    def predict_proba(self, records: list) -> np.ndarray:
        attr_probs = self.attr_attack.predict_proba(records, num_classes=self.num_classes)
        x = self._features(records, attr_probs)
        local = self.final_clf.predict_proba(x)
        probs = np.zeros((len(records), self.num_classes), dtype=np.float64)
        for j, c in enumerate(self.classes_):
            if 0 <= int(c) < self.num_classes:
                probs[:, int(c)] = local[:, j]
        row_sum = probs.sum(axis=1, keepdims=True)
        return np.where(row_sum > 0, probs / row_sum, 1.0 / self.num_classes)

    def evaluate(self, records: list) -> AttackResult:
        probs = self.predict_proba(records)
        y = labels_from_records(records)
        result = metrics_from_probs(self.name, probs, y)
        result.extra["stage1_attr_top1"] = self.attr_attack.evaluate(records, num_classes=self.num_classes).top1_asr
        return result
