from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cpa_lppf.attacks.base import AttackResult, labels_from_records, metrics_from_probs, vectorize_update


class PropertyInferenceAttack:
    """Client-specific property inference from update features."""

    name = "Property Inf."

    def __init__(self, feature_dim: int = 4096, source: str = "low"):
        self.feature_dim = int(feature_dim)
        self.source = source
        self.clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("lr", LogisticRegression(max_iter=1000, multi_class="auto")),
            ]
        )
        self.classes_: np.ndarray | None = None

    def _features(self, records: list) -> np.ndarray:
        feats = []
        for r in records:
            update = r.low_update if self.source == "low" else r.high_update
            feats.append(vectorize_update(update, self.feature_dim))
        return np.stack(feats) if feats else np.empty((0, self.feature_dim), dtype=np.float32)

    def fit(self, records: list):
        x = self._features(records)
        y = labels_from_records(records)
        if len(np.unique(y)) < 2:
            # LogisticRegression needs at least two classes; duplicate a tiny dummy class.
            y = y.copy()
            y[0] = (y[0] + 1) % max(2, int(y.max()) + 2)
        self.clf.fit(x, y)
        self.classes_ = self.clf.named_steps["lr"].classes_
        return self

    def predict_proba(self, records: list, num_classes: int | None = None) -> np.ndarray:
        x = self._features(records)
        local = self.clf.predict_proba(x)
        if num_classes is None:
            return local
        probs = np.zeros((len(records), num_classes), dtype=np.float64)
        for j, c in enumerate(self.classes_):
            if 0 <= int(c) < num_classes:
                probs[:, int(c)] = local[:, j]
        row_sum = probs.sum(axis=1, keepdims=True)
        probs = np.where(row_sum > 0, probs / row_sum, 1.0 / num_classes)
        return probs

    def evaluate(self, records: list, num_classes: int | None = None) -> AttackResult:
        probs = self.predict_proba(records, num_classes=num_classes)
        y = labels_from_records(records)
        return metrics_from_probs(self.name, probs, y)
