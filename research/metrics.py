from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def rank_at_k(results: pd.DataFrame, *, k: int) -> float:
    registered = results.loc[results["probe_type"] == "registered"]
    if registered.empty:
        return 0.0
    hits = []
    for _, row in registered.iterrows():
        ranked = list(row["ranked_identities"])[:k]
        hits.append(str(row["query_identity_id"]) in {str(value) for value in ranked})
    return float(np.mean(hits))


def brier_score(labels, probabilities) -> float:
    return float(brier_score_loss(labels, probabilities))


def expected_calibration_error(labels, probabilities, *, n_bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1.0 else probabilities <= upper)
        if not mask.any():
            continue
        confidence = float(np.mean(probabilities[mask]))
        accuracy = float(np.mean(labels[mask]))
        ece += float(np.mean(mask)) * abs(accuracy - confidence)
    return ece


def auroc(labels, probabilities) -> float:
    labels = np.asarray(labels, dtype=int)
    if len(set(labels.tolist())) < 2:
        return 0.0
    return float(roc_auc_score(labels, probabilities))
