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


def open_set_identification_metrics(
    results: pd.DataFrame,
    *,
    threshold: float | None = None,
    predicted_identity_column: str = "top1_identity",
) -> dict[str, float | int]:
    """Compute standard rank-1 DIR/FNIR and FPIR without label conflation."""

    required = {"probe_type", "query_identity_id", predicted_identity_column}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"missing open-set metric columns: {sorted(missing)}")
    if threshold is None:
        if "accepted" not in results.columns:
            raise ValueError("accepted column is required when threshold is not provided")
        accepted = results["accepted"].astype(bool).to_numpy()
    else:
        if "top1_score" not in results.columns:
            raise ValueError("top1_score column is required when threshold is provided")
        accepted = results["top1_score"].astype(float).to_numpy() >= float(threshold)

    is_mated = results["probe_type"].astype(str).eq("registered").to_numpy()
    top1_correct = (
        results[predicted_identity_column].astype(str).to_numpy()
        == results["query_identity_id"].astype(str).to_numpy()
    )
    non_mated = ~is_mated
    dir_rank1 = float(np.mean(accepted[is_mated] & top1_correct[is_mated])) if is_mated.any() else 0.0
    fpir = float(np.mean(accepted[non_mated])) if non_mated.any() else 0.0
    return {
        "mated_count": int(np.sum(is_mated)),
        "non_mated_count": int(np.sum(non_mated)),
        "dir_rank1": dir_rank1,
        "fnir_rank1": 1.0 - dir_rank1,
        "fpir": fpir,
    }
