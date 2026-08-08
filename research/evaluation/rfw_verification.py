from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.datasets.rfw import (
    RFW_GROUPS,
    RFW_OFFICIAL_FOLD_COUNT,
    RFW_OFFICIAL_GENUINE_PER_FOLD,
)


_PAIR_COLUMNS = {
    "pair_id",
    "rfw_group",
    "fold_index",
    "left_image_id",
    "right_image_id",
    "is_genuine",
}


@dataclass(frozen=True)
class RFWVerificationResult:
    """Official-style RFW 9-fold calibration and held-out evaluation."""

    pair_scores: pd.DataFrame
    fold_metrics: pd.DataFrame
    group_summary: pd.DataFrame
    summary: dict[str, Any]


def _validate_inputs(
    pairs: pd.DataFrame,
    image_ids: Sequence[str],
    embeddings: np.ndarray,
    *,
    strict_official: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    missing = sorted(_PAIR_COLUMNS - set(pairs.columns))
    if missing:
        raise ValueError(f"RFW pairs are missing required columns: {missing}")
    if pairs.empty:
        raise ValueError("RFW pairs must be non-empty")
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 512:
        raise ValueError("RFW embeddings must have shape (n_images, 512)")
    if len(image_ids) != len(matrix):
        raise ValueError("image_ids and embeddings must have equal length")
    resolved_ids = [str(value) for value in image_ids]
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ValueError("RFW image_ids must be unique")
    if not np.isfinite(matrix).all():
        raise ValueError("RFW embeddings must be finite")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("RFW embeddings must have positive L2 norm")
    lookup = {image_id: index for index, image_id in enumerate(resolved_ids)}
    referenced = set(pairs["left_image_id"].astype(str)).union(
        pairs["right_image_id"].astype(str)
    )
    absent = sorted(referenced - set(lookup))
    if absent:
        raise ValueError(
            f"RFW pair images are missing embeddings: {absent[:3]}"
        )
    if strict_official:
        if set(pairs["rfw_group"].astype(str)) != set(RFW_GROUPS):
            raise ValueError("strict RFW evaluation requires all four official groups")
        counts = pairs.groupby(
            ["rfw_group", "fold_index", "is_genuine"]
        ).size()
        expected_index = pd.MultiIndex.from_product(
            [RFW_GROUPS, range(RFW_OFFICIAL_FOLD_COUNT), [False, True]],
            names=["rfw_group", "fold_index", "is_genuine"],
        )
        expected = pd.Series(
            RFW_OFFICIAL_GENUINE_PER_FOLD,
            index=expected_index,
            dtype=np.int64,
        )
        if not counts.reindex(expected_index, fill_value=0).equals(expected):
            raise ValueError(
                "strict RFW evaluation requires 300 genuine and 300 impostor "
                "pairs per group/fold"
            )
    return matrix / norms[:, None], lookup


def _accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    return float(np.mean((scores >= threshold) == labels))


def _select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    predictions = scores[:, None] >= thresholds[None, :]
    accuracies = np.mean(predictions == labels[:, None], axis=0)
    return float(thresholds[int(np.argmax(accuracies))])


def evaluate_rfw_10fold(
    pairs: pd.DataFrame,
    *,
    image_ids: Sequence[str],
    embeddings: np.ndarray,
    thresholds: Sequence[float] | None = None,
    strict_official: bool = True,
) -> RFWVerificationResult:
    """Evaluate cosine verification without fitting a codec on RFW.

    For each demographic group and held-out fold, the threshold is selected
    from the other folds only. Callers may pass origin embeddings or embeddings
    produced by a codec fitted on an external development dataset. This
    function deliberately implements neither PCA/PQ fitting nor open-set
    DIR/FPIR metrics.
    """

    normalized, lookup = _validate_inputs(
        pairs,
        image_ids,
        embeddings,
        strict_official=strict_official,
    )
    grid = np.asarray(
        np.linspace(-1.0, 1.0, 4001) if thresholds is None else thresholds,
        dtype=np.float64,
    )
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all():
        raise ValueError("thresholds must be a finite 1D sequence with >=2 values")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("thresholds must be strictly increasing")

    scored = pairs.copy().reset_index(drop=True)
    left_indices = np.asarray(
        [lookup[str(value)] for value in scored["left_image_id"]], dtype=np.int64
    )
    right_indices = np.asarray(
        [lookup[str(value)] for value in scored["right_image_id"]], dtype=np.int64
    )
    scored["cosine_score"] = np.sum(
        normalized[left_indices] * normalized[right_indices], axis=1
    )

    fold_rows: list[dict[str, Any]] = []
    for group, group_rows in scored.groupby("rfw_group", sort=True):
        folds = sorted(int(value) for value in group_rows["fold_index"].unique())
        if len(folds) < 2:
            raise ValueError(f"RFW group {group!r} requires at least two folds")
        for heldout_fold in folds:
            train = group_rows.loc[group_rows["fold_index"] != heldout_fold]
            test = group_rows.loc[group_rows["fold_index"] == heldout_fold]
            if train.empty or test.empty:
                raise ValueError("RFW train and held-out fold partitions must be non-empty")
            train_scores = train["cosine_score"].to_numpy(dtype=np.float64)
            train_labels = train["is_genuine"].to_numpy(dtype=bool)
            test_scores = test["cosine_score"].to_numpy(dtype=np.float64)
            test_labels = test["is_genuine"].to_numpy(dtype=bool)
            threshold = _select_threshold(train_scores, train_labels, grid)
            accepted = test_scores >= threshold
            genuine = test_labels
            impostor = ~genuine
            fold_rows.append(
                {
                    "rfw_group": str(group),
                    "heldout_fold": heldout_fold,
                    "threshold": threshold,
                    "train_pair_count": int(len(train)),
                    "test_pair_count": int(len(test)),
                    "accuracy": _accuracy(test_scores, test_labels, threshold),
                    "tar": float(np.mean(accepted[genuine])),
                    "far": float(np.mean(accepted[impostor])),
                }
            )

    fold_metrics = pd.DataFrame(fold_rows).sort_values(
        ["rfw_group", "heldout_fold"]
    ).reset_index(drop=True)
    group_summary = (
        fold_metrics.groupby("rfw_group", sort=True)
        .agg(
            fold_count=("heldout_fold", "size"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", lambda values: float(np.std(values, ddof=0))),
            mean_tar=("tar", "mean"),
            mean_far=("far", "mean"),
        )
        .reset_index()
    )
    group_accuracies = group_summary["mean_accuracy"].to_numpy(dtype=np.float64)
    summary = {
        "protocol": "rfw_official_groupwise_10fold_verification",
        "threshold_policy": "other_9_folds_when_strict_official",
        "open_set_protocol": False,
        "codec_fit_on_rfw": False,
        "pair_count": int(len(scored)),
        "image_count": int(len(set(scored["left_image_id"]).union(scored["right_image_id"]))),
        "macro_group_accuracy": float(np.mean(group_accuracies)),
        "group_accuracy_gap": float(np.max(group_accuracies) - np.min(group_accuracies)),
    }
    return RFWVerificationResult(
        pair_scores=scored,
        fold_metrics=fold_metrics,
        group_summary=group_summary,
        summary=summary,
    )
