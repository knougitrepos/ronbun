"""Fixed-gallery/threshold, query-weighted identity bootstrap (not FPIR)."""

from __future__ import annotations

import numpy as np
import pandas as pd

CLUSTER_CI_CONTRACT = "query-weighted-mated-identity-percentile-v1"


def _cluster_counts(identity_ids, events):
    ids = pd.Series(identity_ids, dtype=object).reset_index(drop=True)
    values = np.asarray(events)
    if (values.ndim != 2 or values.shape[1] == 0 or len(ids) != len(values)
            or ids.isna().any() or ids.astype(str).str.strip().eq("").any()):
        raise ValueError("aligned non-blank identities and a 2D event matrix required")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("binary events required")
    codes, unique = pd.factorize(ids.astype(str), sort=True)
    counts = np.column_stack([
        np.bincount(codes, minlength=len(unique)),
        *[np.bincount(codes, weights=values[:, i], minlength=len(unique))
          for i in range(values.shape[1])],
    ])
    return unique, counts


def _draw_counts(counts, *, resamples=2000, seed=8972):
    if (isinstance(resamples, (bool, np.bool_))
            or not isinstance(resamples, (int, np.integer)) or resamples < 100):
        raise ValueError("at least 100 integer resamples required")
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a non-negative integer")
    if len(counts) < 2:
        raise ValueError("at least two genuine identity clusters required")
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, counts.shape[1] - 1))
    for start in range(0, resamples, 64):
        weights = rng.multinomial(
            len(counts), np.full(len(counts), 1 / len(counts)),
            size=min(64, resamples - start),
        )
        draws[start:start + len(weights)] = (
            weights @ counts[:, 1:] / (weights @ counts[:, 0])[:, None]
        )
    return draws


def cluster_rate_draws(identity_ids, events, *, resamples=2000, seed=8972):
    """Jointly resample identities, then divide total successes by total queries.

    This does not average per-identity rates. Columns use identical draws so
    their differences are paired. Identity labels must be genuine, not query IDs.
    """
    _, counts = _cluster_counts(identity_ids, events)
    return _draw_counts(counts, resamples=resamples, seed=seed)


class TpirClusterAccumulator:
    """Bounded sufficient statistics, mergeable across query/ledger chunks."""

    def __init__(self):
        self.counts = {}
        self.labels_missing = False

    def update(self, identity_ids, events):
        if np.asarray(events).ndim != 2 or np.asarray(events).shape[1] != 2:
            raise ValueError("exactly two paired TPIR event columns required")
        if identity_ids is None:
            if len(events):
                self.labels_missing = True
            return
        unique, counts = _cluster_counts(identity_ids, events)
        for identity, row in zip(unique, counts):
            self.counts.setdefault(identity, np.zeros(3))[:] += row

    def summary(self):
        status = ("identity_labels_unavailable" if self.labels_missing else
                  "no_mated_queries" if not self.counts else
                  "insufficient_identity_clusters" if len(self.counts) < 2 else "ok")
        result = {
            "tpir_cluster_ci_contract": CLUSTER_CI_CONTRACT,
            "tpir_cluster_ci_status": status,
            "tpir_cluster_ci_identity_count": len(self.counts),
            "tpir_cluster_ci_resamples": 2000,
            "tpir_cluster_ci_seed": 8972,
            "tpir_cluster_ci_unit": "mated_identity_cluster",
            "tpir_cluster_ci_estimand": "query_weighted_rate",
            "tpir_cluster_ci_threshold_uncertainty_included": False,
            "tpir_cluster_ci_gallery_uncertainty_included": False,
            "tpir_cluster_ci_multiple_comparison_adjustment": "none",
            "fpir_ci_unit": "query",
        }
        intervals = np.full((2, 3), np.nan)
        if status == "ok":
            counts = np.stack([self.counts[key] for key in sorted(self.counts)])
            draws = _draw_counts(counts)
            intervals = np.quantile(
                np.column_stack([draws, draws[:, 1] - draws[:, 0]]), [.025, .975], axis=0,
            )
        for i, prefix in enumerate(("origin", "compressed", "compressed_minus_origin")):
            for j, suffix in enumerate(("low", "high")):
                result[f"{prefix}_tpir_at_rank_k_identity_cluster95_{suffix}"] = intervals[j, i]
        return result
