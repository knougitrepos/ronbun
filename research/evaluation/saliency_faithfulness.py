from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pandas as pd


FAITHFULNESS_METRICS = (
    "high_saliency_occlusion_score_drop",
    "low_saliency_occlusion_score_drop",
    "random_occlusion_score_drop",
    "faithfulness_gain_over_low_saliency",
    "faithfulness_gain_over_random",
)


def _quartile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("faithfulness stratification values must be finite")
    percentile = numeric.rank(method="average", pct=True).to_numpy()
    bins = np.minimum((percentile * 4.0).astype(np.int64), 3)
    return pd.Series([f"q{value + 1}" for value in bins], index=values.index)


def select_stratified_faithfulness_sample(
    candidates: pd.DataFrame,
    *,
    maximum_samples: int,
    seed: int,
    role_column: str = "protocol_role",
    norm_column: str = "raw_embedding_norm",
    score_column: str = "gradcam_target_score",
) -> pd.DataFrame:
    """Deterministically balance roles, raw-norm quartiles, and score quartiles.

    ``role_column`` may be a dataset-specific composite balance key (for
    example ``rfw_group|protocol_role``).  The original protocol columns stay
    in the returned frame for reporting.
    """

    required = {
        "sample_id",
        "identity_id",
        role_column,
        norm_column,
        score_column,
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"faithfulness candidates are missing columns: {missing}")
    limit = int(maximum_samples)
    if isinstance(maximum_samples, (bool, np.bool_)) or limit != maximum_samples:
        raise ValueError("maximum_samples must be a positive integer")
    if limit <= 0:
        raise ValueError("maximum_samples must be a positive integer")
    frame = candidates.copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("faithfulness candidates must not be empty")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("faithfulness candidate sample_id must be unique")
    frame[role_column] = frame[role_column].astype(str)
    frame["raw_norm_bin"] = (
        frame.groupby(role_column, sort=True, group_keys=False)[norm_column]
        .apply(_quartile)
        .sort_index()
    )
    frame["target_score_bin"] = (
        frame.groupby(role_column, sort=True, group_keys=False)[score_column]
        .apply(_quartile)
        .sort_index()
    )
    frame["faithfulness_stratum"] = (
        frame[role_column]
        + "|norm="
        + frame["raw_norm_bin"]
        + "|score="
        + frame["target_score_bin"]
    )
    frame["selection_hash"] = frame["sample_id"].astype(str).map(
        lambda sample_id: hashlib.sha256(
            f"{int(seed)}\x1f{sample_id}".encode("utf-8")
        ).hexdigest()
    )
    groups = {
        str(name): group.sort_values("selection_hash", kind="stable").index.tolist()
        for name, group in frame.groupby("faithfulness_stratum", sort=True)
    }
    selected: list[int] = []
    positions = {name: 0 for name in groups}
    names = sorted(groups)
    target = min(limit, len(frame))
    while len(selected) < target:
        progressed = False
        for name in names:
            position = positions[name]
            if position >= len(groups[name]):
                continue
            selected.append(groups[name][position])
            positions[name] += 1
            progressed = True
            if len(selected) == target:
                break
        if not progressed:
            break
    result = frame.loc[selected].copy().reset_index(drop=True)
    result["faithfulness_selection_index"] = np.arange(len(result), dtype=np.int64)
    return result


def _cluster_bootstrap_mean_ci(
    frame: pd.DataFrame,
    metric: str,
    *,
    identity_column: str,
    repeats: int,
    confidence_level: float,
    seed: int,
    group_name: str,
) -> tuple[float, float]:
    identities = frame[identity_column].astype(str).to_numpy()
    values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(values)
    identities = identities[valid]
    values = values[valid]
    clusters, codes = np.unique(identities, return_inverse=True)
    if len(values) == 0 or len(clusters) == 0 or repeats == 0:
        return np.nan, np.nan
    digest = hashlib.sha256(
        f"{int(seed)}\x1f{group_name}\x1f{metric}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    estimates = np.empty(repeats, dtype=np.float64)
    probabilities = np.full(len(clusters), 1.0 / len(clusters))
    for index in range(repeats):
        weights = rng.multinomial(len(clusters), probabilities)
        row_weights = weights[codes]
        denominator = int(row_weights.sum())
        estimates[index] = (
            np.sum(values * row_weights) / denominator
            if denominator
            else np.nan
        )
    alpha = (1.0 - confidence_level) / 2.0
    return (
        float(np.nanquantile(estimates, alpha)),
        float(np.nanquantile(estimates, 1.0 - alpha)),
    )


def summarize_faithfulness(
    rows: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("protocol_role",),
    identity_column: str = "identity_id",
    bootstrap_repeats: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarize paired occlusion effects with identity-cluster bootstrap CIs."""

    missing = sorted(
        {identity_column, *FAITHFULNESS_METRICS}.difference(rows.columns)
    )
    if missing:
        raise ValueError(f"faithfulness rows are missing columns: {missing}")
    repeats = int(bootstrap_repeats)
    if isinstance(bootstrap_repeats, (bool, np.bool_)) or repeats != bootstrap_repeats:
        raise ValueError("bootstrap_repeats must be a non-negative integer")
    if repeats < 0:
        raise ValueError("bootstrap_repeats must be a non-negative integer")
    level = float(confidence_level)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")

    group_specs: list[tuple[str, pd.DataFrame]] = [("all", rows)]
    for column in group_columns:
        if column not in rows:
            raise ValueError(f"faithfulness rows are missing group column: {column}")
        for value, group in rows.groupby(column, sort=True, dropna=False):
            group_specs.append((f"{column}={value}", group))

    records: list[dict[str, object]] = []
    for group_name, group in group_specs:
        for metric in FAITHFULNESS_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            valid = values.notna() & np.isfinite(values)
            usable = group.loc[valid]
            lower, upper = _cluster_bootstrap_mean_ci(
                usable,
                metric,
                identity_column=identity_column,
                repeats=repeats,
                confidence_level=level,
                seed=int(seed),
                group_name=group_name,
            )
            numeric = values.loc[valid].to_numpy(np.float64)
            records.append(
                {
                    "group": group_name,
                    "metric": metric,
                    "sample_count": int(len(numeric)),
                    "identity_count": int(
                        usable[identity_column].astype(str).nunique()
                    ),
                    "mean": float(np.mean(numeric)) if len(numeric) else np.nan,
                    "median": float(np.median(numeric)) if len(numeric) else np.nan,
                    "positive_fraction": (
                        float(np.mean(numeric > 0.0)) if len(numeric) else np.nan
                    ),
                    "mean_ci_lower": lower,
                    "mean_ci_upper": upper,
                    "confidence_level": level,
                    "bootstrap_method": "identity_cluster",
                    "bootstrap_repeats": repeats,
                }
            )
    return pd.DataFrame.from_records(records)
