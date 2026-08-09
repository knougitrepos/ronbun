from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd


CASE_GROUPS = (
    "stable",
    "high_error",
    "rank_flip",
    "threshold_crossing",
)


def _required_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    name: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _strict_bool(series: pd.Series, *, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    converted: list[bool] = []
    for value in series.tolist():
        if isinstance(value, (bool, np.bool_)):
            converted.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            converted.append(value.strip().lower() == "true")
        else:
            raise ValueError(f"{name} must contain strict boolean values")
    return pd.Series(converted, index=series.index, dtype=bool)


def _tie_key(row: pd.Series, *, seed: int) -> str:
    value = "\x1f".join(
        (
            str(seed),
            str(row["query_id"]),
            str(row["compression_family"]),
            str(row["compression_profile"]),
            str(row.get("search_mode", "legacy_unspecified")),
            str(row.get("target_fpir", "legacy_unspecified")),
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _case_id(row: pd.Series) -> str:
    value = "\x1f".join(
        (
            str(row["selection_seed"]),
            str(row["query_id"]),
            str(row["compression_family"]),
            str(row["compression_profile"]),
            str(row.get("search_mode", "legacy_unspecified")),
            str(row.get("target_fpir", "legacy_unspecified")),
            str(row["case_group"]),
        )
    )
    return f"gradcam-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def _take(
    candidates: pd.DataFrame,
    *,
    count: int,
    group: str,
    ascending: tuple[bool, bool, bool],
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.assign(
            case_group=pd.Series(dtype="object"),
            case_priority_rank=pd.Series(dtype="int64"),
        )
    ordered = candidates.sort_values(
        ["_error", "_absolute_drift", "_tie_key"],
        ascending=list(ascending),
        kind="stable",
    ).head(count)
    ordered = ordered.copy()
    ordered["case_group"] = group
    ordered["case_priority_rank"] = np.arange(1, len(ordered) + 1)
    return ordered


def select_gradcam_cases(
    paired_metrics: pd.DataFrame,
    retrieval_metrics: pd.DataFrame,
    *,
    cases_per_group: int = 20,
    seed: int = 0,
    error_column: str = "angular_error_rad",
    score_drift_column: str = "top1_score_drift",
) -> pd.DataFrame:
    """Select deterministic, exclusive Grad-CAM cases per compression profile.

    Threshold crossings take precedence over rank flips.  Stable cases are the
    lowest-error/no-event rows, and high-error cases are selected from the
    remaining no-event rows.  The helper never changes Step 1 artifacts and
    rejects ambiguous duplicate rows (for example, unfiltered evaluation
    policies) instead of silently choosing one.
    """

    if isinstance(cases_per_group, bool) or int(cases_per_group) != cases_per_group:
        raise ValueError("cases_per_group must be a positive integer")
    cases_per_group = int(cases_per_group)
    if cases_per_group <= 0:
        raise ValueError("cases_per_group must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer")
    seed = int(seed)

    join_columns = ["query_id", "compression_family", "compression_profile"]
    _required_columns(
        paired_metrics,
        [
            "sample_id",
            "compression_family",
            "compression_profile",
            error_column,
            "origin_fallback_used",
        ],
        name="paired_metrics",
    )
    _required_columns(
        retrieval_metrics,
        [
            *join_columns,
            score_drift_column,
            "agreement_with_origin",
            "threshold_crossing",
            "origin_fallback_used",
        ],
        name="retrieval_metrics",
    )

    paired = paired_metrics.rename(columns={"sample_id": "query_id"}).copy()
    retrieval = retrieval_metrics.copy()
    paired_fallback = _strict_bool(
        paired["origin_fallback_used"],
        name="paired_metrics.origin_fallback_used",
    )
    retrieval_fallback = _strict_bool(
        retrieval["origin_fallback_used"],
        name="retrieval_metrics.origin_fallback_used",
    )
    if paired_fallback.any() or retrieval_fallback.any():
        raise ValueError("Grad-CAM case selection requires fallback-free artifacts")
    if paired.duplicated(join_columns).any():
        raise ValueError("paired_metrics rows must be unique by query and profile")
    retrieval_unique_columns = [*join_columns]
    if "search_mode" in retrieval:
        retrieval_unique_columns.append("search_mode")
    if "target_fpir" in retrieval:
        retrieval_unique_columns.append("target_fpir")
    if retrieval.duplicated(retrieval_unique_columns).any():
        raise ValueError(
            "retrieval_metrics rows must be unique by query, profile, search "
            "mode, and target FPIR; filter to one evaluation policy first"
        )

    paired_columns = [*join_columns, error_column]
    merged = retrieval.merge(
        paired.loc[:, paired_columns],
        on=join_columns,
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("paired and retrieval metrics have no matching cases")

    merged["_agreement"] = _strict_bool(
        merged["agreement_with_origin"],
        name="agreement_with_origin",
    )
    merged["_crossing"] = _strict_bool(
        merged["threshold_crossing"],
        name="threshold_crossing",
    )
    merged["_error"] = pd.to_numeric(merged[error_column], errors="coerce")
    drift = pd.to_numeric(merged[score_drift_column], errors="coerce")
    merged["_absolute_drift"] = drift.abs()
    merged["_tie_key"] = merged.apply(_tie_key, axis=1, seed=seed)
    finite = np.isfinite(merged["_error"]) & np.isfinite(merged["_absolute_drift"])
    merged = merged.loc[finite].copy()
    if merged.empty:
        raise ValueError("matching cases do not contain finite error and drift metrics")

    selected_parts: list[pd.DataFrame] = []
    profile_columns = ["compression_family", "compression_profile"]
    if "search_mode" in merged:
        profile_columns.append("search_mode")
    if "target_fpir" in merged:
        profile_columns.append("target_fpir")
    for _, profile_rows in merged.groupby(
        profile_columns,
        sort=True,
        dropna=False,
    ):
        crossing = _take(
            profile_rows.loc[profile_rows["_crossing"]],
            count=cases_per_group,
            group="threshold_crossing",
            ascending=(False, False, True),
        )
        used = set(crossing.index)

        rank_flip = _take(
            profile_rows.loc[~profile_rows["_crossing"] & ~profile_rows["_agreement"]],
            count=cases_per_group,
            group="rank_flip",
            ascending=(False, False, True),
        )
        used.update(rank_flip.index)

        stable_pool = profile_rows.loc[
            profile_rows["_agreement"]
            & ~profile_rows["_crossing"]
            & ~profile_rows.index.isin(used)
        ]
        stable = _take(
            stable_pool,
            count=cases_per_group,
            group="stable",
            ascending=(True, True, True),
        )
        used.update(stable.index)

        high_error = _take(
            profile_rows.loc[
                ~profile_rows["_crossing"]
                & profile_rows["_agreement"]
                & ~profile_rows.index.isin(used)
            ],
            count=cases_per_group,
            group="high_error",
            ascending=(False, False, True),
        )
        selected_parts.extend((stable, high_error, rank_flip, crossing))

    selected = pd.concat(selected_parts, ignore_index=True)
    selected["selection_seed"] = seed
    selected["cases_per_group"] = cases_per_group
    selected.insert(0, "case_id", selected.apply(_case_id, axis=1))
    selected = selected.sort_values(
        [
            "compression_family",
            "compression_profile",
            "case_group",
            "case_priority_rank",
            "_tie_key",
        ],
        kind="stable",
    )
    return selected.drop(
        columns=[
            "_agreement",
            "_crossing",
            "_error",
            "_absolute_drift",
            "_tie_key",
        ],
    ).reset_index(drop=True)


def select_representative_cases(
    paired_metrics: pd.DataFrame,
    retrieval_metrics: pd.DataFrame,
    **kwargs: object,
) -> pd.DataFrame:
    """Select post-analysis examples for visualization only.

    This compatibility wrapper deliberately runs after population saliency and
    compression artifacts have already been materialized. It must not be used
    to decide which samples receive Grad-CAM extraction.
    """

    return select_gradcam_cases(
        paired_metrics,
        retrieval_metrics,
        **kwargs,
    )


def select_population_representative_cases(
    joined_metrics: pd.DataFrame,
    *,
    cases_per_group: int = 20,
    seed: int = 0,
    error_column: str = "angular_error_rad",
    score_drift_column: str = "top1_score_drift",
) -> pd.DataFrame:
    """Select visualization rows only after the population-level strict join."""

    if isinstance(cases_per_group, bool) or int(cases_per_group) != cases_per_group:
        raise ValueError("cases_per_group must be a positive integer")
    count = int(cases_per_group)
    if count <= 0:
        raise ValueError("cases_per_group must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer")
    seed_value = int(seed)
    required = [
        "extraction_uid",
        "dataset_id",
        "sample_id",
        "model_uid",
        "compression_family",
        "compression_profile",
        error_column,
        score_drift_column,
        "agreement_with_origin",
        "threshold_crossing",
        "origin_fallback_used",
        "saliency_target_eligible",
        "heatmap_available",
    ]
    _required_columns(joined_metrics, required, name="joined_metrics")
    fallback = _strict_bool(
        joined_metrics["origin_fallback_used"],
        name="joined_metrics.origin_fallback_used",
    )
    if fallback.any():
        raise ValueError("representative cases require fallback-free artifacts")

    frame = joined_metrics.copy()
    if "retrieval_metrics_available" in frame:
        retrieval_available = _strict_bool(
            frame["retrieval_metrics_available"],
            name="retrieval_metrics_available",
        )
        frame = frame.loc[retrieval_available].copy()
        if frame.empty:
            raise ValueError("no joined rows have retrieval sensitivity metrics")
    frame["_eligible"] = _strict_bool(
        frame["saliency_target_eligible"],
        name="saliency_target_eligible",
    )
    frame["_heatmap"] = _strict_bool(
        frame["heatmap_available"],
        name="heatmap_available",
    )
    frame["_agreement"] = _strict_bool(
        frame["agreement_with_origin"],
        name="agreement_with_origin",
    )
    frame["_crossing"] = _strict_bool(
        frame["threshold_crossing"],
        name="threshold_crossing",
    )
    frame = frame.loc[frame["_eligible"] & frame["_heatmap"]].copy()
    if frame.empty:
        raise ValueError("no joined rows have an eligible, available heatmap")
    key_columns = [
        "extraction_uid",
        "dataset_id",
        "sample_id",
        "model_uid",
        "compression_family",
        "compression_profile",
    ]
    if "search_mode" in frame:
        key_columns.append("search_mode")
    if "target_fpir" in frame:
        key_columns.append("target_fpir")
    if frame.duplicated(key_columns).any():
        raise ValueError(
            "joined_metrics must be filtered to one retrieval policy per "
            "sample, model, compression profile, search mode, and target FPIR"
        )
    frame["_error"] = pd.to_numeric(frame[error_column], errors="coerce")
    frame["_absolute_drift"] = pd.to_numeric(
        frame[score_drift_column],
        errors="coerce",
    ).abs()
    tie_columns = [
        "extraction_uid",
        "dataset_id",
        "sample_id",
        "model_uid",
        "compression_family",
        "compression_profile",
    ]
    if "search_mode" in frame:
        tie_columns.append("search_mode")
    if "target_fpir" in frame:
        tie_columns.append("target_fpir")
    frame["_tie_key"] = frame.apply(
        lambda row: hashlib.sha256(
            "\x1f".join(
                [str(seed_value), *(str(row[column]) for column in tie_columns)]
            ).encode("utf-8")
        ).hexdigest(),
        axis=1,
    )
    frame = frame.loc[
        np.isfinite(frame["_error"]) & np.isfinite(frame["_absolute_drift"])
    ].copy()
    if frame.empty:
        raise ValueError("eligible joined rows have no finite error and drift")

    profile_columns = [
        "extraction_uid",
        "dataset_id",
        "model_uid",
        "compression_family",
        "compression_profile",
    ]
    if "search_mode" in frame:
        profile_columns.append("search_mode")
    if "target_fpir" in frame:
        profile_columns.append("target_fpir")
    selected_parts: list[pd.DataFrame] = []
    for _, group in frame.groupby(profile_columns, sort=True, dropna=False):
        crossing = _take(
            group.loc[group["_crossing"]],
            count=count,
            group="threshold_crossing",
            ascending=(False, False, True),
        )
        used = set(crossing.index)
        rank_flip = _take(
            group.loc[~group["_crossing"] & ~group["_agreement"]],
            count=count,
            group="rank_flip",
            ascending=(False, False, True),
        )
        used.update(rank_flip.index)
        stable = _take(
            group.loc[
                group["_agreement"] & ~group["_crossing"] & ~group.index.isin(used)
            ],
            count=count,
            group="stable",
            ascending=(True, True, True),
        )
        used.update(stable.index)
        high_error = _take(
            group.loc[
                group["_agreement"] & ~group["_crossing"] & ~group.index.isin(used)
            ],
            count=count,
            group="high_error",
            ascending=(False, False, True),
        )
        selected_parts.extend((stable, high_error, rank_flip, crossing))

    selected = pd.concat(selected_parts, ignore_index=True)
    selected["selection_seed"] = seed_value
    selected["cases_per_group"] = count
    selected.insert(
        0,
        "case_id",
        selected.apply(
            lambda row: (
                "gradcam-"
                + hashlib.sha256(
                    "\x1f".join(
                        [
                            str(seed_value),
                            *(str(row[column]) for column in tie_columns),
                            str(row["case_group"]),
                        ]
                    ).encode("utf-8")
                ).hexdigest()[:20]
            ),
            axis=1,
        ),
    )
    return (
        selected.sort_values(
            [*profile_columns, "case_group", "case_priority_rank", "_tie_key"],
            kind="stable",
        )
        .drop(
            columns=[
                "_eligible",
                "_heatmap",
                "_agreement",
                "_crossing",
                "_error",
                "_absolute_drift",
                "_tie_key",
            ]
        )
        .reset_index(drop=True)
    )
