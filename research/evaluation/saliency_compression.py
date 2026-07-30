from __future__ import annotations

from collections.abc import Callable
import hashlib
from typing import Sequence

import numpy as np
import pandas as pd


JOIN_KEYS = (
    "extraction_uid",
    "dataset_id",
    "sample_id",
    "model_uid",
)
PROFILE_KEYS = (
    "compression_family",
    "compression_profile",
)
LINEAGE_COLUMNS = ("origin_embedding_artifact_uid",)
DEFAULT_SALIENCY_FEATURES = (
    "quadrant_top_left",
    "quadrant_top_right",
    "quadrant_bottom_left",
    "quadrant_bottom_right",
    "left_eye_attention",
    "right_eye_attention",
    "nose_attention",
    "mouth_attention",
    "left_cheek_attention",
    "right_cheek_attention",
    "jaw_attention",
    "face_attention",
    "outside_face_attention",
    "saliency_entropy",
    "maximum_region_concentration",
    "left_right_asymmetry",
    "saliency_center_x",
    "saliency_center_y",
    "saliency_spread",
)
DEFAULT_GEOMETRY_METRICS = (
    "angular_error_rad",
    "reconstruction_mse",
    "cosine_to_origin",
)
DEFAULT_RETRIEVAL_METRICS = (
    "top1_score_drift",
    "agreement_with_origin",
    "threshold_crossing",
)
# Backward-compatible union for callers that intentionally analyze a table
# containing one retrieval row per geometry row. New code should use the
# geometry/retrieval-specific wrappers below.
DEFAULT_SENSITIVITY_METRICS = (
    *DEFAULT_GEOMETRY_METRICS,
    *DEFAULT_RETRIEVAL_METRICS,
)
BASE_ASSOCIATION_GROUP_COLUMNS = (
    "dataset_id",
    "model_uid",
    "compression_family",
    "compression_profile",
)
RESAMPLED_RANK_STRATEGY = "resampled"
WEIGHTED_RERANK_STRATEGY = "weighted_rerank"
WEIGHTED_RERANK_ALGORITHM_VERSION = "identity-cluster-weighted-rerank-v1"
RESAMPLED_RANK_ALGORITHM_VERSION = "identity-cluster-resampled-rank-v1"

AssociationProgressCallback = Callable[[str, dict[str, object]], None]


def annotate_compression_lineage(
    frame: pd.DataFrame,
    *,
    extraction_uid: str,
    dataset_id: str,
    model_uid: str,
    origin_embedding_artifact_uid: str,
) -> pd.DataFrame:
    """Attach frozen origin-population lineage to a compression result table.

    Existing non-empty lineage columns must already equal the supplied value;
    the helper never overwrites conflicting provenance. Retrieval tables may
    retain ``query_id`` because the strict join normalizes it to ``sample_id``.
    """

    result = frame.copy()
    if "sample_id" not in result and "query_id" not in result:
        raise ValueError("compression result must contain sample_id or query_id")
    values = {
        "extraction_uid": extraction_uid,
        "dataset_id": dataset_id,
        "model_uid": model_uid,
        "origin_embedding_artifact_uid": origin_embedding_artifact_uid,
    }
    for column, raw_value in values.items():
        value = str(raw_value).strip()
        if not value:
            raise ValueError(f"{column} must not be empty")
        if column in result:
            existing = result[column]
            if existing.isna().any() or not existing.astype(str).eq(value).all():
                raise ValueError(
                    f"existing {column} conflicts with frozen origin lineage"
                )
        else:
            result[column] = value
    return result


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    name: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _strict_false(series: pd.Series, *, name: str) -> None:
    if series.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    if pd.api.types.is_bool_dtype(series.dtype):
        converted = series.astype(bool)
    else:
        normalized = series.astype(str).str.strip().str.lower()
        if not normalized.isin({"true", "false"}).all():
            raise ValueError(f"{name} must contain strict boolean values")
        converted = normalized == "true"
    if converted.any():
        raise ValueError("saliency/compression join requires fallback-free artifacts")


def _strict_boolean(series: pd.Series, *, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{name} must contain strict boolean values")
    return normalized == "true"


def _validate_unique(
    frame: pd.DataFrame,
    keys: Sequence[str],
    *,
    name: str,
) -> None:
    if frame.loc[:, list(keys)].isna().any().any():
        raise ValueError(f"{name} join keys must not contain missing values")
    if frame.duplicated(list(keys)).any():
        examples = (
            frame.loc[frame.duplicated(list(keys), keep=False), list(keys)]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(f"{name} rows are not unique by {list(keys)}: {examples}")


def join_population_saliency_with_compression(
    saliency_features: pd.DataFrame,
    embedding_distortion: pd.DataFrame,
    *,
    retrieval_sensitivity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join all-sample origin saliency to one-or-more compression profiles.

    Saliency remains an origin-image feature table. It is never concatenated to
    the 512D vector or supplied to PCA/PQ. The long compression table is the
    left side, so each sample can repeat once per profile while the saliency
    row must remain unique.
    """

    _require_columns(
        saliency_features,
        [
            *JOIN_KEYS,
            *LINEAGE_COLUMNS,
            "saliency_spec_uid",
            "saliency_target_eligible",
            "heatmap_available",
        ],
        name="saliency_features",
    )
    _require_columns(
        embedding_distortion,
        [
            *JOIN_KEYS,
            *PROFILE_KEYS,
            *LINEAGE_COLUMNS,
            "origin_fallback_used",
        ],
        name="embedding_distortion",
    )
    _validate_unique(saliency_features, JOIN_KEYS, name="saliency_features")
    distortion_keys = (*JOIN_KEYS, *PROFILE_KEYS)
    _validate_unique(
        embedding_distortion,
        distortion_keys,
        name="embedding_distortion",
    )
    _strict_false(
        embedding_distortion["origin_fallback_used"],
        name="embedding_distortion.origin_fallback_used",
    )

    distortion = embedding_distortion.copy()
    if retrieval_sensitivity is not None:
        retrieval = retrieval_sensitivity.copy()
        if "query_id" in retrieval and "sample_id" not in retrieval:
            retrieval = retrieval.rename(columns={"query_id": "sample_id"})
        _require_columns(
            retrieval,
            [
                *JOIN_KEYS,
                *PROFILE_KEYS,
                *LINEAGE_COLUMNS,
                "origin_fallback_used",
            ],
            name="retrieval_sensitivity",
        )
        retrieval_extra_keys = [
            column
            for column in ("threshold_policy", "protocol_uid")
            if column in retrieval
        ]
        retrieval_keys = (*JOIN_KEYS, *PROFILE_KEYS, *retrieval_extra_keys)
        _validate_unique(
            retrieval,
            retrieval_keys,
            name="retrieval_sensitivity",
        )
        _strict_false(
            retrieval["origin_fallback_used"],
            name="retrieval_sensitivity.origin_fallback_used",
        )
        base_retrieval_keys = (*JOIN_KEYS, *PROFILE_KEYS)
        if retrieval.duplicated(list(base_retrieval_keys)).any():
            raise ValueError(
                "retrieval_sensitivity contains multiple policy/protocol rows per "
                "sample-profile and would duplicate geometry metrics; use "
                "join_population_saliency_with_retrieval for retrieval analysis"
            )
        for column in LINEAGE_COLUMNS:
            left_values = set(distortion[column].astype(str))
            right_values = set(retrieval[column].astype(str))
            if left_values != right_values:
                raise ValueError(
                    f"distortion/retrieval lineage differs for {column}: "
                    f"{left_values} != {right_values}"
                )
        retrieval_payload = retrieval.drop(
            columns=[
                *LINEAGE_COLUMNS,
                "origin_fallback_used",
            ],
        )
        retrieval_coverage = retrieval.loc[
            :,
            list((*JOIN_KEYS, *PROFILE_KEYS)),
        ].merge(
            distortion.loc[:, list((*JOIN_KEYS, *PROFILE_KEYS))],
            on=list((*JOIN_KEYS, *PROFILE_KEYS)),
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        if (retrieval_coverage["_merge"] != "both").any():
            missing = int((retrieval_coverage["_merge"] != "both").sum())
            raise ValueError(
                f"{missing} retrieval rows have no embedding distortion match"
            )
        distortion = distortion.merge(
            retrieval_payload,
            on=list((*JOIN_KEYS, *PROFILE_KEYS)),
            how="left",
            validate="one_to_one",
            indicator="_retrieval_merge",
            suffixes=("", "_retrieval"),
        )
        distortion["retrieval_metrics_available"] = (
            distortion["_retrieval_merge"] == "both"
        )
        distortion = distortion.drop(columns="_retrieval_merge")

    saliency_payload = saliency_features.copy()
    joined = distortion.merge(
        saliency_payload,
        on=list(JOIN_KEYS),
        how="left",
        validate="many_to_one",
        indicator="_saliency_merge",
        suffixes=("", "_saliency"),
    )
    if (joined["_saliency_merge"] != "both").any():
        missing = int((joined["_saliency_merge"] != "both").sum())
        raise ValueError(f"{missing} compression rows have no saliency sample row")
    joined = joined.drop(columns="_saliency_merge")
    for column in LINEAGE_COLUMNS:
        saliency_column = f"{column}_saliency"
        mismatch = joined[column].astype(str) != joined[saliency_column].astype(str)
        if mismatch.any():
            raise ValueError(
                f"{column} differs between compression and saliency artifacts"
            )
        joined = joined.drop(columns=saliency_column)
    return joined


def join_population_saliency_with_retrieval(
    saliency_features: pd.DataFrame,
    retrieval_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Join saliency to retrieval rows without copying geometry measurements.

    Retrieval rows intentionally remain long by threshold policy, protocol,
    and mated status. This separation prevents a multi-policy retrieval table
    from weighting the embedding-distortion population more than once.
    """

    _require_columns(
        saliency_features,
        [
            *JOIN_KEYS,
            *LINEAGE_COLUMNS,
            "saliency_spec_uid",
            "saliency_target_eligible",
            "heatmap_available",
        ],
        name="saliency_features",
    )
    _validate_unique(saliency_features, JOIN_KEYS, name="saliency_features")

    retrieval = retrieval_sensitivity.copy()
    if "query_id" in retrieval and "sample_id" not in retrieval:
        retrieval = retrieval.rename(columns={"query_id": "sample_id"})
    _require_columns(
        retrieval,
        [
            *JOIN_KEYS,
            *PROFILE_KEYS,
            *LINEAGE_COLUMNS,
            "origin_fallback_used",
            "threshold_policy",
            "is_mated",
        ],
        name="retrieval_sensitivity",
    )
    if (
        retrieval["threshold_policy"].isna().any()
        or retrieval["threshold_policy"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(
            "retrieval_sensitivity.threshold_policy must contain non-empty values"
        )
    retrieval["is_mated"] = _strict_boolean(
        retrieval["is_mated"],
        name="retrieval_sensitivity.is_mated",
    )
    retrieval_extra_keys = [
        column
        for column in (
            "protocol_uid",
            "threshold_source_split",
            "evaluation_split",
            "threshold_policy",
        )
        if column in retrieval
    ]
    retrieval_keys = (*JOIN_KEYS, *PROFILE_KEYS, *retrieval_extra_keys)
    _validate_unique(
        retrieval,
        retrieval_keys,
        name="retrieval_sensitivity",
    )
    _strict_false(
        retrieval["origin_fallback_used"],
        name="retrieval_sensitivity.origin_fallback_used",
    )

    joined = retrieval.merge(
        saliency_features,
        on=list(JOIN_KEYS),
        how="left",
        validate="many_to_one",
        indicator="_saliency_merge",
        suffixes=("", "_saliency"),
    )
    if (joined["_saliency_merge"] != "both").any():
        missing = int((joined["_saliency_merge"] != "both").sum())
        raise ValueError(f"{missing} retrieval rows have no saliency sample row")
    joined = joined.drop(columns="_saliency_merge")
    for column in LINEAGE_COLUMNS:
        saliency_column = f"{column}_saliency"
        mismatch = joined[column].astype(str) != joined[saliency_column].astype(str)
        if mismatch.any():
            raise ValueError(
                f"{column} differs between retrieval and saliency artifacts"
            )
        joined = joined.drop(columns=saliency_column)
    joined["retrieval_metrics_available"] = True
    return joined


def _spearman(left: pd.Series, right: pd.Series) -> float:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    valid = left_numeric.notna() & right_numeric.notna()
    if int(valid.sum()) < 3:
        return np.nan
    left_rank = left_numeric.loc[valid].rank(method="average")
    right_rank = right_numeric.loc[valid].rank(method="average")
    if left_rank.nunique() < 2 or right_rank.nunique() < 2:
        return np.nan
    return float(left_rank.corr(right_rank, method="pearson"))


def _pair_seed(
    seed: int,
    group_key: tuple[object, ...],
    saliency_feature: str,
    sensitivity_metric: str,
) -> int:
    payload = "\x1f".join(
        (
            str(seed),
            *(str(value) for value in group_key),
            saliency_feature,
            sensitivity_metric,
        )
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8],
        "big",
        signed=False,
    )


def _bootstrap_seed(
    seed: int,
    group_key: tuple[object, ...],
    cluster_identities: Sequence[object],
) -> int:
    digest = hashlib.sha256()
    for value in (seed, *group_key, *cluster_identities):
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def _emit_association_progress(
    progress: AssociationProgressCallback | None,
    message: str,
    **details: object,
) -> None:
    if progress is not None:
        progress(message, details)


_RankSpec = tuple[np.ndarray, np.ndarray | None, np.ndarray | None]


def _rank_spec(values: np.ndarray) -> _RankSpec:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    tie_starts_mask = np.concatenate(
        (
            np.ones(1, dtype=bool),
            sorted_values[1:] != sorted_values[:-1],
        )
    )
    if bool(tie_starts_mask.all()):
        return order, None, None
    tie_starts = np.flatnonzero(tie_starts_mask).astype(np.intp, copy=False)
    tie_codes = np.cumsum(tie_starts_mask, dtype=np.intp) - 1
    return order, tie_starts, tie_codes


def _weighted_average_rank_batch(
    row_weights: np.ndarray,
    rank_spec: _RankSpec,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct average ranks for integer-weighted expanded samples."""

    order, tie_starts, tie_codes = rank_spec
    if row_weights.ndim != 2 or row_weights.shape[1] != len(order):
        raise ValueError("row_weights shape does not match the rank specification")
    sorted_weights = row_weights[:, order].astype(np.float64, copy=False)
    if tie_starts is None:
        sorted_ranks = np.cumsum(sorted_weights, axis=1)
        sorted_ranks -= (sorted_weights - 1.0) / 2.0
    else:
        assert tie_codes is not None
        group_weights = np.add.reduceat(
            sorted_weights,
            tie_starts,
            axis=1,
        )
        group_ranks = np.cumsum(group_weights, axis=1)
        group_ranks -= (group_weights - 1.0) / 2.0
        sorted_ranks = group_ranks[:, tie_codes]
    if out is None:
        out = np.empty(sorted_weights.shape, dtype=np.float64)
    elif out.shape != sorted_weights.shape:
        raise ValueError("rank output shape does not match row_weights")
    out[:, order] = sorted_ranks
    return out


def _weighted_rerank_group_records(
    group: pd.DataFrame,
    *,
    group_key: tuple[object, ...],
    group_columns: Sequence[str],
    saliency_features: Sequence[str],
    sensitivity_metrics: Sequence[str],
    identity_column: str,
    bootstrap_repeats: int,
    confidence_level: float,
    seed: int,
    bootstrap_batch_size: int,
) -> list[dict[str, object]]:
    """Return exact Spearman statistics with scalable cluster-bootstrap CIs.

    Each sampled identity count becomes an integer weight on its rows. Average
    ranks are reconstructed exactly for that weighted expanded sample, including
    cross-identity ties, and weighted Pearson correlation is then evaluated.
    This is the same reranked statistic as concatenating sampled identity frames,
    without materializing repeated rows.
    """

    identities = group[identity_column].astype(str).to_numpy(copy=False)
    numeric_columns = tuple(
        dict.fromkeys((*saliency_features, *sensitivity_metrics))
    )
    numeric = {
        column: pd.to_numeric(group[column], errors="coerce").to_numpy(
            dtype=np.float64,
            copy=False,
        )
        for column in numeric_columns
    }
    pair_specs: list[dict[str, object]] = []
    mask_groups: dict[bytes, list[int]] = {}
    mask_examples: dict[bytes, np.ndarray] = {}
    for saliency_feature in saliency_features:
        left = numeric[str(saliency_feature)]
        for sensitivity_metric in sensitivity_metrics:
            right = numeric[str(sensitivity_metric)]
            valid = np.isfinite(left) & np.isfinite(right)
            valid_count = int(valid.sum())
            if valid_count >= 3:
                left_rank = (
                    pd.Series(left[valid], copy=False)
                    .rank(method="average")
                    .to_numpy(dtype=np.float64, copy=False)
                )
                right_rank = (
                    pd.Series(right[valid], copy=False)
                    .rank(method="average")
                    .to_numpy(dtype=np.float64, copy=False)
                )
                if np.unique(left_rank).size >= 2 and np.unique(right_rank).size >= 2:
                    observed = float(np.corrcoef(left_rank, right_rank)[0, 1])
                else:
                    observed = np.nan
            else:
                observed = np.nan
            pair_index = len(pair_specs)
            pair_specs.append(
                {
                    "saliency_feature": str(saliency_feature),
                    "sensitivity_metric": str(sensitivity_metric),
                    "valid": valid,
                    "sample_count": valid_count,
                    "identity_count": int(np.unique(identities[valid]).size),
                    "observed": observed,
                }
            )
            digest = hashlib.sha256(np.packbits(valid).tobytes()).digest()
            existing = mask_examples.get(digest)
            if existing is not None and not np.array_equal(existing, valid):
                raise RuntimeError("association validity-mask hash collision")
            mask_examples.setdefault(digest, valid)
            mask_groups.setdefault(digest, []).append(pair_index)

    bootstrap_values = np.full(
        (bootstrap_repeats, len(pair_specs)),
        np.nan,
        dtype=np.float64,
    )
    if bootstrap_repeats:
        for pair_indices in sorted(mask_groups.values(), key=lambda values: values[0]):
            valid = pair_specs[pair_indices[0]]["valid"]
            assert isinstance(valid, np.ndarray)
            valid_row_count = int(valid.sum())
            if valid_row_count < 3:
                continue
            valid_identities = identities[valid]
            cluster_identities, cluster_codes = np.unique(
                valid_identities,
                return_inverse=True,
            )
            cluster_count = int(len(cluster_identities))
            if cluster_count < 1:
                continue

            pair_names = [
                (
                    str(pair_specs[index]["saliency_feature"]),
                    str(pair_specs[index]["sensitivity_metric"]),
                )
                for index in pair_indices
            ]
            left_names = tuple(
                dict.fromkeys(name[0] for name in pair_names)
            )
            right_names = tuple(
                dict.fromkeys(name[1] for name in pair_names)
            )
            column_names = tuple(dict.fromkeys((*left_names, *right_names)))
            column_positions = {
                name: position for position, name in enumerate(column_names)
            }
            left_column_positions = [
                column_positions[name] for name in left_names
            ]
            right_column_positions = [
                column_positions[name] for name in right_names
            ]
            left_positions = {
                name: position for position, name in enumerate(left_names)
            }
            right_positions = {
                name: position for position, name in enumerate(right_names)
            }
            rank_specs = {
                name: _rank_spec(numeric[name][valid])
                for name in column_names
            }
            left_nonconstant = np.array(
                [
                    np.unique(numeric[name][valid]).size >= 2
                    for name in left_names
                ],
                dtype=bool,
            )
            right_nonconstant = np.array(
                [
                    np.unique(numeric[name][valid]).size >= 2
                    for name in right_names
                ],
                dtype=bool,
            )
            pair_targets = [
                (
                    pair_index,
                    left_positions[pair_name[0]],
                    right_positions[pair_name[1]],
                )
                for pair_index, pair_name in zip(pair_indices, pair_names)
            ]

            rng = np.random.default_rng(
                _bootstrap_seed(int(seed), group_key, cluster_identities)
            )
            probabilities = np.full(
                cluster_count,
                1.0 / cluster_count,
                dtype=np.float64,
            )
            for start in range(0, bootstrap_repeats, bootstrap_batch_size):
                stop = min(start + bootstrap_batch_size, bootstrap_repeats)
                counts = rng.multinomial(
                    cluster_count,
                    probabilities,
                    size=stop - start,
                ).astype(np.float64, copy=False)
                row_weights = counts[:, cluster_codes]
                ranked = np.empty(
                    (
                        stop - start,
                        len(column_names),
                        valid_row_count,
                    ),
                    dtype=np.float64,
                )
                for column_name in column_names:
                    _weighted_average_rank_batch(
                        row_weights,
                        rank_specs[column_name],
                        out=ranked[
                            :,
                            column_positions[column_name],
                            :,
                        ],
                    )
                correlations = np.full(
                    (
                        stop - start,
                        len(left_names),
                        len(right_names),
                    ),
                    np.nan,
                    dtype=np.float64,
                )
                for batch_index in range(stop - start):
                    weights = row_weights[batch_index]
                    sample_count = float(weights.sum())
                    if sample_count < 3.0:
                        continue
                    left_rank = ranked[
                        batch_index,
                        left_column_positions,
                        :,
                    ]
                    right_rank = ranked[
                        batch_index,
                        right_column_positions,
                        :,
                    ]
                    weighted_left = left_rank * weights
                    weighted_right = right_rank * weights
                    sum_left = weighted_left.sum(axis=1)
                    sum_right = weighted_right.sum(axis=1)
                    covariance = (
                        weighted_left @ right_rank.T
                        - np.outer(sum_left, sum_right) / sample_count
                    )
                    second_left = (weighted_left * left_rank).sum(axis=1)
                    second_right = (weighted_right * right_rank).sum(axis=1)
                    variance_left = (
                        second_left - sum_left * sum_left / sample_count
                    )
                    variance_right = (
                        second_right - sum_right * sum_right / sample_count
                    )
                    with np.errstate(divide="ignore", invalid="ignore"):
                        batch_correlations = covariance / np.sqrt(
                            np.outer(variance_left, variance_right)
                        )
                    tolerance_scale = 16.0 * np.finfo(np.float64).eps
                    invalid_left = (
                        ~left_nonconstant
                        | (
                            variance_left
                            <= tolerance_scale * np.maximum(second_left, 1.0)
                        )
                    )
                    invalid_right = (
                        ~right_nonconstant
                        | (
                            variance_right
                            <= tolerance_scale * np.maximum(second_right, 1.0)
                        )
                    )
                    invalid = (
                        invalid_left[:, np.newaxis]
                        | invalid_right[np.newaxis, :]
                    )
                    batch_correlations[invalid] = np.nan
                    finite = np.isfinite(batch_correlations)
                    batch_correlations[finite] = np.clip(
                        batch_correlations[finite],
                        -1.0,
                        1.0,
                    )
                    correlations[batch_index] = batch_correlations
                for pair_index, left_position, right_position in pair_targets:
                    bootstrap_values[start:stop, pair_index] = correlations[
                        :,
                        left_position,
                        right_position,
                    ]

    alpha = (1.0 - confidence_level) / 2.0
    records: list[dict[str, object]] = []
    for pair_index, pair_spec in enumerate(pair_specs):
        finite = bootstrap_values[:, pair_index]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            lower, upper = np.quantile(
                finite,
                [alpha, 1.0 - alpha],
            )
        else:
            lower = upper = np.nan
        record = {
            column: value for column, value in zip(group_columns, group_key)
        }
        record.update(
            {
                "saliency_feature": pair_spec["saliency_feature"],
                "sensitivity_metric": pair_spec["sensitivity_metric"],
                "sample_count": int(pair_spec["sample_count"]),
                "identity_count": int(pair_spec["identity_count"]),
                "spearman_rho": float(pair_spec["observed"]),
                "bootstrap_confidence_level": confidence_level,
                "bootstrap_ci_low": float(lower),
                "bootstrap_ci_high": float(upper),
                "bootstrap_valid_repeats": int(finite.size),
                "bootstrap_rank_strategy": WEIGHTED_RERANK_STRATEGY,
                "association_algorithm_version": (
                    WEIGHTED_RERANK_ALGORITHM_VERSION
                ),
            }
        )
        records.append(record)
    return records


def saliency_compression_associations(
    joined: pd.DataFrame,
    *,
    saliency_features: Sequence[str] = DEFAULT_SALIENCY_FEATURES,
    sensitivity_metrics: Sequence[str] = DEFAULT_SENSITIVITY_METRICS,
    group_columns: Sequence[str] = (
        "dataset_id",
        "model_uid",
        "compression_family",
        "compression_profile",
    ),
    identity_column: str = "identity_id",
    bootstrap_repeats: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
    bootstrap_rank_strategy: str = RESAMPLED_RANK_STRATEGY,
    bootstrap_batch_size: int = 4,
    progress: AssociationProgressCallback | None = None,
) -> pd.DataFrame:
    """Estimate profile-specific Spearman associations with identity bootstrap."""

    required = [
        *group_columns,
        identity_column,
        *saliency_features,
        *sensitivity_metrics,
    ]
    _require_columns(joined, required, name="joined")
    repeats = int(bootstrap_repeats)
    if isinstance(bootstrap_repeats, bool) or repeats != bootstrap_repeats:
        raise ValueError("bootstrap_repeats must be a non-negative integer")
    if repeats < 0:
        raise ValueError("bootstrap_repeats must be a non-negative integer")
    rank_strategy = str(bootstrap_rank_strategy)
    if rank_strategy not in {
        RESAMPLED_RANK_STRATEGY,
        WEIGHTED_RERANK_STRATEGY,
    }:
        raise ValueError(
            "bootstrap_rank_strategy must be 'resampled' or 'weighted_rerank'"
        )
    if (
        isinstance(bootstrap_batch_size, bool)
        or int(bootstrap_batch_size) != bootstrap_batch_size
        or int(bootstrap_batch_size) <= 0
    ):
        raise ValueError("bootstrap_batch_size must be a positive integer")
    batch_size = int(bootstrap_batch_size)
    level = float(confidence_level)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    alpha = (1.0 - level) / 2.0

    records: list[dict[str, object]] = []
    grouped = joined.groupby(
        list(group_columns),
        dropna=False,
        sort=True,
    )
    total_groups = int(grouped.ngroups)
    for group_index, (raw_group_key, group) in enumerate(grouped, start=1):
        group_key = (
            raw_group_key if isinstance(raw_group_key, tuple) else (raw_group_key,)
        )
        if rank_strategy == WEIGHTED_RERANK_STRATEGY:
            records.extend(
                _weighted_rerank_group_records(
                    group,
                    group_key=group_key,
                    group_columns=group_columns,
                    saliency_features=saliency_features,
                    sensitivity_metrics=sensitivity_metrics,
                    identity_column=identity_column,
                    bootstrap_repeats=repeats,
                    confidence_level=level,
                    seed=int(seed),
                    bootstrap_batch_size=batch_size,
                )
            )
            _emit_association_progress(
                progress,
                "association groups",
                completed=group_index,
                total=total_groups,
            )
            continue
        identities = sorted(
            str(identity) for identity in group[identity_column].drop_duplicates()
        )
        for saliency_feature in saliency_features:
            for sensitivity_metric in sensitivity_metrics:
                pair_frame = group[
                    [identity_column, saliency_feature, sensitivity_metric]
                ].copy()
                valid = (
                    pd.to_numeric(
                        pair_frame[saliency_feature],
                        errors="coerce",
                    ).notna()
                    & pd.to_numeric(
                        pair_frame[sensitivity_metric],
                        errors="coerce",
                    ).notna()
                )
                pair_frame = pair_frame.loc[valid]
                observed = _spearman(
                    pair_frame[saliency_feature],
                    pair_frame[sensitivity_metric],
                )
                bootstrap_values: list[float] = []
                if repeats and identities and len(pair_frame) >= 3:
                    rng = np.random.default_rng(
                        _pair_seed(
                            int(seed),
                            group_key,
                            str(saliency_feature),
                            str(sensitivity_metric),
                        )
                    )
                    pair_by_identity = {
                        str(identity): rows
                        for identity, rows in pair_frame.groupby(
                            identity_column,
                            dropna=False,
                            sort=True,
                        )
                    }
                    available_identities = sorted(pair_by_identity)
                    for _ in range(repeats):
                        sampled = rng.choice(
                            available_identities,
                            size=len(available_identities),
                            replace=True,
                        )
                        boot = pd.concat(
                            [pair_by_identity[str(identity)] for identity in sampled],
                            ignore_index=True,
                        )
                        value = _spearman(
                            boot[saliency_feature],
                            boot[sensitivity_metric],
                        )
                        if np.isfinite(value):
                            bootstrap_values.append(value)
                if bootstrap_values:
                    lower, upper = np.quantile(
                        bootstrap_values,
                        [alpha, 1.0 - alpha],
                    )
                else:
                    lower = upper = np.nan
                record = {
                    column: value for column, value in zip(group_columns, group_key)
                }
                record.update(
                    {
                        "saliency_feature": str(saliency_feature),
                        "sensitivity_metric": str(sensitivity_metric),
                        "sample_count": int(len(pair_frame)),
                        "identity_count": int(
                            pair_frame[identity_column].nunique(dropna=False)
                        ),
                        "spearman_rho": observed,
                        "bootstrap_confidence_level": level,
                        "bootstrap_ci_low": float(lower),
                        "bootstrap_ci_high": float(upper),
                        "bootstrap_valid_repeats": len(bootstrap_values),
                        "bootstrap_rank_strategy": RESAMPLED_RANK_STRATEGY,
                        "association_algorithm_version": (
                            RESAMPLED_RANK_ALGORITHM_VERSION
                        ),
                    }
                )
                records.append(record)
        _emit_association_progress(
            progress,
            "association groups",
            completed=group_index,
            total=total_groups,
        )
    return pd.DataFrame.from_records(records)


def saliency_geometry_associations(
    joined_geometry: pd.DataFrame,
    *,
    saliency_features: Sequence[str] = DEFAULT_SALIENCY_FEATURES,
    sensitivity_metrics: Sequence[str] = DEFAULT_GEOMETRY_METRICS,
    identity_column: str = "identity_id",
    bootstrap_repeats: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
    bootstrap_rank_strategy: str = RESAMPLED_RANK_STRATEGY,
    bootstrap_batch_size: int = 4,
    progress: AssociationProgressCallback | None = None,
) -> pd.DataFrame:
    """Estimate saliency associations for one geometry row per sample/profile."""

    _validate_unique(
        joined_geometry,
        (*JOIN_KEYS, *PROFILE_KEYS),
        name="joined_geometry",
    )
    result = saliency_compression_associations(
        joined_geometry,
        saliency_features=saliency_features,
        sensitivity_metrics=sensitivity_metrics,
        group_columns=BASE_ASSOCIATION_GROUP_COLUMNS,
        identity_column=identity_column,
        bootstrap_repeats=bootstrap_repeats,
        confidence_level=confidence_level,
        seed=seed,
        bootstrap_rank_strategy=bootstrap_rank_strategy,
        bootstrap_batch_size=bootstrap_batch_size,
        progress=progress,
    )
    result.insert(0, "analysis_scope", "geometry")
    return result


def saliency_retrieval_associations(
    joined_retrieval: pd.DataFrame,
    *,
    saliency_features: Sequence[str] = DEFAULT_SALIENCY_FEATURES,
    sensitivity_metrics: Sequence[str] = DEFAULT_RETRIEVAL_METRICS,
    identity_column: str = "identity_id",
    bootstrap_repeats: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
    bootstrap_rank_strategy: str = RESAMPLED_RANK_STRATEGY,
    bootstrap_batch_size: int = 4,
    progress: AssociationProgressCallback | None = None,
) -> pd.DataFrame:
    """Estimate retrieval associations separately by policy and mated status."""

    _require_columns(
        joined_retrieval,
        ("threshold_policy", "is_mated"),
        name="joined_retrieval",
    )
    normalized = joined_retrieval.copy()
    normalized["is_mated"] = _strict_boolean(
        normalized["is_mated"],
        name="joined_retrieval.is_mated",
    )
    optional_groups = tuple(
        column
        for column in (
            "protocol_uid",
            "threshold_source_split",
            "evaluation_split",
        )
        if column in normalized
    )
    group_columns = (
        *BASE_ASSOCIATION_GROUP_COLUMNS,
        *optional_groups,
        "threshold_policy",
        "is_mated",
    )
    result = saliency_compression_associations(
        normalized,
        saliency_features=saliency_features,
        sensitivity_metrics=sensitivity_metrics,
        group_columns=group_columns,
        identity_column=identity_column,
        bootstrap_repeats=bootstrap_repeats,
        confidence_level=confidence_level,
        seed=seed,
        bootstrap_rank_strategy=bootstrap_rank_strategy,
        bootstrap_batch_size=bootstrap_batch_size,
        progress=progress,
    )
    result.insert(0, "analysis_scope", "retrieval")
    return result
