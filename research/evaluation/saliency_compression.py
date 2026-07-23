from __future__ import annotations

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
    "outside_face_attention",
    "saliency_entropy",
    "maximum_region_concentration",
    "left_right_asymmetry",
    "saliency_center_x",
    "saliency_center_y",
    "saliency_spread",
)
DEFAULT_SENSITIVITY_METRICS = (
    "angular_error_rad",
    "reconstruction_mse",
    "top1_score_drift",
    "agreement_with_origin",
    "threshold_crossing",
)


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
            validate=("one_to_one" if not retrieval_extra_keys else "one_to_many"),
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
    level = float(confidence_level)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    alpha = (1.0 - level) / 2.0

    records: list[dict[str, object]] = []
    for raw_group_key, group in joined.groupby(
        list(group_columns),
        dropna=False,
        sort=True,
    ):
        group_key = (
            raw_group_key if isinstance(raw_group_key, tuple) else (raw_group_key,)
        )
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
                    }
                )
                records.append(record)
    return pd.DataFrame.from_records(records)
