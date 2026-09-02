"""Leakage-safe quality-conditioned threshold calibration for open-set search.

The module treats FIQA as a conditioning covariate, never as a replacement for
the retrieval score.  All quality cut-points and non-mated score thresholds are
fit on calibration rows.  Test rows are used only once for frozen evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from research.calibration.rejection import choose_non_mated_fpir_threshold
from research.evaluation.metrics import (
    PAIRED_BOOTSTRAP_RANDOM_SEED,
    PAIRED_BOOTSTRAP_RESAMPLES,
    paired_binary_rate_difference_bootstrap_interval,
    wilson_score_interval,
)
from research.runtime.hashing import canonical_sha256


@dataclass(frozen=True)
class ThresholdGroup:
    name: str
    fit_non_mated_count: int
    safety_non_mated_count: int
    raw_threshold: float | None
    shrinkage_weight: float
    threshold_before_safety: float
    safety_threshold: float | None
    final_threshold: float
    used_global_fallback: bool
    fit_fpir_at_final_threshold: float | None
    safety_fpir_at_final_threshold: float | None
    fit_target_met: bool | None
    safety_target_met: bool | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fit_non_mated_count": self.fit_non_mated_count,
            "safety_non_mated_count": self.safety_non_mated_count,
            "raw_threshold": self.raw_threshold,
            "shrinkage_weight": self.shrinkage_weight,
            "threshold_before_safety": self.threshold_before_safety,
            "safety_threshold": self.safety_threshold,
            "final_threshold": self.final_threshold,
            "used_global_fallback": self.used_global_fallback,
            "fit_fpir_at_final_threshold": self.fit_fpir_at_final_threshold,
            "safety_fpir_at_final_threshold": self.safety_fpir_at_final_threshold,
            "fit_target_met": self.fit_target_met,
            "safety_target_met": self.safety_target_met,
        }


@dataclass(frozen=True)
class ConditionalThresholdModel:
    method: str
    target_fpir: float
    score_column: str
    quality_column: str | None
    group_labels: tuple[str, ...]
    quality_cutpoints: tuple[float, ...]
    global_fit_threshold: float
    global_safety_threshold: float | None
    global_final_threshold: float
    groups: tuple[ThresholdGroup, ...]
    calibration_partition: dict[str, Any]
    score_space: str
    threshold_comparator: str = ">="

    @property
    def model_uid(self) -> str:
        return "conditional-threshold-" + canonical_sha256(self.as_dict())[:24]

    @property
    def thresholds(self) -> dict[str, float]:
        return {item.name: item.final_threshold for item in self.groups}

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": self.method,
            "target_fpir": self.target_fpir,
            "score_column": self.score_column,
            "quality_column": self.quality_column,
            "group_labels": list(self.group_labels),
            "quality_cutpoints": list(self.quality_cutpoints),
            "global_fit_threshold": self.global_fit_threshold,
            "global_safety_threshold": self.global_safety_threshold,
            "global_final_threshold": self.global_final_threshold,
            "groups": [item.as_dict() for item in self.groups],
            "calibration_partition": dict(self.calibration_partition),
            "score_space": self.score_space,
            "threshold_comparator": self.threshold_comparator,
        }


@dataclass(frozen=True)
class ThresholdEvaluation:
    model: ConditionalThresholdModel
    decisions: pd.DataFrame
    summary: dict[str, Any]


def _target_fpir(value: float) -> float:
    target = float(value)
    if not np.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("target_fpir must be finite and inside (0, 1)")
    return target


def _validated_rows(
    frame: pd.DataFrame,
    *,
    score_column: str,
    quality_column: str | None,
) -> pd.DataFrame:
    required = {"sample_id", "is_mated", score_column}
    if quality_column is not None:
        required.add(quality_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"threshold rows are missing columns: {missing}")
    if frame.empty:
        raise ValueError("threshold rows must not be empty")
    rows = frame.copy().reset_index(drop=True)
    rows["sample_id"] = rows["sample_id"].astype(str)
    if rows["sample_id"].duplicated().any():
        raise ValueError("threshold rows require unique sample_id values")
    scores = rows[score_column].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("retrieval scores must be finite")
    rows["is_mated"] = rows["is_mated"].astype(bool)
    if rows["is_mated"].all() or (~rows["is_mated"]).all():
        raise ValueError("threshold rows require both mated and non-mated probes")
    if quality_column is not None:
        quality = rows[quality_column].to_numpy(dtype=np.float64)
        if not np.isfinite(quality).all():
            raise ValueError("quality scores must be finite")
    return rows


def deterministic_calibration_partition(
    calibration: pd.DataFrame,
    *,
    safety_fraction: float = 0.30,
    seed: int = 42,
    partition_column: str | None = None,
) -> pd.Series:
    """Return stable identity/query-cluster fit/safety labels."""

    fraction = float(safety_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError("safety_fraction must be inside [0, 1)")
    key = (
        str(partition_column)
        if partition_column is not None
        else "identity_id"
        if "identity_id" in calibration
        else "sample_id"
    )
    if key not in calibration:
        raise ValueError(f"calibration partition requires {key}")
    identifiers = calibration[key].astype(str)
    if identifiers.eq("").any():
        raise ValueError(f"calibration partition {key} values must not be empty")
    if fraction == 0.0:
        labels = pd.Series("fit", index=calibration.index, dtype="string")
        labels.attrs["partition_column"] = key
        return labels

    denominator = float(2**64)

    def assignment(sample_id: str) -> str:
        digest = hashlib.sha256(f"{int(seed)}:{sample_id}".encode("utf-8")).digest()
        uniform = int.from_bytes(digest[:8], "big") / denominator
        return "safety" if uniform < fraction else "fit"

    labels = identifiers.map(assignment).astype("string")
    if set(labels.astype(str)) != {"fit", "safety"}:
        raise ValueError("calibration partition did not produce fit and safety rows")
    labels.attrs["partition_column"] = key
    return labels


def _group_labels(bin_count: int) -> tuple[str, ...]:
    count = int(bin_count)
    if count == 2:
        return ("low", "high")
    if count == 3:
        return ("low", "mid", "high")
    if count < 2:
        raise ValueError("quality bin_count must be at least two")
    return tuple(f"q{index + 1}" for index in range(count))


def _fit_cutpoints(values: np.ndarray, *, bin_count: int) -> tuple[float, ...]:
    probabilities = np.arange(1, int(bin_count), dtype=np.float64) / int(bin_count)
    cutpoints = tuple(float(value) for value in np.quantile(values, probabilities))
    if len(set(cutpoints)) != len(cutpoints):
        raise ValueError(
            "quality quantiles collapsed; use fewer bins or inspect FIQA scores"
        )
    return cutpoints


def assign_quality_groups(
    quality_scores: np.ndarray | pd.Series,
    *,
    cutpoints: tuple[float, ...],
    labels: tuple[str, ...],
) -> np.ndarray:
    values = np.asarray(quality_scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("quality scores must be finite")
    if len(labels) != len(cutpoints) + 1:
        raise ValueError("quality labels and cutpoints are inconsistent")
    indices = np.searchsorted(np.asarray(cutpoints), values, side="right")
    return np.asarray(labels, dtype=object)[indices].astype(str)


def _threshold(frame: pd.DataFrame, *, score_column: str, target_fpir: float) -> float:
    return float(
        choose_non_mated_fpir_threshold(
            frame[score_column].to_numpy(dtype=np.float64),
            frame["is_mated"].to_numpy(dtype=bool),
            target_fpir=target_fpir,
        )
    )


def _empirical_non_mated_fpir(
    frame: pd.DataFrame,
    *,
    score_column: str,
    threshold: float,
) -> float | None:
    scores = frame.loc[~frame["is_mated"], score_column].to_numpy(
        dtype=np.float64
    )
    if len(scores) == 0:
        return None
    return float(np.mean(scores >= float(threshold)))


def fit_conditional_threshold(
    calibration: pd.DataFrame,
    *,
    target_fpir: float,
    score_column: str = "score",
    quality_column: str = "fiqa_score",
    bin_count: int = 2,
    shrinkage_strength: float = 200.0,
    minimum_group_non_mated: int = 100,
    safety_fraction: float = 0.30,
    partition_seed: int = 42,
    partition_column: str | None = None,
    score_space: str,
) -> ConditionalThresholdModel:
    """Fit quality bins, partial pooling, and a held-out safety threshold."""

    target = _target_fpir(target_fpir)
    rows = _validated_rows(
        calibration,
        score_column=score_column,
        quality_column=quality_column,
    )
    strength = float(shrinkage_strength)
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("shrinkage_strength must be finite and non-negative")
    minimum = int(minimum_group_non_mated)
    if minimum <= 0:
        raise ValueError("minimum_group_non_mated must be positive")
    partition = deterministic_calibration_partition(
        rows,
        safety_fraction=safety_fraction,
        seed=partition_seed,
        partition_column=partition_column,
    )
    fit = rows.loc[partition.eq("fit")].copy()
    safety = rows.loc[partition.eq("safety")].copy()
    if fit.empty:
        raise ValueError("calibration fit partition is empty")
    labels = _group_labels(bin_count)
    cutpoints = _fit_cutpoints(
        fit[quality_column].to_numpy(dtype=np.float64),
        bin_count=len(labels),
    )
    fit["quality_group"] = assign_quality_groups(
        fit[quality_column], cutpoints=cutpoints, labels=labels
    )
    if not safety.empty:
        safety["quality_group"] = assign_quality_groups(
            safety[quality_column], cutpoints=cutpoints, labels=labels
        )

    global_fit = _threshold(fit, score_column=score_column, target_fpir=target)
    global_safety = (
        _threshold(safety, score_column=score_column, target_fpir=target)
        if not safety.empty and (~safety["is_mated"]).any()
        else None
    )
    global_final = max(
        global_fit,
        global_safety if global_safety is not None else global_fit,
    )

    group_models: list[ThresholdGroup] = []
    for label in labels:
        fit_group = fit.loc[fit["quality_group"].eq(label)]
        fit_non_mated = int((~fit_group["is_mated"]).sum())
        safety_group = (
            safety.loc[safety["quality_group"].eq(label)]
            if not safety.empty
            else safety
        )
        safety_non_mated = int((~safety_group["is_mated"]).sum())
        fallback = fit_non_mated < minimum
        raw_threshold = (
            None
            if fallback
            else _threshold(
                fit_group,
                score_column=score_column,
                target_fpir=target,
            )
        )
        weight = (
            0.0
            if raw_threshold is None
            else float(fit_non_mated / (fit_non_mated + strength))
            if strength > 0.0
            else 1.0
        )
        pooled_threshold = (
            global_fit
            if raw_threshold is None
            else float(weight * raw_threshold + (1.0 - weight) * global_fit)
        )
        before_safety = (
            global_fit
            if raw_threshold is None
            else max(float(raw_threshold), pooled_threshold)
        )
        group_safety = (
            _threshold(
                safety_group,
                score_column=score_column,
                target_fpir=target,
            )
            if safety_non_mated >= minimum
            else None
        )
        if fallback:
            final = max(
                global_final,
                group_safety if group_safety is not None else global_final,
            )
        elif safety.empty:
            final = before_safety
        else:
            final = max(
                before_safety,
                group_safety if group_safety is not None else global_final,
            )
        fit_fpir = _empirical_non_mated_fpir(
            fit_group,
            score_column=score_column,
            threshold=final,
        )
        safety_fpir = _empirical_non_mated_fpir(
            safety_group,
            score_column=score_column,
            threshold=final,
        )
        group_models.append(
            ThresholdGroup(
                name=label,
                fit_non_mated_count=fit_non_mated,
                safety_non_mated_count=safety_non_mated,
                raw_threshold=raw_threshold,
                shrinkage_weight=weight,
                threshold_before_safety=before_safety,
                safety_threshold=group_safety,
                final_threshold=float(final),
                used_global_fallback=fallback,
                fit_fpir_at_final_threshold=fit_fpir,
                safety_fpir_at_final_threshold=safety_fpir,
                fit_target_met=(
                    bool(fit_fpir <= target) if fit_fpir is not None else None
                ),
                safety_target_met=(
                    bool(safety_fpir <= target)
                    if safety_fpir is not None
                    else None
                ),
            )
        )

    return ConditionalThresholdModel(
        method=(
            f"fiqa_{len(labels)}bin_conservative_shrunk_safe"
            if safety_fraction > 0.0
            else f"fiqa_{len(labels)}bin_conservative_shrunk"
        ),
        target_fpir=target,
        score_column=score_column,
        quality_column=quality_column,
        group_labels=labels,
        quality_cutpoints=cutpoints,
        global_fit_threshold=global_fit,
        global_safety_threshold=global_safety,
        global_final_threshold=global_final,
        groups=tuple(group_models),
        calibration_partition={
            "method": "sha256_partition_key",
            "seed": int(partition_seed),
            "safety_fraction": float(safety_fraction),
            "partition_column": str(partition.attrs["partition_column"]),
            "partition_unit": (
                "identity_cluster"
                if partition.attrs["partition_column"] == "identity_id"
                else "query"
            ),
            "fit_rows": int(partition.eq("fit").sum()),
            "safety_rows": int(partition.eq("safety").sum()),
            "cutpoints_fit_on": "fit_only",
            "thresholds_fit_on": "fit_only",
            "safety_rule": "max(base_threshold, heldout_group_threshold)",
            "one_sided_shrinkage_rule": (
                "threshold_before_safety=max(raw_group_threshold, "
                "linear_partial_pooling_threshold)"
            ),
        },
        score_space=str(score_space),
    )


def fit_global_threshold(
    calibration: pd.DataFrame,
    *,
    target_fpir: float,
    score_column: str = "score",
    safety_fraction: float = 0.30,
    partition_seed: int = 42,
    partition_column: str | None = None,
    score_space: str,
) -> ConditionalThresholdModel:
    """Fit a global comparator with the same fit/safety partition contract."""

    target = _target_fpir(target_fpir)
    rows = _validated_rows(
        calibration,
        score_column=score_column,
        quality_column=None,
    )
    partition = deterministic_calibration_partition(
        rows,
        safety_fraction=safety_fraction,
        seed=partition_seed,
        partition_column=partition_column,
    )
    fit = rows.loc[partition.eq("fit")]
    safety = rows.loc[partition.eq("safety")]
    fit_threshold = _threshold(fit, score_column=score_column, target_fpir=target)
    safety_threshold = (
        _threshold(safety, score_column=score_column, target_fpir=target)
        if not safety.empty and (~safety["is_mated"]).any()
        else None
    )
    final = max(
        fit_threshold,
        safety_threshold if safety_threshold is not None else fit_threshold,
    )
    fit_fpir = _empirical_non_mated_fpir(
        fit,
        score_column=score_column,
        threshold=final,
    )
    safety_fpir = _empirical_non_mated_fpir(
        safety,
        score_column=score_column,
        threshold=final,
    )
    group = ThresholdGroup(
        name="all",
        fit_non_mated_count=int((~fit["is_mated"]).sum()),
        safety_non_mated_count=int((~safety["is_mated"]).sum()),
        raw_threshold=fit_threshold,
        shrinkage_weight=1.0,
        threshold_before_safety=fit_threshold,
        safety_threshold=safety_threshold,
        final_threshold=final,
        used_global_fallback=False,
        fit_fpir_at_final_threshold=fit_fpir,
        safety_fpir_at_final_threshold=safety_fpir,
        fit_target_met=bool(fit_fpir <= target) if fit_fpir is not None else None,
        safety_target_met=(
            bool(safety_fpir <= target) if safety_fpir is not None else None
        ),
    )
    return ConditionalThresholdModel(
        method="global_safe" if safety_fraction > 0.0 else "global_empirical",
        target_fpir=target,
        score_column=score_column,
        quality_column=None,
        group_labels=("all",),
        quality_cutpoints=(),
        global_fit_threshold=fit_threshold,
        global_safety_threshold=safety_threshold,
        global_final_threshold=final,
        groups=(group,),
        calibration_partition={
            "method": "sha256_partition_key",
            "seed": int(partition_seed),
            "safety_fraction": float(safety_fraction),
            "partition_column": str(partition.attrs["partition_column"]),
            "partition_unit": (
                "identity_cluster"
                if partition.attrs["partition_column"] == "identity_id"
                else "query"
            ),
            "fit_rows": int(partition.eq("fit").sum()),
            "safety_rows": int(partition.eq("safety").sum()),
            "thresholds_fit_on": "fit_only",
            "safety_rule": "max(fit_threshold, heldout_threshold)",
        },
        score_space=str(score_space),
    )


def apply_threshold_model(
    frame: pd.DataFrame,
    model: ConditionalThresholdModel,
    *,
    top_k_correct_column: str = "top_k_correct",
) -> ThresholdEvaluation:
    """Apply one frozen model and report FPIR/TPIR@K with Wilson intervals."""

    rows = _validated_rows(
        frame,
        score_column=model.score_column,
        quality_column=model.quality_column,
    )
    if top_k_correct_column not in rows:
        raise ValueError(
            f"threshold evaluation is missing {top_k_correct_column!r}"
        )
    if model.quality_column is None:
        groups = np.full(len(rows), "all", dtype=object)
    else:
        groups = assign_quality_groups(
            rows[model.quality_column],
            cutpoints=model.quality_cutpoints,
            labels=model.group_labels,
        )
    thresholds = pd.Series(groups).map(model.thresholds).to_numpy(dtype=np.float64)
    if not np.isfinite(thresholds).all():
        raise RuntimeError("one or more quality groups has no frozen threshold")
    scores = rows[model.score_column].to_numpy(dtype=np.float64)
    accepted = scores >= thresholds
    is_mated = rows["is_mated"].to_numpy(dtype=bool)
    top_k_correct = rows[top_k_correct_column].astype(bool).to_numpy()
    non_mated = ~is_mated
    false_accept = accepted & non_mated
    true_identification = accepted & is_mated & top_k_correct
    fpir_successes = int(false_accept.sum())
    fpir_total = int(non_mated.sum())
    tpir_successes = int(true_identification.sum())
    tpir_total = int(is_mated.sum())
    fpir_ci = wilson_score_interval(fpir_successes, fpir_total)
    tpir_ci = wilson_score_interval(tpir_successes, tpir_total)
    decisions = rows.copy()
    decisions["quality_group"] = groups
    decisions["applied_threshold"] = thresholds
    decisions["accepted"] = accepted
    decisions["false_accept"] = false_accept
    decisions["true_identification_at_rank_k"] = true_identification
    summary = {
        "schema_version": 1,
        "model_uid": model.model_uid,
        "method": model.method,
        "target_fpir": model.target_fpir,
        "score_space": model.score_space,
        "test_probe_count": int(len(rows)),
        "test_non_mated_count": fpir_total,
        "test_mated_count": tpir_total,
        "false_accept_count": fpir_successes,
        "true_identification_at_rank_k_count": tpir_successes,
        "realized_fpir": float(fpir_successes / fpir_total),
        "fpir_wilson95_low": fpir_ci[0],
        "fpir_wilson95_high": fpir_ci[1],
        "tpir_at_rank_k": float(tpir_successes / tpir_total),
        "tpir_at_rank_k_wilson95_low": tpir_ci[0],
        "tpir_at_rank_k_wilson95_high": tpir_ci[1],
        "target_met_on_test": bool(fpir_successes / fpir_total <= model.target_fpir),
        "target_met_by_wilson_upper": bool(fpir_ci[1] <= model.target_fpir),
        "threshold_fit_on_test": False,
    }
    return ThresholdEvaluation(model=model, decisions=decisions, summary=summary)


def paired_method_comparison(
    reference: ThresholdEvaluation,
    candidate: ThresholdEvaluation,
) -> dict[str, Any]:
    """Return paired candidate-minus-reference FPIR and TPIR@K evidence."""

    left = reference.decisions
    right = candidate.decisions
    if not np.array_equal(
        left["sample_id"].astype(str).to_numpy(),
        right["sample_id"].astype(str).to_numpy(),
    ):
        raise ValueError("paired threshold evaluations use different sample order")
    if reference.model.score_space != candidate.model.score_space:
        raise ValueError("paired methods must use the same score space")
    if reference.model.target_fpir != candidate.model.target_fpir:
        raise ValueError("paired methods must use the same target FPIR")

    def evidence(column: str, mask: np.ndarray) -> dict[str, Any]:
        ref = left.loc[mask, column].astype(bool).to_numpy()
        cand = right.loc[mask, column].astype(bool).to_numpy()
        ref_count = int(ref.sum())
        cand_count = int(cand.sum())
        both_count = int((ref & cand).sum())
        low, high = paired_binary_rate_difference_bootstrap_interval(
            ref_count,
            cand_count,
            both_count,
            len(ref),
            resamples=PAIRED_BOOTSTRAP_RESAMPLES,
            random_seed=PAIRED_BOOTSTRAP_RANDOM_SEED,
        )
        return {
            "reference_successes": ref_count,
            "candidate_successes": cand_count,
            "both_successes": both_count,
            "total": int(len(ref)),
            "candidate_minus_reference": float((cand_count - ref_count) / len(ref)),
            "paired_bootstrap95_low": low,
            "paired_bootstrap95_high": high,
        }

    is_mated = left["is_mated"].to_numpy(dtype=bool)
    return {
        "schema_version": 1,
        "reference_method": reference.model.method,
        "candidate_method": candidate.model.method,
        "target_fpir": reference.model.target_fpir,
        "score_space": reference.model.score_space,
        "fpir": evidence("false_accept", ~is_mated),
        "tpir_at_rank_k": evidence("true_identification_at_rank_k", is_mated),
        "confidence_interval_method": "paired_nonparametric_bootstrap_percentile",
        "resamples": PAIRED_BOOTSTRAP_RESAMPLES,
        "random_seed": PAIRED_BOOTSTRAP_RANDOM_SEED,
    }
