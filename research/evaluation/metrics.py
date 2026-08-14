from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


PAIRED_BOOTSTRAP_RESAMPLES = 2_000
PAIRED_BOOTSTRAP_RANDOM_SEED = 42
COMPACT_CSV_SIGNIFICANT_DIGITS = 12


def rate_ratio_matches_counts_or_compact_csv(
    observed,
    *,
    reference_successes,
    reference_totals,
    candidate_successes,
    candidate_totals,
    significant_digits: int = COMPACT_CSV_SIGNIFICANT_DIGITS,
) -> bool:
    """Check a rate ratio against integer counts before or after CSV rounding.

    Compact research tables use ``%.12g``. Dividing two independently rounded
    rates can amplify harmless error when the reference rate is small, so the
    ratio is reconstructed from the integer sufficient statistics instead.
    Both the full-precision value and its compact-CSV representation are valid.
    """

    if isinstance(significant_digits, (bool, np.bool_)):
        raise ValueError("significant_digits must be a positive integer")
    digits = int(significant_digits)
    if digits != significant_digits or digits <= 0:
        raise ValueError("significant_digits must be a positive integer")

    vectors = {
        "observed": pd.to_numeric(pd.Series(observed), errors="raise").to_numpy(
            dtype=np.float64
        ),
        "reference_successes": pd.to_numeric(
            pd.Series(reference_successes), errors="raise"
        ).to_numpy(dtype=np.float64),
        "reference_totals": pd.to_numeric(
            pd.Series(reference_totals), errors="raise"
        ).to_numpy(dtype=np.float64),
        "candidate_successes": pd.to_numeric(
            pd.Series(candidate_successes), errors="raise"
        ).to_numpy(dtype=np.float64),
        "candidate_totals": pd.to_numeric(
            pd.Series(candidate_totals), errors="raise"
        ).to_numpy(dtype=np.float64),
    }
    lengths = {values.size for values in vectors.values()}
    if len(lengths) != 1:
        raise ValueError("rate-ratio vectors must have the same length")

    for prefix in ("reference", "candidate"):
        successes = vectors[f"{prefix}_successes"]
        totals = vectors[f"{prefix}_totals"]
        if (
            not np.isfinite(successes).all()
            or not np.isfinite(totals).all()
            or not np.equal(successes, np.floor(successes)).all()
            or not np.equal(totals, np.floor(totals)).all()
            or (totals <= 0).any()
            or (successes < 0).any()
            or (successes > totals).any()
        ):
            raise ValueError(f"{prefix} binomial counts are invalid")

    observed_values = vectors["observed"]
    if np.isinf(observed_values).any():
        return False
    reference_rate = (
        vectors["reference_successes"] / vectors["reference_totals"]
    )
    candidate_rate = (
        vectors["candidate_successes"] / vectors["candidate_totals"]
    )
    expected = np.full(reference_rate.shape, np.nan, dtype=np.float64)
    np.divide(
        candidate_rate,
        reference_rate,
        out=expected,
        where=reference_rate != 0.0,
    )
    compact_expected = np.asarray(
        [
            np.nan if np.isnan(value) else float(format(float(value), f".{digits}g"))
            for value in expected
        ],
        dtype=np.float64,
    )
    full_precision_match = np.isclose(
        observed_values,
        expected,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=True,
    )
    compact_match = np.isclose(
        observed_values,
        compact_expected,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    return bool(np.all(full_precision_match | compact_match))


def _validated_binomial_counts(successes: int, total: int) -> tuple[int, int]:
    if isinstance(successes, (bool, np.bool_)) or isinstance(
        total, (bool, np.bool_)
    ):
        raise ValueError("binomial counts must be integers")
    try:
        successes_value = int(successes)
        total_value = int(total)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("binomial counts must be integers") from exc
    if successes_value != successes or total_value != total:
        raise ValueError("binomial counts must be integers")
    if total_value <= 0 or not 0 <= successes_value <= total_value:
        raise ValueError("binomial counts are invalid")
    return successes_value, total_value


def _validated_confidence_level(confidence_level: float) -> float:
    level = float(confidence_level)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    return level


def wilson_score_interval(
    successes: int,
    total: int,
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial rate."""

    successes_value, total_value = _validated_binomial_counts(successes, total)
    level = _validated_confidence_level(confidence_level)
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    proportion = successes_value / total_value
    denominator = 1.0 + z * z / total_value
    center = (
        proportion + z * z / (2.0 * total_value)
    ) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total_value
            + z * z / (4.0 * total_value * total_value)
        )
        / denominator
    )
    lower = 0.0 if successes_value == 0 else float(max(0.0, center - margin))
    upper = (
        1.0
        if successes_value == total_value
        else float(min(1.0, center + margin))
    )
    return lower, upper


def paired_binary_rate_difference_bootstrap_interval(
    reference_successes: int,
    candidate_successes: int,
    both_successes: int,
    total: int,
    *,
    confidence_level: float = 0.95,
    resamples: int = PAIRED_BOOTSTRAP_RESAMPLES,
    random_seed: int = PAIRED_BOOTSTRAP_RANDOM_SEED,
) -> tuple[float, float]:
    """Return a deterministic paired-bootstrap CI for candidate minus reference.

    The four joint binary-outcome cell counts are sufficient to reproduce a
    query-level non-parametric bootstrap. Keeping those counts lets streaming
    compact-summary generation preserve the paired origin/compressed design
    without retaining a row-level ledger in memory.
    """

    reference_value, total_value = _validated_binomial_counts(
        reference_successes,
        total,
    )
    candidate_value, candidate_total = _validated_binomial_counts(
        candidate_successes,
        total,
    )
    both_value, both_total = _validated_binomial_counts(both_successes, total)
    if candidate_total != total_value or both_total != total_value:
        raise RuntimeError("paired binary totals drifted")
    if both_value > min(reference_value, candidate_value):
        raise ValueError("both_successes exceeds a marginal success count")
    neither_value = (
        total_value - reference_value - candidate_value + both_value
    )
    if neither_value < 0:
        raise ValueError("paired binary joint counts are inconsistent")
    if isinstance(resamples, (bool, np.bool_)):
        raise ValueError("resamples must be a positive integer")
    try:
        resample_count = int(resamples)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("resamples must be a positive integer") from exc
    if resample_count != resamples or resample_count <= 0:
        raise ValueError("resamples must be a positive integer")
    level = _validated_confidence_level(confidence_level)
    reference_only = reference_value - both_value
    candidate_only = candidate_value - both_value
    probabilities = np.asarray(
        [neither_value, reference_only, candidate_only, both_value],
        dtype=np.float64,
    ) / total_value
    rng = np.random.default_rng(int(random_seed))
    joint_samples = rng.multinomial(
        total_value,
        probabilities,
        size=resample_count,
    )
    differences = (
        joint_samples[:, 2].astype(np.float64)
        - joint_samples[:, 1].astype(np.float64)
    ) / total_value
    alpha = 1.0 - level
    lower, upper = np.quantile(
        differences,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return float(max(-1.0, lower)), float(min(1.0, upper))


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


def certified_open_set_metrics(
    results: pd.DataFrame,
    *,
    decision_column: str = "certified_decision",
    predicted_identity_column: str = "certified_identity",
) -> dict[str, object]:
    """Score certificate-only decisions against open-set ground truth.

    Deferred probes are included in the DIR/FPIR denominators but are excluded
    from accept precision and reject accuracy because no certificate decision
    was issued for them.
    """

    required = {
        "probe_type",
        "query_identity_id",
        decision_column,
        predicted_identity_column,
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"missing certified metric columns: {sorted(missing)}")

    decisions = results[decision_column].astype(str)
    certified_accept = decisions.eq("accept").to_numpy()
    certified_reject = decisions.eq("reject").to_numpy()
    is_mated = results["probe_type"].astype(str).eq("registered").to_numpy()
    non_mated = ~is_mated
    identity_correct = (
        results[predicted_identity_column].astype(str).to_numpy()
        == results["query_identity_id"].astype(str).to_numpy()
    )

    accept_correct = certified_accept & is_mated & identity_correct
    reject_correct = certified_reject & non_mated
    accept_count = int(certified_accept.sum())
    reject_count = int(certified_reject.sum())
    accept_correct_count = int(accept_correct.sum())
    reject_correct_count = int(reject_correct.sum())
    mated_count = int(is_mated.sum())
    non_mated_count = int(non_mated.sum())

    return {
        "certified_accept_precision": {
            "count": accept_count,
            "correct": accept_correct_count,
            "rate": accept_correct_count / accept_count if accept_count else None,
        },
        "certified_reject_accuracy": {
            "count": reject_count,
            "correct": reject_correct_count,
            "rate": reject_correct_count / reject_count if reject_count else None,
        },
        "mated_count": mated_count,
        "non_mated_count": non_mated_count,
        "certified_DIR": accept_correct_count / mated_count if mated_count else 0.0,
        "certified_FPIR": int((certified_accept & non_mated).sum()) / non_mated_count
        if non_mated_count
        else 0.0,
    }
