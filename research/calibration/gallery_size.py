from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GallerySizeThresholdResult:
    method: str
    threshold_comparator: str
    target_fpir: float
    target_gallery_size: int
    effective_gallery_size: float
    effective_gallery_ratio: float
    target_pair_false_match_rate: float
    impostor_pair_count: int
    threshold: float
    empirical_pair_false_match_count: int
    empirical_pair_false_match_rate: float
    independence_assumption: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EffectiveGalleryRatioEstimate:
    gallery_size: int
    target_fpir: float
    maximum_score_threshold: float
    realized_maximum_fpir: float
    pair_false_match_rate_at_threshold: float
    effective_gallery_size: float
    effective_gallery_ratio: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sidak_pair_false_match_rate(*, target_fpir: float, gallery_size: int) -> float:
    """Convert a gallery-level FPIR target into a per-template tail rate."""

    target = float(target_fpir)
    size = int(gallery_size)
    if not 0.0 < target < 1.0:
        raise ValueError("target_fpir must be strictly inside (0, 1)")
    if size <= 0:
        raise ValueError("gallery_size must be positive")
    return float(-np.expm1(np.log1p(-target) / size))


def _empirical_tail_threshold(
    scores: np.ndarray,
    *,
    target_rate: float,
) -> tuple[float, int, float]:
    grouped = (
        pd.Series(scores, name="score")
        .value_counts(sort=False)
        .rename_axis("score")
        .reset_index(name="count")
        .sort_values("score", ascending=False, kind="stable")
    )
    grouped["cumulative_count"] = grouped["count"].cumsum()
    feasible = grouped.loc[
        grouped["cumulative_count"] / len(scores) <= target_rate + 1e-15
    ]
    if feasible.empty:
        observed_maximum = float(np.max(scores))
        threshold = float(np.nextafter(observed_maximum, np.inf))
        if threshold > 1.0:
            raise RuntimeError(
                "no finite cosine threshold can satisfy the requested tail rate"
            )
        false_match_count = 0
    else:
        selected = feasible.iloc[-1]
        threshold = float(selected["score"])
        false_match_count = int(selected["cumulative_count"])
    empirical_rate = float(false_match_count / len(scores))
    if empirical_rate > target_rate + 1e-15:
        raise RuntimeError("selected threshold exceeds its empirical tail budget")
    return threshold, false_match_count, empirical_rate


def sidak_gallery_fpir(*, pair_false_match_rate: float, gallery_size: int) -> float:
    """Map a per-template false-match rate to gallery-level FPIR."""

    pair_rate = float(pair_false_match_rate)
    size = int(gallery_size)
    if not 0.0 <= pair_rate <= 1.0:
        raise ValueError("pair_false_match_rate must be inside [0, 1]")
    if size <= 0:
        raise ValueError("gallery_size must be positive")
    return float(-np.expm1(size * np.log1p(-pair_rate)))


def choose_sidak_gallery_threshold(
    impostor_pair_scores: np.ndarray,
    *,
    target_fpir: float,
    target_gallery_size: int,
) -> GallerySizeThresholdResult:
    """Choose a conservative empirical Sidak threshold for ``score >= t``.

    The method assumes independent template comparisons when converting the
    gallery FPIR target to a pairwise tail probability. Equal-score groups are
    kept intact. If even the maximum-score tie exceeds the pairwise budget, the
    threshold is moved one floating-point step above the observed maximum.
    """

    scores = np.asarray(impostor_pair_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("impostor_pair_scores must be a non-empty 1D array")
    if not np.all(np.isfinite(scores)):
        raise ValueError("impostor_pair_scores must be finite")
    if np.any(scores < -1.000001) or np.any(scores > 1.000001):
        raise ValueError("impostor pair cosine scores must be inside [-1, 1]")

    pair_target = sidak_pair_false_match_rate(
        target_fpir=target_fpir,
        gallery_size=target_gallery_size,
    )
    threshold, false_match_count, empirical_rate = _empirical_tail_threshold(
        scores,
        target_rate=pair_target,
    )
    return GallerySizeThresholdResult(
        method="empirical_sidak_pair_tail_v1",
        threshold_comparator=">=",
        target_fpir=float(target_fpir),
        target_gallery_size=int(target_gallery_size),
        effective_gallery_size=float(target_gallery_size),
        effective_gallery_ratio=1.0,
        target_pair_false_match_rate=pair_target,
        impostor_pair_count=int(len(scores)),
        threshold=threshold,
        empirical_pair_false_match_count=false_match_count,
        empirical_pair_false_match_rate=empirical_rate,
        independence_assumption=True,
    )


def estimate_effective_gallery_ratio(
    impostor_pair_scores: np.ndarray,
    maximum_impostor_scores: np.ndarray,
    *,
    target_fpir: float,
    gallery_size: int,
) -> EffectiveGalleryRatioEstimate:
    """Estimate the comparison multiplicity implied by observed maxima."""

    pair_scores = np.asarray(impostor_pair_scores, dtype=np.float64)
    maximum_scores = np.asarray(maximum_impostor_scores, dtype=np.float64)
    if pair_scores.ndim != 1 or len(pair_scores) == 0:
        raise ValueError("impostor_pair_scores must be a non-empty 1D array")
    if maximum_scores.ndim != 1 or len(maximum_scores) == 0:
        raise ValueError("maximum_impostor_scores must be a non-empty 1D array")
    if not np.all(np.isfinite(pair_scores)) or not np.all(np.isfinite(maximum_scores)):
        raise ValueError("effective gallery calibration scores must be finite")
    size = int(gallery_size)
    if size <= 0:
        raise ValueError("gallery_size must be positive")
    threshold, _, realized_fpir = _empirical_tail_threshold(
        maximum_scores,
        target_rate=float(target_fpir),
    )
    pair_rate = float(np.mean(pair_scores >= threshold))
    if not 0.0 < realized_fpir < 1.0 or not 0.0 < pair_rate < 1.0:
        raise RuntimeError("effective gallery ratio requires non-degenerate tail rates")
    effective_size = float(np.log1p(-realized_fpir) / np.log1p(-pair_rate))
    ratio = float(effective_size / size)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise RuntimeError("effective gallery ratio is not positive and finite")
    return EffectiveGalleryRatioEstimate(
        gallery_size=size,
        target_fpir=float(target_fpir),
        maximum_score_threshold=threshold,
        realized_maximum_fpir=realized_fpir,
        pair_false_match_rate_at_threshold=pair_rate,
        effective_gallery_size=effective_size,
        effective_gallery_ratio=ratio,
    )


def choose_effective_gallery_threshold(
    impostor_pair_scores: np.ndarray,
    *,
    target_fpir: float,
    target_gallery_size: int,
    effective_gallery_ratio: float,
) -> GallerySizeThresholdResult:
    """Choose a pair-tail threshold using a calibration-derived multiplicity."""

    scores = np.asarray(impostor_pair_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("impostor_pair_scores must be a finite non-empty 1D array")
    size = int(target_gallery_size)
    ratio = float(effective_gallery_ratio)
    if size <= 0:
        raise ValueError("target_gallery_size must be positive")
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("effective_gallery_ratio must be positive and finite")
    effective_size = float(size * ratio)
    target = float(target_fpir)
    if not 0.0 < target < 1.0:
        raise ValueError("target_fpir must be strictly inside (0, 1)")
    pair_target = float(-np.expm1(np.log1p(-target) / effective_size))
    threshold, false_match_count, empirical_rate = _empirical_tail_threshold(
        scores,
        target_rate=pair_target,
    )
    return GallerySizeThresholdResult(
        method="empirical_effective_gallery_pair_tail_v1",
        threshold_comparator=">=",
        target_fpir=target,
        target_gallery_size=size,
        effective_gallery_size=effective_size,
        effective_gallery_ratio=ratio,
        target_pair_false_match_rate=pair_target,
        impostor_pair_count=int(len(scores)),
        threshold=threshold,
        empirical_pair_false_match_count=false_match_count,
        empirical_pair_false_match_rate=empirical_rate,
        independence_assumption=False,
    )
