import numpy as np
import pytest

from research.calibration import (
    choose_effective_gallery_threshold,
    choose_sidak_gallery_threshold,
    estimate_effective_gallery_ratio,
    sidak_gallery_fpir,
    sidak_pair_false_match_rate,
)


def test_sidak_pair_rate_round_trips_gallery_fpir() -> None:
    pair_rate = sidak_pair_false_match_rate(target_fpir=0.10, gallery_size=3000)

    assert pair_rate == pytest.approx(3.511964e-05, rel=1e-5)
    assert sidak_gallery_fpir(
        pair_false_match_rate=pair_rate,
        gallery_size=3000,
    ) == pytest.approx(0.10)


def test_sidak_threshold_preserves_ties_and_pair_budget() -> None:
    scores = np.array([0.9, 0.8, 0.8, 0.7, 0.6], dtype=np.float64)

    result = choose_sidak_gallery_threshold(
        scores,
        target_fpir=0.75,
        target_gallery_size=2,
    )

    assert result.threshold == pytest.approx(0.9)
    assert result.empirical_pair_false_match_count == 1
    assert result.empirical_pair_false_match_rate == pytest.approx(0.2)
    assert result.threshold_comparator == ">="


def test_sidak_threshold_moves_above_maximum_when_tail_budget_is_subsample() -> None:
    result = choose_sidak_gallery_threshold(
        np.array([0.9, 0.8, 0.7], dtype=np.float64),
        target_fpir=0.01,
        target_gallery_size=1000,
    )

    assert 0.9 < result.threshold <= 1.0
    assert result.empirical_pair_false_match_count == 0
    assert result.empirical_pair_false_match_rate == 0.0


def test_effective_gallery_ratio_recovers_independent_simulation() -> None:
    rng = np.random.default_rng(42)
    score_matrix = rng.random((20_000, 10))
    estimate = estimate_effective_gallery_ratio(
        score_matrix.reshape(-1),
        score_matrix.max(axis=1),
        target_fpir=0.10,
        gallery_size=10,
    )

    assert estimate.effective_gallery_ratio == pytest.approx(1.0, abs=0.08)
    adjusted = choose_effective_gallery_threshold(
        score_matrix.reshape(-1),
        target_fpir=0.10,
        target_gallery_size=20,
        effective_gallery_ratio=estimate.effective_gallery_ratio,
    )
    assert adjusted.method == "empirical_effective_gallery_pair_tail_v1"
    assert adjusted.independence_assumption is False
    assert adjusted.effective_gallery_size == pytest.approx(
        20 * estimate.effective_gallery_ratio
    )
