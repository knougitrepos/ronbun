import pandas as pd
import pytest

from research.evaluation.metrics import (
    brier_score,
    certified_open_set_metrics,
    expected_calibration_error,
    open_set_identification_metrics,
    paired_binary_rate_difference_bootstrap_interval,
    rank_at_k,
    wilson_score_interval,
)


def test_metrics_cover_rank_and_calibration_quality():
    rows = pd.DataFrame(
        {
            "query_identity_id": ["a", "b", "u"],
            "probe_type": ["registered", "registered", "known_unknown"],
            "ranked_identities": [["a", "b"], ["a", "b"], ["a", "b"]],
            "y_true_accept": [1, 1, 0],
            "accept_probability": [0.9, 0.4, 0.2],
        }
    )

    assert rank_at_k(rows, k=1) == 0.5
    assert rank_at_k(rows, k=2) == 1.0
    assert brier_score(rows["y_true_accept"], rows["accept_probability"]) > 0.0
    assert expected_calibration_error(rows["y_true_accept"], rows["accept_probability"], n_bins=2) >= 0.0


def test_open_set_metrics_separate_mated_identification_from_non_mated_acceptance():
    rows = pd.DataFrame(
        {
            "probe_type": ["registered", "registered", "known_unknown", "unknown_unknown"],
            "query_identity_id": ["a", "b", "u", "x"],
            "top1_identity": ["a", "a", "a", "b"],
            "accepted": [True, True, False, True],
        }
    )

    metrics = open_set_identification_metrics(rows)

    assert metrics == {
        "mated_count": 2,
        "non_mated_count": 2,
        "dir_rank1": 0.5,
        "fnir_rank1": 0.5,
        "fpir": 0.5,
    }


def test_open_set_metrics_can_score_exact_fallback_identity():
    rows = pd.DataFrame(
        {
            "probe_type": ["registered", "known_unknown"],
            "query_identity_id": ["a", "u"],
            "top1_identity": ["wrong", "a"],
            "final_identity": ["a", None],
            "accepted": [True, False],
        }
    )

    metrics = open_set_identification_metrics(
        rows,
        predicted_identity_column="final_identity",
    )

    assert metrics["dir_rank1"] == 1.0
    assert metrics["fpir"] == 0.0


def test_certified_metrics_use_ground_truth_instead_of_origin_agreement():
    rows = pd.DataFrame(
        {
            "probe_type": [
                "registered",
                "registered",
                "registered",
                "known_unknown",
                "unknown_unknown",
                "known_unknown",
            ],
            "query_identity_id": ["a", "b", "c", "u", "x", "y"],
            "certified_decision": [
                "accept",
                "reject",
                "defer",
                "accept",
                "reject",
                "defer",
            ],
            "certified_identity": ["a", None, None, "a", None, None],
        }
    )

    metrics = certified_open_set_metrics(rows)

    assert metrics["certified_accept_precision"] == {
        "count": 2,
        "correct": 1,
        "rate": 0.5,
    }
    assert metrics["certified_reject_accuracy"] == {
        "count": 2,
        "correct": 1,
        "rate": 0.5,
    }
    assert metrics["certified_DIR"] == 1 / 3
    assert metrics["certified_FPIR"] == 1 / 3


def test_wilson_score_interval_handles_sparse_false_accepts() -> None:
    low, high = wilson_score_interval(0, 100)

    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.0369935, rel=1e-5)
    with pytest.raises(ValueError, match="counts"):
        wilson_score_interval(2, 1)


def test_paired_bootstrap_interval_preserves_origin_compressed_pairing() -> None:
    first = paired_binary_rate_difference_bootstrap_interval(
        reference_successes=4,
        candidate_successes=6,
        both_successes=3,
        total=10,
    )
    second = paired_binary_rate_difference_bootstrap_interval(
        reference_successes=4,
        candidate_successes=6,
        both_successes=3,
        total=10,
    )

    assert first == second
    assert first[0] <= 0.2 <= first[1]
    assert paired_binary_rate_difference_bootstrap_interval(
        reference_successes=10,
        candidate_successes=10,
        both_successes=10,
        total=10,
    ) == (0.0, 0.0)
    with pytest.raises(ValueError, match="joint counts"):
        paired_binary_rate_difference_bootstrap_interval(9, 9, 0, 10)
