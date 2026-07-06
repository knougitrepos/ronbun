import pandas as pd

from research.metrics import brier_score, expected_calibration_error, rank_at_k


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
