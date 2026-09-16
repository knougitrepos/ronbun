import json

import numpy as np
import pandas as pd
import pytest

from research.experiments.compression_fpir_failure_diagnosis import (
    diagnose_decision, event_counts, verified_path, write_diagnosis,
)
from research.runtime.hashing import sha256_file


@pytest.fixture
def frame():
    data = pd.DataFrame({
        "query_id": ["a", "b", "c", "d", "e", "f"],
        "is_mated": [False, False, False, False, True, True],
        "origin_top1_score": [.9, .5, .9, .2, .9, .9],
        "compressed_top1_score": [.95, .85, .7, .3, .95, .95],
        "compressed_score_at_origin_top1": [.95, .85, .7, .2, .8, .8],
        "origin_top1_gallery_id": ["g"]*6,
        "compressed_top1_gallery_id": ["g", "g", "g", "h", "h", "h"],
        "origin_top_k_correct": [False]*4+[True, True],
        "compressed_top_k_correct": [False]*4+[True, True],
        "origin_true_identity_score": [np.nan]*4+[.85, .75],
        "compressed_true_identity_score": [np.nan]*4+[.75, .85],
        "origin_true_identity_rank": [np.nan]*4+[1, 2],
        "compressed_true_identity_rank": [np.nan]*4+[2, 1],
    })
    for col, value in dict(compression_profile="pq_512_m16_b8", search_mode="pq_reconstruction_cosine",
                           threshold_policy="frozen_origin", target_fpir=.01, protocol_uid="test-protocol",
                           model_uid="test-model", gallery_template_count=3, top_k=20,
                           origin_score_space="cosine_similarity", compressed_score_space="cosine_similarity",
                           query_representation="pq_reconstructed_float32", gallery_representation="pq_reconstructed_float32",
                           distance_function="cosine_similarity",
                           score_spaces_comparable=True, frozen_origin_threshold_applicable=True,
                           origin_fallback_used=False, origin_decision_threshold=.8,
                           compressed_decision_threshold=.8).items():
        data[col] = value
    for side in ("origin", "compressed"):
        data[f"{side}_accepted"] = data[f"{side}_top1_score"] >= .8
        data[f"{side}_tpir_at_rank_k"] = data.is_mated & (data[f"{side}_true_identity_score"] >= .8)
    return data


def test_nonmated_events_and_genuine_metric(frame):
    row, tails, winners = diagnose_decision(frame, resamples=100)
    assert row["n"] == 4 and row["mated_count"] == 2
    assert (row["both_fa"], row["neither_fa"], row["new_fa"], row["lost_fa"]) == (1, 1, 1, 1)
    assert row["new_fa_same_winner"] == 1
    assert row["compressed_tpir20"] == .5  # max score would wrongly give 1
    assert row["compressed_rank20"] == 1
    assert row["score_effect_at_origin_threshold"] + row["threshold_effect_after_score_change"] == row["delta_fpir"]
    assert len(tails) == 10 and len(winners) == 4


@pytest.mark.parametrize("column,value", [("compressed_accepted", False), ("compressed_tpir_at_rank_k", True)])
def test_rejects_wrong_stored_metric(frame, column, value):
    frame[column] = value
    with pytest.raises(ValueError):
        diagnose_decision(frame)


def test_duplicate_query_rejected(frame):
    frame.loc[1, "query_id"] = "a"
    with pytest.raises(ValueError, match="unique"):
        diagnose_decision(frame)


@pytest.mark.parametrize("column,value", [
    ("origin_fallback_used", True),
    ("is_mated", None),
    ("compressed_true_identity_rank", 21),
    ("compressed_decision_threshold", .7),
    ("query_representation", "origin_float32"),
])
def test_invalid_contract_rejected(frame, column, value):
    frame[column] = value
    with pytest.raises(ValueError):
        diagnose_decision(frame)


def test_adc_has_no_cross_space_decomposition(frame):
    frame["search_mode"] = "pq_adc_exhaustive"
    frame["compressed_score_space"] = "negative_squared_l2_adc"
    frame["score_spaces_comparable"] = False
    frame["frozen_origin_threshold_applicable"] = False
    frame["query_representation"] = "origin_float32"
    frame["gallery_representation"] = "pq_code"
    frame["distance_function"] = "squared_l2_asymmetric"
    with pytest.raises(ValueError, match="not applicable"):
        diagnose_decision(frame)
    frame["threshold_policy"] = "recalibrated_compressed"
    frame["compressed_score_at_origin_top1"] = np.nan
    row, _, winners = diagnose_decision(frame)
    assert np.isnan(row["score_effect_at_origin_threshold"]) and not winners


def test_float_tolerance_preserves_raw_negative(frame):
    frame.loc[0, "compressed_score_at_origin_top1"] += 5e-7
    _, _, winners = diagnose_decision(frame)
    assert winners[0]["selection_gain_min_raw"] < 0
    frame.loc[0, "compressed_score_at_origin_top1"] += 1e-4
    with pytest.raises(ValueError, match="invariant"):
        diagnose_decision(frame)


def test_empty_new_cohort_mean_is_nan(frame):
    frame["compressed_top1_score"] = frame.origin_top1_score
    frame["compressed_score_at_origin_top1"] = frame.origin_top1_score
    frame["compressed_accepted"] = frame.origin_accepted
    _, _, winners = diagnose_decision(frame)
    assert winners[1]["count"] == 0 and np.isnan(winners[1]["fixed_winner_drift_mean"])


def test_paired_counts_reproducible():
    a = [False, True, True, False]
    b = [True, False, True, False]
    assert event_counts(a, b) == event_counts(a, b)
    with pytest.raises(ValueError):
        event_counts([], [])


def test_verified_path_and_immutable_export(tmp_path):
    result = {name: pd.DataFrame({"x": [1]}) for name in ("summary", "tails", "winners", "adjacent")}
    result["provenance"] = {"seed": 8972}
    first = write_diagnosis(result, tmp_path)
    before = sha256_file(first / "manifest.json")
    second = write_diagnosis(result, tmp_path)
    assert first != second and sha256_file(first / "manifest.json") == before
    manifest = json.loads((first / "manifest.json").read_text())
    entry = manifest["output_files"]["summary.csv"]
    assert verified_path(first, entry).is_file()
    with pytest.raises(ValueError, match="SHA"):
        verified_path(first, {**entry, "sha256": "0"*64})
    with pytest.raises(ValueError, match="escapes"):
        verified_path(first, {**entry, "path": "../outside.csv"})
