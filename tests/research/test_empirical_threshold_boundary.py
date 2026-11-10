"""Empirical ceilings and score-gap acceptance in the shared calibration engine."""
import numpy as np
import pandas as pd
import pytest

from research.calibration.rejection import EMPIRICAL_THRESHOLD_RULE, choose_non_mated_fpir_threshold
from research.calibration.conditional import fit_global_threshold, apply_threshold_model
from research.calibration.continuous import fit_continuous_threshold, _heldout_safety_offset


@pytest.mark.parametrize("scores,target", [
    ([-1, -2, -3, -4], .25),
    ([.8, .8, .7, .6], .5),
    ([.8, .8, .7, .6], .25),
    ([0, 0, 0], 0),
    ([0, 0, 0], 1),
    ([-1, -2, -3], np.nextafter(1., 0.)),
    (-np.arange(1, 11), np.nextafter(.1, 0.)),
    ([np.finfo(float).max], 0),
    (np.array([.8, .7, .6], dtype=np.float32), .34),
])
def test_threshold_obeys_exact_ceiling_and_cannot_be_relaxed(scores, target):
    scores = np.asarray(scores, dtype=np.float64)
    tau = choose_non_mated_fpir_threshold(scores, np.zeros(len(scores), bool), target)
    assert np.mean(scores >= tau) <= target
    if target == 1:
        assert tau == -np.inf
    else:
        # The immediately less restrictive representable boundary violates the
        # budget. This checks maximality independently of the selector's sort.
        with np.errstate(over="ignore"):
            relaxed = np.nextafter(tau, -np.inf)
        assert np.mean(scores >= relaxed) > target


def calibration_rows(count=10, safety_shift=0):
    rows = []
    labels = []
    for part in ("fit", "safety"):
        shift = safety_shift if part == "safety" else 0
        for i in range(count):
            rows.append(dict(sample_id=f"{part}-n{i}", identity_id=f"{part}-n{i}",
                             score=-float(i+1)+shift, is_mated=False, fiqa_score=.5))
            labels.append(part)
        rows.append(dict(sample_id=f"{part}-m", identity_id=f"{part}-m",
                         score=-1.5+shift, is_mated=True, fiqa_score=.5))
        labels.append(part)
    return pd.DataFrame(rows), labels


def test_public_global_safe_admits_genuine_gap_without_extra_false_accepts():
    calibration, labels = calibration_rows()
    model = fit_global_threshold(calibration, target_fpir=.1,
                                 score_space="negative_squared_l2_adc", partition_labels=labels)
    assert model.groups[0].fit_fpir_at_final_threshold == .1
    assert model.groups[0].safety_fpir_at_final_threshold == .1
    assert model.calibration_partition["empirical_threshold_rule"] == EMPIRICAL_THRESHOLD_RULE
    test = calibration.iloc[:11].copy()
    test["sample_id"] = "test-" + test.sample_id
    test["identity_id"] = "test-" + test.identity_id
    test["top_k"] = 20
    test["top_k_correct"] = test.is_mated
    test["true_identity_score"] = np.where(test.is_mated, test.score, np.nan)
    test["true_identity_rank"] = np.where(test.is_mated, 1., np.nan)
    result = apply_threshold_model(test, model)
    assert result.summary["false_accept_count"] == 1
    assert result.summary["true_identification_at_rank_k_count"] == 1
    assert result.summary["target_met_on_test"]
    changed = calibration.copy()
    changed.loc[changed.is_mated, "score"] = 1000
    other = fit_global_threshold(changed, target_fpir=.1,
                                 score_space="negative_squared_l2_adc", partition_labels=labels)
    assert other == model  # Genuine/test scores cannot tune this threshold.


def test_public_continuous_keeps_safety_ceiling_and_accepts_score_gap():
    calibration, labels = calibration_rows(count=20, safety_shift=100)
    model = fit_continuous_threshold(calibration, target_fpir=.1, partition_labels=labels)
    assert model.settings["empirical_threshold_rule"] == EMPIRICAL_THRESHOLD_RULE
    safety = calibration.loc[np.array(labels) == "safety"]
    negative = safety.loc[~safety.is_mated]
    assert np.count_nonzero(negative.score.to_numpy(dtype=float) >= model.predict(negative)) == 2
    genuine = safety.loc[safety.is_mated].copy()
    genuine["score"] = 97.5  # Between the first rejected score 97 and accepted 98.
    assert (genuine.score.to_numpy(dtype=float) >= model.predict(genuine)).all()


@pytest.mark.parametrize("base", [0., 1e8, -1e8])
def test_continuous_guard_checks_actual_rounded_threshold_without_tolerance(base):
    scores = base + np.arange(1, 11, dtype=float)
    target = np.nextafter(.1, 0.)  # One in ten exceeds this target.
    offset = _heldout_safety_offset(scores, np.full(10, base), target_fpir=target)
    assert np.mean(scores >= base + offset) <= target
