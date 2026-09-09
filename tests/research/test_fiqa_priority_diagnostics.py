import json

import numpy as np
import pandas as pd
import pytest

from research.experiments.fiqa_priority_diagnostics import (
    cluster_rate_draws, quality_tail_transfer,
    run_fiqa_priority_diagnostics, write_fiqa_priority_diagnostics,
)
from research.calibration.conditional import fit_conditional_threshold
from research.experiments.fiqa_threshold_calibration import ConditionScoreTables
from research.fiqa import FIQAScoreArtifact
from research.runtime.hashing import sha256_file
from test_fiqa_threshold_calibration import (
    _calibration_rows, _condition_manifest, _fiqa_manifest,
)


def test_cluster_bootstrap_is_paired_and_query_weighted():
    ids = ["a"] * 9 + ["b"]
    events = np.array([[1, 1]] * 9 + [[0, 0]])
    draws = cluster_rate_draws(ids, events, resamples=500)
    assert np.array_equal(draws[:, 0], draws[:, 1])
    assert set(np.unique(draws)) == {0, .9, 1}  # NOT equal-identity mean .5
    assert np.array_equal(draws, cluster_rate_draws(ids, events, resamples=500))
    with pytest.raises(ValueError, match="two genuine"):
        cluster_rate_draws(["a"] * 10, events)


def test_tail_diagnostic_does_not_refit_on_test():
    cal = _calibration_rows("calibration", 400)
    test = _calibration_rows("test", 200)
    model = fit_conditional_threshold(cal, target_fpir=.1, minimum_group_non_mated=5,
                                      score_space="negative_squared_l2_adc")
    before = model.as_dict()
    first = quality_tail_transfer(cal, test, model, safety_fraction=.3, seed=8972)
    test["score"] = 100
    shifted = quality_tail_transfer(cal, test, model, safety_fraction=.3, seed=8972)
    assert model.as_dict() == before
    pd.testing.assert_frame_equal(first[first.split != "test"], shifted[shifted.split != "test"])
    assert shifted.loc[shifted.split == "test", "realized_fpir"].eq(1).all()


def test_same_split_comparison_and_immutable_artifact(tmp_path):
    cal = _calibration_rows("calibration", 400)
    test = _calibration_rows("test", 200)
    raw = pd.concat([cal, test], ignore_index=True)
    artifacts = []
    for variant in ("S", "L"):
        uid = f"cr-fiqa-{variant.lower()}-test"
        scores = raw[["sample_id", "fiqa_score", "aligned_content_sha256"]].copy()
        scores["fiqa_model_uid"] = uid
        manifest = {**_fiqa_manifest(), "variant": variant,
                    "fiqa_model_uid": uid, "fiqa_uid": uid}
        artifacts.append(FIQAScoreArtifact(tmp_path, scores, manifest))
    condition = ConditionScoreTables(
        cal.drop(columns=["fiqa_score", "fiqa_model_uid"]),
        test.drop(columns=["fiqa_score", "fiqa_model_uid"]), _condition_manifest())
    result = run_fiqa_priority_diagnostics(
        condition, *artifacts, target_fpirs=(.1,), minimum_group_non_mated=5,
        resamples=100)
    paired = result["paired_comparisons"]
    sl = paired[(paired.reference_method == "fiqa_s") & (paired.candidate_method == "fiqa_l")]
    assert sl.candidate_minus_reference.eq(0).all()
    assert sl.paired_bootstrap95_low.eq(0).all()
    assert sl.paired_bootstrap95_high.eq(0).all()
    assert paired.loc[paired.metric == "tpir_at_rank_k", "resampling_unit"].eq("mated_identity_cluster").all()
    path = write_fiqa_priority_diagnostics(tmp_path, result)
    manifest = json.loads((path / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert sha256_file(path / name) == digest
    with pytest.raises(FileExistsError):
        write_fiqa_priority_diagnostics(tmp_path, result)
    with pytest.raises(ValueError, match="variant"):
        run_fiqa_priority_diagnostics(condition, artifacts[1], artifacts[0])
