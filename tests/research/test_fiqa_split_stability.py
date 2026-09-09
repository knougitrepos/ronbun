import copy
import json

import numpy as np
import pandas as pd
import pytest

from research.experiments.fiqa_split_stability import (
    _partition_inventory, _validated_seeds, run_fiqa_split_stability,
    summarize_split_stability, write_fiqa_split_stability,
)
from research.experiments.fiqa_threshold_calibration import ConditionScoreTables
from research.fiqa import FIQAScoreArtifact
from research.runtime.hashing import sha256_file
from test_fiqa_threshold_calibration import _calibration_rows, _condition_manifest, _fiqa_manifest


def _inputs(tmp_path):
    cal, test = _calibration_rows("calibration", 400), _calibration_rows("test", 200)
    raw = pd.concat([cal, test], ignore_index=True)
    artifacts = []
    for variant in ("S", "L"):
        uid = f"cr-fiqa-{variant.lower()}-test"
        scores = raw[["sample_id", "fiqa_score", "aligned_content_sha256"]].copy()
        scores["fiqa_model_uid"] = uid
        manifest = {**_fiqa_manifest(), "variant": variant, "fiqa_model_uid": uid, "fiqa_uid": uid}
        artifacts.append(FIQAScoreArtifact(tmp_path, scores, manifest))
    return ConditionScoreTables(
        cal.drop(columns=["fiqa_score", "fiqa_model_uid"]),
        test.drop(columns=["fiqa_score", "fiqa_model_uid"]), _condition_manifest()), artifacts


def _run(condition, artifacts, **kwargs):
    return run_fiqa_split_stability(condition, *artifacts, partition_seeds=(0, 42, 8972),
                                    target_fpirs=(.1,), resamples=100,
                                    minimum_group_non_mated=5, **kwargs)


def test_shared_panel_conditional_ci_reproducibility_and_immutable_output(tmp_path):
    condition, artifacts = _inputs(tmp_path)
    progress = []
    result = _run(condition, artifacts, progress=progress.append)
    again = _run(condition, artifacts)
    assert progress[-1] == {"completed": 3, "total": 3, "partition_seed": 8972}
    assert result["manifest"]["stability_uid"] == again["manifest"]["stability_uid"]
    pd.testing.assert_frame_equal(result["seed_metrics"], again["seed_metrics"], check_exact=True)
    assert len(result["seed_metrics"]) == 12
    paired = result["seed_paired_comparisons"]
    sl = paired[(paired.reference_method == "fiqa_s") & (paired.candidate_method == "fiqa_l")]
    assert sl.candidate_minus_reference.eq(0).all()
    assert sl.paired_bootstrap95_low.eq(0).all()
    assert sl.paired_bootstrap95_high.eq(0).all()
    assert result["stability_summary"].between_split_statistics_are_ci.eq(False).all()
    assert result["stability_summary"].split_count.eq(3).all()
    assert result["manifest"]["uncertainty"]["joint_split_test_ci"] is False
    baseline = result["seed_metrics"].query("method == 'global_empirical'")
    assert baseline.realized_fpir.nunique() == 1
    assert baseline.tpir_at_rank_k.nunique() == 1
    path = write_fiqa_split_stability(tmp_path / "out", result)
    manifest = json.loads((path / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert sha256_file(path / name) == digest
    with pytest.raises(FileExistsError):
        write_fiqa_split_stability(tmp_path / "out", result)


def test_test_shift_cannot_change_partition_or_fitted_thresholds(tmp_path):
    condition, artifacts = _inputs(tmp_path)
    before = _run(condition, artifacts)
    shifted = copy.deepcopy(condition)
    shifted.test.loc[~shifted.test.is_mated, "score"] = 100
    after = _run(shifted, artifacts)
    for table in ("partition_inventory", "seed_thresholds", "threshold_stability_summary"):
        pd.testing.assert_frame_equal(before[table], after[table], check_exact=True)
    for result in (before, after):
        result["calibration_tails"] = result["seed_group_tail_transfer"].query("split != 'test'")
    pd.testing.assert_frame_equal(before["calibration_tails"], after["calibration_tails"], check_exact=True)
    assert after["seed_metrics"].realized_fpir.eq(1).all()
    assert after["manifest"]["stability_uid"] != before["manifest"]["stability_uid"]


@pytest.mark.parametrize("seeds", [(0,), (0, 0), (True, 2), (-1, 2), (1.1, 2)])
def test_invalid_seed_panel(seeds):
    with pytest.raises(ValueError, match="partition seeds"):
        _validated_seeds(seeds)


def test_inventory_and_missing_seed_fail_closed(tmp_path):
    condition, artifacts = _inputs(tmp_path)
    a = _partition_inventory(condition.calibration, 42, .3)
    b = _partition_inventory(condition.calibration.iloc[::-1], 42, .3)
    assert a == b
    condition.calibration.loc[0, "identity_id"] = None
    with pytest.raises(ValueError, match="identity"):
        _partition_inventory(condition.calibration, 42, .3)
    condition, artifacts = _inputs(tmp_path)
    result = _run(condition, artifacts)
    with pytest.raises(ValueError, match="incomplete"):
        summarize_split_stability(result["seed_metrics"].iloc[1:],
                                  result["seed_paired_comparisons"], result["seed_thresholds"],
                                  (0, 42, 8972))


def test_all_mixed_and_no_target_attainment_are_descriptive(tmp_path):
    condition, artifacts = _inputs(tmp_path)
    result = _run(condition, artifacts)
    metrics = result["seed_metrics"].copy()
    for method, flags in (("global_safe", [False]*3), ("fiqa_s", [False, True, False]),
                          ("fiqa_l", [True]*3)):
        mask = metrics.method.eq(method)
        metrics.loc[mask, "target_met_on_test"] = flags
    summary, _, _ = summarize_split_stability(metrics, result["seed_paired_comparisons"],
                                              result["seed_thresholds"], (0, 42, 8972))
    summary = summary.set_index("method")
    assert summary.loc["global_safe", "observed_pattern"] == "all_observed_splits_exceed"
    assert summary.loc["fiqa_s", "observed_pattern"] == "split_sensitive_target_attainment"
    assert summary.loc["fiqa_l", "observed_pattern"] == "all_observed_splits_meet"
    assert np.isfinite(summary.fpir_median).all()
