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
from research.experiments.fiqa_priority_diagnostics import (
    run_fiqa_priority_diagnostics, write_fiqa_priority_diagnostics,
)
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
    return run_fiqa_split_stability(condition, *artifacts, partition_seeds=(0, 1, 8972),
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
    a = _partition_inventory(condition.calibration, 8972, .3)
    b = _partition_inventory(condition.calibration.iloc[::-1], 8972, .3)
    assert a == b
    condition.calibration.loc[0, "identity_id"] = None
    with pytest.raises(ValueError, match="identity"):
        _partition_inventory(condition.calibration, 8972, .3)
    condition, artifacts = _inputs(tmp_path)
    result = _run(condition, artifacts)
    with pytest.raises(ValueError, match="incomplete"):
        summarize_split_stability(result["seed_metrics"].iloc[1:],
                                  result["seed_paired_comparisons"], result["seed_thresholds"],
                                  (0, 1, 8972))


def test_all_mixed_and_no_target_attainment_are_descriptive(tmp_path):
    condition, artifacts = _inputs(tmp_path)
    result = _run(condition, artifacts)
    metrics = result["seed_metrics"].copy()
    for method, flags in (("global_safe", [False]*3), ("fiqa_s", [False, True, False]),
                          ("fiqa_l", [True]*3)):
        mask = metrics.method.eq(method)
        metrics.loc[mask, "target_met_on_test"] = flags
    summary, _, _ = summarize_split_stability(metrics, result["seed_paired_comparisons"],
                                              result["seed_thresholds"], (0, 1, 8972))
    summary = summary.set_index("method")
    assert summary.loc["global_safe", "observed_pattern"] == "all_observed_splits_exceed"
    assert summary.loc["fiqa_s", "observed_pattern"] == "split_sensitive_target_attainment"
    assert summary.loc["fiqa_l", "observed_pattern"] == "all_observed_splits_meet"
    assert np.isfinite(summary.fpir_median).all()


def test_multibin_panel_pairs_bins_and_shares_partitions(tmp_path):
    condition, artifacts = _inputs(tmp_path)
    result = _run(condition, artifacts, bin_counts=(2, 5))
    assert len(result["seed_metrics"]) == 18
    assert result["stability_summary"].split_count.eq(3).all()
    paired = result["seed_paired_comparisons"]
    assert len(paired.query("reference_method == 'fiqa_l_2bin' and candidate_method == 'fiqa_l_5bin'")) == 6
    assert result["partition_inventory"].shared_by.str.contains("fiqa_l_5bin").all()
    shifted = copy.deepcopy(condition)
    shifted.test.loc[~shifted.test.is_mated, "score"] = 100
    after = _run(shifted, artifacts, bin_counts=(2, 5))
    pd.testing.assert_frame_equal(result["seed_thresholds"], after["seed_thresholds"])
    pd.testing.assert_frame_equal(result["partition_inventory"], after["partition_inventory"])
    path = write_fiqa_split_stability(tmp_path / "multi", result)
    assert write_fiqa_split_stability(tmp_path / "multi", result, reuse_existing=True) == path
    (path / "seed_metrics.csv").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        write_fiqa_split_stability(tmp_path / "multi", result, reuse_existing=True)


def test_multibin_matches_standalone_and_exposes_sparse_group_fallback(tmp_path):
    condition, artifacts = _inputs(tmp_path)
    kwargs = dict(target_fpirs=(.1,), minimum_group_non_mated=5, resamples=100)
    combined = run_fiqa_priority_diagnostics(condition, *artifacts, bin_counts=(2, 5), **kwargs)
    for bins in (2, 5):
        single = run_fiqa_priority_diagnostics(condition, *artifacts, bin_count=bins, **kwargs)
        for name in ("global_empirical", "global_safe", "fiqa_s", "fiqa_l"):
            candidate = name if name.startswith("global") else f"{name}_{bins}bin"
            left = single["method_summary"].query("method == @name").reset_index(drop=True)
            right = combined["method_summary"].loc[
                combined["method_summary"].method.eq(candidate)
            ].reset_index(drop=True)
            right["method"] = name
            pd.testing.assert_frame_equal(left, right)
    paired = combined["paired_comparisons"]
    sl = paired.query("reference_method == 'fiqa_s_5bin' and candidate_method == 'fiqa_l_5bin'")
    assert sl.paired_bootstrap95_low.eq(0).all()
    assert sl.paired_bootstrap95_high.eq(0).all()
    sparse = run_fiqa_priority_diagnostics(
        condition, *artifacts, bin_counts=(2, 5),
        **{**kwargs, "minimum_group_non_mated": 10000})
    thresholds = sparse["thresholds"].query("method.str.startswith('fiqa')", engine="python")
    assert thresholds.used_global_fallback.all()
    assert np.isfinite(thresholds.final_threshold).all()
    path = write_fiqa_priority_diagnostics(tmp_path / "multi", combined)
    assert write_fiqa_priority_diagnostics(tmp_path / "multi", combined, reuse_existing=True) == path
    (path / "thresholds.csv").write_text("corrupt", encoding="utf8")
    with pytest.raises(ValueError, match="hash mismatch"):
        write_fiqa_priority_diagnostics(tmp_path / "multi", combined, reuse_existing=True)


@pytest.mark.parametrize("bins", [(), (2, 2), (True, 5), (2, 1), (2, 5.5)])
def test_invalid_bin_plan_fails_before_fitting(tmp_path, bins):
    condition, artifacts = _inputs(tmp_path)
    with pytest.raises(ValueError, match="bin_counts"):
        run_fiqa_priority_diagnostics(condition, *artifacts, bin_counts=bins)
