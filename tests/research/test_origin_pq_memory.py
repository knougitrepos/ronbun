"""Numerical parity while dropping repeated per-query provenance from working copies."""

import json
import weakref

import numpy as np
import pandas as pd

import research.experiments.origin_vs_pq_calibration as module
from research.calibration.conditional import apply_threshold_model
from research.calibration.continuous import apply_continuous_threshold
from research.evaluation.cluster_bootstrap import cluster_rate_draws
from research.evaluation.metrics import paired_binary_rate_difference_bootstrap_interval
from test_origin_vs_pq_calibration import inputs, options


def test_compact_execution_matches_full_frame_canonical_evaluation(tmp_path, monkeypatch):
    conditions, fiqa = inputs(tmp_path)
    for index, condition in enumerate(conditions.values()):
        for frame in (condition.calibration, condition.test):
            frame["irrelevant_large_metadata"] = "source-details-" * 1000
        test = condition.test
        mated_ids = test.index[test.is_mated]
        # Include genuine-score threshold failures, rank exits, and distinct
        # compression effects; maxima must never replace genuine scores.
        test.loc[mated_ids[::3], "true_identity_score"] -= .20 + index * .03
        test.loc[mated_ids[1::7], ["true_identity_score", "true_identity_rank"]] = np.nan
        test.loc[mated_ids[1::7], "top_k_correct"] = False
        test.loc[~test.is_mated, "score"] += index * .015
    settings = options()
    joined = module._joined_frames(conditions, fiqa)
    target, seed = settings.target_fpirs[0], 8972
    evaluations, fits = {}, {}
    for profile, (calibration, test) in joined.items():
        for method, model in module._fit_models(
                calibration, conditions[profile].manifest["score_space"], seed, target, settings).items():
            key = profile, method
            fits[key] = model
            evaluations[key] = (apply_continuous_threshold(test, model) if method == "continuous_fiqa"
                                else apply_threshold_model(test, model))

    live_evaluations = []

    def bounded_apply(original):
        def apply(frame, model):
            assert "irrelevant_large_metadata" not in frame
            assert "aligned_content_sha256" not in frame
            # Only one canonical evaluation frame is alive at a time.
            assert all(ref() is None for ref in live_evaluations)
            result = original(frame, model)
            live_evaluations.append(weakref.ref(result))
            return result
        return apply

    monkeypatch.setattr(module, "apply_threshold_model", bounded_apply(apply_threshold_model))
    monkeypatch.setattr(module, "apply_continuous_threshold", bounded_apply(apply_continuous_threshold))
    results = module.run_origin_pq_split(conditions, fiqa, settings=settings)
    assert len(live_evaluations) == len(evaluations)
    assert all(ref() is None for ref in live_evaluations)
    for row in results["models"].itertuples(index=False):
        assert json.loads(row.model_json) == json.loads(json.dumps(fits[row.compression_profile, row.method].as_dict()))
    for row in results["method_summary"].to_dict("records"):
        key = row["compression_profile"], row["method"]
        expected = {**evaluations[key].summary, **module.failure_decomposition(evaluations[key])}
        # Campaign labels deliberately shorten the canonical fit method name.
        expected["method"] = key[1]
        assert all(row[name] == value for name, value in expected.items())
        curve = module.frozen_score_diagnostic(evaluations[key], settings.diagnostic_fpir_grid)
        actual = results["diagnostic_curves"]
        actual = actual.loc[actual.compression_profile.eq(key[0]) & actual.method.eq(key[1]), curve.columns]
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), curve, check_exact=True)

    keys = list(evaluations)
    anchor = joined[module.ORIGIN_PROFILE][1]
    mated = anchor.is_mated.to_numpy(dtype=bool)
    events = np.column_stack([evaluations[k].decisions.true_identification_at_rank_k.to_numpy()[mated] for k in keys])
    draws = cluster_rate_draws(anchor.loc[mated, "identity_id"], events,
                               resamples=settings.resamples, seed=settings.bootstrap_seed)
    for row in results["paired_comparisons"].itertuples(index=False):
        left = row.reference_profile, row.reference_method
        right = row.candidate_profile, row.candidate_method
        tpir = row.metric == "tpir_at_rank_k"
        column = "true_identification_at_rank_k" if tpir else "false_accept"
        mask = mated if tpir else ~mated
        a = evaluations[left].decisions[column].to_numpy(dtype=bool)[mask]
        b = evaluations[right].decisions[column].to_numpy(dtype=bool)[mask]
        ci = (np.quantile(draws[:, keys.index(right)] - draws[:, keys.index(left)], [.025, .975]) if tpir
              else paired_binary_rate_difference_bootstrap_interval(
                  int(a.sum()), int(b.sum()), int((a & b).sum()), len(a),
                  resamples=settings.resamples, random_seed=settings.bootstrap_seed))
        assert row.reference_successes == int(a.sum())
        assert row.candidate_successes == int(b.sum())
        assert row.both_successes == int((a & b).sum())
        assert row.candidate_minus_reference == float(b.mean() - a.mean())
        np.testing.assert_array_equal([row.paired_bootstrap95_low, row.paired_bootstrap95_high], ci)
    for row in results["interactions"].itertuples(index=False):
        order = [(row.compression_profile, row.method), (row.compression_profile, "global_safe"),
                 (module.ORIGIN_PROFILE, row.method), (module.ORIGIN_PROFILE, "global_safe")]
        di = [draws[:, keys.index(key)] for key in order]
        ci = np.quantile(di[0] - di[1] - di[2] + di[3], [.025, .975])
        np.testing.assert_array_equal([row.paired_bootstrap95_low, row.paired_bootstrap95_high], ci)


def test_prevalidated_join_is_reused_without_repeating_merge(tmp_path, monkeypatch):
    conditions, fiqa = inputs(tmp_path)
    joined = module._joined_frames(conditions, fiqa)

    def repeated_join(*args, **kwargs):
        raise AssertionError("full FIQA/provenance join repeated")

    monkeypatch.setattr(module, "_joined_frames", repeated_join)
    result = module.run_origin_pq_split(conditions, fiqa, settings=options(), joined_frames=joined)
    assert len(result["method_summary"]) == 6
