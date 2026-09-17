import copy
import json

import numpy as np
import pandas as pd
import pytest

from research.calibration.conditional import deterministic_calibration_partition
from research.calibration.continuous import apply_continuous_threshold, fit_continuous_threshold
from research.calibration.saliency_fallback import (
    SALIENCY_FEATURES, SaliencyFallbackModel, fit_saliency_fallback, saliency_valid_mask,
)
from research.experiments.fiqa_threshold_calibration import join_fiqa_score_artifacts
from research.experiments.saliency_incremental_calibration import (
    assess_incremental_gate, load_saliency_incremental_inputs,
    run_saliency_incremental_calibration, write_saliency_incremental_result,
)
from test_saliency_incremental_calibration import _sources, _receipt


def _invalid_sources(tmp_path, *, all_invalid=False, corrupt=None):
    c, f, sr, fr = _sources(tmp_path)
    rows = pd.read_csv(sr/'saliency_features.csv')
    chosen = rows.index if all_invalid else rows.groupby('split').head(3).index
    rows.loc[chosen, 'gradcam_valid_heatmap'] = False
    rows.loc[chosen, list(SALIENCY_FEATURES)] = np.nan
    if corrupt == 'unflagged_nan':
        rows.loc[chosen, 'gradcam_valid_heatmap'] = True
    elif corrupt == 'infinity':
        rows.loc[chosen, SALIENCY_FEATURES[0]] = np.inf
    elif corrupt == 'null_flag':
        rows['gradcam_valid_heatmap'] = rows.gradcam_valid_heatmap.astype(object)
        rows.loc[chosen, 'gradcam_valid_heatmap'] = None
    rows.to_csv(sr/'saliency_features.csv', index=False)
    manifest = json.loads((sr/'manifest.json').read_text())
    manifest['saliency_features'] = _receipt(sr/'saliency_features.csv')
    (sr/'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    return c, f, load_saliency_incremental_inputs(c, sr, fr)


def test_invalid_maps_continue_with_full_cohort_and_joint_safety(tmp_path):
    c, f, inputs = _invalid_sources(tmp_path)
    gate = assess_incremental_gate(c, inputs)
    assert gate['comparison_enabled'] and gate['invalid_saliency_counts'] == {'calibration': 3, 'test': 3}
    assert gate['readiness']['status'] == 'ready'
    before = inputs['saliency'].copy(deep=True)
    result = run_saliency_incremental_calibration(c, f, inputs, target_fpirs=(.1,),
                                                 partition_seeds=(0, 8972), resamples=100)
    pd.testing.assert_frame_equal(before, inputs['saliency'])
    metrics = result['method_summary']
    assert metrics.test_probe_count.eq(len(c.test)).all()
    assert metrics.loc[metrics.method.ne('baseline'), 'fallback_query_count'].eq(3).all()
    assert metrics.loc[metrics.method.eq('baseline'), 'fallback_query_count'].eq(0).all()
    assert len(result['fallback_queries']) == 6
    routing = result['fallback_diagnostics']
    for (_, _, _), group in routing.loc[routing.split.eq('test')].groupby(['method', 'partition_seed', 'target_fpir']):
        assert group.query_count.sum() == len(c.test)
    for (_, _), group in routing.loc[routing.partition.eq('safety')].groupby(['method', 'partition_seed']):
        assert group.false_accept_count.sum() / group.non_mated_count.sum() <= .1 + 1e-12
    for _, row in metrics.iterrows():
        group = routing.loc[routing.split.eq('test') & routing.method.eq(row.method)
                            & routing.partition_seed.eq(row.partition_seed)]
        assert group.false_accept_count.sum() == row.false_accept_count
        assert group.true_identification_at_rank_k_count.sum() == row.true_identification_at_rank_k_count
    saved = write_saliency_incremental_result(tmp_path/'out', result)
    assert (saved/'fallback_queries.csv').is_file()
    assert (saved/'fallback_diagnostics.csv').is_file()
    assert write_saliency_incremental_result(tmp_path/'out', result) == saved
    changed = copy.deepcopy(c)
    changed.test.loc[~changed.test.is_mated, 'score'] += .3
    again = run_saliency_incremental_calibration(changed, f, inputs, target_fpirs=(.1,),
                                                partition_seeds=(0, 8972), resamples=100)
    assert result['models'].model_json.equals(again['models'].model_json)


@pytest.mark.parametrize('corrupt', ['unflagged_nan', 'infinity', 'null_flag'])
def test_corruption_is_not_silently_routed(tmp_path, corrupt):
    c, f, inputs = _invalid_sources(tmp_path, corrupt=corrupt)
    assert not assess_incremental_gate(c, inputs)['comparison_enabled']
    with pytest.raises(ValueError, match='gate blocked'):
        run_saliency_incremental_calibration(c, f, inputs)


def test_calibration_cache_without_genuine_scores_has_fpir_only_diagnostics(tmp_path):
    c, f, inputs = _invalid_sources(tmp_path)
    # Real ADC calibration caches omit these test-only fields.
    c.calibration.drop(columns=['true_identity_score', 'true_identity_rank'], inplace=True)
    result = run_saliency_incremental_calibration(c, f, inputs, target_fpirs=(.1,),
                                                 partition_seeds=(8972,), resamples=100)
    routing = result['fallback_diagnostics']
    cal = routing.loc[routing.split.eq('calibration')]
    test = routing.loc[routing.split.eq('test')]
    assert not cal.tpir_available.any()
    assert cal[['true_identification_at_rank_k_count', 'tpir_at_rank_k']].isna().all().all()
    assert cal.false_accept_count.notna().all()
    assert test.tpir_available.all()
    assert test.true_identification_at_rank_k_count.notna().all()
    assert result['method_summary'].test_probe_count.eq(len(c.test)).all()


def _frames(tmp_path):
    c, f, inputs = _invalid_sources(tmp_path)
    cal, test = join_fiqa_score_artifacts(c, f)
    sal = inputs['saliency'].set_index('sample_id')
    for frame in (cal, test):
        for col in (*SALIENCY_FEATURES, 'heatmap_available', 'gradcam_valid_heatmap'):
            frame[col] = sal.loc[frame.sample_id, col].to_numpy()
    return cal, test


def test_routing_serialization_genuine_score_and_valid_only_fit(tmp_path):
    cal, test = _frames(tmp_path)
    opts = dict(features=('fiqa_score', *SALIENCY_FEATURES), method='plus_both', target_fpir=.1)
    model = fit_saliency_fallback(cal, **opts)
    restored = SaliencyFallbackModel.from_dict(json.loads(json.dumps(model.as_dict(), allow_nan=False)))
    np.testing.assert_array_equal(model.predict(test), restored.predict(test))
    assert model.model_uid == restored.model_uid
    partition = deterministic_calibration_partition(cal, seed=8972, safety_fraction=.3, partition_column='identity_id')
    fit = cal.loc[partition.eq('fit') & ~cal.is_mated & saliency_valid_mask(cal)]
    np.testing.assert_array_equal(model.saliency_model.centers, np.median(fit[list(opts['features'])], axis=0))
    invalid = model.fallback_mask(test)
    np.testing.assert_array_equal(model.predict(test)[invalid], model.fiqa_model.predict(test)[invalid] + model.safety_offset)
    # Change only safety scores: fit coefficients must remain unchanged.
    altered = cal.copy()
    altered.loc[partition.eq('safety'), 'score'] += .4
    changed = fit_saliency_fallback(altered, **opts)
    assert model.saliency_model.coefficients == changed.saliency_model.coefficients
    assert model.fiqa_model.coefficients == changed.fiqa_model.coefficients
    tau = model.predict(test)
    test.loc[test.is_mated, 'score'] = tau[test.is_mated] + .1
    test.loc[test.is_mated, 'true_identity_score'] = tau[test.is_mated] - .1
    test.loc[test.is_mated, 'true_identity_rank'] = 2
    result = apply_continuous_threshold(test, model)
    assert result.decisions.loc[test.is_mated, 'accepted'].all()
    assert not result.decisions.true_identification_at_rank_k.any()


def test_zero_invalid_matches_original_saliency_and_all_invalid_matches_fiqa(tmp_path):
    cal, test = _frames(tmp_path)
    for frame in (cal, test):
        frame['gradcam_valid_heatmap'] = True
        frame[list(SALIENCY_FEATURES)] = frame[list(SALIENCY_FEATURES)].fillna(.5)
    opts = dict(features=('fiqa_score', *SALIENCY_FEATURES), method='plus_both', target_fpir=.1)
    original = fit_continuous_threshold(cal, **opts)
    routed = fit_saliency_fallback(cal, **opts)
    np.testing.assert_array_equal(original.predict(test), routed.predict(test))
    for frame in (cal, test):
        frame['gradcam_valid_heatmap'] = False
        frame[list(SALIENCY_FEATURES)] = np.nan
    routed = fit_saliency_fallback(cal, **opts)
    baseline = fit_continuous_threshold(cal, target_fpir=.1)
    assert routed.saliency_model is None and routed.fallback_mask(test).all()
    assert routed.settings['branch_disabled_reason'] == 'fewer_than_20_valid_fit_non_mated'
    np.testing.assert_array_equal(baseline.predict(test), routed.predict(test))
