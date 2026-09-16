import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research.calibration.continuous import fit_continuous_threshold, apply_continuous_threshold
from research.calibration.conditional import deterministic_calibration_partition
from research.experiments.fiqa_retrieval_features import adc_feature_frame, join_retrieval_features
from research.experiments.fiqa_continuous_calibration import (
    run_continuous_calibration, write_continuous_calibration, summarize_continuous_ablations,
)
from research.runtime.hashing import canonical_sha256
from research.evaluation.metrics import paired_binary_rate_difference_bootstrap_interval
from test_fiqa_split_stability import _inputs as _base_inputs
from test_fiqa_threshold_calibration import _calibration_rows


def _inputs(tmp_path):
    condition, artifacts = _base_inputs(tmp_path)
    condition.manifest['source_run_id'] = 'source'
    return condition, artifacts


def _features(rows):
    s1 = rows.score.to_numpy()
    gap = .01 + rows.fiqa_score.to_numpy() * .03 if 'fiqa_score' in rows else np.linspace(.01, .04, len(rows))
    return pd.DataFrame({"sample_id": rows.sample_id, "adc_s1": s1, "adc_s2": s1-gap,
                         "adc_margin": gap, "top1_gallery_id": "gallery",
                         "top1_gallery_pq_distortion": np.linspace(.03, .15, len(rows))})


def test_continuous_safety_bound_and_margin_monotonicity():
    cal = _calibration_rows("calibration", 1000)
    cal = join_retrieval_features(cal, _features(cal))
    model = fit_continuous_threshold(cal, target_fpir=.1,
                                    features=("fiqa_score", "adc_margin", "top1_gallery_pq_distortion"))
    partition = deterministic_calibration_partition(cal, seed=8972, safety_fraction=.3,
                                                     partition_column="identity_id")
    safety = cal.loc[partition.eq("safety") & ~cal.is_mated]
    assert np.mean(safety.score.to_numpy() >= model.predict(safety)) <= .1 + 1e-12
    x = cal.iloc[:2].copy()
    x.iloc[1] = x.iloc[0]
    x.loc[x.index[1], "adc_margin"] += .1
    thresholds = model.predict(x)
    assert 0 <= thresholds[1] - thresholds[0] <= .095 + 1e-12
    assert model.settings["formal_fpir_guarantee"] is False
    assert model == type(model)(**model.as_dict())


def test_continuous_evaluation_uses_genuine_score_and_frozen_quality():
    cal = _calibration_rows("calibration", 1000)
    model = fit_continuous_threshold(cal, target_fpir=.1)
    test = _calibration_rows("test", 200)
    tau = model.predict(test)
    # Accepted top-1 for every mated query does not imply genuine identity acceptance.
    test.loc[test.is_mated, "score"] = tau[test.is_mated] + .1
    test.loc[test.is_mated, "true_identity_score"] = tau[test.is_mated] - .1
    test.loc[test.is_mated, "true_identity_rank"] = 2.
    evaluated = apply_continuous_threshold(test, model)
    assert evaluated.decisions.loc[test.is_mated, "accepted"].all()
    assert not evaluated.decisions.true_identification_at_rank_k.any()
    assert evaluated.decisions.true_identity_score.equals(test.true_identity_score)
    before = model.as_dict()
    test["fiqa_score"] = 1e6
    assert np.isfinite(model.predict(test)).all()
    assert model.as_dict() == before


def test_full_ablation_is_reproducible_and_never_fits_on_test(tmp_path):
    condition, artifacts = _inputs(tmp_path)
    retrieval = {split: _features(getattr(condition, split)) for split in ("calibration", "test")}
    retrieval["manifest"] = {"status": "completed", "spec": {
        "condition_manifest_sha256": canonical_sha256(condition.manifest)}}
    options = dict(retrieval=retrieval, methods=("continuous_fiqa", "continuous_fiqa_margin",
                   "continuous_fiqa_margin_distortion", "continuous_fiqa_runnerup"),
                   target_fpirs=(.1,), resamples=100, minimum_group_non_mated=5)
    result = run_continuous_calibration(condition, artifacts[1], **options)
    assert len(result["method_summary"]) == 7
    assert result['manifest']['schema_version'] == 2
    assert set(result['ablation_summary'].stage) == {
        'continuous_vs_2bin', 'continuous_vs_5bin', 'add_margin',
        'add_gallery_distortion', 'runnerup_control'}
    assert result['paired_comparisons'].resamples.eq(100).all()
    assert result["models"].model_json.map(json.loads).notna().all()
    changed = copy.deepcopy(condition)
    changed.test.loc[~changed.test.is_mated, "score"] += .05
    changed_retrieval = copy.deepcopy(retrieval)
    nm = ~changed.test.is_mated
    for col in ("adc_s1", "adc_s2"):
        changed_retrieval["test"].loc[nm, col] += .05
    after = run_continuous_calibration(changed, artifacts[1], **{**options, "retrieval": changed_retrieval})
    assert result["models"].model_json.equals(after["models"].model_json)
    assert result["manifest"]["result_uid"] != after["manifest"]["result_uid"]
    path = write_continuous_calibration(tmp_path, result)
    assert (path/'ablation_summary.csv').is_file()
    assert write_continuous_calibration(tmp_path, result) == path
    (path/"models.csv").write_text("damaged", encoding="utf8")
    with pytest.raises(ValueError, match="hash mismatch"):
        write_continuous_calibration(tmp_path, result)


def test_feature_adapter_distortion_and_strict_join():
    rows = _calibration_rows("test", 200)
    frame = adc_feature_frame(np.array([[.1, .3], [.2, .25]]), np.array([[1, 0], [0, 1]]),
                              ["a", "b"], ["g0", "g1"], [.4, .8])
    np.testing.assert_allclose(frame.adc_margin, [.2, .05])
    np.testing.assert_allclose(frame.top1_gallery_pq_distortion, [.8, .4])
    features = _features(rows)
    out = join_retrieval_features(rows, features.iloc[::-1])
    assert rows.sample_id.equals(out.sample_id)
    with pytest.raises(ValueError, match="coverage"):
        join_retrieval_features(rows, features.iloc[:-1])
    features.loc[0, "adc_s1"] += .1
    with pytest.raises(ValueError, match="disagree"):
        join_retrieval_features(rows, features)


def test_optimizer_failure_does_not_publish_model():
    with pytest.raises(RuntimeError, match="converge"):
        fit_continuous_threshold(_calibration_rows("calibration", 1000), target_fpir=.1,
                                 max_iterations=1)


def test_zero_threshold_adapter_matches_direct_decisions():
    rows = _calibration_rows("calibration", 1000)
    model = fit_continuous_threshold(rows, target_fpir=.1)
    # A fitted model with zero slopes is an explicit scalar control.
    model = replace(model, coefficients=tuple(np.zeros(len(model.coefficients))), safety_offset=0)
    evaluated = apply_continuous_threshold(rows, model)
    assert np.array_equal(evaluated.decisions.accepted, rows.score >= model.score_center)


def test_feature_replay_resumes_verified_shards_and_rejects_tampering(tmp_path, monkeypatch):
    import research.experiments.fiqa_retrieval_features as module
    from research.runtime.hashing import sha256_file
    condition, _ = _inputs(tmp_path)
    workflow = tmp_path / 'workflow'
    workflow.mkdir()
    (workflow / 'prepared_population').mkdir()
    source_files = {
        tmp_path / 'run_manifest.json': 'source_run_manifest_sha256',
        workflow / 'freeze_manifest.json': 'source_freeze_manifest_sha256',
        workflow / 'selected_manifest.csv': 'selected_manifest_sha256',
        workflow / 'prepared_population/manifest.json': 'prepared_population_manifest_sha256',
    }
    for path, key in source_files.items():
        path.write_text('sample_id\na\n' if path.suffix == '.csv' else '{}', encoding='utf8')
        condition.manifest[key] = sha256_file(path)
    condition.manifest.update(source_run_id='source', calibration_seed=8972, frozen_codec={'sha256': 'codec'})
    run = {'run_id': 'source', 'config': {'step4': {'evaluation': {'survface_calibration_gallery_identities': 2}}}}
    monkeypatch.setattr(module, '_completed_run', lambda p: (tmp_path, run, workflow))
    monkeypatch.setattr(module, 'read_prepared_population_artifact', lambda p: None)
    monkeypatch.setattr(module, 'prepared_population_frame', lambda *a: pd.DataFrame({'protocol_role': ['gallery']}))
    monkeypatch.setattr(module, 'build_survface_matched_calibration_protocol', lambda *a, **k: 'calibration')
    monkeypatch.setattr(module, 'build_survface_official_protocol', lambda *a: 'test')
    def arrays(split, population):
        rows = getattr(condition, split)
        return {'query_ids': rows.sample_id.to_numpy(), 'query_identity_ids': rows.identity_id.to_numpy(),
                'queries': rows.score.to_numpy()[:, None], 'gallery': np.ones((2, 1)),
                'gallery_ids': np.array(['g0', 'g1'])}
    monkeypatch.setattr(module, 'open_set_protocol_arrays', arrays)
    calls = []
    class Codec:
        def encode(self, x):
            return np.zeros((2, 1), dtype=np.uint8)

        def decode(self, codes):
            return np.full((2, 1), .8)

        def search_adc_with_metrics(self, queries, codes, top_k):
            calls.append(len(queries))
            return np.column_stack([-queries[:, 0], -queries[:, 0]+.1]), np.tile([0, 1], (len(queries), 1)), {}
    monkeypatch.setattr(module, '_frozen_pq_codec', lambda *a, **k: (Codec(), {'artifact_sha256': 'codec'}, {'fit_seed': 8972}))
    def interrupt(event):
        raise InterruptedError('simulated interruption')
    with pytest.raises(InterruptedError):
        module.build_retrieval_features(tmp_path, condition, tmp_path/'out', batch_size=128, progress=interrupt)
    assert len(calls) == 1
    result = module.build_retrieval_features(tmp_path, condition, tmp_path/'out', batch_size=128)
    assert len(calls) == 6  # 4 calibration + 2 test shards; first reused
    assert len(result['calibration']) == 400 and len(result['test']) == 200
    again = module.build_retrieval_features(tmp_path, condition, tmp_path/'out', batch_size=128)
    assert len(calls) == 6
    assert result['directory'] == again['directory']
    shard = result['directory'] / result['manifest']['shards']['test'][0]['file']
    shard.write_bytes(b'broken')
    with pytest.raises(ValueError, match='hash'):
        module.load_retrieval_features(result['directory'], condition)


def test_continuous_split_panel_keeps_full_method_coverage(tmp_path):
    condition, artifacts = _inputs(tmp_path)
    result = run_continuous_calibration(condition, artifacts[1], partition_seeds=(0, 8972),
                                        target_fpirs=(.1,), resamples=100, minimum_group_non_mated=5)
    assert len(result['method_summary']) == 8
    assert result['split_summary'].split_count.eq(2).all()
    assert result['manifest']['uncertainty']['test_based_selection'] is False


def test_continuous_notebook_settings_and_execution_contract():
    import ast
    from pathlib import Path
    import nbformat
    path = Path(__file__).parents[2] / 'notebooks/calibration/01_fiqa_continuous_retrieval_conditioned_calibration.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == 'code']
    assert code[0].id == 'user-configuration'
    ids = [c.id for c in notebook.cells]
    assert len(ids) == len(set(ids))
    for cell in code:
        # Saved user execution history is valid; editing must not erase it.
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        assert cell.id == 'user-configuration', target.id
    assert 'BUILD_RETRIEVAL_FEATURES = False' in code[0].source
    assert 'RUN_CONTINUOUS_CALIBRATION = False' in code[0].source
    assert 'WRITE_CONTINUOUS_RESULTS = False' in code[0].source
    assert "'edgeface'" in code[0].source
    assert 'incremental-ablation' in ids


def test_ci_settings_apply_to_both_metrics_and_do_not_change_fits(tmp_path):
    condition, artifacts = _inputs(tmp_path)
    options = dict(target_fpirs=(.1,), minimum_group_non_mated=5)
    first = run_continuous_calibration(condition, artifacts[1], resamples=101, bootstrap_seed=11, **options)
    second = run_continuous_calibration(condition, artifacts[1], resamples=171, bootstrap_seed=13, **options)
    assert first['paired_comparisons'].resamples.eq(101).all()
    assert first['paired_comparisons'].bootstrap_seed.eq(11).all()
    for _, evidence in first['paired_comparisons'].query("metric == 'fpir'").iterrows():
        expected = paired_binary_rate_difference_bootstrap_interval(
            int(evidence.reference_successes), int(evidence.candidate_successes),
            int(evidence.both_successes), int(evidence.total), resamples=101, random_seed=11)
        assert (evidence.paired_bootstrap95_low, evidence.paired_bootstrap95_high) == expected
    assert second['paired_comparisons'].resamples.eq(171).all()
    assert second['paired_comparisons'].bootstrap_seed.eq(13).all()
    assert first['models'].model_json.equals(second['models'].model_json)
    assert first['manifest']['result_uid'] != second['manifest']['result_uid']


def test_ablations_keep_target_miss_visible_and_reject_incomplete_evidence(tmp_path):
    condition, artifacts = _inputs(tmp_path)
    result = run_continuous_calibration(condition, artifacts[1], target_fpirs=(.1,),
                                        resamples=101, minimum_group_non_mated=5)
    summary, paired = result['method_summary'].copy(), result['paired_comparisons']
    summary.loc[summary.method.eq('continuous_fiqa'), 'target_met_on_test'] = False
    table = summarize_continuous_ablations(summary, paired)
    assert table.operating_point_status.eq('candidate_misses_target').all()
    assert table.same_realized_fpir_enforced.eq(False).all()
    with pytest.raises(ValueError, match='incomplete'):
        summarize_continuous_ablations(summary, paired.loc[paired.metric.ne('fpir')])
    with pytest.raises(ValueError, match='duplicate'):
        summarize_continuous_ablations(pd.concat([summary, summary.iloc[:1]]), paired)


@pytest.mark.parametrize('settings', [
    {'resamples': 0}, {'resamples': True}, {'bootstrap_seed': -1},
    {'bootstrap_seed': 1.5},
    {'methods': ('continuous_fiqa', 'continuous_fiqa_margin_distortion')},
])
def test_invalid_continuous_execution_plan(tmp_path, settings):
    condition, artifacts = _inputs(tmp_path)
    with pytest.raises(ValueError):
        run_continuous_calibration(condition, artifacts[1], **settings)
