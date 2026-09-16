import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.evaluation.saliency_faithfulness import FAITHFULNESS_METRICS
from research.experiments.saliency_incremental_calibration import (
    load_saliency_incremental_inputs, assess_incremental_gate,
    run_saliency_incremental_calibration, write_saliency_incremental_result,
)
from research.runtime.hashing import canonical_sha256, sha256_file
from research.calibration.continuous import fit_continuous_threshold
from research.experiments.fiqa_threshold_calibration import join_fiqa_score_artifacts
from test_fiqa_continuous import _inputs, _features


def _receipt(path):
    return dict(path=path.name, bytes=path.stat().st_size, sha256=sha256_file(path))


def _sources(tmp_path):
    condition, fiqa = _inputs(tmp_path)
    cm = condition.manifest
    sal_root, faith_root = tmp_path/'saliency', tmp_path/'faithfulness'
    sal_root.mkdir()
    faith_root.mkdir()
    cal, test = condition.calibration, condition.test
    sal = pd.concat([rows[['sample_id', 'aligned_content_sha256']].assign(split=split)
                     for split, rows in [('calibration', cal), ('test', test)]], ignore_index=True)
    sal['saliency_target_name'] = 'origin_top1_gallery_cosine'
    sal['heatmap_available'] = sal['gradcam_valid_heatmap'] = True
    sal['outside_face_attention'] = np.linspace(.05, .9, len(sal))
    sal['saliency_entropy'] = 1 + np.sin(np.arange(len(sal)))**2
    sal.to_csv(sal_root/'saliency_features.csv', index=False)
    sm = dict(artifact_type='saliency_calibration_features', schema_version=1, status='completed',
              condition_manifest_sha256=canonical_sha256(cm), gallery_contract='split_matched_origin_top1',
              saliency_spec_uid='saliency-test', target_name='origin_top1_gallery_cosine',
              **{k:cm[k] for k in ('dataset_id', 'model_uid', 'extraction_uid', 'origin_embedding_artifact_uid')},
              saliency_features=_receipt(sal_root/'saliency_features.csv'))
    (sal_root/'manifest.json').write_text(json.dumps(sm), encoding='utf8')
    values = dict(zip(FAITHFULNESS_METRICS, [.3, .1, .15, .2, .15]))
    rows = pd.DataFrame({'sample_id': [f'development-{i}' for i in range(60)], **values})
    rows.to_csv(faith_root/'faithfulness_rows.csv', index=False)
    summary = pd.DataFrame([dict(group='all', metric=k, sample_count=len(rows), mean=v,
                                 mean_ci_lower=v-.01, mean_ci_upper=v+.01) for k,v in values.items()])
    summary.to_csv(faith_root/'faithfulness_summary.csv', index=False)
    fm = dict(artifact_type='open_set_gradcam_faithfulness', schema_version=2,
              evaluation_split='development', saliency_spec_uid='saliency-test',
              saliency_target_name='origin_top1_gallery_cosine',
              **{k:cm[k] for k in ('dataset_id', 'model_uid', 'source_run_id', 'origin_embedding_artifact_uid')},
              statistics={'bootstrap_method':'identity_cluster', 'confidence_level':.95},
              outputs=[_receipt(faith_root/n) for n in ('faithfulness_rows.csv','faithfulness_summary.csv')])
    (faith_root/'manifest.json').write_text(json.dumps(fm), encoding='utf8')
    return condition, fiqa[1], sal_root, faith_root


def test_ready_full_incremental_and_frozen_test_fit(tmp_path):
    c, fiqa, sr, fr = _sources(tmp_path)
    inputs = load_saliency_incremental_inputs(c, sr, fr)
    assert assess_incremental_gate(c, inputs)['comparison_enabled']
    options = dict(target_fpirs=(.1,), partition_seeds=(0,8972), resamples=100)
    result = run_saliency_incremental_calibration(c, fiqa, inputs, **options)
    assert len(result['method_summary']) == 8
    assert len(result['paired_comparisons']) == 12
    assert result['split_summary'].split_count.eq(2).all()
    assert result['paired_comparisons'].resamples.eq(100).all()
    for model in result['models'].model_json.map(json.loads):
        assert not any('occlusion' in col for col in model['features'])
    cal, _ = join_fiqa_score_artifacts(c,fiqa)
    baseline = fit_continuous_threshold(cal,target_fpir=.1,partition_seed=0)
    recorded = json.loads(result['models'].iloc[0].model_json)
    assert recorded['coefficients'] == list(baseline.coefficients)
    assert recorded['safety_offset'] == baseline.safety_offset
    changed = copy.deepcopy(c)
    changed.test.loc[~changed.test.is_mated, 'score'] += .3
    after = run_saliency_incremental_calibration(changed, fiqa, inputs, **options)
    assert result['models'].model_json.equals(after['models'].model_json)
    assert result['manifest']['result_uid'] != after['manifest']['result_uid']
    path = write_saliency_incremental_result(tmp_path/'out', result)
    assert write_saliency_incremental_result(tmp_path/'out', result) == path
    (path/'models.csv').write_text('damaged',encoding='utf8')
    with pytest.raises(ValueError,match='hash'):
        write_saliency_incremental_result(tmp_path/'out', result)


def test_retrieval_baseline_requires_matching_artifact(tmp_path):
    c, fiqa, sr, fr = _sources(tmp_path)
    inputs = load_saliency_incremental_inputs(c, sr, fr)
    opts = dict(baseline_method='continuous_fiqa_margin_distortion', target_fpirs=(.1,), resamples=100)
    with pytest.raises(ValueError,match='retrieval'):
        run_saliency_incremental_calibration(c, fiqa, inputs, **opts)
    retrieval = {s:_features(getattr(c,s)) for s in ('calibration','test')}
    retrieval['manifest'] = dict(status='completed', spec={'condition_manifest_sha256':canonical_sha256(c.manifest)})
    result = run_saliency_incremental_calibration(c, fiqa, inputs, retrieval=retrieval, **opts)
    base = json.loads(result['models'].iloc[0].model_json)
    assert base['features'] == ['fiqa_score','adc_margin','top1_gallery_pq_distortion']


@pytest.mark.parametrize('case', ['coverage','alignment','split','test_faithfulness','weak_random','missing_gallery_contract'])
def test_gate_blocks_without_fitting(tmp_path, monkeypatch, case):
    c, fiqa, sr, fr = _sources(tmp_path)
    sm = json.loads((sr/'manifest.json').read_text())
    fm = json.loads((fr/'manifest.json').read_text())
    sal = pd.read_csv(sr/'saliency_features.csv')
    if case == 'coverage':
        sal.loc[sal.split.eq('calibration'),'heatmap_available'] = False
    if case == 'alignment':
        sal.loc[0,'aligned_content_sha256'] = 'wrong'
    if case == 'split':
        sal.loc[0,'split'] = 'test'
    if case == 'missing_gallery_contract':
        sm.pop('gallery_contract')
    if case == 'test_faithfulness':
        rows = pd.read_csv(fr/'faithfulness_rows.csv')
        rows.loc[0,'sample_id'] = c.test.sample_id.iloc[0]
        rows.to_csv(fr/'faithfulness_rows.csv',index=False)
    if case == 'weak_random':
        summary = pd.read_csv(fr/'faithfulness_summary.csv')
        summary.loc[summary.metric.eq('faithfulness_gain_over_random'),'mean_ci_lower'] = -.01
        summary.to_csv(fr/'faithfulness_summary.csv',index=False)
    sal.to_csv(sr/'saliency_features.csv',index=False)
    sm['saliency_features'] = _receipt(sr/'saliency_features.csv')
    fm['outputs'] = [_receipt(fr/n) for n in ('faithfulness_rows.csv','faithfulness_summary.csv')]
    (sr/'manifest.json').write_text(json.dumps(sm),encoding='utf8')
    (fr/'manifest.json').write_text(json.dumps(fm),encoding='utf8')
    inputs = load_saliency_incremental_inputs(c,sr,fr)
    gate = assess_incremental_gate(c,inputs)
    assert not gate['comparison_enabled'] and gate['reasons']
    def forbidden(*args, **kwargs):
        raise AssertionError('blocked inputs must not fit')
    monkeypatch.setattr('research.experiments.saliency_incremental_calibration.fit_continuous_threshold',forbidden)
    with pytest.raises(ValueError,match='gate blocked'):
        run_saliency_incremental_calibration(c,fiqa,inputs)


def test_hash_lineage_and_in_memory_mutation(tmp_path):
    c, fiqa, sr, fr = _sources(tmp_path)
    inputs = load_saliency_incremental_inputs(c,sr,fr)
    inputs['saliency'].loc[0,'saliency_entropy'] = 9
    with pytest.raises(ValueError,match='modified'):
        assess_incremental_gate(c,inputs)
    inputs = load_saliency_incremental_inputs(c,sr,fr)
    inputs['faithfulness_manifest']['evaluation_split'] = 'test'
    with pytest.raises(ValueError,match='modified'):
        assess_incremental_gate(c,inputs)
    changed = copy.deepcopy(c)
    changed.manifest['model_uid'] = 'other-model'
    with pytest.raises(ValueError,match='lineage'):
        load_saliency_incremental_inputs(changed,sr,fr)
    (sr/'saliency_features.csv').write_text('corrupt',encoding='utf8')
    with pytest.raises(ValueError,match='hash'):
        load_saliency_incremental_inputs(c,sr,fr)


def test_notebook_is_restartable_and_defaults_read_only():
    import ast
    import nbformat
    n=nbformat.read(Path(__file__).parents[2]/'notebooks/calibration/02_saliency_incremental_threshold_calibration.ipynb',as_version=4)
    nbformat.validate(n)
    codes=[c for c in n.cells if c.cell_type=='code']
    assert codes[0].id=='user-configuration'
    assert 'RUN_INCREMENTAL_CALIBRATION = False' in codes[0].source
    for c in codes:
        assert c.outputs==[] and c.execution_count is None
        ast.parse(c.source)
