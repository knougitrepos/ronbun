import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.experiments import saliency_calibration_inputs as producer
from research.experiments.saliency_incremental_calibration import (
    load_saliency_incremental_inputs, assess_incremental_gate,
)
from test_fiqa_continuous import _inputs


def test_selection_is_prespecified_and_excludes_test_content(tmp_path):
    condition, _ = _inputs(tmp_path)
    condition.calibration.loc[0, 'aligned_content_sha256'] = condition.test.aligned_content_sha256.iloc[0]
    condition.calibration.loc[1, 'identity_id'] = condition.test.identity_id.iloc[0]
    selected = producer.select_calibration_faithfulness(condition, maximum_samples=30)
    assert len(selected) == 30
    assert not selected.sample_id.isin(condition.calibration.sample_id.iloc[:2]).any()
    changed = copy.deepcopy(condition)
    changed.test['score'] = 1e6
    changed = replace(changed, calibration=changed.calibration.sample(frac=1, random_state=7))
    assert selected.sample_id.tolist() == producer.select_calibration_faithfulness(
        changed, maximum_samples=30).sample_id.tolist()


def _mock_producer(tmp_path, monkeypatch):
    condition, _ = _inputs(tmp_path)
    condition = replace(condition, calibration=condition.calibration.iloc[:60].copy(),
                        test=condition.test.iloc[:40].copy())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *_: 'test GPU')
    context = {'root': tmp_path/'immutable-run', 'spec': None}
    monkeypatch.setattr(producer, '_source_context', lambda *_: context)
    monkeypatch.setattr(producer, 'create_pytorch_adapter_from_spec',
                        lambda *a, **k: SimpleNamespace(device=SimpleNamespace(type='cuda')))
    calls = []

    def generate(context, adapter, rows, split, start, settings, faith_ids):
        calls.append((split, start))
        out = rows[['sample_id', 'identity_id', 'aligned_content_sha256']].reset_index(drop=True).copy()
        out['split'] = split
        out['saliency_target_name'] = producer.TARGET
        out['heatmap_available'] = out['gradcam_valid_heatmap'] = True
        out['outside_face_attention'] = np.linspace(.1, .8, len(out))
        out['saliency_entropy'] = np.linspace(1, 2, len(out))
        for name, value in zip(producer.FAITHFULNESS_METRICS, [.3, .1, .15, .2, .15]):
            out[name] = np.where(out.sample_id.isin(faith_ids), value, np.nan)
        return out

    monkeypatch.setattr(producer, '_generate_chunk', generate)
    return condition, calls, generate


def test_resume_completed_reuse_and_corruption_detection(tmp_path, monkeypatch):
    condition, calls, generate = _mock_producer(tmp_path, monkeypatch)
    options = dict(reuse_test_saliency=False, chunk_size=20, bootstrap_repeats=100,
                   faithfulness_maximum_samples=None)

    def interrupted(*args):
        if args[3] == 'calibration' and args[4] == 20:
            raise RuntimeError('simulated interruption')
        return generate(*args)

    monkeypatch.setattr(producer, '_generate_chunk', interrupted)
    with pytest.raises(RuntimeError, match='simulated'):
        producer.build_saliency_calibration_inputs('source', condition, tmp_path/'out', **options)
    assert calls == [('calibration', 0)]
    monkeypatch.setattr(producer, '_generate_chunk', generate)
    result = producer.build_saliency_calibration_inputs('source', condition, tmp_path/'out', **options)
    assert calls.count(('calibration', 0)) == 1
    assert len(calls) == 5
    inputs = load_saliency_incremental_inputs(condition, result['saliency_directory'], result['faithfulness_directory'])
    assert assess_incremental_gate(condition, inputs)['comparison_enabled']
    again = producer.build_saliency_calibration_inputs('source', condition, tmp_path/'out', **options)
    assert again['directory'] == result['directory'] and len(calls) == 5
    (result['saliency_directory']/'saliency_features.csv').write_text('corrupted', encoding='utf8')
    with pytest.raises(ValueError, match='hash mismatch'):
        producer.build_saliency_calibration_inputs('source', condition, tmp_path/'out', **options)


def test_limited_build_never_authorizes_full_comparison(tmp_path, monkeypatch):
    condition, _, _ = _mock_producer(tmp_path, monkeypatch)
    result = producer.build_saliency_calibration_inputs('source', condition, tmp_path/'out',
        reuse_test_saliency=False, chunk_size=20, bootstrap_repeats=100, max_queries_per_split=30)
    inputs = load_saliency_incremental_inputs(condition, result['saliency_directory'], result['faithfulness_directory'])
    gate = assess_incremental_gate(condition, inputs)
    assert not gate['comparison_enabled']
    assert any('smoke' in reason for reason in gate['reasons'])


def test_cuda_required_before_reading_source(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CPU fallback is disabled'):
        producer.build_saliency_calibration_inputs('missing', None, tmp_path)
