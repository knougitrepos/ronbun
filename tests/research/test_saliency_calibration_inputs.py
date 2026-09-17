import copy
import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.experiments import saliency_calibration_inputs as producer
from research.explainability.gradcam import artifacts as gradcam_artifacts
from research.runtime.hashing import canonical_sha256, sha256_file
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


@pytest.mark.parametrize('permanent', [False, True])
def test_atomic_json_retries_windows_permission_error_without_deleting_old_file(tmp_path, monkeypatch, permanent):
    path = tmp_path/'progress.json'
    path.write_text('{"old": true}', encoding='utf8')
    real_replace = gradcam_artifacts.os.replace
    attempts = []

    def locked_replace(source, destination):
        attempts.append(1)
        if permanent or len(attempts) <= 3:
            raise PermissionError(13, 'simulated Windows file lock')
        return real_replace(source, destination)

    monkeypatch.setattr(gradcam_artifacts.os, 'replace', locked_replace)
    monkeypatch.setattr(gradcam_artifacts.time, 'sleep', lambda _: None)
    if permanent:
        with pytest.raises(PermissionError):
            producer._atomic_json(path, {'new': True})
        assert json.loads(path.read_text()) == {'old': True}
        assert json.loads(path.with_suffix('.tmp').read_text()) == {'new': True}
        assert len(attempts) == 8
    else:
        producer._atomic_json(path, {'new': True})
        assert json.loads(path.read_text()) == {'new': True}
        assert len(attempts) == 4


def test_verified_legacy_recovery_includes_pending_receipt_without_recomputing(tmp_path, monkeypatch):
    condition, calls, _ = _mock_producer(tmp_path, monkeypatch)
    options = dict(reuse_test_saliency=False, chunk_size=20, bootstrap_repeats=100,
                   faithfulness_maximum_samples=None)
    original = producer.build_saliency_calibration_inputs('source', condition, tmp_path/'initial', **options)
    journal = json.loads((original['directory']/'progress.json').read_text())
    journal['spec'].pop('generation_contract_sha256')
    journal['spec']['implementation_sha256']['saliency_calibration_inputs.py'] = producer._LEGACY_PRODUCER_SHA256
    legacy = tmp_path / ('saliency-inputs-' + canonical_sha256(journal['spec'])[:24])
    legacy.mkdir()
    for split in ('calibration', 'test'):
        for entry in journal['shards'][split]:
            shutil.copyfile(original['directory']/entry['path'], legacy/entry['path'])
    (legacy/'progress.tmp').write_text(json.dumps(journal), encoding='utf8')
    committed = copy.deepcopy(journal)
    committed['shards']['test'].pop()
    (legacy/'progress.json').write_text(json.dumps(committed), encoding='utf8')
    before = {p.name:sha256_file(p) for p in legacy.iterdir()}
    recovered = producer.build_saliency_calibration_inputs('source', condition, tmp_path/'recovered',
        resume_from=legacy, **options)
    assert len(calls) == 5  # Every calibration/test shard was reused, including pending test shard.
    assert recovered['manifest']['resume_from']['journal'] == 'progress.tmp'
    assert before == {p.name:sha256_file(p) for p in legacy.iterdir()}
    assert sha256_file(original['saliency_directory']/'saliency_features.csv') == sha256_file(
        recovered['saliency_directory']/'saliency_features.csv')
    with pytest.raises(ValueError, match='settings/data/runtime'):
        producer.build_saliency_calibration_inputs('source', condition, tmp_path/'different',
            resume_from=legacy, **{**options, 'seed':7})
    monkeypatch.setattr(producer, '_generation_contract_hash', lambda: 'scientific-logic-changed')
    with pytest.raises(ValueError, match='not compatible'):
        producer.build_saliency_calibration_inputs('source', condition, tmp_path/'changed',
            resume_from=legacy, **options)
