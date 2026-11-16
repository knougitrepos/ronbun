import copy
import json
from pathlib import Path

import pytest
import yaml

from research.experiments import facex_inventory as f


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    from research.protocols import facex_handoff
    exported = tmp_path / 'model.pt'
    exported.write_bytes(b'export')
    planned = dict(run_id='scface_arcface_test_s1', dataset='scface', head='arcface', seed=1, recipe='two_eye_alignment')
    row = dict(planned, completed=True, export=str(exported), sha256=f.sha256_file(exported), source_sha256='a'*64,
               training_dataset='scface', evaluation_datasets=['scface'])
    meta = dict(dataset='scface', head='arcface', seed=1, source_sha256='a'*64,
                source_config={'head': {'margin_am':0}, 'dataset': {'num_classes':50}},
                protocol_record={'images':800})
    path = tmp_path / 'inventory.json'
    scope = dict(purpose='local_protocol_baseline', exact_ucface_reproduction=False, origin_comparison_required=True,
                 populations={'scface': {'local_training':[50,800]}})
    data = dict(schema='facex-model-inventory-v2', status='completed',
                selection=dict(schema='facex-selection-v2', rows=[planned], purpose='local_protocol_baseline', reproduction_scope=scope), models=[row])
    path.write_text(json.dumps(data), encoding='utf-8')
    monkeypatch.setattr(f, '_read_validated_export', lambda p: (exported, meta, {}))
    monkeypatch.setattr(f, 'register_facex_export', lambda *a: tmp_path / 'spec.json')
    monkeypatch.setattr(f, '_config', lambda p: dict(project_root=str(tmp_path), handoff='h', models={}, datasets=['scface'],
                                                   compression={'scface':[{'family':'origin'}, {'family':'pca','dimensions':256}]}))
    monkeypatch.setattr(facex_handoff, 'load_handoff', lambda p: {})
    monkeypatch.setattr(f, '_models', lambda *a: {})
    return path, data, meta


def test_inventory_config_is_explicit_and_immutable(tmp_path, inventory):
    path, data, meta = inventory
    output = tmp_path / 'experiment.yaml'
    args = (path, 'baseline.yaml', tmp_path / 'registry', output)
    assert f.write_inventory_config(*args) == output
    original = output.read_bytes()
    assert f.write_inventory_config(*args) == output
    cfg = yaml.safe_load(original)
    assert cfg['datasets'] == ['scface']
    assert cfg['models'][data['models'][0]['run_id']]['evaluation_datasets'] == ['scface']
    assert cfg['model_inventory']['sha256'] == f.sha256_file(path)
    assert cfg['model_scope']['exact_ucface_reproduction'] is False
    assert cfg['compression'] == {'scface':[{'family':'origin'}]}
    assert cfg['model_scope']['evaluation_phase'] == 'origin_reference'
    output.write_text('different config', encoding='utf-8')
    with pytest.raises(FileExistsError):
        f.write_inventory_config(*args)
    assert output.read_text() == 'different config'


@pytest.mark.parametrize('corruption', ['partial', 'duplicate', 'bytes', 'scope', 'head', 'geometry', 'margin', 'claim', 'population', 'images', 'purpose'])
def test_inventory_rejects_invalid_artifacts_before_registration(tmp_path, monkeypatch, inventory, corruption):
    path, data, meta = inventory
    data = copy.deepcopy(data)
    if corruption == 'partial':
        data['selection']['rows'].append(dict(data['selection']['rows'][0], run_id='untrained'))
    elif corruption == 'duplicate':
        data['models'] *= 2
    elif corruption == 'bytes':
        data['models'][0]['sha256'] = '0'*64
    elif corruption == 'scope':
        data['models'][0]['evaluation_datasets'] = ['lfw']
    elif corruption == 'head':
        meta['head'] = 'cosface'
    elif corruption == 'geometry':
        meta['input_geometry'] = dict(file='../escape.csv', sha256='0'*64)
    elif corruption == 'margin':
        meta['source_config']['head']['margin_am'] = .175
    elif corruption == 'claim':
        data['selection']['reproduction_scope']['exact_ucface_reproduction'] = True
    elif corruption == 'population':
        data['selection']['reproduction_scope']['populations']['scface']['local_training'][0] = 130
    elif corruption == 'images':
        data['selection']['reproduction_scope']['populations']['scface']['local_training'][1] = 2080
    elif corruption == 'purpose':
        data['selection']['purpose'] = 'diagnostic_control'
    path.write_text(json.dumps(data), encoding='utf-8')
    monkeypatch.setattr(f, 'register_facex_export', lambda *a: pytest.fail('Registration before full validation'))
    with pytest.raises(ValueError):
        f.write_inventory_config(path, 'baseline.yaml', tmp_path, tmp_path / 'config.yaml')


def test_legacy_inventory_is_only_an_explicit_local_control(tmp_path, inventory):
    path, data, meta = inventory
    data['schema'] = 'facex-model-inventory-v1'
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='Legacy inventory'):
        f.write_inventory_config(path, 'baseline', tmp_path, tmp_path / 'config.yaml')
    output = f.write_inventory_config(path, 'baseline', tmp_path, tmp_path / 'config.yaml', purpose='legacy_local_control')
    assert yaml.safe_load(output.read_text())['model_scope']['exact_ucface_reproduction'] is False


def test_compression_is_an_explicit_followup_config(tmp_path, inventory):
    path, _, _ = inventory
    output = f.write_inventory_config(path, 'baseline', tmp_path, tmp_path/'compression.yaml', origin_only=False)
    cfg = yaml.safe_load(output.read_text())
    assert cfg['model_scope']['evaluation_phase'] == 'compression_and_calibration'
    assert cfg['compression']['scface'][1]['family'] == 'pca'


def test_notebook_forwards_inventory_purpose_and_origin_first(tmp_path, monkeypatch):
    from research.experiments import facex_workflow, facex_comparison
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    calls = []
    def write(*args, **kwargs):
        calls.append(kwargs)
        return tmp_path / 'origin.yaml'
    monkeypatch.setattr(f, 'write_inventory_config', write)
    monkeypatch.setattr(facex_workflow, 'prepare_inputs', lambda *a, **kw: {'input_index':'unused'})
    monkeypatch.setattr(facex_comparison, 'verify_alignment_ranks', lambda *a: {})
    cells = [c for c in json.loads((root/'notebooks/02_inputs.ipynb').read_text(encoding='utf-8'))['cells'] if c['cell_type']=='code']
    scope = {'display':lambda *a: None}
    exec(compile(''.join(cells[0]['source']), '02:settings', 'exec'), scope)
    scope.update(EXECUTE=True, MODEL_INVENTORY=tmp_path/'inventory.json')
    for cell in cells[1:]:
        exec(compile(''.join(cell['source']), '02:run', 'exec'), scope)
    assert calls == [dict(purpose='local_protocol_baseline', origin_only=True)]
