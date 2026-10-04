"""Cold-start routing and frozen input contracts; neural inference is stubbed here.

Actual CUDA extraction is a separate integration check, not claimed by these tests.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json

import pandas as pd
import pytest
import yaml
from PIL import Image

from research.experiments import lfw_pair_inputs as module
from research.runtime.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cold(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / module.CONFIG_PATH).read_text(encoding="utf8"))
    config["inputs"]["raw_image_root"] = "raw"
    config["inputs"]["expected_images"] = 3
    config["inputs"]["expected_identities"] = 2
    config["source_runs"] = {"arcface":config["source_runs"]["arcface"]}
    config["inputs"]["models"] = {"arcface":config["inputs"]["models"]["arcface"]}
    for person, count in [("Alice", 2), ("Unused", 1)]:
        folder = tmp_path / "raw" / person
        folder.mkdir(parents=True)
        for number in range(1, count+1):
            Image.new("RGB", (250, 250), (number*30, 50, 70)).save(folder / f"{person}_{number:04d}.jpg")
    cp = tmp_path / config["resize_inputs"]["fiqa_checkpoint"]
    cp.parent.mkdir(parents=True); cp.write_bytes(b"test-weights")
    variant = SimpleNamespace(model_uid="fixture-fiqa", variant="L", expected_sha256=sha256_file(cp))
    monkeypatch.setattr(module, "CRFIQA_VARIANTS", {"L":variant})
    # These stubs replace neural/checkpoint validation only; the filesystem,
    # inventory, resizing, publication and restart contracts are real.
    monkeypatch.setattr(module, "model_specs", lambda *a: {"arcface":SimpleNamespace(model_uid="fixture-fr")})
    def write_spec(path, spec):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fixture")
    monkeypatch.setattr(module, "write_model_spec", write_spec)
    monkeypatch.setattr(module, "load_pairs", lambda p, f, **k: (pd.DataFrame([{}]*6000),f[f.identity_id.eq("lfw:Unused")]))
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "fixture-cuda")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(module, "check_resources", lambda **kw: None)
    config_path=tmp_path / "config.yaml"; config_path.write_text(yaml.safe_dump(config), encoding="utf8")
    def extract(root, config, model, spec, bundle, *args):
        from research.preprocessing.aligned_crops import validate_aligned_crop_bundle, LFW_DEEPFUNNELED_RESIZE
        validate_aligned_crop_bundle(bundle, dataset_id="lfw", expected_source_count=3,
            preprocessing_mode=LFW_DEEPFUNNELED_RESIZE, require_full_coverage=True)
        directory=root/config["source_runs"][model]
        if directory.exists():
            return
        directory.mkdir(parents=True,exist_ok=True)
        (directory/"_SUCCESS").write_text("fixture")
        (directory/"manifest.json").write_text('{"row_count":3}')
        (root/config["resize_inputs"]["checkpoint_file"]).write_bytes(b"shards")
    monkeypatch.setattr(module, "_extract_source", extract)
    monkeypatch.setattr(module, "load_resize_source", lambda *a:(None,{"row_count":3,"model_uid":"fixture-fr"},None,None,None,
        {"aligned_bundle_manifest_sha256":sha256_file(tmp_path/config["resize_inputs"]["bundle_dir"] / "bundle_manifest.json")}))
    monkeypatch.setattr(module, "load_cr_fiqa", lambda *a,**k:(object(),variant))
    def fiqa(bundle, destination, **kw):
        destination.mkdir(parents=True)
        (destination/"manifest.json").write_text(json.dumps(dict(aligned_bundle_manifest_sha256=sha256_file(bundle/"bundle_manifest.json"),checkpoint_sha256=variant.expected_sha256)))
    monkeypatch.setattr(module, "materialize_aligned_bundle_score_artifact", fiqa)
    monkeypatch.setattr(module, "load_fiqa_score_artifact", lambda p: SimpleNamespace(
        manifest=json.loads((p/"manifest.json").read_text()), scores=pd.DataFrame([{}]*3)))
    return tmp_path, config


@pytest.mark.parametrize("keep", [False, True])
def test_blank_derived_state_to_reusable_inputs(cold, keep):
    root, config=cold
    kwargs=dict(config_path="config.yaml", models=("arcface",),keep_raw_results=keep)
    state=module.prepare_pair_inputs(root, **kwargs)
    assert not state["ready"]
    assert not (root/config["image_manifest"]).exists()
    assert not (root/config["inputs"]["registry_root"]).exists()
    assert not (root/"data/interim/lfw/face_manifest.csv").exists()
    result=module.prepare_pair_inputs(root,execute=True,**kwargs)
    assert result["ready"] and result["completed"]
    original=(root/config["image_manifest"]).read_bytes()
    assert set(pd.read_csv(root/config["image_manifest"]).split)=={"population"}
    assert (root/config["resize_inputs"]["checkpoint_file"]).exists() is keep
    assert module.prepare_pair_inputs(root,**kwargs)["ready"]
    module.prepare_pair_inputs(root,execute=True,**kwargs)
    assert (root/config["image_manifest"]).read_bytes()==original
    # Source/checkpoint files are protected even when detailed shards are pruned.
    assert len(list((root/"raw").glob("*/*.jpg")))==3
    assert (root/config["resize_inputs"]["fiqa_checkpoint"]).is_file()
    assert not list(root.rglob("*.mat"))


def test_changed_raw_source_cannot_reuse_frozen_population(cold):
    root,config=cold
    module.prepare_pair_inputs(root,config_path="config.yaml",models=("arcface",),execute=True)
    image=next((root/"raw").glob("*/*.jpg")); image.write_bytes(image.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="raw population changed"):
        module.prepare_pair_inputs(root,config_path="config.yaml",models=("arcface",))


def test_cuda_failure_does_not_publish_partial_inputs(cold, monkeypatch):
    import torch
    root,config=cold
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="requires CUDA"):
        module.prepare_pair_inputs(root,config_path="config.yaml",models=("arcface",),execute=True)
    assert not (root/config["image_manifest"]).exists()


def test_retention_change_does_not_delete_previous_shards(cold):
    root,config=cold
    module.prepare_pair_inputs(root,config_path="config.yaml",keep_raw_results=True,execute=True)
    saved=(root/config["resize_inputs"]["checkpoint_file"]).read_bytes()
    result=module.prepare_pair_inputs(root,config_path="config.yaml",keep_raw_results=False,execute=True)
    assert (root/config["resize_inputs"]["checkpoint_file"]).read_bytes()==saved
    assert result["extraction_checkpoint_retained"]


def test_partial_other_model_keeps_shared_resume_shards(cold):
    root,config=cold
    config["source_runs"]["adaface"]="results/lfw_pair_inputs_v1/sources/adaface"
    config["inputs"]["models"]["adaface"] = dict(config["inputs"]["models"]["arcface"])
    (root/"config.yaml").write_text(yaml.safe_dump(config),encoding="utf8")
    result=module.prepare_pair_inputs(root,config_path="config.yaml",models=("arcface",),execute=True)
    assert result["completed"] and result["extraction_checkpoint_retained"]
    assert (root/config["resize_inputs"]["checkpoint_file"]).is_file()


def test_failed_fiqa_keeps_extraction_resume_data(cold,monkeypatch):
    root,config=cold
    monkeypatch.setattr(module,"load_cr_fiqa",Mock(side_effect=RuntimeError("interrupted")))
    with pytest.raises(RuntimeError,match="interrupted"):
        module.prepare_pair_inputs(root,config_path="config.yaml",execute=True)
    assert (root/config["resize_inputs"]["checkpoint_file"]).is_file()
    assert not list((root/config["inputs"]["registry_root"]).glob("preparation-*.json"))


def test_workflow_automatically_prepares_before_evaluation(cold, monkeypatch):
    root,_=cold
    from research.experiments import lfw_pair_verification
    evaluate=Mock(return_value=dict(completed=True))
    monkeypatch.setattr(lfw_pair_verification,"run_verification",evaluate)
    plan=module.run_pair_workflow(root,config_path="config.yaml",models=("arcface",))
    assert plan["expected_jobs"]==200 and not plan["ready"]
    evaluate.assert_not_called()
    result=module.run_pair_workflow(root,config_path="config.yaml",models=("arcface",),execute=True,fold_ids=(1,),partition_seeds=(8972,))
    assert result["completed"] and result["input_preparation"]["ready"]
    assert evaluate.call_args.kwargs["fold_ids"]==(1,)


@pytest.mark.parametrize("relative",[
    "notebooks/lfw/00_data_preparation/00_data_preparation.ipynb",
    "notebooks/common/orchestration/01_batch_fiqa_saliency_calibration_compact.ipynb",
    "notebooks/common/orchestration/cross_dataset_calibration_transfer.ipynb",
])
def test_auxiliary_notebooks_default_without_old_inputs(relative,monkeypatch):
    import nbformat
    called=Mock(return_value=dict(coverage=pd.DataFrame(),ready=False))
    monkeypatch.setattr(module,"prepare_pair_inputs",called)
    notebook=nbformat.read(ROOT/relative,as_version=4); nbformat.validate(notebook)
    ns={}
    for cell in notebook.cells:
        if cell.cell_type=="code": exec(compile(cell.source,relative,"exec"),ns)
    assert ns["LFW_PROTOCOL_MODE"].startswith("pair_verification")
    if "/lfw/" in relative:
        called.assert_called_once()
        assert called.call_args.kwargs["execute"] is False
