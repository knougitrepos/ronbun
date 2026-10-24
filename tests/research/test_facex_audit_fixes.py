"""Regression checks for the independent repository/Argon audit findings."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import cv2
import numpy as np
import pandas as pd
import pytest

from research.experiments import facex_inputs as inputs, facex_workflow as workflow
from research.runtime import source_fingerprints as fingerprints
from test_facex_review_regressions import stub_extraction, metadata_policy


def selection_fixture(tmp_path, monkeypatch):
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text("{}")
    config = dict(handoff=str(handoff_path), project_root=str(tmp_path), models={"surv": {}},
                  datasets=["survface", "tinyface", "scface", "lfw", "rfw"], quality=dict(source="raw_norm"))
    handoff = {d: (dict(num_classes=1, images=2, role_counts={}), pd.DataFrame())
               for d in ("survface", "scface", "tinyface")}
    models = {"surv": (SimpleNamespace(model_uid="surv-model"), "survface"),
              "tiny": (SimpleNamespace(model_uid="tiny-model"), "tinyface")}
    monkeypatch.setattr(workflow, "_config", lambda *_: config)
    monkeypatch.setattr(workflow, "load_handoff", lambda *_: handoff)
    monkeypatch.setattr(workflow, "_models", lambda *_: models)
    monkeypatch.setattr(workflow, "_population", lambda *_: (pd.DataFrame(dict(image_id=["x"])), None))
    return config


def test_readiness_and_execution_reject_missing_selected_dataset_model(tmp_path, monkeypatch):
    selection_fixture(tmp_path, monkeypatch)
    table = workflow.inspect_workflow("fixture", datasets=["scface", "survface"]).set_index("dataset")
    assert table.loc["scface", "selected_models"] == 0
    assert table.loc["scface", "status"] == "model_specs_required"
    assert table.loc["survface", "model_aliases"] == ["surv"]
    for action in (workflow.prepare_inputs, workflow.run_workflow):
        with pytest.raises(ValueError, match="No applicable.*scface"):
            action("fixture", datasets=["survface", "scface"])


def test_unused_models_filtered_but_pair_only_keeps_external_codec_inputs(tmp_path, monkeypatch):
    config = selection_fixture(tmp_path, monkeypatch)
    _, models, _, jobs, _ = workflow._workload(config, ["tinyface"])
    assert set(models) == {"tiny"}
    assert [(a, d) for a, _, _, d in jobs] == [("tiny", "tinyface")]
    _, models, _, jobs, _ = workflow._workload(config, ["lfw"])
    assert set(models) == {"surv", "tiny"}
    assert {(a, d) for a, _, _, d in jobs} == {("surv", "survface"), ("tiny", "tinyface"), ("surv", "lfw"), ("tiny", "lfw")}


def test_input_index_tracks_planning_and_population_loader_code(tmp_path, monkeypatch):
    config = selection_fixture(tmp_path, monkeypatch)
    before = workflow._workload(config, ["tinyface"])[-1]
    assert "research/protocols/facex_handoff.py" in before["input_code"]
    assert "research/datasets/lfw_pairs.py" in before["input_code"]
    old = fingerprints.sha256_file
    monkeypatch.setattr(fingerprints, "sha256_file", lambda p: "changed" if Path(p).name == "facex_workflow.py" else old(p))
    assert workflow._workload(config, ["tinyface"])[-1] != before


@pytest.mark.parametrize("module", ["rejection.py", "origin_vs_pq_calibration.py", "sample_efficiency.py", "metrics.py"])
def test_downstream_science_change_changes_report_contract(monkeypatch, module):
    before = workflow._science_hash()
    old = fingerprints.sha256_file
    monkeypatch.setattr(fingerprints, "sha256_file", lambda p: "changed" if Path(p).name == module else old(p))
    assert workflow._science_hash() != before
    assert all(not key.startswith(("C:", "F:")) for key in before)


@pytest.mark.parametrize("module", ["adapter.py", "facex_resnet.py", "facex_backbone.py", "cr_fiqa.py"])
def test_downstream_input_change_changes_input_namespace(tmp_path, monkeypatch, module):
    args = (tmp_path, "model", "scface", "population", {}, {})
    before = inputs.input_path(*args)
    old = fingerprints.sha256_file
    monkeypatch.setattr(fingerprints, "sha256_file", lambda p: "changed" if Path(p).name == module else old(p))
    assert inputs.input_path(*args) != before


def test_completed_inputs_reject_changed_source_bytes_and_preserve_old_artifact(tmp_path, monkeypatch):
    pop, spec = stub_extraction(tmp_path, monkeypatch)
    args = (tmp_path, pop, spec, "scface", tmp_path / "outputs", dict(source="raw_norm"))
    path = inputs.materialize_inputs(*args)
    manifest = (path / "manifest.json").read_bytes()
    assert inputs.materialize_inputs(*args) == path
    cv2.imwrite(pop.iloc[0].image_path, np.full((12, 15, 3), 231, dtype=np.uint8))
    with pytest.raises(ValueError, match="Raw input changed"):
        inputs.materialize_inputs(*args)
    assert (path / "manifest.json").read_bytes() == manifest
    # An explicitly pinned artifact remains readable independently of source files.
    inputs.read_inputs(path)


def test_gzip_tar_reader_is_one_forward_pass_without_member_name_scans(tmp_path, monkeypatch):
    path = tmp_path / "images.tar.gz"
    with tarfile.open(path, "w:gz") as handle:
        for i in range(30):
            payload = cv2.imencode(".png", np.full((8, 8, 3), i, dtype=np.uint8))[1].tobytes()
            info = tarfile.TarInfo(f"face{i}.png")
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    original_open, opened = tarfile.open, []
    def opened_once(*args, **kwargs):
        opened.append(args)
        return original_open(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("No getmember/getmembers scans allowed")
    monkeypatch.setattr(tarfile, "open", opened_once)
    monkeypatch.setattr(tarfile.TarFile, "getmember", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "getmembers", forbidden)
    reader = inputs.ImageReader(tmp_path)
    try:
        for i in range(30):
            assert np.all(reader.read(f"tar://images.tar.gz#face{i}.png")[0] == i)
        assert len(opened) == 1
        assert np.all(reader.read("tar://images.tar.gz#face0.png")[0] == 0)
        assert len(opened) == 2  # explicit backward access remains correct
        with pytest.raises(FileNotFoundError):
            reader.read("tar://images.tar.gz#absent.png")
    finally:
        reader.close()


def quality_fixture(tmp_path, monkeypatch):
    from research.fiqa import cr_fiqa
    pop, spec = stub_extraction(tmp_path, monkeypatch)
    checkpoint = tmp_path / "quality.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    calls = []
    monkeypatch.setattr(cr_fiqa, "load_cr_fiqa", lambda *a, **k: (object(), {}))
    def infer(model, images, **kwargs):
        calls.extend(images[:, 0, 0, 0].tolist())
        return images[:, 0, 0, 0].astype(np.float32) + 0.5
    monkeypatch.setattr(cr_fiqa, "infer_cr_fiqa_scores", infer)
    return pop, spec, dict(source="cr_fiqa", checkpoint=str(checkpoint), variant="L"), calls


def test_quality_is_shared_across_fr_checkpoints_and_verified_against_bytes(tmp_path, monkeypatch):
    pop, spec, quality, calls = quality_fixture(tmp_path, monkeypatch)
    root = tmp_path / "outputs"
    first = inputs.materialize_inputs(tmp_path, pop, spec, "survface", root, quality, batch_size=2)
    spec.model_uid = "different-fr-checkpoint"
    second = inputs.materialize_inputs(tmp_path, pop, spec, "survface", root, quality, batch_size=3)
    assert first != second and len(calls) == len(pop)
    np.testing.assert_array_equal(np.load(first / "quality.npy"), np.load(second / "quality.npy"))
    cache = next((root / "quality").rglob("manifest.json"))
    assert "model_uid" not in json.loads(cache.read_text())["contract"]
    cv2.imwrite(pop.iloc[4].image_path, np.full((12, 15, 3), 44, dtype=np.uint8))
    spec.model_uid = "third-fr-checkpoint"
    with pytest.raises(ValueError, match="Raw input changed relative to shared quality"):
        inputs.materialize_inputs(tmp_path, pop, spec, "survface", root, quality)
    assert len(calls) == len(pop)


@pytest.mark.parametrize("change", ["checkpoint", "transform", "tamper"])
def test_quality_cache_separates_policies_and_refuses_tampering(tmp_path, monkeypatch, change):
    pop, spec, quality, calls = quality_fixture(tmp_path, monkeypatch)
    root = tmp_path / "outputs"
    inputs.materialize_inputs(tmp_path, pop, spec, "lfw", root, quality)
    spec.model_uid = "second"
    if change == "checkpoint":
        Path(quality["checkpoint"]).write_bytes(b"different quality checkpoint")
    elif change == "transform":
        monkeypatch.setattr(inputs, "resolve_input_transform", lambda _: inputs.input_transform_from_metadata(metadata_policy("bicubic")))
    else:
        next((root / "quality").rglob("quality.npy")).write_bytes(b"tampered")
        with pytest.raises(ValueError, match="Quality cache changed"):
            inputs.materialize_inputs(tmp_path, pop, spec, "lfw", root, quality)
        return
    inputs.materialize_inputs(tmp_path, pop, spec, "lfw", root, quality)
    assert len(calls) == 2 * len(pop)


@pytest.mark.parametrize("dataset", ["tinyface", "scface"])
def test_rank_inputs_skip_quality_model_and_quality_artifact(tmp_path, monkeypatch, dataset):
    pop, spec = stub_extraction(tmp_path, monkeypatch)
    path = inputs.materialize_inputs(tmp_path, pop, spec, dataset, tmp_path / "outputs",
                                    dict(source="cr_fiqa", checkpoint="absent.pt"), quality_required=False)
    manifest, _, _, _, quality = inputs.read_inputs(path)
    assert quality is None and manifest["quality"]["source"] == "not_required"
    assert not (path / "quality.npy").exists()


def test_input_preparation_requests_quality_only_for_threshold_evaluations(tmp_path, monkeypatch):
    config = selection_fixture(tmp_path, monkeypatch)
    config["output_root"] = str(tmp_path / "output")
    config["quality"] = dict(source="cr_fiqa", checkpoint="absent.pt")
    calls = []
    def materialize(project, pop, spec, dataset, root, quality, **kw):
        calls.append((dataset, kw["quality_required"]))
        path = root / spec.model_uid / dataset
        path.mkdir(parents=True, exist_ok=True)
        (path / "manifest.json").write_text("{}")
        return path
    monkeypatch.setattr(workflow, "materialize_inputs", materialize)
    workflow.prepare_inputs("fixture", datasets=["tinyface"])
    assert calls == [("tinyface", False)]  # missing quality checkpoint is irrelevant
    calls.clear()
    (tmp_path / "absent.pt").write_bytes(b"quality fixture")
    workflow.prepare_inputs("fixture", datasets=["lfw"])
    assert sorted(calls) == [("lfw", True), ("lfw", True), ("survface", False), ("tinyface", False)]


def test_interrupted_quality_is_not_published_and_resume_can_use_completed_shared_quality(tmp_path, monkeypatch):
    from research.embeddings import registry
    pop, spec, quality, calls = quality_fixture(tmp_path, monkeypatch)
    adapter = registry.create_pytorch_adapter_from_spec(spec)
    real_embed = adapter.embed
    def fail_second(images):
        if adapter.calls == 1:
            adapter.calls += 1
            raise RuntimeError("interrupted quality fixture")
        return real_embed(images)
    monkeypatch.setattr(adapter, "embed", fail_second)
    root = tmp_path / "outputs"
    with pytest.raises(RuntimeError, match="interrupted quality fixture"):
        inputs.materialize_inputs(tmp_path, pop, spec, "rfw", root, quality, batch_size=2)
    assert not list((root / "quality").rglob("COMPLETED"))
    other = SimpleNamespace(**vars(spec))
    other.model_uid = "other-checkpoint"
    second = inputs.materialize_inputs(tmp_path, pop, other, "rfw", root, quality, batch_size=3)
    call_count = len(calls)
    first = inputs.materialize_inputs(tmp_path, pop, spec, "rfw", root, quality, batch_size=2)
    assert len(calls) == call_count
    np.testing.assert_array_equal(np.load(first / "quality.npy"), np.load(second / "quality.npy"))
    assert len((first / "image_hashes.jsonl").read_text().splitlines()) == len(pop)
    assert json.loads((first / "manifest.json").read_text())["quality_cache_reused"] is True
