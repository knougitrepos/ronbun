"""Regression coverage for execution, retention and exported input policies."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from research.experiments import facex_inputs as inputs
from research.experiments import facex_workflow as workflow
from research.runtime.hashing import canonical_sha256, sha256_file


@pytest.mark.parametrize("payload", [b"", b"\n", b"\r\n", b"\r\n\r\n"])
def test_optional_csv_accepts_empty_windows_and_unix_tables(tmp_path, payload):
    path = tmp_path / "paired.csv"
    path.write_bytes(payload)
    assert workflow._read_optional_csv(path).empty
    path.unlink()
    with pytest.raises(FileNotFoundError):
        workflow._read_optional_csv(path)


def rank_workflow(tmp_path, monkeypatch, dataset):
    pop = pd.DataFrame([
        dict(image_id="dev", identity_id="train:001", role="pq_development", distance=""),
        dict(image_id="g", identity_id="test:002", role="test_gallery", distance=""),
        *[dict(image_id=f"p{d}", identity_id="test:002", role="test_registered_probe", distance=str(d)) for d in (1, 2, 3)],
        dict(image_id="distractor", identity_id="", role="test_gallery_distractor", distance=""),
    ])
    vectors = np.zeros((len(pop), 1024), dtype=np.float32)
    vectors[:-1, 0], vectors[-1, 1] = 1, 1
    source = tmp_path / "source"
    source.mkdir()
    pop.to_csv(source / "population.csv", index=False)
    for name, values in (("embeddings.npy", vectors), ("raw_norm.npy", np.ones(len(pop))), ("quality.npy", np.ones(len(pop)))):
        np.save(source / name, values)
    inputs.write_json(source / "manifest.json", dict(status="completed", model_uid="fixture", files={
        p.name: sha256_file(p) for p in source.iterdir()}))
    (source / "COMPLETED").write_text("completed")
    config = dict(project_root=str(Path(__file__).resolve().parents[2]), embedding_dim=1024,
                  models={}, datasets=[dataset], target_rates=[.1], quality=dict(source="raw_norm"),
                  compression={dataset: [dict(family="origin")]})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    models = {"model": (SimpleNamespace(model_uid="fixture"), dataset)}
    identity = {"fixture": dataset}
    monkeypatch.setattr(workflow, "_workload", lambda *a: ({}, models, {dataset: (pop, None)}, [], identity))
    root = tmp_path / "output"
    inputs.write_json(root / "input_indexes" / (canonical_sha256(identity) + ".json"),
        dict(identity=identity, inputs={f"model/{dataset}": dict(path=str(source), manifest_sha256=sha256_file(source / "manifest.json"))}))
    return path, root


@pytest.mark.parametrize("dataset", ["scface", "tinyface"])
@pytest.mark.parametrize("keep", [False, True])
def test_rank_only_publication_reuse_resume_and_retention(tmp_path, monkeypatch, dataset, keep):
    path, root = rank_workflow(tmp_path, monkeypatch, dataset)
    original_rank, original_summary = workflow.evaluate_rank, workflow.write_analysis_summary
    evaluations = []
    def rank(*args, **kwargs):
        evaluations.append(1)
        return original_rank(*args, **kwargs)
    monkeypatch.setattr(workflow, "evaluate_rank", rank)
    def interrupt(*args):
        raise RuntimeError("interrupt after rank job checkpoint")
    monkeypatch.setattr(workflow, "write_analysis_summary", interrupt)
    with pytest.raises(RuntimeError, match="interrupt after rank"):
        workflow.run_workflow(path, output_root=root, keep_raw_results=keep)
    work = next((root / "reports").glob("*.partial"))
    # Force the exact CRLF-only legacy file, keeping the checkpoint hash valid.
    pair_path = work / f"model_{dataset}_paired.csv"
    pair_path.write_bytes(b"\r\n")
    done_path = work / f"model_{dataset}_COMPLETED.json"
    done = json.loads(done_path.read_text())
    done[pair_path.name] = sha256_file(pair_path)
    inputs.write_json(done_path, done)
    monkeypatch.setattr(workflow, "write_analysis_summary", original_summary)
    result = workflow.run_workflow(path, output_root=root, keep_raw_results=keep)
    report = Path(result["report_dir"])
    assert len(evaluations) == 1  # completed rank job was resumed, not rerun
    loaded = workflow.read_results(report)
    assert loaded["paired"].empty and not loaded["metrics"].empty
    details = list((report / "details").rglob("*.csv"))
    assert len(details) == ((2 if dataset == "tinyface" else 1) if keep else 0)
    for file in details:
        raw = pd.read_csv(file)
        assert len(raw) == 3 and raw.sample_id.nunique() == 3
        assert raw.true_identity_rank.eq(1).all()
        assert file.relative_to(report).as_posix() in loaded["manifest"]["files"]
    assert workflow.run_workflow(path, output_root=root, keep_raw_results=keep)["reused"]
    assert not list(root.rglob("RUNNING.lock"))


@pytest.mark.parametrize("dataset", ["lfw", "rfw"])
@pytest.mark.parametrize("keep", [False, True])
def test_pair_detail_retention_matches_aggregate_metrics(tmp_path, dataset, keep):
    rows, pairs = [], []
    group = "all" if dataset == "lfw" else "African"
    for fold in (1, 2):
        for i in range(200):
            identity = f"f{fold}:id{i}"
            for j in range(2):
                rows.append(dict(image_id=f"{identity}:{j}", identity_id=identity))
            for genuine, right in ((True, identity), (False, f"f{fold}:id{(i+1)%200}")):
                pairs.append(dict(pair_id=f"{fold}-{i}-{genuine}", fold=fold, group=group,
                    left_image_id=f"{identity}:0", right_image_id=f"{right}:1",
                    left_identity=identity, right_identity=right, is_genuine=genuine))
    population = pd.DataFrame(rows)
    rng = np.random.default_rng(12)
    vectors = workflow.normalize(rng.normal(size=(len(population), 32)))
    details = tmp_path / "details"
    metrics, _, _ = workflow.evaluate_pairs(population, vectors, rng.normal(size=len(population)),
        pd.DataFrame(pairs), {"origin_32": ("origin", None, {})},
        dict(target_rates=[.1], fold_ids=[1], partition_seed=8972, bootstrap_resamples=100),
        details=details if keep else None)
    files = list(details.glob("*.csv"))
    assert len(files) == int(keep)
    if keep:
        raw = pd.read_csv(files[0])
        assert len(raw) == 400 * 3
        assert raw[["left_image_id", "right_image_id", "fiqa_score"]].notna().all().all()
        for row in metrics.itertuples():
            sample = raw.loc[raw.method.eq(row.method)]
            assert int((sample.accepted & sample.is_genuine).sum()) == row.true_accepts
            assert int((sample.accepted & ~sample.is_genuine).sum()) == row.false_accepts


def metadata_policy(interpolation="bicubic"):
    return dict(preprocessing=dict(alignment_policy=dict(method="resize_only", interpolation=interpolation,
        input_size=[112, 112], color="BGR", geometry="full_crop_stretch", face_detection="none")))


@pytest.mark.parametrize("interpolation,cv_mode", [("bicubic", cv2.INTER_CUBIC), ("bilinear", cv2.INTER_LINEAR)])
def test_reader_applies_export_resize_policy_and_namespaces_cache(tmp_path, interpolation, cv_mode):
    bgr = np.random.default_rng(42).integers(0, 256, size=(27, 35, 3), dtype=np.uint8)
    path = tmp_path / "face.png"
    cv2.imwrite(str(path), bgr)
    transform = inputs.input_transform_from_metadata(metadata_policy(interpolation))
    reader = inputs.ImageReader(tmp_path, interpolation=transform["interpolation"])
    image, _ = reader.read(path)
    reader.close()
    np.testing.assert_array_equal(image, cv2.cvtColor(cv2.resize(bgr, (112, 112), interpolation=cv_mode), cv2.COLOR_BGR2RGB))
    other = inputs.input_transform_from_metadata(metadata_policy("bilinear" if interpolation == "bicubic" else "bicubic"))
    assert inputs.input_path(tmp_path, "model", "scface", "population", {}, transform) != inputs.input_path(
        tmp_path, "model", "scface", "population", {}, other)


@pytest.mark.parametrize("field,value", [("method", "landmark_alignment"), ("interpolation", "nearest"),
    ("input_size", [224, 224]), ("color", "RGB"), ("geometry", "center_crop"), ("face_detection", "retinaface")])
def test_unsupported_export_transform_is_rejected(field, value):
    meta = metadata_policy()
    meta["preprocessing"]["alignment_policy"][field] = value
    with pytest.raises(ValueError, match="Export must declare"):
        inputs.input_transform_from_metadata(meta)


def test_missing_export_transform_is_rejected():
    with pytest.raises(ValueError, match="Export must declare"):
        inputs.input_transform_from_metadata({})


def stub_extraction(tmp_path, monkeypatch, fail_second=False):
    import torch
    from research.embeddings import registry
    pop = pd.DataFrame(dict(image_id=[f"im{i}" for i in range(7)], identity_id=["scface:train:001"] * 7,
        source_identity_id=["001"] * 7, image_path=[str(tmp_path / f"{i}.png") for i in range(7)]))
    for i, path in enumerate(pop.image_path):
        cv2.imwrite(path, np.full((12, 15, 3), i, dtype=np.uint8))
    spec = SimpleNamespace(embedding_dim=1024, preprocessing=SimpleNamespace(source_color_order="rgb"), model_uid="fixture")
    monkeypatch.setattr(inputs, "resolve_input_transform", lambda spec: inputs.input_transform_from_metadata(metadata_policy("bilinear")))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "stub-only unit test")
    class Adapter:
        calls = 0
        def embed(self, images):
            self.calls += 1
            if fail_second and self.calls == 2:
                raise RuntimeError("interrupted fixture extraction")
            vector = np.zeros((len(images), 1024), dtype=np.float32)
            vector[:, 0] = 1
            return SimpleNamespace(normalized_embedding=vector, raw_norm=np.full(len(images), 2.))
    adapter = Adapter()
    monkeypatch.setattr(registry, "create_pytorch_adapter_from_spec", lambda *a, **k: adapter)
    return pop, spec


@pytest.mark.parametrize("batch_size", [1, 3, 7])
def test_population_hash_once_per_extraction_not_per_batch(tmp_path, monkeypatch, batch_size):
    pop, spec = stub_extraction(tmp_path, monkeypatch)
    original, calls = inputs.population_hash, []
    expected = original(pop)
    def counted(frame):
        calls.append(len(frame))
        return original(frame)
    monkeypatch.setattr(inputs, "population_hash", counted)
    path = inputs.materialize_inputs(tmp_path, pop, spec, "scface", tmp_path / "outputs", dict(source="raw_norm"), batch_size=batch_size)
    manifest, frame, *_ = inputs.read_inputs(path)
    assert calls == [7]
    assert manifest["population_hash"] == expected
    assert manifest["transform"]["interpolation"] == "bilinear"
    assert frame.source_identity_id.eq("001").all()


def test_input_resume_retains_manifest_hash_and_image_ledger(tmp_path, monkeypatch):
    pop, spec = stub_extraction(tmp_path, monkeypatch, fail_second=True)
    root = tmp_path / "outputs"
    with pytest.raises(RuntimeError, match="interrupted fixture"):
        inputs.materialize_inputs(tmp_path, pop, spec, "scface", root, dict(source="raw_norm"), batch_size=2)
    progress = json.loads(next(root.rglob("progress.json")).read_text())
    assert progress["completed_images"] == 2
    assert progress["population_hash"] == inputs.population_hash(pop)
    path = inputs.materialize_inputs(tmp_path, pop, spec, "scface", root, dict(source="raw_norm"), batch_size=2)
    manifest, frame, vectors, _, _ = inputs.read_inputs(path)
    assert len(frame) == len(pop) == len((path / "image_hashes.jsonl").read_text().splitlines())
    assert manifest["population_hash"] == progress["population_hash"]
    assert np.isfinite(vectors).all()
