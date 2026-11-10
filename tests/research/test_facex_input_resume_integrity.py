"""Committed input prefixes and incremental cache allocation, using CPU fixtures."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from research.experiments import facex_inputs as inputs
from research.runtime.hashing import sha256_file
from test_facex_review_regressions import stub_extraction


def interrupted(tmp_path, monkeypatch):
    pop, spec = stub_extraction(tmp_path, monkeypatch, fail_second=True)
    root = tmp_path / "outputs"
    with pytest.raises(RuntimeError, match="interrupted fixture"):
        inputs.materialize_inputs(tmp_path, pop, spec, "scface", root, dict(source="raw_norm"), batch_size=2)
    progress = next(root.rglob("progress.json"))
    assert json.loads(progress.read_text())["completed_images"] == 2
    return pop, spec, root, progress.parent


@pytest.mark.parametrize("target", ["population", "embeddings.npy", "raw_norm.npy", "quality.npy",
                                   "image_hashes.jsonl", "batch_integrity.jsonl", "chain", "legacy"])
def test_committed_corruption_is_rejected_without_publication(tmp_path, monkeypatch, target):
    pop, spec, root, work = interrupted(tmp_path, monkeypatch)
    if target == "population":
        path = work / "population.csv"
        path.write_text(path.read_text().replace("scface:train:001", "scface:train:999", 1))
    elif target.endswith(".npy"):
        array = np.load(work / target, mmap_mode="r+")
        array[0] = 3
        array.flush()
        array._mmap.close()
    elif target.endswith(".jsonl"):
        path = work / target
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["sha256" if target == "image_hashes.jsonl" else "previous"] = "changed"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    else:
        path = work / "progress.json"
        progress = json.loads(path.read_text())
        if target == "legacy":
            progress.pop("schema")
        else:
            progress["batch_chain_sha256"] = "changed"
        path.write_text(json.dumps(progress))
    before = {p.name: sha256_file(p) for p in work.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="changed|partial|commit"):
        inputs.materialize_inputs(tmp_path, pop, spec, "scface", root, dict(source="raw_norm"), batch_size=2)
    assert not list(root.rglob("COMPLETED"))
    assert before == {p.name: sha256_file(p) for p in work.iterdir() if p.is_file()}


def test_uncommitted_corrupt_tails_are_recomputed_after_prefix_validation(tmp_path, monkeypatch):
    pop, spec, root, work = interrupted(tmp_path, monkeypatch)
    for name in ("embeddings.npy", "raw_norm.npy", "quality.npy"):
        array = np.load(work / name, mmap_mode="r+")
        array[2:] = np.nan
        array.flush()
        array._mmap.close()
    for name in ("image_hashes.jsonl", "batch_integrity.jsonl"):
        with (work / name).open("ab") as stream:
            stream.write(b'{"crashed":\xff')
    done = inputs.materialize_inputs(tmp_path, pop, spec, "scface", root, dict(source="raw_norm"), batch_size=3)
    manifest, actual, embeddings, norms, quality = inputs.read_inputs(done)
    np.testing.assert_array_equal(embeddings[:, 0], np.ones(7))
    np.testing.assert_array_equal(norms, np.full(7, 2))
    np.testing.assert_array_equal(quality, np.full(7, 2))
    assert manifest["population_hash"] == inputs.population_hash(actual)
    assert len((done / "image_hashes.jsonl").read_text().splitlines()) == 7
    inputs._close_arrays((embeddings, norms, quality))


def test_receipt_written_but_progress_not_committed_is_a_recoverable_tail(tmp_path, monkeypatch):
    pop, spec = stub_extraction(tmp_path, monkeypatch)
    root = tmp_path / "outputs"
    original = inputs.write_json
    def fault(path, payload):
        if path.name == "progress.json" and payload["completed_images"] == 4:
            raise OSError("simulated power interruption before commit")
        original(path, payload)
    with monkeypatch.context() as m:
        m.setattr(inputs, "write_json", fault)
        with pytest.raises(OSError, match="before commit"):
            inputs.materialize_inputs(tmp_path, pop, spec, "scface", root, dict(source="raw_norm"), batch_size=2)
    progress = json.loads(next(root.rglob("progress.json")).read_text())
    assert progress["completed_images"] == 2
    done = inputs.materialize_inputs(tmp_path, pop, spec, "scface", root, dict(source="raw_norm"), batch_size=3)
    assert json.loads((done / "progress.json").read_text())["completed_images"] == 7
    receipts = [json.loads(line) for line in (done / "batch_integrity.jsonl").read_text().splitlines()]
    assert [(r["start"], r["stop"]) for r in receipts] == [(0, 2), (2, 5), (5, 7)]


def test_new_batch_hashing_is_linear_not_full_prefix_per_batch(tmp_path, monkeypatch):
    pop, spec = stub_extraction(tmp_path, monkeypatch)
    real, sizes = inputs._batch_receipt, []
    def counted(arrays, start, stop, records, previous):
        sizes.append(stop-start)
        return real(arrays, start, stop, records, previous)
    monkeypatch.setattr(inputs, "_batch_receipt", counted)
    inputs.materialize_inputs(tmp_path, pop, spec, "scface", tmp_path / "outputs", dict(source="raw_norm"), batch_size=2)
    assert sizes == [2, 2, 2, 1]


def test_completed_population_must_match_contract_even_when_file_hash_is_resigned(tmp_path, monkeypatch):
    pop, spec = stub_extraction(tmp_path, monkeypatch)
    path = inputs.materialize_inputs(tmp_path, pop, spec, "scface", tmp_path / "outputs", dict(source="raw_norm"))
    frame = inputs._read_population(path / "population.csv")
    frame.loc[0, "identity_id"] = "scface:train:999"
    frame.to_csv(path / "population.csv", index=False)
    metadata = json.loads((path / "manifest.json").read_text())
    metadata["files"]["population.csv"] = sha256_file(path / "population.csv")
    inputs.write_json(path / "manifest.json", metadata)
    with pytest.raises(ValueError, match="population hash"):
        inputs.read_inputs(path)
    metadata["population_hash"] = inputs.population_hash(frame)
    inputs.write_json(path / "manifest.json", metadata)
    with pytest.raises(ValueError, match="population hash"):
        inputs.materialize_inputs(tmp_path, pop, spec, "scface", tmp_path / "outputs", dict(source="raw_norm"))


def test_completed_cache_reuse_does_not_require_rebuild_space(tmp_path, monkeypatch):
    pop, spec = stub_extraction(tmp_path, monkeypatch)
    root = tmp_path / "outputs"
    path = inputs.materialize_inputs(tmp_path, pop, spec, "scface", root, dict(source="raw_norm"))
    monkeypatch.setattr(inputs.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    assert inputs.materialize_inputs(tmp_path, pop, spec, "scface", root, dict(source="raw_norm")) == path


def test_partial_cache_counts_existing_arrays_not_entire_rebuild(tmp_path, monkeypatch):
    pop, spec, root, work = interrupted(tmp_path, monkeypatch)
    # All .npy arrays are already allocated. Only five rows of journals and final
    # metadata need new space; this is less than a complete 28,728-byte rebuild.
    monkeypatch.setattr(inputs.shutil, "disk_usage", lambda _: SimpleNamespace(free=25000))
    path = inputs.materialize_inputs(tmp_path, pop, spec, "scface", root, dict(source="raw_norm"), batch_size=2)
    assert (path / "COMPLETED").is_file()


def test_new_cache_checks_space_before_allocating_arrays(tmp_path, monkeypatch):
    pop, spec = stub_extraction(tmp_path, monkeypatch)
    monkeypatch.setattr(inputs.shutil, "disk_usage", lambda _: SimpleNamespace(free=25000))
    root = tmp_path / "outputs"
    with pytest.raises(RuntimeError, match="additional input cache"):
        inputs.materialize_inputs(tmp_path, pop, spec, "scface", root, dict(source="raw_norm"), batch_size=2)
    assert not list(root.rglob("*.npy"))

