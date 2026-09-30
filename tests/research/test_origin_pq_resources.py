import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.runtime.hashing import sha256_file

# Resource and persistence units do not require optional FR/sklearn imports.
spec = importlib.util.spec_from_file_location("origin_pq_resources_unit",
    Path(__file__).parents[2] / "research/experiments/origin_pq_resources.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def setup_scores(root):
    files = {}
    for name in ("calibration_scores.parquet", "test_scores.parquet"):
        (root / name).write_bytes(b"verified-final")
        files[name] = dict(sha256=sha256_file(root / name), rows=2)
    shard = root / "calibration-000000.parquet"
    shard.write_bytes(b"replay-scratch")
    manifest = dict(status="completed", artifact_type="origin_calibration_test_score_tables",
                    spec={"x": 1}, files=files)
    journal = dict(spec={"x": 1}, shards=[dict(name=shard.name, rows=2, sha256=sha256_file(shard))])
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "progress.json").write_text(json.dumps(journal))
    return shard


def test_verified_cleanup_preserves_completed_scores_and_unrelated_files(tmp_path):
    setup_scores(tmp_path)
    (tmp_path / "unrelated.txt").write_text("keep")
    before = {n: sha256_file(tmp_path / n) for n in (
        "manifest.json", "calibration_scores.parquet", "test_scores.parquet", "unrelated.txt")}
    assert module.prune_origin_replay_scratch(tmp_path) == 2
    assert module.prune_origin_replay_scratch(tmp_path) == 0
    assert before == {n: sha256_file(tmp_path / n) for n in before}


@pytest.mark.parametrize("bad", ["final", "shard", "path"])
def test_cleanup_refuses_bad_hash_or_path_before_deleting(tmp_path, bad):
    shard = setup_scores(tmp_path)
    if bad == "final":
        (tmp_path / "test_scores.parquet").write_bytes(b"tampered")
    elif bad == "shard":
        shard.write_bytes(b"tampered")
    else:
        journal = json.loads((tmp_path / "progress.json").read_text())
        journal["shards"][0]["name"] = "../calibration-000000.parquet"
        (tmp_path / "progress.json").write_text(json.dumps(journal))
    with pytest.raises(ValueError):
        module.prune_origin_replay_scratch(tmp_path)
    assert shard.exists() and (tmp_path / "progress.json").exists()


def test_interrupted_cleanup_resumes_and_incomplete_replay_is_retained(tmp_path):
    shard = setup_scores(tmp_path)
    shard.unlink()
    assert module.prune_origin_replay_scratch(tmp_path) == 1
    setup_scores(tmp_path)
    (tmp_path / "manifest.json").unlink()
    assert module.prune_origin_replay_scratch(tmp_path) == 0
    assert shard.exists() and (tmp_path / "progress.json").exists()


def test_memory_guard_before_more_work(monkeypatch):
    monkeypatch.setattr(module.psutil, "virtual_memory", lambda: SimpleNamespace(available=7 * 2**30))
    monkeypatch.setattr(module.psutil, "Process", lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=2 * 2**30)))
    with pytest.raises(module.ResourceBudgetExceeded, match="available=7.00"):
        module.check_resources()
    assert module.check_resources(minimum_available_gb=6)["process_gb"] == 2
    with pytest.raises(ValueError):
        module.check_resources(maximum_process_gb=float("nan"))
