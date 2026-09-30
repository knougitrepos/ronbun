"""Storage tests load the module independently of optional experiment imports."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[2] / "research/experiments/origin_pq_storage.py"
_spec = importlib.util.spec_from_file_location("origin_pq_storage_isolated", MODULE_PATH)
storage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(storage)


def tables():
    return {
        "summary": pd.DataFrame({
            "score": [np.nextafter(.1, 0.), .1, np.nan],
            "count": pd.Series([1, 2, 3], dtype="int64"),
            "flag": [True, False, True],
            "label": ["원본", "PQ", ""],
        }),
        "model": pd.DataFrame({"model_json": ['{"threshold":0.1}', '{"threshold":-2.5}']}),
    }


def assert_tables(actual, expected):
    assert set(actual) == set(expected)
    for name in actual:
        pd.testing.assert_frame_equal(actual[name], expected[name], check_exact=True)


def test_split_lossless_reuse_and_one_file(tmp_path):
    path, frames = tmp_path / "splits.sqlite3", tables()
    with storage.SplitStore(path) as store:
        assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert store.connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert not store.has({"seed": 0})
        for seed in range(3):
            manifest = store.put({"seed": seed}, frames)
            assert manifest["status"] == "completed"
            assert manifest["formal_fpir_guarantee"] is False
        assert store.has({"seed": 0})
        assert store.put({"seed": 0}, frames) == store.get({"seed": 0})[1]
    with storage.SplitStore(path) as store:
        restored, _ = store.get({"seed": 1})
        assert_tables(restored, frames)
    assert [p.name for p in tmp_path.iterdir()] == ["splits.sqlite3"]


def test_completed_split_cannot_be_replaced(tmp_path):
    with storage.SplitStore(tmp_path / "splits.sqlite3") as store:
        frames = tables()
        store.put({"seed": 0}, frames)
        frames["summary"].loc[0, "score"] = 99.
        with pytest.raises(ValueError, match="cannot replace completed"):
            store.put({"seed": 0}, frames)
        assert_tables(store.get({"seed": 0})[0], tables())


def test_failed_write_rolls_back_entire_split(tmp_path, monkeypatch):
    original = storage._encode_frame
    calls = []

    def fail_second(frame):
        calls.append(1)
        if len(calls) == 2:
            raise InterruptedError("simulated interruption")
        return original(frame)

    with storage.SplitStore(tmp_path / "splits.sqlite3") as store:
        store.put({"seed": 0}, tables())
        monkeypatch.setattr(storage, "_encode_frame", fail_second)
        with pytest.raises(InterruptedError):
            store.put({"seed": 1}, tables())
        assert store.get({"seed": 1}) is None
        assert store.connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
        assert_tables(store.get({"seed": 0})[0], tables())
        monkeypatch.setattr(storage, "_encode_frame", original)
        store.put({"seed": 1}, tables())
        assert store.has({"seed": 1})


def test_hard_process_exit_leaves_no_completed_partial_split(tmp_path):
    path = tmp_path / "splits.sqlite3"
    script = """
import importlib.util, os, sys
import pandas as pd
spec = importlib.util.spec_from_file_location('storage_under_test', sys.argv[2])
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)
store = storage.SplitStore(sys.argv[1])
store.put({'seed': 0}, {'summary': pd.DataFrame({'x': [1.]})})
original = storage._encode_frame
calls = []
def interrupted(frame):
    calls.append(1)
    if len(calls) == 2:
        os._exit(19)
    return original(frame)
storage._encode_frame = interrupted
store.put({'seed': 1}, {'a': pd.DataFrame({'x': [2.]}), 'b': pd.DataFrame({'x': [3.]})})
"""
    completed = subprocess.run([sys.executable, "-c", script, str(path), str(MODULE_PATH)],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 19, completed.stderr
    with storage.SplitStore(path) as store:
        assert store.has({"seed": 0})
        assert store.get({"seed": 1}) is None
        store.put({"seed": 1}, tables())
        assert store.has({"seed": 1})


@pytest.mark.parametrize("target", ["blob", "manifest", "inventory", "spec"])
def test_split_tamper_is_rejected(tmp_path, target):
    path, spec = tmp_path / "splits.sqlite3", {"seed": 0}
    with storage.SplitStore(path) as store:
        store.put(spec, tables())
    with sqlite3.connect(path) as connection:
        if target == "blob":
            connection.execute("UPDATE tables SET payload=? WHERE name='summary'", (b"broken",))
        elif target == "manifest":
            connection.execute("UPDATE jobs SET manifest_json='{}'")
        elif target == "inventory":
            connection.execute("DELETE FROM tables WHERE name='summary'")
        else:
            connection.execute("UPDATE jobs SET spec_json='{}'")
    with storage.SplitStore(path) as store, pytest.raises(ValueError):
        store.get(spec)


def test_unpublished_row_ignored_and_replaced(tmp_path):
    spec = {"seed": 0}
    with storage.SplitStore(tmp_path / "splits.sqlite3") as store:
        store.connection.execute("INSERT INTO jobs(uid,spec_json,status) VALUES(?,?,'writing')",
                                 (storage.split_uid(spec), json.dumps(spec)))
        assert store.get(spec) is None
        store.put(spec, tables())
        assert store.has(spec)


def test_report_two_files_lossless_selective_read_and_reuse(tmp_path):
    frames, spec = tables(), {"seeds": [0, 1], "settings": {"target": .1}}
    directory = storage.write_report(tmp_path, frames, spec)
    assert sorted(p.name for p in directory.iterdir()) == ["manifest.json", "results.zip"]
    restored, manifest = storage.read_report(directory)
    assert_tables(restored, frames)
    assert manifest["uncertainty"]["between_splits"] == "descriptive_not_independent_replications"
    assert set(storage.read_report(directory, names=["summary"])[0]) == {"summary"}
    assert storage.write_report(tmp_path, frames, spec) == directory
    with pytest.raises(ValueError, match="selection"):
        storage.read_report(directory, names=["unknown"])
    frames["summary"].loc[0, "score"] = 7.
    with pytest.raises(ValueError, match="cannot replace completed"):
        storage.write_report(tmp_path, frames, spec)


def test_report_archive_corruption_rejected_even_for_subset(tmp_path):
    directory = storage.write_report(tmp_path, tables(), {"seed": 0})
    path = directory / "results.zip"
    blob = bytearray(path.read_bytes())
    blob[40] ^= 1
    path.write_bytes(blob)
    with pytest.raises(ValueError, match="archive hash"):
        storage.read_report(directory, names=["model"])


def test_report_internal_hash_and_inventory_checked(tmp_path):
    directory = storage.write_report(tmp_path, tables(), {"seed": 0})
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf8"))
    manifest["tables"]["summary"]["sha256"] = "bad"
    path.write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(ValueError, match="table blob hash"):
        storage.read_report(directory)


def test_unsafe_names_rejected_before_writing(tmp_path):
    with storage.SplitStore(tmp_path / "splits.sqlite3") as store:
        with pytest.raises(ValueError, match="safe table names"):
            store.put({}, {"../outside": pd.DataFrame()})
    with pytest.raises(ValueError, match="safe table names"):
        storage.write_report(tmp_path, {"../outside": pd.DataFrame()}, {})
