from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from research.runtime import ACTIVE_RUN_POINTER, RunStore, resolve_active_run


KST = ZoneInfo("Asia/Seoul")


def _create_run(root: Path, *, sequence_minute: int, name: str) -> RunStore:
    return RunStore.create(
        experiment_name=name,
        config={"run": {"name": name}, "seed": sequence_minute},
        root=root,
        now=datetime(2026, 7, 14, 13, sequence_minute, tzinfo=KST),
        repo_root=root.parent,
    )


def test_create_writes_validated_active_run_pointer(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("RONBUN_RUN_DIR", raising=False)
    root = tmp_path / "runs"
    run = _create_run(root, sequence_minute=1, name="active-test")

    pointer_path = root / ACTIVE_RUN_POINTER
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))

    assert pointer["run_dir"] == str(run.run_dir.resolve())
    assert pointer["run_id"] == run.run_id
    assert pointer["config_hash"] == run.config_hash
    assert resolve_active_run(root) == run.run_dir.resolve()


def test_explicit_environment_override_has_priority(tmp_path: Path, monkeypatch):
    root = tmp_path / "runs"
    first = _create_run(root, sequence_minute=1, name="first")
    second = _create_run(root, sequence_minute=2, name="second")
    monkeypatch.setenv("RONBUN_RUN_DIR", str(first.run_dir))

    assert resolve_active_run(root) == first.run_dir.resolve()
    pointer = json.loads((root / ACTIVE_RUN_POINTER).read_text(encoding="utf-8"))
    assert pointer["run_id"] == second.run_id


def test_single_legacy_active_run_is_discovered_without_pointer(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("RONBUN_RUN_DIR", raising=False)
    root = tmp_path / "runs"
    run = _create_run(root, sequence_minute=1, name="legacy")
    (root / ACTIVE_RUN_POINTER).unlink()

    assert resolve_active_run(root) == run.run_dir.resolve()


def test_legacy_fallback_never_guesses_between_multiple_runs(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("RONBUN_RUN_DIR", raising=False)
    root = tmp_path / "runs"
    _create_run(root, sequence_minute=1, name="first")
    _create_run(root, sequence_minute=2, name="second")
    (root / ACTIVE_RUN_POINTER).unlink()

    with pytest.raises(RuntimeError, match="multiple active runs"):
        resolve_active_run(root)


def test_completed_pointer_is_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("RONBUN_RUN_DIR", raising=False)
    root = tmp_path / "runs"
    run = _create_run(root, sequence_minute=1, name="completed")
    run.complete()

    with pytest.raises(RuntimeError, match="already completed"):
        resolve_active_run(root)

    assert resolve_active_run(root, allow_completed=True) == run.run_dir.resolve()
