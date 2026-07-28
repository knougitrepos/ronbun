from __future__ import annotations

from datetime import datetime
import subprocess
from zoneinfo import ZoneInfo

import pytest

from research.runtime.run_store import RunStore


KST = ZoneInfo("Asia/Seoul")


def _git(repo, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def test_paper_run_rejects_dirty_git_tree_unless_explicitly_allowed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "RunStore Test")
    tracked = repo / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")
    tracked.write_text("dirty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="clean Git working tree"):
        RunStore.create(
            experiment_name="paper-run",
            config={"seed": 42},
            root=tmp_path / "rejected",
            repo_root=repo,
            now=datetime(2026, 7, 28, 9, 0, tzinfo=KST),
            allow_dirty=False,
        )

    run = RunStore.create(
        experiment_name="development-run",
        config={"seed": 42},
        root=tmp_path / "allowed",
        repo_root=repo,
        now=datetime(2026, 7, 28, 9, 0, tzinfo=KST),
        allow_dirty=True,
    )
    manifest = run._read_manifest()
    assert manifest["git"]["dirty"] is True
    assert manifest["git"]["dirty_run_explicitly_allowed"] is True


def test_run_root_untracked_artifacts_do_not_block_a_clean_code_resume(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "RunStore Test")
    tracked = repo / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")
    run_root = repo / "runs" / "lfw_20260728"
    run_root.mkdir(parents=True)
    (run_root / "partial-artifact.csv").write_text(
        "sample_id\none\n",
        encoding="utf-8",
    )

    run = RunStore.create(
        experiment_name="resumable-paper-run",
        config={"seed": 42},
        root=run_root,
        repo_root=repo,
        now=datetime(2026, 7, 28, 9, 0, tzinfo=KST),
        allow_dirty=False,
    )

    manifest = run._read_manifest()
    assert manifest["git"]["dirty"] is False
    assert manifest["git"]["allowed_untracked_roots"] == ["runs"]
