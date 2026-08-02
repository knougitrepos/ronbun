from __future__ import annotations

from datetime import datetime
import json
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
    assert manifest["git"]["allowed_untracked_roots"] == [
        "runs",
        "results",
    ]


def test_tracked_run_and_result_artifacts_do_not_dirty_source_contract(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "RunStore Test")
    tracked = repo / "tracked.txt"
    run_manifest = repo / "runs" / "lfw_20260803" / "run_manifest.json"
    result_summary = repo / "results" / "paper" / "summary.csv"
    run_manifest.parent.mkdir(parents=True)
    result_summary.parent.mkdir(parents=True)
    tracked.write_text("clean\n", encoding="utf-8")
    run_manifest.write_text('{"status":"running"}\n', encoding="utf-8")
    result_summary.write_text("metric,value\nfpir,0.1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    run_manifest.write_text('{"status":"completed"}\n', encoding="utf-8")
    result_summary.write_text("metric,value\nfpir,0.2\n", encoding="utf-8")

    run = RunStore.create(
        experiment_name="artifact-dirty-source-clean",
        config={"seed": 42},
        root=repo / "runs" / "survface_20260803",
        repo_root=repo,
        now=datetime(2026, 8, 3, 5, 40, tzinfo=KST),
        allow_dirty=False,
    )

    git = run._read_manifest()["git"]
    assert git["dirty"] is False
    assert git["tracked_source_changes"] == []
    assert git["ignored_tracked_artifact_paths"] == [
        "results/paper/summary.csv",
        "runs/lfw_20260803/run_manifest.json",
    ]


def _write_notebook(path, *, source: str, executed: bool) -> None:
    payload = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1 if executed else None,
                "id": "source-cell",
                "metadata": {"trusted": executed},
                "outputs": (
                    [{"name": "stdout", "output_type": "stream", "text": ["ok\n"]}]
                    if executed
                    else []
                ),
                "source": [source],
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": (
                {
                    "name": "python",
                    "version": "3.11.9",
                    "file_extension": ".py",
                }
                if executed
                else {"name": "python", "version": "3.11"}
            ),
            "ronbun": {"restart_policy": "restart_kernel_and_run_all"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def test_paper_run_ignores_notebook_execution_record_only_changes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "RunStore Test")
    notebook = repo / "experiment.ipynb"
    _write_notebook(notebook, source="result = 42\n", executed=False)
    _git(repo, "add", "experiment.ipynb")
    _git(repo, "commit", "-m", "initial")
    _write_notebook(notebook, source="result = 42\n", executed=True)

    run = RunStore.create(
        experiment_name="notebook-runtime-record",
        config={"seed": 42},
        root=repo / "runs" / "notebook-runtime-record",
        repo_root=repo,
        now=datetime(2026, 7, 29, 22, 30, tzinfo=KST),
        allow_dirty=False,
    )

    manifest = run._read_manifest()
    assert manifest["git"]["dirty"] is False
    assert manifest["git"]["tracked_source_changes"] == []
    assert manifest["git"]["ignored_notebook_runtime_paths"] == [
        "experiment.ipynb"
    ]


def test_paper_run_still_rejects_notebook_source_changes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "RunStore Test")
    notebook = repo / "experiment.ipynb"
    _write_notebook(notebook, source="result = 42\n", executed=False)
    _git(repo, "add", "experiment.ipynb")
    _git(repo, "commit", "-m", "initial")
    _write_notebook(notebook, source="result = 43\n", executed=True)

    with pytest.raises(RuntimeError, match="tracked_source_changes"):
        RunStore.create(
            experiment_name="notebook-source-change",
            config={"seed": 42},
            root=repo / "runs" / "notebook-source-change",
            repo_root=repo,
            now=datetime(2026, 7, 29, 22, 30, tzinfo=KST),
            allow_dirty=False,
        )
