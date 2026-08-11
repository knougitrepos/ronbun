from __future__ import annotations

from pathlib import Path

import pytest

from scripts import derive_survface_faithfulness as module


def test_provenance_uses_resolved_source_run_before_expensive_work(
    tmp_path,
    monkeypatch,
):
    source_run = (tmp_path / "source-run").resolve()
    observed: dict[str, Path] = {}

    monkeypatch.setattr(
        module,
        "_source_context",
        lambda run_dir: {
            "run_dir": source_run,
            "dataset_id": "survface",
            "run_id": "S001",
        },
    )
    monkeypatch.setattr(
        module,
        "_verify_existing_output",
        lambda output, **kwargs: None,
    )

    def inspect(repo_root, *, run_root):
        observed["repo_root"] = Path(repo_root)
        observed["run_root"] = Path(run_root)
        return {"dirty": False}

    monkeypatch.setattr(module, "inspect_git_provenance", inspect)

    class CandidateReadReached(RuntimeError):
        pass

    def stop_before_expensive_work(context):
        raise CandidateReadReached

    monkeypatch.setattr(module, "_read_candidates", stop_before_expensive_work)

    with pytest.raises(CandidateReadReached):
        module.derive_survface_faithfulness(
            source_run,
            output_dir=tmp_path / "output",
        )

    assert observed == {
        "repo_root": module.PROJECT_ROOT,
        "run_root": source_run,
    }
