from datetime import datetime
import json
from zoneinfo import ZoneInfo

import pytest

from research.runtime import RunStore, export_paper_results
from research.runtime.hashing import sha256_file


KST = ZoneInfo("Asia/Seoul")


def _run(tmp_path):
    return RunStore.create(
        experiment_name="paper-results-test",
        config={"dataset": "lfw", "seed": 42},
        root=tmp_path / "runs",
        now=datetime(2026, 7, 14, 18, 0, tzinfo=KST),
        repo_root=tmp_path,
    )


def test_export_paper_results_creates_canonical_bundle_and_manifest(tmp_path):
    run = _run(tmp_path)
    metrics = tmp_path / "evaluation_metrics_A001.json"
    figure = tmp_path / "decisions_by_probe_A001.png"
    metrics.write_text('{"dir": 0.9}', encoding="utf-8")
    figure.write_bytes(b"png-placeholder")
    run.complete()

    result = export_paper_results(
        run=run,
        dataset="LFW",
        source_phase="05_evaluation_and_visualization",
        source_attempt="A001",
        files={
            "evaluation_metrics.json": metrics,
            "decisions_by_probe.png": figure,
        },
        output_root=tmp_path / "results" / "paper",
    )

    destination = tmp_path / "results" / "paper" / "lfw" / run.run_id
    assert result["status"] == "created"
    assert sorted(path.name for path in destination.iterdir()) == [
        "decisions_by_probe.png",
        "evaluation_metrics.json",
        "result_manifest.json",
    ]
    manifest = json.loads((destination / "result_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"] == "lfw"
    assert manifest["run_id"] == run.run_id
    assert manifest["config_hash"] == run.config_hash
    assert manifest["source"]["attempt"] == "A001"
    assert manifest["files"]["evaluation_metrics.json"]["sha256"] == sha256_file(metrics)


def test_export_paper_results_is_idempotent_but_never_overwrites_conflicts(tmp_path):
    run = _run(tmp_path)
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"dir": 0.9}', encoding="utf-8")
    arguments = {
        "run": run,
        "dataset": "lfw",
        "source_phase": "05_evaluation_and_visualization",
        "source_attempt": "A001",
        "files": {"evaluation_metrics.json": metrics},
        "output_root": tmp_path / "results" / "paper",
    }

    export_paper_results(**arguments)
    assert export_paper_results(**arguments)["status"] == "unchanged"

    destination = tmp_path / "results" / "paper" / "lfw" / run.run_id
    (destination / "evaluation_metrics.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(FileExistsError, match="different contents"):
        export_paper_results(**arguments)

    (destination / "evaluation_metrics.json").write_text('{"dir": 0.9}', encoding="utf-8")
    metrics.write_text('{"dir": 0.8}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="different contents"):
        export_paper_results(**arguments)


@pytest.mark.parametrize("dataset", ["../lfw", "lfw/results", ""])
def test_export_paper_results_rejects_unsafe_dataset_names(tmp_path, dataset):
    run = _run(tmp_path)
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset"):
        export_paper_results(
            run=run,
            dataset=dataset,
            source_phase="05_evaluation_and_visualization",
            source_attempt="A001",
            files={"evaluation_metrics.json": metrics},
            output_root=tmp_path / "results" / "paper",
        )
