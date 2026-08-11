from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from research.evaluation.faithfulness_artifacts import (
    load_selected_faithfulness_artifacts,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(root: Path, *, dataset: str, run_id: str, model_uid: str) -> None:
    target = (
        root
        / "results"
        / "paper"
        / dataset
        / run_id
        / "faithfulness_v2_n10000"
    )
    target.mkdir(parents=True)
    rows = pd.DataFrame(
        {
            "dataset": [dataset],
            "model_uid": [model_uid],
            "source_run_id": [run_id],
        }
    )
    summary = pd.DataFrame(
        {
            "dataset": [dataset],
            "model_uid": [model_uid],
            "source_run_id": [run_id],
            "evaluation_mode": ["derived_stratified"],
            "group": ["all"],
            "metric": ["faithfulness_gain_over_random"],
            "sample_count": [1],
        }
    )
    rows_path = target / "faithfulness_rows.csv"
    summary_path = target / "faithfulness_summary.csv"
    rows.to_csv(rows_path, index=False, lineterminator="\n")
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    outputs = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in (rows_path, summary_path)
    ]
    manifest = {
        "artifact_type": "open_set_gradcam_faithfulness",
        "schema_version": 2,
        "dataset_id": dataset,
        "source_run_id": run_id,
        "model_uid": model_uid,
        "threshold_independent": True,
        "sampling": {
            "candidate_count": 1,
            "selected_count": 1,
            "maximum_samples": 10000,
        },
        "outputs": outputs,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def test_loads_one_verified_summary_per_selected_dataset(tmp_path: Path) -> None:
    _artifact(tmp_path, dataset="lfw", run_id="L001", model_uid="adaface-test")
    _artifact(
        tmp_path,
        dataset="rfw_custom",
        run_id="R001",
        model_uid="adaface-test",
    )

    result = load_selected_faithfulness_artifacts(
        tmp_path,
        datasets=("lfw", "rfw_custom"),
        model_uids={"lfw": "adaface-test", "rfw_custom": "adaface-test"},
        run_ids={"lfw": "L001", "rfw_custom": "R001"},
    )

    assert set(result.summary["dataset"]) == {"lfw", "rfw_custom"}
    assert set(result.manifests) == {"lfw", "rfw_custom"}


def test_rejects_output_hash_mismatch(tmp_path: Path) -> None:
    _artifact(tmp_path, dataset="lfw", run_id="L001", model_uid="adaface-test")
    rows = (
        tmp_path
        / "results/paper/lfw/L001/faithfulness_v2_n10000/faithfulness_rows.csv"
    )
    rows.write_text("corrupt\n", encoding="utf-8")

    with pytest.raises(ValueError, match="byte mismatch|SHA-256 mismatch"):
        load_selected_faithfulness_artifacts(
            tmp_path,
            datasets=("lfw",),
            model_uids={"lfw": "adaface-test"},
            run_ids={"lfw": "L001"},
        )
