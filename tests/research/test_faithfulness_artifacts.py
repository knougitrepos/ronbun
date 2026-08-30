from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from research.evaluation.faithfulness_artifacts import (
    faithfulness_artifact_directory_name,
    load_selected_faithfulness_artifacts,
    normalize_faithfulness_maximum_samples,
    resolve_common_faithfulness_maximum_samples,
    resolve_faithfulness_selected_count,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(
    root: Path,
    *,
    dataset: str,
    run_id: str,
    model_uid: str,
    maximum_samples: int | None = 10000,
) -> None:
    target = (
        root
        / "results"
        / "paper"
        / dataset
        / run_id
        / faithfulness_artifact_directory_name(maximum_samples)
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
            "maximum_samples": maximum_samples,
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


def test_loads_custom_maximum_samples_directory(tmp_path: Path) -> None:
    _artifact(
        tmp_path,
        dataset="lfw",
        run_id="L001",
        model_uid="adaface-test",
        maximum_samples=321,
    )

    result = load_selected_faithfulness_artifacts(
        tmp_path,
        datasets=("lfw",),
        model_uids={"lfw": "adaface-test"},
        run_ids={"lfw": "L001"},
        maximum_samples=321,
    )

    assert result.manifests["lfw"]["sampling"]["maximum_samples"] == 321
    assert result.roots["lfw"].name == "faithfulness_v2_n321"


def test_loads_unlimited_faithfulness_directory(tmp_path: Path) -> None:
    _artifact(
        tmp_path,
        dataset="lfw",
        run_id="L001",
        model_uid="adaface-test",
        maximum_samples=None,
    )

    result = load_selected_faithfulness_artifacts(
        tmp_path,
        datasets=("lfw",),
        model_uids={"lfw": "adaface-test"},
        run_ids={"lfw": "L001"},
        maximum_samples=None,
    )

    assert result.manifests["lfw"]["sampling"]["maximum_samples"] is None
    assert result.roots["lfw"].name == "faithfulness_v2_all"
    assert resolve_faithfulness_selected_count(123, None) == 123
    assert resolve_faithfulness_selected_count(123, 100) == 100


def test_auto_selects_largest_common_finite_faithfulness_cap(
    tmp_path: Path,
) -> None:
    for dataset, run_id in (("lfw", "L001"), ("survface", "S001")):
        _artifact(
            tmp_path,
            dataset=dataset,
            run_id=run_id,
            model_uid="arcface-test",
            maximum_samples=1000,
        )
        _artifact(
            tmp_path,
            dataset=dataset,
            run_id=run_id,
            model_uid="arcface-test",
            maximum_samples=10000,
        )

    selected = resolve_common_faithfulness_maximum_samples(
        tmp_path,
        datasets=("lfw", "survface"),
        run_ids={"lfw": "L001", "survface": "S001"},
    )

    assert selected == 10000


def test_auto_prefers_common_unlimited_faithfulness_artifact(
    tmp_path: Path,
) -> None:
    for dataset, run_id in (("lfw", "L001"), ("survface", "S001")):
        _artifact(
            tmp_path,
            dataset=dataset,
            run_id=run_id,
            model_uid="adaface-test",
            maximum_samples=10000,
        )
        _artifact(
            tmp_path,
            dataset=dataset,
            run_id=run_id,
            model_uid="adaface-test",
            maximum_samples=None,
        )

    selected = resolve_common_faithfulness_maximum_samples(
        tmp_path,
        datasets=("lfw", "survface"),
        run_ids={"lfw": "L001", "survface": "S001"},
    )

    assert selected is None


def test_auto_rejects_runs_without_a_common_faithfulness_contract(
    tmp_path: Path,
) -> None:
    _artifact(
        tmp_path,
        dataset="lfw",
        run_id="L001",
        model_uid="arcface-test",
        maximum_samples=10000,
    )
    _artifact(
        tmp_path,
        dataset="survface",
        run_id="S001",
        model_uid="arcface-test",
        maximum_samples=None,
    )

    with pytest.raises(FileNotFoundError, match="no common faithfulness"):
        resolve_common_faithfulness_maximum_samples(
            tmp_path,
            datasets=("lfw", "survface"),
            run_ids={"lfw": "L001", "survface": "S001"},
        )


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "10000"])
def test_rejects_invalid_faithfulness_maximum_samples(invalid: object) -> None:
    with pytest.raises(ValueError, match="None or a positive integer"):
        normalize_faithfulness_maximum_samples(invalid)


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


def test_missing_policy_omit_skips_only_absent_manifests(tmp_path: Path) -> None:
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
        missing_policy="omit",
    )

    assert set(result.summary["dataset"]) == {"rfw_custom"}
    assert set(result.manifests) == {"rfw_custom"}
    assert set(result.missing_manifests) == {"lfw"}
    assert result.missing_manifests["lfw"].name == "manifest.json"


def test_missing_manifest_remains_fail_closed_by_default(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_selected_faithfulness_artifacts(
            tmp_path,
            datasets=("lfw",),
            model_uids={"lfw": "adaface-test"},
            run_ids={"lfw": "L001"},
        )


def test_missing_policy_omit_still_rejects_invalid_existing_artifact(
    tmp_path: Path,
) -> None:
    _artifact(tmp_path, dataset="lfw", run_id="L001", model_uid="wrong-model")

    with pytest.raises(ValueError, match="faithfulness manifest mismatch"):
        load_selected_faithfulness_artifacts(
            tmp_path,
            datasets=("lfw",),
            model_uids={"lfw": "adaface-test"},
            run_ids={"lfw": "L001"},
            missing_policy="omit",
        )


def test_rejects_unknown_missing_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported missing_policy"):
        load_selected_faithfulness_artifacts(
            tmp_path,
            datasets=("lfw",),
            model_uids={"lfw": "adaface-test"},
            run_ids={"lfw": "L001"},
            missing_policy="ignore",  # type: ignore[arg-type]
        )
