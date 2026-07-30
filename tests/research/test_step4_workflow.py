from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pandas as pd
import pytest
import yaml

from research.evaluation.saliency_compression import DEFAULT_SALIENCY_FEATURES
from research.experiments import step4_workflow
from research.experiments.step4_workflow import (
    analyze_step4_saliency_compression,
    freeze_step4_source_and_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakePhase:
    def __init__(self) -> None:
        self.details: dict[str, object] = {}
        self.events: list[tuple[str, dict[str, object]]] = []

    def record(self, event: str, **details: object) -> None:
        self.events.append((event, details))

    def record_counts(self, **counts: int) -> None:
        self.details["counts"] = counts


class _FakeRun:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_id = "synthetic-run"
        self.last_phase: _FakePhase | None = None

    @contextmanager
    def phase(self, name: str) -> Iterator[_FakePhase]:
        assert name == "05_saliency_compression_join"
        phase = _FakePhase()
        self.last_phase = phase
        yield phase


def _saliency_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "extraction_uid": ["extract-1"] * 3,
            "dataset_id": ["survface"] * 3,
            "sample_id": ["sample-1", "sample-2", "sample-3"],
            "model_uid": ["model-a"] * 3,
            "identity_id": ["identity-1", "identity-1", "identity-2"],
            "origin_embedding_artifact_uid": ["origin-a"] * 3,
            "saliency_spec_uid": ["saliency-v1"] * 3,
            "saliency_target_eligible": [True] * 3,
            "heatmap_available": [True] * 3,
        }
    )
    for index, column in enumerate(DEFAULT_SALIENCY_FEATURES, start=1):
        frame[column] = [index / 100.0, index / 50.0, index / 25.0]
    return frame


def _geometry_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for profile_index, profile in enumerate(("pca_128", "pca_64"), start=1):
        for sample_index, sample_id in enumerate(
            ("sample-1", "sample-2", "sample-3"),
            start=1,
        ):
            rows.append(
                {
                    "extraction_uid": "extract-1",
                    "dataset_id": "survface",
                    "sample_id": sample_id,
                    "model_uid": "model-a",
                    "compression_family": "pca",
                    "compression_profile": profile,
                    "origin_embedding_artifact_uid": "origin-a",
                    "origin_fallback_used": False,
                    "angular_error_rad": profile_index * sample_index / 100.0,
                    "reconstruction_mse": profile_index
                    * sample_index
                    / 1000.0,
                    "cosine_to_origin": 1.0
                    - profile_index * sample_index / 100.0,
                }
            )
    return pd.DataFrame.from_records(rows)


def _retrieval_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy in ("frozen_origin", "recalibrated_compressed"):
        for sample_index, (sample_id, is_mated) in enumerate(
            (
                ("sample-1", True),
                ("sample-2", True),
                ("sample-3", False),
            ),
            start=1,
        ):
            rows.append(
                {
                    "extraction_uid": "extract-1",
                    "dataset_id": "survface",
                    "query_id": sample_id,
                    "model_uid": "model-a",
                    "compression_family": "pca",
                    "compression_profile": "pca_128",
                    "origin_embedding_artifact_uid": "origin-a",
                    "origin_fallback_used": False,
                    "protocol_uid": "survface-official-v1",
                    "threshold_source_split": "calibration",
                    "evaluation_split": "official_test",
                    "threshold_policy": policy,
                    "is_mated": is_mated,
                    "top1_score_drift": sample_index / 100.0,
                    "agreement_with_origin": sample_index != 3,
                    "threshold_crossing": sample_index == 2,
                }
            )
    return pd.DataFrame.from_records(rows)


def _prepare_join_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, _FakeRun, dict[str, Path]]:
    geometry = _geometry_frame()
    retrieval = _retrieval_frame()
    geometry.to_csv(tmp_path / "paired.csv", index=False)
    retrieval.to_csv(tmp_path / "retrieval.csv", index=False)
    output_names = {
        "geometry_joined_metrics_path": "geometry_join.csv",
        "retrieval_joined_metrics_path": "retrieval_join.csv",
        "geometry_association_path": "geometry_associations.csv",
        "retrieval_association_path": "retrieval_associations.csv",
    }
    config = {
        "execution": {"overwrite": False, "seed": 1701},
        "workflow": {
            "saliency_population_dir": "saliency",
            "paired_metrics_path": "paired.csv",
            "retrieval_metrics_path": "retrieval.csv",
            **output_names,
        },
        "joint_analysis": {
            "association": {
                "bootstrap_method": "identity_cluster",
                "bootstrap_unit": "identity_id",
                "bootstrap_repeats": 2,
            }
        },
    }
    config_path = tmp_path / "step4.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    run = _FakeRun(tmp_path / "run")
    phase04 = (
        run.run_dir
        / "phases"
        / "04_step2_compression_characterization"
        / "attempts"
        / "A001"
    )
    phase04.mkdir(parents=True)
    (phase04 / "phase_manifest.json").write_text(
        json.dumps(
            {
                "attempt": 1,
                "status": "completed",
                "details": {
                    "counts": {
                        "paired_rows": len(geometry),
                        "retrieval_rows": len(retrieval),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        step4_workflow,
        "_open_step4_run",
        lambda *args, **kwargs: (
            run,
            tmp_path,
            SimpleNamespace(dataset_id="survface"),
        ),
    )
    monkeypatch.setattr(
        step4_workflow,
        "read_population_saliency_features",
        lambda path: _saliency_frame(),
    )
    return (
        config_path,
        run,
        {key: tmp_path / name for key, name in output_names.items()},
    )


def test_step4_execution_requires_explicit_runtime_acknowledgement(tmp_path):
    config = tmp_path / "step4.yaml"
    config.write_text("execution: {}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="explicit local acknowledgement"):
        freeze_step4_source_and_model(
            config,
            project_root=tmp_path,
            dataset_id="lfw",
        )


def test_saliency_compression_workflow_streams_and_publishes_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path, run, outputs = _prepare_join_workflow(tmp_path, monkeypatch)
    progress: list[tuple[str, dict[str, object]]] = []

    result = analyze_step4_saliency_compression(
        config_path,
        project_root=PROJECT_ROOT,
        dataset_id="survface",
        execution_acknowledged=True,
        progress=lambda message, details: progress.append((message, details)),
    )

    assert result["geometry_join_rows"] == 6
    assert result["retrieval_join_rows"] == 6
    assert all(path.is_file() for path in outputs.values())
    assert not list(tmp_path.glob(".05_saliency_compression_join.*"))
    assert run.last_phase is not None
    implementation = run.last_phase.details["implementation"]
    assert implementation["bootstrap_rank_strategy"] == "weighted_rerank"
    assert implementation["bootstrap_batch_size"] == 4
    assert len(implementation["source_git_commit"]) == 40
    assert set(implementation["source_sha256"]) == {
        "association",
        "streaming_join",
        "workflow",
    }
    assert any(message.startswith("geometry ") for message, _ in progress)
    assert any(message.startswith("retrieval ") for message, _ in progress)


def test_saliency_compression_workflow_does_not_publish_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path, _, outputs = _prepare_join_workflow(tmp_path, monkeypatch)

    def fail_association(*args: object, **kwargs: object) -> pd.DataFrame:
        raise RuntimeError("synthetic association failure")

    monkeypatch.setattr(
        step4_workflow,
        "saliency_geometry_associations",
        fail_association,
    )
    with pytest.raises(RuntimeError, match="synthetic association failure"):
        analyze_step4_saliency_compression(
            config_path,
            project_root=PROJECT_ROOT,
            dataset_id="survface",
            execution_acknowledged=True,
        )

    assert all(not path.exists() for path in outputs.values())
    assert not list(tmp_path.glob(".05_saliency_compression_join.*"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bootstrap_method", "row", "identity_cluster"),
        ("bootstrap_unit", "sample_id", "bootstrap_unit=identity_id"),
    ],
)
def test_saliency_compression_workflow_rejects_bootstrap_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
):
    config_path, _, outputs = _prepare_join_workflow(tmp_path, monkeypatch)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["joint_analysis"]["association"][field] = value
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        analyze_step4_saliency_compression(
            config_path,
            project_root=PROJECT_ROOT,
            dataset_id="survface",
            execution_acknowledged=True,
        )

    assert all(not path.exists() for path in outputs.values())
