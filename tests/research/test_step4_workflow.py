from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import pandas as pd
import pytest
import yaml

from research.evaluation.saliency_compression import (
    DEFAULT_PRIMARY_THRESHOLD_EVENT_METRICS,
    DEFAULT_PRIMARY_THRESHOLD_SALIENCY_FEATURES,
    DEFAULT_SALIENCY_FEATURES,
)
from research.experiments import step4_workflow
from research.experiments.scope import ExperimentScope
from research.experiments.step4_workflow import (
    analyze_step4_saliency_compression,
    freeze_step4_source_and_model,
    select_step4_source_manifest,
)
from research.protocols import build_survface_official_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_git_readiness_allows_pinned_dirty_quick_but_not_dirty_full() -> None:
    provenance = {
        "commit": "a" * 40,
        "branch": "step5",
        "dirty": True,
        "working_tree_diff_sha256": "b" * 64,
        "untracked_content_sha256": "c" * 64,
    }
    snapshot = dict(provenance)
    quick_config = {
        "execution": {"allow_dirty": True},
        "orchestration": {"source_snapshot": snapshot},
    }
    full_config = {
        "execution": {"allow_dirty": False},
        "orchestration": {"source_snapshot": snapshot},
    }

    quick = step4_workflow._git_source_readiness(
        quick_config,
        provenance,
    )
    full = step4_workflow._git_source_readiness(
        full_config,
        provenance,
    )
    changed = step4_workflow._git_source_readiness(
        quick_config,
        {
            **provenance,
            "working_tree_diff_sha256": "d" * 64,
        },
    )

    assert quick["git_policy_satisfied"] is True
    assert quick["source_snapshot_matches"] is True
    assert full["git_policy_satisfied"] is False
    assert changed["git_policy_satisfied"] is True
    assert changed["source_snapshot_matches"] is False


def test_survface_quick_selection_records_source_and_local_protocol_indexes():
    rows: list[dict[str, object]] = []
    for index in range(4):
        registered_id = f"registered-{index}"
        unknown_id = f"unknown-{index}"
        rows.extend(
            [
                {
                    "image_id": f"gallery-{index}",
                    "identity_id": registered_id,
                    "split": "test",
                    "image_path": f"gallery-{index}.jpg",
                    "protocol_role": "gallery",
                    "protocol_index": index,
                },
                {
                    "image_id": f"probe-{index}",
                    "identity_id": registered_id,
                    "split": "test",
                    "image_path": f"probe-{index}.jpg",
                    "protocol_role": "registered_probe",
                    "protocol_index": index,
                },
                {
                    "image_id": f"unknown-{index}",
                    "identity_id": unknown_id,
                    "split": "test",
                    "image_path": f"unknown-{index}.jpg",
                    "protocol_role": "unknown_unknown_probe",
                    "protocol_index": index,
                },
            ]
        )
    for split in ("development", "calibration"):
        for index in range(4):
            rows.append(
                {
                    "image_id": f"{split}-{index}",
                    "identity_id": f"{split}-identity-{index}",
                    "split": split,
                    "image_path": f"{split}-{index}.jpg",
                    "protocol_role": None,
                    "protocol_index": None,
                }
            )
    source = pd.DataFrame.from_records(rows)

    selected = select_step4_source_manifest(
        source,
        dataset_id="survface",
        scope=ExperimentScope(mode="real", data_fraction=0.5, seed=42),
    )
    official_mask = selected["protocol_role"].astype(str).isin(
        {"gallery", "registered_probe", "unknown_unknown_probe"}
    )
    official = selected.loc[official_mask].copy()
    protocol = build_survface_official_protocol(official)
    source_indexes = source.set_index("image_id")["protocol_index"]

    assert official["source_protocol_index"].notna().all()
    assert all(
        int(row.source_protocol_index)
        == int(source_indexes.loc[row.image_id])
        for row in official.itertuples()
    )
    for frame in (
        protocol.gallery,
        protocol.registered_probes,
        protocol.unknown_unknown_probes,
    ):
        assert frame["protocol_index"].tolist() == list(range(len(frame)))


def test_rfw_custom_quick_selection_uses_role_and_group_preserving_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = pd.DataFrame({"image_id": ["rfw-1"]})
    selected = pd.DataFrame({"image_id": ["rfw-selected"]})
    observed: dict[str, object] = {}

    def fake_select(frame, *, data_fraction, seed):
        observed.update(
            frame=frame,
            data_fraction=data_fraction,
            seed=seed,
        )
        return selected

    monkeypatch.setattr(
        step4_workflow,
        "select_rfw_custom_protocol_fraction",
        fake_select,
    )
    result = select_step4_source_manifest(
        source,
        dataset_id="rfw_custom",
        scope=ExperimentScope(mode="real", data_fraction=0.10, seed=42),
    )

    assert observed == {
        "frame": source,
        "data_fraction": 0.10,
        "seed": 42,
    }
    pd.testing.assert_frame_equal(result, selected)


class _FakePhase:
    def __init__(self, run_dir: Path) -> None:
        self.details: dict[str, object] = {}
        self.events: list[tuple[str, dict[str, object]]] = []
        self.attempt = 1
        self.attempt_dir = run_dir / "attempt"
        self.attempt_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = run_dir / "artifacts" / "phase"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **details: object) -> None:
        self.events.append((event, details))

    def record_counts(self, **counts: int) -> None:
        self.details["counts"] = counts

    def publish_artifact(self, source: Path) -> Path:
        destination = self.artifact_dir / source.name
        destination.write_bytes(source.read_bytes())
        return destination


class _FakeRun:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_id = "synthetic-run"
        self.last_phase: _FakePhase | None = None

    @contextmanager
    def phase(self, name: str) -> Iterator[_FakePhase]:
        assert name == "05_saliency_compression_join"
        phase = _FakePhase(self.run_dir)
        self.last_phase = phase
        yield phase


class _FakeCompressionRun:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_id = "synthetic-compression-run"
        self.last_phase: _FakePhase | None = None

    @contextmanager
    def phase(self, name: str) -> Iterator[_FakePhase]:
        assert name == "04_step2_compression_characterization"
        phase = _FakePhase(self.run_dir)
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
            origin_score = (0.80, 0.66, 0.40)[sample_index - 1]
            compressed_score = (0.79, 0.64, 0.43)[sample_index - 1]
            crossing = sample_index == 2
            compressed_tpir = is_mated and not crossing
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
                    "top_k": 20,
                    "origin_top1_score": origin_score,
                    "compressed_top1_score": compressed_score,
                    "compressed_score_at_origin_top1": compressed_score,
                    "top1_score_drift": compressed_score - origin_score,
                    "origin_winner_score_drift": compressed_score - origin_score,
                    "origin_decision_threshold": 0.65,
                    "compressed_decision_threshold": 0.65,
                    "origin_accepted": is_mated,
                    "compressed_accepted": compressed_tpir,
                    "origin_true_identity_rank": 1 if is_mated else np.nan,
                    "compressed_true_identity_rank": 1 if is_mated else np.nan,
                    "origin_true_identity_score": (
                        origin_score if is_mated else np.nan
                    ),
                    "compressed_true_identity_score": (
                        compressed_score if is_mated else np.nan
                    ),
                    "origin_tpir_at_rank_k": is_mated,
                    "compressed_tpir_at_rank_k": compressed_tpir,
                    "score_spaces_comparable": True,
                    "agreement_with_origin": sample_index != 3,
                    "threshold_crossing": crossing,
                    "threshold_crossing_direction": (
                        "accept_to_reject" if crossing else "none"
                    ),
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
        "threshold_instability_association_path": (
            "threshold_instability_associations.csv"
        ),
        "threshold_policy_comparison_path": "threshold_policy_comparisons.csv",
        "threshold_policy_saliency_rho_path": "threshold_policy_rho.csv",
        "representative_case_candidates_path": (
            "representative_case_candidates.csv"
        ),
    }
    config = {
        "execution": {"overwrite": False, "seed": 1701},
        "workflow": {
            "saliency_population_dir": "saliency",
            "paired_metrics_path": "paired.csv",
            "retrieval_metrics_path": "retrieval.csv",
            "retrieval_ledger_manifest_path": "retrieval_ledger/manifest.json",
            "artifact_storage_mode": "results_only",
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
    run = _FakeRun(tmp_path)
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


def test_tracked_step4_config_pins_paired_threshold_analysis_contract() -> None:
    config = yaml.safe_load(
        (
            PROJECT_ROOT / "configs/experiments/step2_pytorch_gradcam.yaml"
        ).read_text(encoding="utf-8")
    )
    association = config["joint_analysis"]["association"]

    assert association["paired_saliency_features"] == list(
        DEFAULT_PRIMARY_THRESHOLD_SALIENCY_FEATURES
    )
    assert association["paired_event_metrics"] == list(
        DEFAULT_PRIMARY_THRESHOLD_EVENT_METRICS
    )
    assert association["paired_minimum_event_count"] == 5
    assert association["paired_confidence_level"] == pytest.approx(0.95)


def test_step4_execution_requires_explicit_runtime_acknowledgement(tmp_path):
    config = tmp_path / "step4.yaml"
    config.write_text("execution: {}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="explicit local acknowledgement"):
        freeze_step4_source_and_model(
            config,
            project_root=tmp_path,
            dataset_id="lfw",
        )


def test_compression_phase_persists_origin_calibration_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_root = tmp_path / "workflow"
    prepared_dir = workflow_root / "prepared"
    prepared_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "sample_id": ["sample-1"],
            "identity_id": ["identity-1"],
            "split": ["calibration"],
        }
    ).to_csv(workflow_root / "selected.csv", index=False)
    config = {
        "execution": {"overwrite": False, "seed": 42},
        "workflow": {
            "prepared_population_dir": "prepared",
            "selected_manifest_path": "selected.csv",
            "paired_metrics_path": "paired.csv",
            "retrieval_metrics_path": "retrieval.csv",
            "retrieval_ledger_manifest_path": "retrieval_ledger/manifest.json",
            "artifact_storage_mode": "results_only",
            "origin_score_audit_path": "origin_score_audit.csv",
            "calibration_diagnostics_path": "calibration_diagnostics.json",
        },
        "compression": {
            "families": {
                "pca": {"dimensions": [2]},
                "pq": {"settings": [{"m": 8, "nbits": 1}]},
            }
        },
        "evaluation": {
            "survface_target_fpir": 0.10,
            "survface_calibration_gallery_identities": 3000,
            "survface_calibration_protocol": "training_3000_half_gallery_v2",
            "survface_threshold_selection": "non_mated_only",
            "top_k": 1,
        },
    }
    run = _FakeCompressionRun(tmp_path)
    prepared = SimpleNamespace(
        extraction_uid="extract-a",
        dataset_id="survface",
        model_uid="model-a",
        origin_embedding_artifact_uid="origin-a",
    )
    paired = pd.DataFrame(
        {"sample_id": ["sample-1"], "value": [1.0]}
    )
    retrieval = _retrieval_frame().assign(
        search_mode="pca_reconstruction_cosine",
        target_fpir=0.10,
        extraction_uid="extract-a",
    )
    origin_audit = pd.DataFrame(
        {
            "query_id": ["calibration-1", "test-1"],
            "evaluation_split": ["calibration", "test"],
            "origin_top1_score": [0.4, 0.8],
        }
    )
    diagnostics = {
        "schema_version": 1,
        "calibration_transfer_assessment": {
            "status": "failed_target_fpir",
        },
        "splits": {
            "calibration": {"origin_fpir": 0.10},
            "test": {"origin_fpir": 0.50},
        },
    }
    fake_codec = SimpleNamespace(
        fit_count=4,
        save=lambda path: Path(path).write_bytes(b"frozen-pca-codec"),
    )
    monkeypatch.setattr(step4_workflow, "load_step4_config", lambda path: config)
    monkeypatch.setattr(
        step4_workflow,
        "_open_step4_run",
        lambda *args, **kwargs: (
            run,
            workflow_root,
            SimpleNamespace(dataset_id="survface"),
        ),
    )
    monkeypatch.setattr(
        step4_workflow,
        "read_prepared_population_artifact",
        lambda path: prepared,
    )
    def fake_characterize(*args: object, **kwargs: object) -> SimpleNamespace:
        sink = kwargs["retrieval_sink"]
        for _, batch in retrieval.groupby("threshold_policy", sort=True):
            sink(batch.reset_index(drop=True))
        return SimpleNamespace(
            paired_metrics=paired,
            retrieval_metrics=pd.DataFrame(),
            retrieval_row_count=len(retrieval),
            origin_score_audit=origin_audit,
            calibration_diagnostics=diagnostics,
            fitted_codecs=(("pca", "pca_2", fake_codec),),
        )

    monkeypatch.setattr(
        step4_workflow,
        "characterize_step2_survface_compression",
        fake_characterize,
    )

    result = step4_workflow.characterize_step4_compression(
        tmp_path / "config.yaml",
        project_root=tmp_path,
        dataset_id="survface",
        execution_acknowledged=True,
    )

    assert result["origin_score_audit_rows"] == 2
    assert result["calibration_origin_fpir"] == pytest.approx(0.10)
    assert result["test_origin_fpir"] == pytest.approx(0.50)
    assert result["origin_calibration_transfer_status"] == "failed_target_fpir"
    assert result["frozen_codec_count"] == 1
    assert result["retrieval_rows"] == len(retrieval)
    assert result["retrieval_ledger_condition_count"] == 1
    assert result["retrieval_ledger_decision_partition_count"] == 2
    assert result["retrieval_topk_detail_retained"] is False
    assert (workflow_root / "retrieval_ledger" / "manifest.json").is_file()
    assert not (workflow_root / "retrieval.csv").exists()
    assert (workflow_root / "origin_score_audit.csv").is_file()
    payload = json.loads(
        (workflow_root / "calibration_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["lineage"] == {
        "extraction_uid": "extract-a",
        "dataset_id": "survface",
        "model_uid": "model-a",
        "origin_embedding_artifact_uid": "origin-a",
    }
    codec_manifest = json.loads(
        (workflow_root / "frozen_codec_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert codec_manifest["fit_source_run_id"] == run.run_id
    assert codec_manifest["codecs"][0]["profile_name"] == "pca_2"
    assert codec_manifest["codecs"][0]["fit_seed"] == 42
    assert run.last_phase is not None
    assert run.last_phase.details["counts"]["origin_score_audit_rows"] == 2
    assert run.last_phase.details["retrieval_ledger"]["logical_row_count"] == (
        len(retrieval)
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
    assert implementation["paired_saliency_features"] == list(
        DEFAULT_PRIMARY_THRESHOLD_SALIENCY_FEATURES
    )
    assert implementation["paired_event_metrics"] == list(
        DEFAULT_PRIMARY_THRESHOLD_EVENT_METRICS
    )
    assert implementation["paired_minimum_event_count"] == 5
    assert implementation["paired_confidence_level"] == pytest.approx(0.95)
    assert implementation["bootstrap_seed"] == 1701
    assert implementation["association_workers"] == 4
    assert implementation["association_max_in_flight"] == 8
    assert implementation["partitioned_association_algorithm_version"] == (
        "condition-partitioned-identity-bootstrap-v1"
    )
    assert (
        implementation["threshold_metric_derivation_version"]
        == "saliency-threshold-metrics-v2"
    )
    assert len(implementation["source_git_commit"]) == 40
    assert set(implementation["source_sha256"]) == {
        "association",
        "partitioned_association",
        "streaming_join",
        "workflow",
    }
    compact_outputs = {
        path.name: path
        for key, path in outputs.items()
        if key
        not in {
            "geometry_joined_metrics_path",
            "retrieval_joined_metrics_path",
        }
    }
    output_artifacts = run.last_phase.details["output_artifacts"]
    assert set(output_artifacts) == set(compact_outputs)
    for filename, output_path in compact_outputs.items():
        metadata = output_artifacts[filename]
        assert metadata == {
            "path": output_path.relative_to(run.run_dir).as_posix(),
            "bytes": output_path.stat().st_size,
            "sha256": step4_workflow.sha256_file(output_path),
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


def test_saliency_compression_workflow_records_explicit_paired_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, run, _ = _prepare_join_workflow(tmp_path, monkeypatch)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    association = config["joint_analysis"]["association"]
    association["paired_saliency_features"] = ["face_attention"]
    association["paired_event_metrics"] = ["threshold_crossing"]
    association["paired_minimum_event_count"] = 2
    association["paired_confidence_level"] = 0.90
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    analyze_step4_saliency_compression(
        config_path,
        project_root=PROJECT_ROOT,
        dataset_id="survface",
        execution_acknowledged=True,
    )

    assert run.last_phase is not None
    implementation = run.last_phase.details["implementation"]
    assert implementation["paired_saliency_features"] == ["face_attention"]
    assert implementation["paired_event_metrics"] == ["threshold_crossing"]
    assert implementation["paired_minimum_event_count"] == 2
    assert implementation["paired_confidence_level"] == pytest.approx(0.90)
    assert implementation["bootstrap_seed"] == 1701


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bootstrap_method", "row", "identity_cluster"),
        ("bootstrap_unit", "sample_id", "bootstrap_unit=identity_id"),
        (
            "paired_saliency_features",
            [],
            "paired_saliency_features must be a non-empty list",
        ),
        (
            "paired_event_metrics",
            ["unknown_event"],
            "paired_event_metrics contains unsupported values",
        ),
        (
            "paired_minimum_event_count",
            True,
            "paired_minimum_event_count must be a positive integer",
        ),
        (
            "paired_confidence_level",
            1.0,
            "paired_confidence_level must be between 0 and 1",
        ),
    ],
)
def test_saliency_compression_workflow_rejects_association_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
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
