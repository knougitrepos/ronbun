from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from research.evaluation import (
    ALL_SEARCH_MODES,
    search_condition,
    threshold_policies_for_search_mode,
)

from research.experiments.cross_model_matrix import (
    FAMILY_ARTIFACT_TYPE,
    MATRIX_ARTIFACT_TYPE,
    SEARCH_SPACE_ARTIFACT_TYPE,
    SEARCH_SPACE_DIRECTORY,
    SEARCH_SPACE_SCHEMA_VERSION,
    SUPPORTED_MODEL_FAMILIES,
    SUPPORTED_OPEN_SET_DATASETS,
    TARGET_FPIRS,
    load_cross_model_open_set_matrix,
    write_cross_model_open_set_matrix,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _named_entry(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _compression_rows(
    *, dataset: str, model_uid: str, run_id: str, extraction_uid: str, origin_uid: str
) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "dataset": dataset,
                "model_uid": model_uid,
                "run_id": run_id,
                "extraction_uid": extraction_uid,
                "origin_embedding_artifact_uid": origin_uid,
                "compression_family": family,
                "compression_profile": profile,
                "sample_count": 10,
                "origin_fallback_count": 0,
                "mean_angular_error_rad": angular_error,
            }
            for family, profile, angular_error in (
                ("pca", "pca_128", 0.02),
                ("pq", "pq_m16_nbits8", 0.03),
            )
        ]
    )


def _retrieval_record(
    *,
    dataset: str,
    model_uid: str,
    run_id: str,
    extraction_uid: str,
    origin_uid: str,
    family: str,
    profile: str,
    search_mode: str,
    threshold_policy: str,
    target_fpir: float,
    bootstrap_resamples: int,
) -> dict[str, object]:
    origin_false_accepts = int(round(target_fpir * 100))
    compressed_false_accepts = max(0, origin_false_accepts - 1)
    origin_fpir = origin_false_accepts / 100
    compressed_fpir = compressed_false_accepts / 100
    condition = search_condition(search_mode)
    return {
        "dataset": dataset,
        "model_uid": model_uid,
        "run_id": run_id,
        "extraction_uid": extraction_uid,
        "origin_embedding_artifact_uid": origin_uid,
        "compression_family": family,
        "compression_profile": profile,
        "search_mode": search_mode,
        "query_representation": condition.query_representation,
        "gallery_representation": condition.gallery_representation,
        "distance_function": condition.distance_function,
        "compressed_score_space": condition.compressed_score_space,
        "score_spaces_comparable": condition.score_spaces_comparable,
        "frozen_origin_threshold_applicable": (
            condition.frozen_origin_threshold_applicable
        ),
        "threshold_policy": threshold_policy,
        "target_fpir": target_fpir,
        "origin_fallback_count": 0,
        "origin_dir_rank1_count": 8,
        "origin_dir_rank1_denominator": 10,
        "origin_dir_rank1": 0.8,
        "origin_dir_rank1_wilson95_low": 0.5,
        "origin_dir_rank1_wilson95_high": 0.95,
        "compressed_dir_rank1_count": 7,
        "compressed_dir_rank1_denominator": 10,
        "compressed_dir_rank1": 0.7,
        "compressed_dir_rank1_wilson95_low": 0.4,
        "compressed_dir_rank1_wilson95_high": 0.9,
        "compressed_minus_origin_dir_rank1": -0.1,
        "compressed_minus_origin_dir_rank1_paired_bootstrap95_low": -0.3,
        "compressed_minus_origin_dir_rank1_paired_bootstrap95_high": 0.1,
        "tpir_rank": 20,
        "origin_tpir20_count": 6,
        "origin_tpir20_denominator": 10,
        "origin_tpir20": 0.6,
        "origin_tpir_at_rank_k_wilson95_low": 0.3,
        "origin_tpir_at_rank_k_wilson95_high": 0.85,
        "compressed_tpir20_count": 5,
        "compressed_tpir20_denominator": 10,
        "compressed_tpir20": 0.5,
        "compressed_tpir_at_rank_k_wilson95_low": 0.2,
        "compressed_tpir_at_rank_k_wilson95_high": 0.8,
        "compressed_tpir20_retention": 5 / 6,
        "origin_closed_set_rank20_recall": 0.9,
        "compressed_closed_set_rank20_recall": 0.8,
        "compressed_minus_origin_tpir_at_rank_k": -0.1,
        "compressed_minus_origin_tpir_at_rank_k_paired_bootstrap95_low": -0.4,
        "compressed_minus_origin_tpir_at_rank_k_paired_bootstrap95_high": 0.2,
        "origin_false_accept_count": origin_false_accepts,
        "origin_fpir_denominator": 100,
        "origin_fpir": origin_fpir,
        "origin_realized_fpir": origin_fpir,
        "origin_fpir_wilson95_low": 0.0,
        "origin_fpir_wilson95_high": 0.5,
        "compressed_false_accept_count": compressed_false_accepts,
        "compressed_fpir_denominator": 100,
        "compressed_fpir": compressed_fpir,
        "compressed_realized_fpir": compressed_fpir,
        "compressed_fpir_wilson95_low": 0.0,
        "compressed_fpir_wilson95_high": 0.5,
        "compressed_minus_origin_fpir": compressed_fpir - origin_fpir,
        "compressed_minus_origin_fpir_paired_bootstrap95_low": -0.1,
        "compressed_minus_origin_fpir_paired_bootstrap95_high": 0.1,
        "confidence_interval_unit": "probe",
        "rate_confidence_interval_method": "wilson_score",
        "difference_confidence_interval_method": (
            "paired_nonparametric_bootstrap_percentile"
        ),
        "difference_confidence_interval_resamples": bootstrap_resamples,
        "difference_confidence_interval_random_seed": 42,
    }


def _retrieval_rows(
    *,
    dataset: str,
    model_uid: str,
    run_id: str,
    extraction_uid: str,
    origin_uid: str,
    bootstrap_resamples: int,
) -> pd.DataFrame:
    specs = tuple(
        (
            search_condition(mode).compression_family,
            (
                "pca_128"
                if search_condition(mode).compression_family == "pca"
                else "pq_m16_nbits8"
            ),
            mode,
        )
        for mode in ALL_SEARCH_MODES
    )
    records: list[dict[str, object]] = []
    for family, profile, search_mode in specs:
        policies = threshold_policies_for_search_mode(search_mode)
        for policy in policies:
            for target in TARGET_FPIRS:
                records.append(
                    _retrieval_record(
                        dataset=dataset,
                        model_uid=model_uid,
                        run_id=run_id,
                        extraction_uid=extraction_uid,
                        origin_uid=origin_uid,
                        family=family,
                        profile=profile,
                        search_mode=search_mode,
                        threshold_policy=policy,
                        target_fpir=target,
                        bootstrap_resamples=bootstrap_resamples,
                    )
                )
    return pd.DataFrame.from_records(records)


def _build_run(
    root: Path,
    *,
    model: str,
    dataset: str,
    bootstrap_resamples: int = 2000,
    fallback_free: bool = True,
) -> Path:
    run_id = f"run-{model}-{dataset}"
    model_uid = f"{model}-{hashlib.sha256(model.encode()).hexdigest()[:20]}"
    checkpoint_sha256 = hashlib.sha256(f"checkpoint:{model}".encode()).hexdigest()
    extraction_uid = f"extract-{model}-{dataset}"
    origin_uid = f"origin-{model}-{dataset}"
    run_dir = root / "runs" / model / dataset / run_id
    workflow_dir = run_dir / "artifacts" / "step2_workflow"
    workflow_dir.mkdir(parents=True)
    (run_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    selected_path = workflow_dir / "selected_manifest.csv"
    selected_path.write_text("sample_id\nsample-1\n", encoding="utf-8")
    prepared_path = workflow_dir / "prepared_population" / "manifest.json"
    _write_json(
        prepared_path,
        {
            "dataset_id": dataset,
            "model_uid": model_uid,
            "extraction_uid": extraction_uid,
            "origin_embedding_artifact_uid": origin_uid,
        },
    )
    freeze_path = workflow_dir / "freeze_manifest.json"
    _write_json(
        freeze_path,
        {
            "run_id": run_id,
            "dataset_id": dataset,
            "model_uid": model_uid,
            "checkpoint_sha256": checkpoint_sha256,
            "selected_manifest_sha256": _sha256(selected_path),
            "fallback_free": fallback_free,
        },
    )
    run_manifest_path = run_dir / "run_manifest.json"
    _write_json(
        run_manifest_path,
        {
            "run_id": run_id,
            "status": "completed",
            "config": {
                "dataset_id": dataset,
                "model_uid": model_uid,
                "step4": {
                    "evaluation": {
                        **(
                            {
                                "rfw_custom_calibration_gallery_policy": (
                                    "evaluation_group_matched"
                                )
                            }
                            if dataset == "rfw_custom"
                            else {}
                        )
                    },
                    "workflow": {
                        "artifact_subdir": "artifacts/step2_workflow",
                        "freeze_manifest_path": "freeze_manifest.json",
                        "prepared_population_dir": "prepared_population",
                        "selected_manifest_path": "selected_manifest.csv",
                    }
                },
            },
        },
    )

    summary_dir = (
        root / "results" / "paper" / dataset / run_id / SEARCH_SPACE_DIRECTORY
    )
    summary_dir.mkdir(parents=True)
    compression = _compression_rows(
        dataset=dataset,
        model_uid=model_uid,
        run_id=run_id,
        extraction_uid=extraction_uid,
        origin_uid=origin_uid,
    )
    retrieval = _retrieval_rows(
        dataset=dataset,
        model_uid=model_uid,
        run_id=run_id,
        extraction_uid=extraction_uid,
        origin_uid=origin_uid,
        bootstrap_resamples=bootstrap_resamples,
    )
    compression_path = summary_dir / "compression_summary.csv"
    retrieval_path = summary_dir / "retrieval_summary.csv"
    compression.to_csv(compression_path, index=False)
    retrieval.to_csv(retrieval_path, index=False)

    family_entries: dict[str, dict[str, object]] = {}
    for family in ("pca", "pq"):
        family_dir = summary_dir / family
        family_dir.mkdir()
        family_compression_path = family_dir / "compression_summary.csv"
        family_retrieval_path = family_dir / "retrieval_summary.csv"
        diagnostics_path = family_dir / "origin_calibration_diagnostics.json"
        compression.loc[compression["compression_family"].eq(family)].to_csv(
            family_compression_path, index=False
        )
        retrieval.loc[retrieval["compression_family"].eq(family)].to_csv(
            family_retrieval_path, index=False
        )
        _write_json(diagnostics_path, {"target_fpirs": list(TARGET_FPIRS)})
        family_manifest_path = family_dir / "family_manifest.json"
        _write_json(
            family_manifest_path,
            {
                "schema_version": SEARCH_SPACE_SCHEMA_VERSION,
                "artifact_type": FAMILY_ARTIFACT_TYPE,
                "family": family,
                "dataset_id": dataset,
                "model_uid": model_uid,
                "source_run_id": run_id,
                "target_fpirs": list(TARGET_FPIRS),
                "outputs": {
                    "compression_summary.csv": _named_entry(family_compression_path),
                    "retrieval_summary.csv": _named_entry(family_retrieval_path),
                    "origin_calibration_diagnostics.json": _named_entry(
                        diagnostics_path
                    ),
                },
            },
        )
        family_entries[family] = _entry(family_manifest_path, root=root)

    _write_json(
        summary_dir / "summary_manifest.json",
        {
            "schema_version": SEARCH_SPACE_SCHEMA_VERSION,
            "artifact_type": SEARCH_SPACE_ARTIFACT_TYPE,
            "dataset_id": dataset,
            "model_uid": model_uid,
            "run_id": run_id,
            "source_run_id": run_id,
            "source_run_preserved_immutable": True,
            "compact_only": True,
            "producer_script": "scripts/refresh_step4_search_spaces.py",
            "target_fpirs": list(TARGET_FPIRS),
            "source_files": {
                "run_manifest.json": _entry(run_manifest_path, root=root),
                "freeze_manifest.json": _entry(freeze_path, root=root),
                "prepared_population/manifest.json": _entry(prepared_path, root=root),
                "selected_manifest.csv": _entry(selected_path, root=root),
            },
            "family_manifests": family_entries,
            "validated_counts": {
                "compression_summary_rows": len(compression),
                "retrieval_summary_rows": len(retrieval),
            },
            "output_files": {
                "compression_summary.csv": _entry(compression_path, root=root),
                "retrieval_summary.csv": _entry(retrieval_path, root=root),
            },
        },
    )
    return run_dir


def _build_matrix(
    root: Path,
    *,
    bootstrap_resamples: int = 2000,
    fallback_selection: tuple[str, str] | None = None,
) -> dict[str, dict[str, Path]]:
    return {
        model: {
            dataset: _build_run(
                root,
                model=model,
                dataset=dataset,
                bootstrap_resamples=(
                    bootstrap_resamples
                    if (model, dataset) == ("arcface", "lfw")
                    else 2000
                ),
                fallback_free=(model, dataset) != fallback_selection,
            )
            for dataset in SUPPORTED_OPEN_SET_DATASETS
        }
        for model in SUPPORTED_MODEL_FAMILIES
    }


def test_loads_complete_explicit_4_by_3_matrix_with_claim_boundaries(
    tmp_path: Path,
) -> None:
    selections = _build_matrix(tmp_path)

    result = load_cross_model_open_set_matrix(selections, project_root=tmp_path)

    assert len(result.compression_summary) == 24
    assert len(result.retrieval_summary) == 600
    assert len(result.joined_summary) == 600
    assert result.selection_manifest["complete_matrix"] is True
    assert result.selection_manifest["matrix_shape"] == {
        "model_count": 4,
        "dataset_count": 3,
        "completed_run_count": 12,
    }
    assert result.selection_manifest["matrix_uid"].startswith("open-set-matrix-")
    assert len(result.selection_manifest["selected_runs"]) == 12
    assert result.selection_manifest["auto_selection_used"] is False
    assert set(result.retrieval_summary["target_fpir"]) == set(TARGET_FPIRS)
    assert result.retrieval_summary["tpir_rank"].eq(20).all()
    assert not result.retrieval_summary["strict_unseen_identity_evidence"].any()
    rfw = result.joined_summary.loc[result.joined_summary["dataset"].eq("rfw_custom")]
    assert set(rfw["checkpoint_training_identity_overlap_status"]) == {"UNKNOWN"}
    assert set(rfw["rfw_protocol_variant"]) == {"RFW-Custom"}
    assert result.selection_manifest["rfw_custom_boundary"] == {
        "task": "1:N_open_set",
        "official_protocol": False,
        "checkpoint_training_identity_overlap_status": "UNKNOWN",
        "strict_unseen_identity_evidence": False,
        "rfw_official_1to1_included": False,
    }
    assert result.selection_manifest["edgeface_rfw_overlap_boundary"] == {
        "checkpoint_training_identity_overlap_status": "UNKNOWN",
        "strict_unseen_identity_evidence": False,
        "permitted_claim": "checkpoint_level_generalization",
    }
    repeated = load_cross_model_open_set_matrix(selections, project_root=tmp_path)
    assert (
        repeated.selection_manifest["matrix_uid"]
        == result.selection_manifest["matrix_uid"]
    )


def test_rejects_incomplete_model_dataset_matrix(tmp_path: Path) -> None:
    selections = _build_matrix(tmp_path)
    del selections["edgeface"]["rfw_custom"]

    with pytest.raises(ValueError, match="edgeface must contain exactly"):
        load_cross_model_open_set_matrix(selections, project_root=tmp_path)


def test_rejects_non_fallback_free_completed_run(tmp_path: Path) -> None:
    selections = _build_matrix(tmp_path, fallback_selection=("edgeface", "rfw_custom"))

    with pytest.raises(ValueError, match="fallback_free=true"):
        load_cross_model_open_set_matrix(selections, project_root=tmp_path)


def test_rejects_legacy_rfw_custom_calibration_in_matrix(tmp_path: Path) -> None:
    selections = _build_matrix(tmp_path)
    run_dir = selections["arcface"]["rfw_custom"]
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["step4"]["evaluation"] = {
        "rfw_custom_calibration_gallery_identities": 80
    }
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="predates gallery-size-matched"):
        load_cross_model_open_set_matrix(selections, project_root=tmp_path)


def test_rejects_tampered_compact_output_hash(tmp_path: Path) -> None:
    selections = _build_matrix(tmp_path)
    run_id = "run-arcface-lfw"
    compression_path = (
        tmp_path
        / "results"
        / "paper"
        / "lfw"
        / run_id
        / SEARCH_SPACE_DIRECTORY
        / "compression_summary.csv"
    )
    compression_path.write_text(
        compression_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="byte-size mismatch"):
        load_cross_model_open_set_matrix(selections, project_root=tmp_path)


def test_rejects_wrong_paired_bootstrap_contract(tmp_path: Path) -> None:
    selections = _build_matrix(tmp_path, bootstrap_resamples=1999)

    with pytest.raises(ValueError, match="2000 resamples and seed 42"):
        load_cross_model_open_set_matrix(selections, project_root=tmp_path)


def test_atomic_writer_hashes_outputs_and_requires_explicit_overwrite(
    tmp_path: Path,
) -> None:
    selections = _build_matrix(tmp_path / "source")
    matrix = load_cross_model_open_set_matrix(
        selections, project_root=tmp_path / "source"
    )

    written = write_cross_model_open_set_matrix(matrix, tmp_path / "published")

    assert written.manifest["artifact_type"] == MATRIX_ARTIFACT_TYPE
    assert written.output_dir.name == matrix.selection_manifest["matrix_uid"]
    assert written.manifest_path.is_file()
    for name, entry in written.manifest["output_files"].items():
        path = written.output_dir / name
        assert path.stat().st_size == entry["bytes"]
        assert _sha256(path) == entry["sha256"]
    with pytest.raises(FileExistsError):
        write_cross_model_open_set_matrix(matrix, tmp_path / "published")

    replaced = write_cross_model_open_set_matrix(
        matrix, tmp_path / "published", overwrite=True
    )
    assert replaced.manifest_path.is_file()
