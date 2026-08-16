from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import nbformat
import pytest

from research.evaluation.cross_dataset_saliency import (
    load_cross_dataset_saliency_associations,
    validate_rfw_custom_calibration_contract,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _synthetic_completed_run(tmp_path: Path) -> Path:
    run_id = "20260810-R001-test"
    dataset = "rfw_custom"
    model_uid = "arcface-test"
    run_dir = tmp_path / "run"
    workflow = run_dir / "artifacts" / "step2_workflow"
    workflow.mkdir(parents=True)
    (run_dir / "COMPLETED").touch()
    _write_json(
        run_dir / "run_manifest.json",
        {
            "status": "completed",
            "run_id": run_id,
            "config": {
                "step4": {
                    "evaluation": {
                        "rfw_custom_calibration_gallery_policy": (
                            "evaluation_group_matched"
                        )
                    }
                }
            },
        },
    )
    identity = {
        "run_id": run_id,
        "dataset_id": dataset,
        "model_uid": model_uid,
    }
    _write_json(
        workflow / "freeze_manifest.json",
        {**identity, "fallback_free": True},
    )
    _write_json(
        workflow / "step4_summary.json",
        {
            **identity,
            "geometry_association_rows": 1,
            "retrieval_association_rows": 5,
            "threshold_instability_association_rows": 5,
            "threshold_policy_comparison_rows": 5,
            "threshold_policy_saliency_rho_rows": 5,
            "representative_cases": 1,
        },
    )
    _write_json(
        workflow / "origin_calibration_diagnostics.json",
        {
            "calibration_contract": {
                "name": "rfw_custom_gallery_group_matched_calibration_v2",
                "score_statistic": "maximum_gallery_score",
                "gallery_matching_policy": "evaluation_group_matched",
                "gallery_size_match_verified": True,
                "gallery_group_match_verified": True,
            },
            "splits": {
                "calibration": {"template_count": 4},
                "test": {"template_count": 4},
            },
        },
    )
    for phase in (
        "05_saliency_compression_join",
        "06_representative_case_visualization",
    ):
        manifest = {"attempt": 1, "phase": phase, "status": "completed"}
        if phase == "05_saliency_compression_join":
            manifest["details"] = {
                "implementation": {
                    "threshold_metric_derivation_version": (
                        "saliency-threshold-metrics-v1"
                    )
                }
            }
        _write_json(
            run_dir / "phases" / phase / "attempts" / "A001" / "phase_manifest.json",
            manifest,
        )

    common = {
        "dataset_id": dataset,
        "model_uid": model_uid,
        "compression_family": "pca",
        "compression_profile": "pca_128",
        "saliency_feature": "saliency_eye",
        "sensitivity_metric": "angular_error_rad",
        "sample_count": 10,
        "identity_count": 5,
        "spearman_rho": 0.2,
        "bootstrap_ci_low": 0.1,
        "bootstrap_ci_high": 0.3,
    }
    pd.DataFrame([{**common, "analysis_scope": "geometry"}]).to_csv(
        workflow / "saliency_geometry_associations.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                **common,
                "analysis_scope": "retrieval",
                "threshold_metric_derivation_version": (
                    "saliency-threshold-metrics-v1"
                ),
                "search_mode": "pca_direct",
                "target_fpir": target_fpir,
                "threshold_policy": "recalibrated_compressed",
                "is_mated": True,
            }
            for target_fpir in (0.01, 0.05, 0.10, 0.20, 0.30)
        ]
    ).to_csv(workflow / "saliency_retrieval_associations.csv", index=False)
    pd.DataFrame(
        [
            {
                **common,
                "analysis_scope": "threshold_instability",
                "threshold_metric_derivation_version": (
                    "saliency-threshold-metrics-v1"
                ),
                "search_mode": "pca_direct",
                "target_fpir": target_fpir,
                "threshold_policy": "recalibrated_compressed",
                "is_mated": True,
                "instability_predictor": "absolute_threshold_margin_shift",
                "sensitivity_metric": "tpir_at_rank_k_loss",
                "event_count": 1,
                "non_event_count": 9,
                "event_rate": 0.1,
            }
            for target_fpir in (0.01, 0.05, 0.10, 0.20, 0.30)
        ]
    ).to_csv(
        workflow / "saliency_threshold_instability_associations.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "analysis_scope": "threshold_policy_comparison",
                "threshold_metric_derivation_version": (
                    "saliency-threshold-metrics-v1"
                ),
                "dataset_id": dataset,
                "model_uid": model_uid,
                "compression_family": "pca",
                "compression_profile": "pca_128",
                "search_mode": "pca_direct",
                "target_fpir": target_fpir,
                "is_mated": True,
                "event_metric": "tpir_at_rank_k_loss",
                "paired_query_count": 10,
                "identity_count": 5,
                "frozen_event_count": 2,
                "frozen_event_rate": 0.2,
                "recalibrated_event_count": 1,
                "recalibrated_event_rate": 0.1,
                "recalibrated_minus_frozen_rate": -0.1,
                "resolved_event_count": 1,
                "introduced_event_count": 0,
                "paired_bootstrap_ci_low": -0.2,
                "paired_bootstrap_ci_high": 0.0,
            }
            for target_fpir in (0.01, 0.05, 0.10, 0.20, 0.30)
        ]
    ).to_csv(
        workflow / "saliency_threshold_policy_comparisons.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "analysis_scope": "threshold_policy_saliency_rho",
                "threshold_metric_derivation_version": (
                    "saliency-threshold-metrics-v1"
                ),
                "dataset_id": dataset,
                "model_uid": model_uid,
                "compression_family": "pca",
                "compression_profile": "pca_128",
                "search_mode": "pca_direct",
                "target_fpir": target_fpir,
                "is_mated": True,
                "saliency_feature": "saliency_entropy",
                "event_metric": "tpir_at_rank_k_loss",
                "paired_query_count": 10,
                "identity_count": 5,
                "frozen_event_count": 2,
                "recalibrated_event_count": 1,
                "frozen_spearman_rho": 0.2,
                "recalibrated_spearman_rho": 0.1,
                "recalibrated_minus_frozen_rho": -0.1,
                "paired_bootstrap_ci_low": -0.2,
                "paired_bootstrap_ci_high": 0.0,
            }
            for target_fpir in (0.01, 0.05, 0.10, 0.20, 0.30)
        ]
    ).to_csv(
        workflow / "saliency_threshold_policy_rho_comparisons.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "case_id": "case-1",
                "dataset_id": dataset,
                "model_uid": model_uid,
                "compression_family": "pca",
                "compression_profile": "pca_128",
                "search_mode": "pca_direct",
                "target_fpir": 0.10,
                "threshold_policy": "recalibrated_compressed",
                "case_group": "high_drift",
            }
        ]
    ).to_csv(workflow / "representative_cases.csv", index=False)
    return run_dir


def test_load_cross_dataset_saliency_associations_preserves_full_rows_and_lineage(
    tmp_path: Path,
) -> None:
    run_dir = _synthetic_completed_run(tmp_path)

    result = load_cross_dataset_saliency_associations(
        {"rfw_custom": run_dir},
        expected_model_uids={"rfw_custom": "arcface-test"},
        expected_run_ids={"rfw_custom": "20260810-R001-test"},
    )

    assert len(result.geometry) == 1
    assert len(result.retrieval) == 5
    assert len(result.threshold_instability) == 5
    assert len(result.threshold_policy_comparison) == 5
    assert len(result.threshold_policy_saliency_rho) == 5
    assert len(result.representative_cases) == 1
    assert set(result.retrieval["target_fpir"]) == {
        0.01,
        0.05,
        0.10,
        0.20,
        0.30,
    }
    assert set(result.geometry["run_id"]) == {"20260810-R001-test"}
    assert set(result.geometry["source_run_dir"]) == {str(run_dir.resolve())}
    assert set(result.source_files["rfw_custom"]) == {
        "run_manifest.json",
        "freeze_manifest.json",
        "step4_summary.json",
        "origin_calibration_diagnostics.json",
        "phase05_manifest.json",
        "phase06_manifest.json",
        "saliency_geometry_associations.csv",
        "saliency_retrieval_associations.csv",
        "saliency_threshold_instability_associations.csv",
        "saliency_threshold_policy_comparisons.csv",
        "saliency_threshold_policy_rho_comparisons.csv",
        "representative_cases.csv",
    }


def test_load_cross_dataset_saliency_associations_rejects_legacy_rfw_calibration(
    tmp_path: Path,
) -> None:
    run_dir = _synthetic_completed_run(tmp_path)
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["config"]["step4"]["evaluation"] = {
        "rfw_custom_calibration_gallery_identities": 80
    }
    _write_json(run_manifest_path, run_manifest)

    with pytest.raises(ValueError, match="predates gallery-size-matched"):
        load_cross_dataset_saliency_associations(
            {"rfw_custom": run_dir},
            expected_model_uids={"rfw_custom": "arcface-test"},
            expected_run_ids={"rfw_custom": "20260810-R001-test"},
        )


def test_validate_rfw_custom_calibration_contract_rejects_legacy_manifest(
    tmp_path: Path,
) -> None:
    run_dir = _synthetic_completed_run(tmp_path)
    run_manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        (
            run_dir
            / "artifacts"
            / "step2_workflow"
            / "origin_calibration_diagnostics.json"
        ).read_text(encoding="utf-8")
    )
    run_manifest["config"]["step4"]["evaluation"] = {
        "rfw_custom_calibration_gallery_identities": 80
    }

    with pytest.raises(ValueError, match="predates gallery-size-matched"):
        validate_rfw_custom_calibration_contract(run_manifest, diagnostics)


def test_load_cross_dataset_saliency_associations_rejects_selector_key_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = _synthetic_completed_run(tmp_path)

    with pytest.raises(ValueError, match="keys must exactly match"):
        load_cross_dataset_saliency_associations(
            {"rfw_custom": run_dir},
            expected_model_uids={"rfw_custom": "arcface-test", "lfw": "arcface-test"},
            expected_run_ids={"rfw_custom": "20260810-R001-test"},
        )


def test_cross_dataset_report_exports_full_saliency_association_contract() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[2]
        / "notebooks"
        / "common"
        / "reports"
        / "00_cross_dataset_results.ipynb"
    )
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )

    assert 'CROSS_DATASET_REPORT_PARAMETERS_INJECTED' in source
    assert 'if not PARAMETERS_INJECTED:' in source
    assert 'DATASETS = ("rfw_custom",)' in source
    assert 'globals().get("DATASETS"' not in source
    assert "keys must exactly match DATASETS" in source
    assert "load_cross_dataset_saliency_associations" in source
    assert "validate_rfw_custom_calibration_contract" in source
    assert '"report_ready": report_ready' in source
    assert 'EXPERIMENT_CANDIDATES["report_ready"]' in source
    assert "보고서 호환 불가" in source
    assert "saliency_geometry_associations_all.csv" in source
    assert "saliency_retrieval_associations_all.csv" in source
    assert "saliency_threshold_instability_associations_all.csv" in source
    assert "saliency_threshold_policy_comparisons_all.csv" in source
    assert "saliency_threshold_policy_rho_comparisons_all.csv" in source
    assert "expected_target_fpirs=tuple(TARGET_FPIRS)" in source
    assert "THRESHOLD_POLICY_COMPARISON_ALL" in source
    assert "representative_cases_all.csv" in source
    assert '"artifact_type": "cross_dataset_results"' in source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
