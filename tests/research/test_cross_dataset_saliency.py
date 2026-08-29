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
from research.evaluation.saliency_compression import (
    SALIENCY_THRESHOLD_METRICS_VERSION,
)
from research.runtime.hashing import sha256_file


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _phase05_manifest_path(run_dir: Path) -> Path:
    return (
        run_dir
        / "phases"
        / "05_saliency_compression_join"
        / "attempts"
        / "A001"
        / "phase_manifest.json"
    )


def _refresh_phase05_output_contract(run_dir: Path) -> None:
    workflow = run_dir / "artifacts" / "step2_workflow"
    phase05_path = _phase05_manifest_path(run_dir)
    phase05 = json.loads(phase05_path.read_text(encoding="utf-8"))
    phase05_outputs = (
        "saliency_geometry_associations.csv",
        "saliency_retrieval_associations.csv",
        "saliency_threshold_instability_associations.csv",
        "saliency_threshold_policy_comparisons.csv",
        "saliency_threshold_policy_rho_comparisons.csv",
        "representative_case_candidates.csv",
    )
    phase05["details"]["output_artifacts"] = {
        name: {
            "path": (Path("artifacts") / "step2_workflow" / name).as_posix(),
            "bytes": (workflow / name).stat().st_size,
            "sha256": sha256_file(workflow / name),
        }
        for name in phase05_outputs
    }
    _write_json(phase05_path, phase05)


def _synthetic_completed_run(
    tmp_path: Path,
    *,
    run_name: str = "run",
    run_id: str = "20260810-R001-test",
    dataset: str = "rfw_custom",
    model_uid: str = "arcface-test",
) -> Path:
    run_dir = tmp_path / run_name
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
                        SALIENCY_THRESHOLD_METRICS_VERSION
                    ),
                    "bootstrap_method": "identity_cluster",
                    "bootstrap_unit": "identity_id",
                    "bootstrap_repeats": 500,
                    "bootstrap_rank_strategy": "weighted_rerank",
                    "bootstrap_batch_size": 4,
                    "bootstrap_seed": 42,
                    "paired_saliency_features": ["saliency_entropy"],
                    "paired_event_metrics": ["threshold_crossing"],
                    "paired_minimum_event_count": 1,
                    "paired_confidence_level": 0.95,
                    "source_git_commit": "a" * 40,
                    "source_sha256": {
                        "association": "b" * 64,
                        "streaming_join": "c" * 64,
                        "workflow": "d" * 64,
                    },
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
                    SALIENCY_THRESHOLD_METRICS_VERSION
                ),
                "analysis_tier": "exploratory",
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
                    SALIENCY_THRESHOLD_METRICS_VERSION
                ),
                "analysis_tier": "prespecified_supporting",
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
                    SALIENCY_THRESHOLD_METRICS_VERSION
                ),
                "analysis_tier": "prespecified_supporting",
                "dataset_id": dataset,
                "model_uid": model_uid,
                "compression_family": "pca",
                "compression_profile": "pca_128",
                "search_mode": "pca_direct",
                "target_fpir": target_fpir,
                "is_mated": True,
                "event_metric": "threshold_crossing",
                "paired_query_count": 10,
                "identity_count": 5,
                "frozen_event_count": 2,
                "frozen_event_rate": 0.2,
                "recalibrated_event_count": 1,
                "recalibrated_event_rate": 0.1,
                "recalibrated_minus_frozen_rate": -0.1,
                "resolved_event_count": 1,
                "introduced_event_count": 0,
                "paired_bootstrap_confidence_level": 0.95,
                "paired_bootstrap_ci_low": -0.2,
                "paired_bootstrap_ci_high": 0.0,
                "paired_bootstrap_valid_repeats": 500,
                "bootstrap_unit": "identity_id",
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
                    SALIENCY_THRESHOLD_METRICS_VERSION
                ),
                "analysis_tier": "prespecified_primary",
                "dataset_id": dataset,
                "model_uid": model_uid,
                "compression_family": "pca",
                "compression_profile": "pca_128",
                "search_mode": "pca_direct",
                "target_fpir": target_fpir,
                "is_mated": True,
                "saliency_feature": "saliency_entropy",
                "event_metric": "threshold_crossing",
                "paired_query_count": 10,
                "identity_count": 5,
                "frozen_event_count": 2,
                "recalibrated_event_count": 1,
                "frozen_spearman_rho": 0.2,
                "recalibrated_spearman_rho": 0.1,
                "recalibrated_minus_frozen_rho": -0.1,
                "event_support_eligible": True,
                "minimum_event_count": 1,
                "paired_bootstrap_confidence_level": 0.95,
                "paired_bootstrap_ci_low": -0.2,
                "paired_bootstrap_ci_high": 0.0,
                "paired_bootstrap_valid_repeats": 500,
                "bootstrap_unit": "identity_id",
            }
            for target_fpir in (0.01, 0.05, 0.10, 0.20, 0.30)
        ]
    ).to_csv(
        workflow / "saliency_threshold_policy_rho_comparisons.csv",
        index=False,
    )
    case_frame = pd.DataFrame(
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
    )
    case_frame.to_csv(workflow / "representative_case_candidates.csv", index=False)
    case_frame.to_csv(workflow / "representative_cases.csv", index=False)
    _refresh_phase05_output_contract(run_dir)
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
        "representative_case_candidates.csv",
        "representative_cases.csv",
    }


def test_load_cross_dataset_saliency_rejects_missing_paired_support_field(
    tmp_path: Path,
) -> None:
    run_dir = _synthetic_completed_run(tmp_path)
    path = (
        run_dir
        / "artifacts"
        / "step2_workflow"
        / "saliency_threshold_policy_rho_comparisons.csv"
    )
    frame = pd.read_csv(path).drop(columns="event_support_eligible")
    frame.to_csv(path, index=False)
    _refresh_phase05_output_contract(run_dir)

    with pytest.raises(ValueError, match="event_support_eligible"):
        load_cross_dataset_saliency_associations(
            {"rfw_custom": run_dir},
            expected_model_uids={"rfw_custom": "arcface-test"},
            expected_run_ids={"rfw_custom": "20260810-R001-test"},
        )


def test_load_cross_dataset_saliency_rejects_inconsistent_paired_support(
    tmp_path: Path,
) -> None:
    run_dir = _synthetic_completed_run(tmp_path)
    path = (
        run_dir
        / "artifacts"
        / "step2_workflow"
        / "saliency_threshold_policy_rho_comparisons.csv"
    )
    frame = pd.read_csv(path)
    frame["event_support_eligible"] = False
    frame.to_csv(path, index=False)
    _refresh_phase05_output_contract(run_dir)

    with pytest.raises(ValueError, match="event_support_eligible is inconsistent"):
        load_cross_dataset_saliency_associations(
            {"rfw_custom": run_dir},
            expected_model_uids={"rfw_custom": "arcface-test"},
            expected_run_ids={"rfw_custom": "20260810-R001-test"},
        )


def test_load_cross_dataset_saliency_rejects_inconsistent_policy_counts(
    tmp_path: Path,
) -> None:
    run_dir = _synthetic_completed_run(tmp_path)
    path = (
        run_dir
        / "artifacts"
        / "step2_workflow"
        / "saliency_threshold_policy_comparisons.csv"
    )
    frame = pd.read_csv(path)
    frame["introduced_event_count"] = 1
    frame.to_csv(path, index=False)
    _refresh_phase05_output_contract(run_dir)

    with pytest.raises(ValueError, match="resolved/introduced counts"):
        load_cross_dataset_saliency_associations(
            {"rfw_custom": run_dir},
            expected_model_uids={"rfw_custom": "arcface-test"},
            expected_run_ids={"rfw_custom": "20260810-R001-test"},
        )


def test_load_cross_dataset_saliency_respects_mated_event_applicability(
    tmp_path: Path,
) -> None:
    run_dir = _synthetic_completed_run(tmp_path)
    workflow = run_dir / "artifacts" / "step2_workflow"
    phase05_path = _phase05_manifest_path(run_dir)
    phase05 = json.loads(phase05_path.read_text(encoding="utf-8"))
    phase05["details"]["implementation"]["paired_event_metrics"] = [
        "threshold_crossing",
        "false_accept_gain",
        "tpir_threshold_loss",
    ]
    _write_json(phase05_path, phase05)

    retrieval_path = workflow / "saliency_retrieval_associations.csv"
    retrieval = pd.read_csv(retrieval_path)
    retrieval_with_non_mated = pd.concat(
        (retrieval, retrieval.assign(is_mated=False)),
        ignore_index=True,
    )
    retrieval_with_non_mated.to_csv(retrieval_path, index=False)

    for filename in (
        "saliency_threshold_policy_comparisons.csv",
        "saliency_threshold_policy_rho_comparisons.csv",
    ):
        path = workflow / filename
        base = pd.read_csv(path)
        mated_tpir = base.copy()
        mated_tpir["event_metric"] = "tpir_threshold_loss"
        non_mated_crossing = base.copy()
        non_mated_crossing["is_mated"] = False
        non_mated_false_accept = non_mated_crossing.copy()
        non_mated_false_accept["event_metric"] = "false_accept_gain"
        combined = pd.concat(
            (base, mated_tpir, non_mated_crossing, non_mated_false_accept),
            ignore_index=True,
        )
        combined.to_csv(path, index=False)

    summary_path = workflow / "step4_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["retrieval_association_rows"] = 10
    summary["threshold_policy_comparison_rows"] = 20
    summary["threshold_policy_saliency_rho_rows"] = 20
    _write_json(summary_path, summary)
    _refresh_phase05_output_contract(run_dir)

    result = load_cross_dataset_saliency_associations(
        {"rfw_custom": run_dir},
        expected_model_uids={"rfw_custom": "arcface-test"},
        expected_run_ids={"rfw_custom": "20260810-R001-test"},
    )

    assert len(result.threshold_policy_comparison) == 20
    assert len(result.threshold_policy_saliency_rho) == 20


def test_load_cross_dataset_saliency_rejects_per_grain_fpir_gap(
    tmp_path: Path,
) -> None:
    run_dir = _synthetic_completed_run(tmp_path)
    workflow = run_dir / "artifacts" / "step2_workflow"
    path = workflow / "saliency_retrieval_associations.csv"
    frame = pd.read_csv(path)
    second_grain = frame.loc[frame["target_fpir"].ne(0.30)].copy()
    second_grain["saliency_feature"] = "outside_face_attention"
    combined = pd.concat((frame, second_grain), ignore_index=True)
    combined.to_csv(path, index=False)
    summary_path = workflow / "step4_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["retrieval_association_rows"] = len(combined)
    _write_json(summary_path, summary)
    _refresh_phase05_output_contract(run_dir)

    with pytest.raises(ValueError, match="scientific grain"):
        load_cross_dataset_saliency_associations(
            {"rfw_custom": run_dir},
            expected_model_uids={"rfw_custom": "arcface-test"},
            expected_run_ids={"rfw_custom": "20260810-R001-test"},
        )


def test_load_cross_dataset_saliency_rejects_missing_paired_base_grain(
    tmp_path: Path,
) -> None:
    run_dir = _synthetic_completed_run(tmp_path)
    workflow = run_dir / "artifacts" / "step2_workflow"
    path = workflow / "saliency_retrieval_associations.csv"
    frame = pd.read_csv(path)
    second_profile = frame.assign(compression_profile="pca_64")
    combined = pd.concat((frame, second_profile), ignore_index=True)
    combined.to_csv(path, index=False)
    summary_path = workflow / "step4_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["retrieval_association_rows"] = len(combined)
    _write_json(summary_path, summary)
    _refresh_phase05_output_contract(run_dir)

    with pytest.raises(ValueError, match="paired base grain coverage mismatch"):
        load_cross_dataset_saliency_associations(
            {"rfw_custom": run_dir},
            expected_model_uids={"rfw_custom": "arcface-test"},
            expected_run_ids={"rfw_custom": "20260810-R001-test"},
        )


def test_load_cross_dataset_saliency_rejects_ci_without_paired_point_rho(
    tmp_path: Path,
) -> None:
    run_dir = _synthetic_completed_run(tmp_path)
    path = (
        run_dir
        / "artifacts"
        / "step2_workflow"
        / "saliency_threshold_policy_rho_comparisons.csv"
    )
    frame = pd.read_csv(path)
    frame[[
        "frozen_spearman_rho",
        "recalibrated_spearman_rho",
        "recalibrated_minus_frozen_rho",
    ]] = float("nan")
    frame.to_csv(path, index=False)
    _refresh_phase05_output_contract(run_dir)

    with pytest.raises(ValueError, match="bootstrap evidence requires paired point rho"):
        load_cross_dataset_saliency_associations(
            {"rfw_custom": run_dir},
            expected_model_uids={"rfw_custom": "arcface-test"},
            expected_run_ids={"rfw_custom": "20260810-R001-test"},
        )


@pytest.mark.parametrize(
    ("filename", "invalid_tier"),
    [
        ("saliency_retrieval_associations.csv", "prespecified_primary"),
        ("saliency_threshold_instability_associations.csv", "exploratory"),
    ],
)
def test_load_cross_dataset_saliency_rejects_analysis_tier_drift(
    tmp_path: Path,
    filename: str,
    invalid_tier: str,
) -> None:
    run_dir = _synthetic_completed_run(tmp_path)
    path = run_dir / "artifacts" / "step2_workflow" / filename
    frame = pd.read_csv(path)
    frame["analysis_tier"] = invalid_tier
    frame.to_csv(path, index=False)
    _refresh_phase05_output_contract(run_dir)

    with pytest.raises(ValueError, match="analysis_tier is inconsistent"):
        load_cross_dataset_saliency_associations(
            {"rfw_custom": run_dir},
            expected_model_uids={"rfw_custom": "arcface-test"},
            expected_run_ids={"rfw_custom": "20260810-R001-test"},
        )


def test_load_cross_dataset_saliency_rejects_phase05_output_hash_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = _synthetic_completed_run(tmp_path)
    path = (
        run_dir
        / "artifacts"
        / "step2_workflow"
        / "saliency_retrieval_associations.csv"
    )
    original = path.read_text(encoding="utf-8")
    tampered = original.replace("0.01", "0.02", 1)
    assert len(tampered.encode("utf-8")) == len(original.encode("utf-8"))
    path.write_text(tampered, encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_cross_dataset_saliency_associations(
            {"rfw_custom": run_dir},
            expected_model_uids={"rfw_custom": "arcface-test"},
            expected_run_ids={"rfw_custom": "20260810-R001-test"},
        )


def test_load_cross_dataset_saliency_rejects_mixed_implementation_fingerprint(
    tmp_path: Path,
) -> None:
    rfw_run = _synthetic_completed_run(
        tmp_path,
        run_name="rfw-run",
    )
    lfw_run = _synthetic_completed_run(
        tmp_path,
        run_name="lfw-run",
        run_id="20260810-R002-test",
        dataset="lfw",
        model_uid="adaface-test",
    )
    phase05_path = _phase05_manifest_path(lfw_run)
    phase05 = json.loads(phase05_path.read_text(encoding="utf-8"))
    phase05["details"]["implementation"]["source_git_commit"] = "e" * 40
    _write_json(phase05_path, phase05)

    with pytest.raises(ValueError, match="mixed Phase05 implementation fingerprint"):
        load_cross_dataset_saliency_associations(
            {"rfw_custom": rfw_run, "lfw": lfw_run},
            expected_model_uids={
                "rfw_custom": "arcface-test",
                "lfw": "adaface-test",
            },
            expected_run_ids={
                "rfw_custom": "20260810-R001-test",
                "lfw": "20260810-R002-test",
            },
        )


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
    assert "inspect_step4_retrieval_source" in source
    assert "retrieval_source is not None" in source
    assert 'retrieval_source["kind"]' in source
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
