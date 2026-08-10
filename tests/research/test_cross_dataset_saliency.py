from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import nbformat
import pytest

from research.evaluation.cross_dataset_saliency import (
    load_cross_dataset_saliency_associations,
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
            "retrieval_association_rows": 2,
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
        _write_json(
            run_dir / "phases" / phase / "attempts" / "A001" / "phase_manifest.json",
            {"attempt": 1, "phase": phase, "status": "completed"},
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
                "search_mode": "pca_direct",
                "target_fpir": target_fpir,
                "threshold_policy": "recalibrated_compressed",
                "is_mated": True,
            }
            for target_fpir in (0.10, 0.01)
        ]
    ).to_csv(workflow / "saliency_retrieval_associations.csv", index=False)
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
    assert len(result.retrieval) == 2
    assert len(result.representative_cases) == 1
    assert set(result.retrieval["target_fpir"]) == {0.10, 0.01}
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
    assert "saliency_geometry_associations_all.csv" in source
    assert "saliency_retrieval_associations_all.csv" in source
    assert "representative_cases_all.csv" in source
    assert '"artifact_type": "cross_dataset_results"' in source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
