from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.summarize_survface_origin_fpir_models import SUMMARY_COLUMNS, summarize


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diagnostic_bundle(
    root: Path,
    *,
    model_uid: str,
    run_id: str,
    target_fpir: float = 0.1,
) -> Path:
    directory = root / model_uid
    directory.mkdir(parents=True)
    row = {column: 0 for column in SUMMARY_COLUMNS}
    row.update(
        {
            "dataset_id": "survface",
            "model_uid": model_uid,
            "protocol_uid": "protocol-1",
            "calibration_condition_uid": "gallery-200_enrollment-1",
            "seed": 42,
            "target_fpir": target_fpir,
            "calibration_gallery_identity_count": 200,
            "calibration_enrollment_count": 1,
            "origin_decision_threshold": 0.6,
            "calibration_non_mated_count": 1000,
            "calibration_false_accept_count": 100,
            "calibration_fpir": 0.1,
            "calibration_fpir_wilson95_low": 0.08,
            "calibration_fpir_wilson95_high": 0.12,
            "matched_test_non_mated_count": 1000,
            "matched_test_false_accept_count": 110,
            "matched_test_fpir": 0.11,
            "matched_test_fpir_wilson95_low": 0.09,
            "matched_test_fpir_wilson95_high": 0.13,
            "test_gallery_template_count": 3000,
            "test_gallery_source_image_count": 60000,
            "test_non_mated_count": 120000,
            "test_false_accept_count": 60000,
            "test_fpir": 0.5,
            "test_fpir_wilson95_low": 0.497,
            "test_fpir_wilson95_high": 0.503,
            "test_to_target_fpir_ratio": 5.0,
        }
    )
    summary_path = directory / "origin_calibration_sweep_summary.csv"
    pd.DataFrame([row]).to_csv(summary_path, index=False)
    manifest = {
        "status": "completed",
        "source": {"model_uid": model_uid, "run_id": run_id},
        "parameters": {
            "conditions": [
                {"gallery_identity_count": 200, "enrollment_count": 1}
            ],
            "target_fpir": target_fpir,
            "seed": 42,
            "test_threshold_recalibration": False,
        },
        "outputs": {"condition_summary": summary_path.name},
        "output_sha256": {summary_path.name: _sha256(summary_path)},
    }
    (directory / "diagnostic_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return directory


def test_cross_model_summary_validates_and_combines_comparable_bundles(
    tmp_path: Path,
) -> None:
    arcface = _diagnostic_bundle(
        tmp_path,
        model_uid="arcface-uid",
        run_id="arc-run",
    )
    adaface = _diagnostic_bundle(
        tmp_path,
        model_uid="adaface-uid",
        run_id="ada-run",
    )
    output_dir = tmp_path / "output"

    manifest = summarize((arcface, adaface), output_dir)
    combined = pd.read_csv(output_dir / "cross_model_origin_fpir_summary.csv")

    assert manifest["status"] == "completed"
    assert manifest["model_count"] == 2
    assert manifest["row_count"] == 2
    assert set(combined["model_family"]) == {"arcface", "adaface"}
    assert set(combined["source_run_id"]) == {"arc-run", "ada-run"}


def test_cross_model_summary_rejects_parameter_mismatch(tmp_path: Path) -> None:
    arcface = _diagnostic_bundle(
        tmp_path,
        model_uid="arcface-uid",
        run_id="arc-run",
    )
    adaface = _diagnostic_bundle(
        tmp_path,
        model_uid="adaface-uid",
        run_id="ada-run",
        target_fpir=0.01,
    )

    with pytest.raises(ValueError, match="different audit parameters"):
        summarize((arcface, adaface), tmp_path / "output")
