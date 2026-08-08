from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_integrated_postprocessing import (
    REPORT_MODEL_NAMES,
    build_report_parameter_source,
    postprocess_completed_run,
)


def _completed_run(
    root: Path,
    *,
    dataset_id: str,
    run_id: str,
    model_uid: str = "magface-test",
) -> Path:
    run_dir = root / dataset_id / run_id
    workflow = run_dir / "artifacts/step2_workflow"
    workflow.mkdir(parents=True)
    (run_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"status": "completed", "run_id": run_id}),
        encoding="utf-8",
    )
    (workflow / "freeze_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "run_id": run_id,
                "model_uid": model_uid,
                "fallback_free": True,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_postprocess_can_validate_completed_run_without_derived_work(tmp_path: Path):
    run_dir = _completed_run(
        tmp_path,
        dataset_id="lfw",
        run_id="R001",
    )

    result = postprocess_completed_run(
        run_dir,
        refresh_search_spaces=False,
        derive_survface_faithfulness=False,
    )

    assert result["status"] == "completed"
    assert result["source"]["dataset_id"] == "lfw"
    assert result["search_space_v3_matched_calibration"] == {
        "status": "disabled"
    }
    assert result["survface_faithfulness"] == {"status": "not_applicable"}


def test_report_parameter_source_uses_verified_run_identity(tmp_path: Path):
    lfw = _completed_run(tmp_path, dataset_id="lfw", run_id="L001")
    survface = _completed_run(tmp_path, dataset_id="survface", run_id="S001")

    source, identities = build_report_parameter_source(
        model_name="mag",
        selected_runs={"lfw": lfw, "survface": survface},
        include_survface_faithfulness=True,
        write_outputs=False,
        overwrite_outputs=False,
    )

    assert identities["lfw"]["run_id"] == "L001"
    assert identities["survface"]["run_id"] == "S001"
    assert "MODEL_NAME = 'magface'" in source
    assert "WRITE_OUTPUTS = False" in source
    assert "INCLUDE_SURVFACE_FAITHFULNESS = True" in source


def test_report_parameter_source_rejects_dataset_key_mismatch(tmp_path: Path):
    lfw = _completed_run(tmp_path, dataset_id="lfw", run_id="L001")

    with pytest.raises(ValueError, match="key/manifest mismatch"):
        build_report_parameter_source(
            model_name="magface",
            selected_runs={"survface": lfw},
            include_survface_faithfulness=False,
            write_outputs=False,
            overwrite_outputs=False,
        )


def test_edgeface_is_a_supported_report_model_alias() -> None:
    assert REPORT_MODEL_NAMES["edge"] == "edgeface"
    assert REPORT_MODEL_NAMES["edgeface"] == "edgeface"
