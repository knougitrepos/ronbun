from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd

from research.runtime.hashing import sha256_file

from scripts.run_integrated_postprocessing import (
    REPORT_MODEL_NAMES,
    build_report_parameter_source,
    inspect_step4_retrieval_source,
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


def _completed_tinyface_run(
    root: Path,
    *,
    run_id: str,
    model_uid: str,
) -> Path:
    run_dir = root / "tinyface" / run_id
    output_root = run_dir / "artifacts/tinyface_official"
    output_root.mkdir(parents=True)
    (run_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"status": "completed", "run_id": run_id}),
        encoding="utf-8",
    )
    summary_path = output_root / "condition_summary.csv"
    per_query_path = output_root / "per_query.csv"
    pd.DataFrame(
        {
            "model_uid": [model_uid],
            "fpir_tpir_metrics_applicable": [False],
            "mean_average_precision": [0.5],
        }
    ).to_csv(summary_path, index=False)
    pd.DataFrame(
        {"query_image_id": ["probe"], "average_precision": [0.5]}
    ).to_csv(per_query_path, index=False)
    outputs = {
        path.name: {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (summary_path, per_query_path)
    }
    (output_root / "tinyface_evaluation_manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "tinyface_official_compression_evaluation_v1",
                "source_run_id": run_id,
                "dataset_id": "tinyface",
                "model_uid": model_uid,
                "open_set_protocol": False,
                "outputs": outputs,
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
    assert result["search_space_v6_query_gallery_conditions"] == {
        "status": "disabled"
    }
    assert result["faithfulness"] == {"status": "disabled"}


def test_report_candidate_accepts_results_only_retrieval_ledger(
    tmp_path: Path,
) -> None:
    run_dir = _completed_run(
        tmp_path,
        dataset_id="lfw",
        run_id="L001",
        model_uid="arcface-test",
    )
    ledger_root = run_dir / "artifacts/step2_workflow/retrieval_ledger"
    core = ledger_root / "core/condition.parquet"
    decision = ledger_root / "decisions/policy.parquet"
    core.parent.mkdir(parents=True)
    decision.parent.mkdir(parents=True)
    core.write_bytes(b"core")
    decision.write_bytes(b"decision")
    manifest_path = ledger_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "normalized_retrieval_ledger",
                "status": "completed",
                "logical_row_count": 30,
                "lineage": {
                    "dataset_id": "lfw",
                    "model_uid": "arcface-test",
                },
                "conditions": [
                    {
                        "core": {"path": "core/condition.parquet", "bytes": 4},
                        "decisions": [
                            {
                                "artifact": {
                                    "path": "decisions/policy.parquet",
                                    "bytes": 8,
                                }
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    inspected = inspect_step4_retrieval_source(
        run_dir,
        expected_dataset_id="lfw",
        expected_model_uid="arcface-test",
        expected_logical_rows=30,
    )

    assert inspected["kind"] == "normalized_retrieval_ledger"
    assert inspected["logical_row_count"] == 30
    assert inspected["bytes"] == manifest_path.stat().st_size + 12


def test_report_candidate_rejects_retrieval_ledger_lineage_drift(
    tmp_path: Path,
) -> None:
    run_dir = _completed_run(
        tmp_path,
        dataset_id="lfw",
        run_id="L001",
        model_uid="arcface-test",
    )
    ledger_root = run_dir / "artifacts/step2_workflow/retrieval_ledger"
    ledger_root.mkdir(parents=True)
    (ledger_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "normalized_retrieval_ledger",
                "status": "completed",
                "logical_row_count": 30,
                "lineage": {
                    "dataset_id": "survface",
                    "model_uid": "arcface-test",
                },
                "conditions": [{}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset mismatch"):
        inspect_step4_retrieval_source(
            run_dir,
            expected_dataset_id="lfw",
            expected_model_uid="arcface-test",
            expected_logical_rows=30,
        )


def test_postprocess_and_report_accept_rfw_custom_open_set_run(
    tmp_path: Path,
) -> None:
    rfw_custom = _completed_run(
        tmp_path,
        dataset_id="rfw_custom",
        run_id="RFWC001",
        model_uid="edgeface-test",
    )

    result = postprocess_completed_run(
        rfw_custom,
        refresh_search_spaces=False,
        derive_survface_faithfulness=False,
    )
    source, identities = build_report_parameter_source(
        model_name="edge",
        selected_runs={"rfw_custom": rfw_custom},
        include_faithfulness=True,
        write_outputs=False,
        overwrite_outputs=False,
    )

    assert result["source"]["dataset_id"] == "rfw_custom"
    assert identities["rfw_custom"]["run_id"] == "RFWC001"
    assert "DATASETS = ('rfw_custom',)" in source
    assert "INCLUDE_FAITHFULNESS = True" in source


def test_report_parameter_source_uses_verified_run_identity(tmp_path: Path):
    lfw = _completed_run(tmp_path, dataset_id="lfw", run_id="L001")
    survface = _completed_run(tmp_path, dataset_id="survface", run_id="S001")

    source, identities = build_report_parameter_source(
        model_name="mag",
        selected_runs={"lfw": lfw, "survface": survface},
        include_faithfulness=True,
        write_outputs=False,
        overwrite_outputs=False,
    )

    assert identities["lfw"]["run_id"] == "L001"
    assert identities["survface"]["run_id"] == "S001"
    assert "MODEL_NAME = 'magface'" in source
    assert source.startswith("CROSS_DATASET_REPORT_PARAMETERS_INJECTED = True\n")
    assert "WRITE_OUTPUTS = False" in source
    assert "GENERATE_MISSING_SEARCH_CONDITION_ARTIFACTS = False" in source
    assert "INCLUDE_FAITHFULNESS = True" in source
    assert "RFW_EVALUATION_DIR = None" in source


def test_report_parameter_source_accepts_verified_matching_rfw_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lfw = _completed_run(
        tmp_path,
        dataset_id="lfw",
        run_id="L001",
        model_uid="edgeface-test",
    )
    rfw_dir = tmp_path / "rfw-evaluation"
    rfw_dir.mkdir()
    monkeypatch.setattr(
        "research.experiments.load_rfw_frozen_codec_evaluation",
        lambda path: SimpleNamespace(
            root=rfw_dir.resolve(),
            manifest={"model_uid": "edgeface-test"},
        ),
    )

    source, _ = build_report_parameter_source(
        model_name="edge",
        selected_runs={"lfw": lfw},
        include_faithfulness=False,
        write_outputs=False,
        overwrite_outputs=False,
        rfw_evaluation_dir=rfw_dir,
    )

    assert f"RFW_EVALUATION_DIR = {str(rfw_dir.resolve())!r}" in source


def test_report_parameter_source_accepts_matching_tinyface_supplement(
    tmp_path: Path,
) -> None:
    lfw = _completed_run(
        tmp_path,
        dataset_id="lfw",
        run_id="L001",
        model_uid="edgeface-test",
    )
    tinyface = _completed_tinyface_run(
        tmp_path,
        run_id="T001",
        model_uid="edgeface-test",
    )

    source, identities = build_report_parameter_source(
        model_name="edge",
        selected_runs={"lfw": lfw, "tinyface": tinyface},
        include_faithfulness=False,
        write_outputs=False,
        overwrite_outputs=False,
    )

    assert f"TINYFACE_EVALUATION_DIR = {str(tinyface.resolve())!r}" in source
    assert identities["tinyface"]["run_id"] == "T001"
    assert "tinyface" not in eval(
        source.split("DATASETS = ", 1)[1].splitlines()[0]
    )


def test_report_parameter_source_rejects_dataset_key_mismatch(tmp_path: Path):
    lfw = _completed_run(tmp_path, dataset_id="lfw", run_id="L001")

    with pytest.raises(ValueError, match="key/manifest mismatch"):
        build_report_parameter_source(
            model_name="magface",
            selected_runs={"survface": lfw},
            include_faithfulness=False,
            write_outputs=False,
            overwrite_outputs=False,
        )


def test_edgeface_is_a_supported_report_model_alias() -> None:
    assert REPORT_MODEL_NAMES["edge"] == "edgeface"
    assert REPORT_MODEL_NAMES["edgeface"] == "edgeface"


def test_report_parameter_source_requires_explicit_complete_four_by_three_matrix(
    tmp_path: Path,
) -> None:
    matrix = {}
    for model_name in ("arcface", "adaface", "magface", "edgeface"):
        matrix[model_name] = {
            dataset: _completed_run(
                tmp_path,
                dataset_id=dataset,
                run_id=f"{model_name}-{dataset}",
                model_uid=f"{model_name}-test",
            )
            for dataset in ("lfw", "survface", "rfw_custom")
        }
    source, _ = build_report_parameter_source(
        model_name="edgeface",
        selected_runs={"lfw": matrix["edgeface"]["lfw"]},
        include_faithfulness=False,
        write_outputs=False,
        overwrite_outputs=False,
        cross_model_run_matrix=matrix,
    )

    assert "REQUIRE_COMPLETE_MODEL_MATRIX = True" in source
    assert "MODEL_RUN_MATRIX =" in source
    assert all(model_name in source for model_name in matrix)
    assert all(dataset in source for dataset in matrix["arcface"])

    incomplete = dict(matrix)
    incomplete.pop("arcface")
    with pytest.raises(ValueError, match="requires ArcFace"):
        build_report_parameter_source(
            model_name="edgeface",
            selected_runs={"lfw": matrix["edgeface"]["lfw"]},
            include_faithfulness=False,
            write_outputs=False,
            overwrite_outputs=False,
            cross_model_run_matrix=incomplete,
        )
