from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"

EXPECTED_NOTEBOOKS = {
    "lfw/00_data_preparation": {
        "00_data_preparation.ipynb",
        "01_aligned_crop_materialization.ipynb",
        "02_landmark_region_materialization.ipynb",
    },
    "lfw/01_embeddings": {
        "00_protocol_and_run_freeze.ipynb",
        "01_arcface_embedding_extraction.ipynb",
    },
    "lfw/02_compression": {
        "00_compressor_fit.ipynb",
        "01_compressed_materialization_and_index.ipynb",
        "02_step1_compression_characterization.ipynb",
    },
    "lfw/03_open_set": {
        "00_probe_search_and_certification.ipynb",
        "01_evaluation_and_visualization.ipynb",
    },
    "lfw/04_gradcam/prerequisite": {
        "00_source_and_model_freeze.ipynb",
        "01_origin_embedding_and_loo_templates.ipynb",
    },
    "lfw/04_gradcam/experiment": {
        "00_population_gradcam_extraction.ipynb",
        "01_saliency_feature_validation.ipynb",
        "02_step2_compression_characterization.ipynb",
        "03_saliency_compression_join.ipynb",
        "04_representative_case_visualization.ipynb",
    },
    "survface/00_data_preparation": {
        "00_data_preparation.ipynb",
        "01_aligned_crop_materialization.ipynb",
        "02_landmark_region_materialization.ipynb",
    },
    "survface/01_embeddings": {
        "00_official_protocol_and_run_freeze.ipynb",
        "01_official_arcface_embedding_extraction.ipynb",
    },
    "survface/02_compression": {
        "00_compressor_fit.ipynb",
        "01_official_compressed_materialization_and_index.ipynb",
        "02_step1_compression_characterization.ipynb",
    },
    "survface/03_open_set": {
        "00_official_probe_search.ipynb",
        "01_official_evaluation_and_visualization.ipynb",
    },
    "survface/04_gradcam/prerequisite": {
        "00_source_and_model_freeze.ipynb",
        "01_origin_embedding_and_top1_gallery_templates.ipynb",
    },
    "survface/04_gradcam/experiment": {
        "00_population_gradcam_extraction.ipynb",
        "01_saliency_feature_validation.ipynb",
        "02_step2_compression_characterization.ipynb",
        "03_saliency_compression_join.ipynb",
        "04_representative_case_visualization.ipynb",
    },
    "rfw/00_data_preparation": {"00_data_preparation.ipynb"},
    "balancedface/00_data_preparation": {"00_data_preparation.ipynb"},
    "common/model_preparation": {
        "00_checkpoint_registration.ipynb",
        "01_preprocessing_and_model_smoke.ipynb",
    },
    "common/reports": {"00_cross_dataset_results.ipynb"},
    "common/maintenance": {"00_selective_cleanup.ipynb"},
    "common/orchestration": {"00_batch_experiment_runner.ipynb"},
}


def _source(path: Path) -> str:
    notebook = nbformat.read(path, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)


def test_notebooks_are_grouped_by_dataset_then_execution_stage() -> None:
    assert not list(NOTEBOOK_ROOT.glob("*.ipynb"))
    assert not (NOTEBOOK_ROOT / "step4").exists()
    actual_directories = {
        path.parent.relative_to(NOTEBOOK_ROOT).as_posix()
        for path in NOTEBOOK_ROOT.rglob("*.ipynb")
    }
    assert actual_directories == set(EXPECTED_NOTEBOOKS)
    for relative, expected_names in EXPECTED_NOTEBOOKS.items():
        directory = NOTEBOOK_ROOT / relative
        assert {path.name for path in directory.glob("*.ipynb")} == expected_names


def test_step4_notebooks_are_dataset_specific_single_stage_runbooks() -> None:
    expected = {
        "00_source_and_model_freeze.ipynb": "freeze_step4_source_and_model",
        "00_population_gradcam_extraction.ipynb": (
            "extract_step4_population_gradcam"
        ),
        "01_saliency_feature_validation.ipynb": "validate_step4_saliency",
        "02_step2_compression_characterization.ipynb": (
            "characterize_step4_compression"
        ),
        "03_saliency_compression_join.ipynb": (
            "analyze_step4_saliency_compression"
        ),
        "04_representative_case_visualization.ipynb": (
            "finalize_step4_representative_cases"
        ),
    }
    origin_names = {
        "lfw": "01_origin_embedding_and_loo_templates.ipynb",
        "survface": "01_origin_embedding_and_top1_gallery_templates.ipynb",
    }
    stage_functions = {
        *expected.values(),
        "extract_step4_origin_embeddings",
    }
    for dataset_id in ("lfw", "survface"):
        root = NOTEBOOK_ROOT / dataset_id / "04_gradcam"
        notebooks = [
            *(root / "prerequisite").glob("*.ipynb"),
            *(root / "experiment").glob("*.ipynb"),
        ]
        for path in notebooks:
            source = _source(path)
            expected_function = (
                "extract_step4_origin_embeddings"
                if path.name == origin_names[dataset_id]
                else expected[path.name]
            )
            assert f'DATASET_ID = "{dataset_id}"' in source
            assert expected_function in source
            assert "execution_acknowledged=True" in source
            assert sum(name in source for name in stage_functions) == 1
            assert "run_step4_experiment" not in source


def test_common_orchestration_notebook_preserves_quick_full_contract() -> None:
    path = (
        NOTEBOOK_ROOT
        / "common"
        / "orchestration"
        / "00_batch_experiment_runner.ipynb"
    )
    source = _source(path)

    for phrase in (
        'DATASET_ID = "survface"',
        'RUN_TIER = "quick"',
        "QUICK_DATA_FRACTIONS",
        "FULL_DATA_FRACTION",
        "build_common_experiment_plan",
        "inspect_common_experiment_plan",
        "run_common_step4_experiment",
        "ACKNOWLEDGE_LOCAL_EXECUTION = False",
        "milestone_percent=10",
        "heartbeat_seconds=None",
    ):
        assert phrase in source
    assert "DATA_FRACTION =" not in source
    assert "%run" not in source


def test_survface_saliency_join_runbook_exposes_long_run_progress() -> None:
    path = (
        NOTEBOOK_ROOT
        / "survface"
        / "04_gradcam"
        / "experiment"
        / "03_saliency_compression_join.ipynb"
    )
    source = _source(path)

    assert "ProgressReporter" in source
    assert "heartbeat_seconds=60" in source
    assert '"bootstrap_batch_size": 4' in source
    assert 'bootstrap_rank_strategy": "weighted_rerank"' in source
    assert 'progress=PROGRESS.callback(key_prefix="step4-05:")' in source


def test_all_notebooks_are_valid_restartable_and_output_free() -> None:
    for path in sorted(NOTEBOOK_ROOT.rglob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert notebook.metadata["ronbun"]["restart_policy"] == (
            "restart_kernel_and_run_all"
        )
        assert notebook.metadata["ronbun"]["outputs_committed"] is False
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            compile(cell.source, f"{path.name}:cell-{index}", "exec")
            assert cell.execution_count is None
            assert cell.outputs == []


def test_experiment_defaults_use_full_data_and_execute() -> None:
    for path in sorted(NOTEBOOK_ROOT.rglob("*.ipynb")):
        if "maintenance" in path.parts:
            continue
        source = _source(path)
        if "DATA_FRACTION =" in source:
            assert (
                "DATA_FRACTION = 1.0" in source
                or 'DATA_FRACTION = float(EXECUTION["data_fraction"])' in source
            )
        if "EXECUTE_STAGE =" in source:
            assert (
                "EXECUTE_STAGE = True" in source
                or 'EXECUTE_STAGE = bool(EXECUTION["execute_stage"])' in source
            )
        if "WRITE_OUTPUTS =" in source:
            assert (
                "WRITE_OUTPUTS = True" in source
                or 'WRITE_OUTPUTS = bool(EXECUTION["write_outputs"])' in source
            )
        if "OVERWRITE =" in source:
            assert (
                "OVERWRITE = True" in source
                or 'OVERWRITE = bool(EXECUTION["overwrite"])' in source
            )


def test_survface_notebooks_preserve_official_protocol_boundaries() -> None:
    paths = [
        *(NOTEBOOK_ROOT / "survface" / "01_embeddings").glob("*.ipynb"),
        *(NOTEBOOK_ROOT / "survface" / "02_compression").glob("*.ipynb"),
        *(NOTEBOOK_ROOT / "survface" / "03_open_set").glob("*.ipynb"),
    ]
    sources = "\n".join(_source(path) for path in paths)
    for phrase in (
        "official_all",
        "known_unknown",
        "TPIR",
        "FPIR",
        "development",
        "fit_on_survface_official_test",
    ):
        assert phrase in sources


def test_step1_characterization_and_report_remain_fallback_free() -> None:
    for dataset in ("lfw", "survface"):
        path = (
            NOTEBOOK_ROOT
            / dataset
            / "02_compression"
            / "02_step1_compression_characterization.ipynb"
        )
        source = _source(path)
        phrases = [
            "paired_embedding_metrics",
            "compare_cosine_retrieval",
            "origin_fallback_used",
            "storage_bytes_per_embedding",
            "codebook_bytes",
        ]
        if dataset == "survface":
            phrases.extend(
                [
                    "apply_retrieval_thresholds",
                    "load_survface_compressor_bundle",
                ]
            )
        else:
            phrases.extend(
                [
                    "origin_threshold=origin_threshold",
                    "compressed_threshold=operating_compressed_threshold",
                ]
            )
        for phrase in phrases:
            assert phrase in source

    lfw_open_set = _source(
        NOTEBOOK_ROOT
        / "lfw"
        / "03_open_set"
        / "00_probe_search_and_certification.ipynb"
    )
    assert "EXECUTE_LEGACY_FALLBACK" not in lfw_open_set
    assert "origin_fallback_used" in lfw_open_set
    assert "verified_read_only" in lfw_open_set
    assert "dir_rank1_recomputed" in lfw_open_set
    assert "fpir_recomputed" in lfw_open_set

    lfw_visualization = _source(
        NOTEBOOK_ROOT
        / "lfw"
        / "03_open_set"
        / "01_evaluation_and_visualization.ipynb"
    )
    assert "visualized_read_only" in lfw_visualization
    assert "origin_fallback_used" in lfw_visualization

    report = _source(
        NOTEBOOK_ROOT / "common" / "reports" / "00_cross_dataset_results.ipynb"
    )
    for phrase in (
        "origin_fallback_used",
        "origin_decision_threshold",
        "compressed_decision_threshold",
        "profile_storage schema mismatch",
    ):
        assert phrase in report
