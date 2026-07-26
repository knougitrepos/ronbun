from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"

EXPECTED_NOTEBOOKS = {
    "prerequisite/datasets": {
        "00_lfw_data_preparation.ipynb",
        "01_survface_data_preparation.ipynb",
        "02_rfw_data_preparation.ipynb",
        "03_balancedface_data_preparation.ipynb",
        "04_lfw_aligned_crop_materialization.ipynb",
    },
    "prerequisite/models": {
        "00_checkpoint_registration.ipynb",
        "01_preprocessing_and_model_smoke.ipynb",
    },
    "prerequisite/embeddings/lfw": {
        "00_protocol_and_run_freeze.ipynb",
        "01_arcface_embedding_extraction.ipynb",
    },
    "prerequisite/embeddings/survface": {
        "00_official_protocol_and_run_freeze.ipynb",
        "01_official_arcface_embedding_extraction.ipynb",
    },
    "experiments/compression/lfw": {
        "00_compressor_fit.ipynb",
        "01_compressed_materialization_and_index.ipynb",
        "02_step1_compression_characterization.ipynb",
    },
    "experiments/compression/survface": {
        "00_external_compressor_import.ipynb",
        "01_official_compressed_materialization_and_index.ipynb",
        "02_step1_compression_characterization.ipynb",
    },
    "experiments/open_set/lfw": {
        "00_probe_search_and_certification.ipynb",
        "01_evaluation_and_visualization.ipynb",
    },
    "experiments/open_set/survface": {
        "00_official_probe_search.ipynb",
        "01_official_evaluation_and_visualization.ipynb",
    },
    "experiments/gradcam/prerequisite": {
        "00_source_and_model_freeze.ipynb",
        "01_origin_embedding_and_loo_templates.ipynb",
    },
    "experiments/gradcam/experiment": {
        "00_population_gradcam_extraction.ipynb",
        "01_saliency_feature_validation.ipynb",
        "02_step2_compression_characterization.ipynb",
        "03_saliency_compression_join.ipynb",
        "04_representative_case_visualization.ipynb",
    },
    "reports": {"00_cross_dataset_results.ipynb"},
    "maintenance": {"00_selective_cleanup.ipynb"},
}


def _source(path: Path) -> str:
    notebook = nbformat.read(path, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)


def test_notebooks_are_grouped_by_workflow_responsibility() -> None:
    assert not list(NOTEBOOK_ROOT.glob("*.ipynb"))
    actual_directories = {
        path.parent.relative_to(NOTEBOOK_ROOT).as_posix()
        for path in NOTEBOOK_ROOT.rglob("*.ipynb")
    }
    assert actual_directories == set(EXPECTED_NOTEBOOKS)
    for relative, expected_names in EXPECTED_NOTEBOOKS.items():
        directory = NOTEBOOK_ROOT / relative
        assert {path.name for path in directory.glob("*.ipynb")} == expected_names


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
            assert "DATA_FRACTION = 1.0" in source
        if "EXECUTE_STAGE =" in source:
            assert "EXECUTE_STAGE = True" in source
        if "WRITE_OUTPUTS =" in source:
            assert "WRITE_OUTPUTS = True" in source
        if "OVERWRITE =" in source:
            assert "OVERWRITE = True" in source


def test_survface_notebooks_preserve_official_protocol_boundaries() -> None:
    paths = [
        *(
            NOTEBOOK_ROOT / "prerequisite" / "embeddings" / "survface"
        ).glob("*.ipynb"),
        *(
            NOTEBOOK_ROOT / "experiments" / "compression" / "survface"
        ).glob("*.ipynb"),
        *(
            NOTEBOOK_ROOT / "experiments" / "open_set" / "survface"
        ).glob("*.ipynb"),
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
            / "experiments"
            / "compression"
            / dataset
            / "02_step1_compression_characterization.ipynb"
        )
        source = _source(path)
        for phrase in (
            "paired_embedding_metrics",
            "compare_cosine_retrieval",
            "origin_fallback_used",
            "origin_threshold=origin_threshold",
            "compressed_threshold=operating_compressed_threshold",
            "storage_bytes_per_embedding",
            "codebook_bytes",
        ):
            assert phrase in source

    lfw_legacy = _source(
        NOTEBOOK_ROOT
        / "experiments"
        / "open_set"
        / "lfw"
        / "00_probe_search_and_certification.ipynb"
    )
    assert "EXECUTE_LEGACY_FALLBACK" in lfw_legacy

    report = _source(
        NOTEBOOK_ROOT / "reports" / "00_cross_dataset_results.ipynb"
    )
    for phrase in (
        "origin_fallback_used",
        "origin_decision_threshold",
        "compressed_decision_threshold",
        "profile_storage schema mismatch",
    ):
        assert phrase in report
