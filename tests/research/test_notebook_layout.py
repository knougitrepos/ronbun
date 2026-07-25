from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"

EXPECTED = {
    "lfw": {
        "00_data_preparation.ipynb",
        "01_protocol_and_run_freeze.ipynb",
        "02_materialize_aligned_crops.ipynb",
        "03_source_and_model_freeze.ipynb",
        "04_origin_embedding_and_loo_templates.ipynb",
        "05_population_gradcam_extraction.ipynb",
        "06_saliency_feature_validation.ipynb",
        "07_compression_characterization.ipynb",
        "08_saliency_compression_join.ipynb",
        "09_representative_case_visualization.ipynb",
    },
    "survface": {
        "data_preparation.ipynb",
        "00_official_protocol_and_run_freeze.ipynb",
        "01_official_arcface_embedding_extraction.ipynb",
        "02_external_compressor_import.ipynb",
        "03_official_compressed_materialization_and_index.ipynb",
        "04_official_probe_search.ipynb",
        "05_official_evaluation_and_visualization.ipynb",
        "06_step1_compression_characterization.ipynb",
    },
}

COMMON_EXPECTED = {"cross_dataset_results.ipynb"}


def test_dataset_notebooks_are_separated_and_valid() -> None:
    assert not list(NOTEBOOK_ROOT.glob("*.ipynb"))
    assert not list((NOTEBOOK_ROOT / "data_preparation").glob("*.ipynb"))

    for dataset, expected_names in EXPECTED.items():
        directory = NOTEBOOK_ROOT / dataset
        actual_names = {path.name for path in directory.glob("*.ipynb")}
        assert actual_names == expected_names

        for path in sorted(directory.glob("*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)

    common_directory = NOTEBOOK_ROOT / "common"
    assert {path.name for path in common_directory.glob("*.ipynb")} == COMMON_EXPECTED
    for path in common_directory.glob("*.ipynb"):
        nbformat.validate(nbformat.read(path, as_version=4))


def test_survface_notebooks_state_official_protocol_safety_rules() -> None:
    sources = "\n".join(
        cell.source
        for path in sorted((NOTEBOOK_ROOT / "survface").glob("*.ipynb"))
        for cell in nbformat.read(path, as_version=4).cells
    )
    assert "official_all" in sources
    assert "known_unknown" in sources
    assert "TPIR" in sources
    assert "FPIR" in sources
    assert "development" in sources
    assert "fit_on_survface_official_test" in sources


def test_step1_notebooks_expose_scope_and_use_fallback_free_evaluation() -> None:
    for dataset in ("lfw", "survface"):
        directory = NOTEBOOK_ROOT / dataset
        all_sources = "\n".join(
            cell.source
            for path in sorted(directory.glob("*.ipynb"))
            for cell in nbformat.read(path, as_version=4).cells
        )
        assert "MODE" in all_sources
        assert "DATA_FRACTION" in all_sources
        assert "SEED" in all_sources

        step1_filename = (
            "07_compression_characterization.ipynb"
            if dataset == "lfw"
            else "06_step1_compression_characterization.ipynb"
        )
        step1_path = directory / step1_filename
        step1_sources = "\n".join(
            cell.source for cell in nbformat.read(step1_path, as_version=4).cells
        )
        assert "origin_fallback_used" in step1_sources

    lfw_sources = "\n".join(
        cell.source
        for cell in nbformat.read(
            NOTEBOOK_ROOT / "lfw" / "07_compression_characterization.ipynb",
            as_version=4,
        ).cells
    )
    assert "origin_fallback_used" in lfw_sources

    common_sources = "\n".join(
        cell.source
        for cell in nbformat.read(
            NOTEBOOK_ROOT / "common" / "cross_dataset_results.ipynb",
            as_version=4,
        ).cells
    )
    assert "origin_fallback_used" in common_sources
    assert "fallback" in common_sources.lower()
    assert "origin_decision_threshold" in common_sources
    assert "compressed_decision_threshold" in common_sources
    assert "profile_storage schema mismatch" in common_sources
