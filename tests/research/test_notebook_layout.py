from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"

EXPECTED = {
    "lfw": {
        "data_preparation.ipynb",
        "00_protocol_and_run_freeze.ipynb",
        "01_arcface_embedding_extraction.ipynb",
        "02_compressor_fit.ipynb",
        "03_compressed_materialization_and_index.ipynb",
        "04_probe_search_and_certification.ipynb",
        "05_evaluation_and_visualization.ipynb",
        "06_step1_compression_characterization.ipynb",
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

        step1_path = directory / "06_step1_compression_characterization.ipynb"
        step1_sources = "\n".join(
            cell.source for cell in nbformat.read(step1_path, as_version=4).cells
        )
        assert "paired_embedding_metrics" in step1_sources
        assert "compare_cosine_retrieval" in step1_sources
        assert "origin_fallback_used" in step1_sources
        assert "origin_threshold=origin_threshold" in step1_sources
        assert (
            "compressed_threshold=operating_compressed_threshold"
            in step1_sources
        )
        assert "storage_bytes_per_embedding" in step1_sources
        assert "codebook_bytes" in step1_sources
        assert "codebook_bytes_source" in step1_sources

    lfw_legacy_sources = "\n".join(
        cell.source
        for cell in nbformat.read(
            NOTEBOOK_ROOT / "lfw" / "04_probe_search_and_certification.ipynb",
            as_version=4,
        ).cells
    )
    assert "EXECUTE_LEGACY_FALLBACK" in lfw_legacy_sources

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
