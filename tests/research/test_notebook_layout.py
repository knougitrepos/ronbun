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
    },
    "survface": {
        "data_preparation.ipynb",
        "00_official_protocol_and_run_freeze.ipynb",
        "01_official_arcface_embedding_extraction.ipynb",
        "02_external_compressor_import.ipynb",
        "03_official_compressed_materialization_and_index.ipynb",
        "04_official_probe_search.ipynb",
        "05_official_evaluation_and_visualization.ipynb",
    },
}


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
