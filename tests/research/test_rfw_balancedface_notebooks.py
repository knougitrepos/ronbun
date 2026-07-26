from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "notebooks" / "prerequisite" / "datasets"


def _load(name: str):
    path = DATASET_ROOT / name
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return path, notebook


def test_rfw_and_balancedface_are_dataset_prerequisites():
    expected = {
        "02_rfw_data_preparation.ipynb",
        "03_balancedface_data_preparation.ipynb",
    }
    assert expected.issubset({path.name for path in DATASET_ROOT.glob("*.ipynb")})
    for name in sorted(expected):
        path, notebook = _load(name)
        assert notebook.metadata["ronbun"]["workflow_role"] == "prerequisite"
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            compile(cell.source, f"{path.name}:cell-{index}", "exec")


def test_dataset_prerequisites_use_full_execution_defaults_and_guards():
    _, rfw = _load("02_rfw_data_preparation.ipynb")
    _, balanced = _load("03_balancedface_data_preparation.ipynb")
    rfw_source = "\n".join(cell.source for cell in rfw.cells)
    balanced_source = "\n".join(cell.source for cell in balanced.cells)

    for source in (rfw_source, balanced_source):
        assert 'MODE = "dev"' in source
        assert "DATA_FRACTION = 1.0" in source
        assert "SEED = 42" in source
        assert "EXECUTE_STAGE = True" in source
        assert "WRITE_OUTPUTS = True" in source
        assert "OVERWRITE = True" in source
        assert "WRITE_OUTPUTS=True requires EXECUTE_STAGE=True" in source

    assert "select_rfw_protocol_scope" in rfw_source
    assert "strict_official=True" in rfw_source
    assert "PCA/PQ를 fit" in rfw_source
    assert "source_identities.txt" in rfw_source

    assert "RFW_SUCCESS_PATH" in balanced_source
    assert "build_balancedface_index_bundle" in balanced_source
    assert "VERIFY_RECORDIO_ARCHIVE = True" in balanced_source
    assert "alignment_and_group_coverage_audit_required" in balanced_source
    assert "RecordIO decoder" in balanced_source
