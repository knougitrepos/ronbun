from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"


def _load(dataset: str):
    path = NOTEBOOK_ROOT / dataset / "data_preparation.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return path, notebook


def test_rfw_and_balancedface_notebooks_are_thin_restartable_runbooks():
    for dataset in ("rfw", "balancedface"):
        directory = NOTEBOOK_ROOT / dataset
        assert {path.name for path in directory.glob("*.ipynb")} == {
            "data_preparation.ipynb"
        }
        path, notebook = _load(dataset)
        assert notebook.metadata["ronbun"]["restart_policy"] == (
            "restart_kernel_and_run_all"
        )
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            compile(cell.source, f"{path.name}:cell-{index}", "exec")
            assert cell.execution_count is None
            assert cell.outputs == []


def test_dataset_runbooks_expose_safe_scope_and_write_guards():
    _, rfw = _load("rfw")
    _, balanced = _load("balancedface")
    rfw_source = "\n".join(cell.source for cell in rfw.cells)
    balanced_source = "\n".join(cell.source for cell in balanced.cells)

    for source in (rfw_source, balanced_source):
        assert 'MODE = "dev"' in source
        assert "DATA_FRACTION = 0.10" in source
        assert "SEED = 42" in source
        assert "EXECUTE_STAGE = False" in source
        assert "WRITE_OUTPUTS = False" in source
        assert "WRITE_OUTPUTS=True requires EXECUTE_STAGE=True" in source

    assert "select_rfw_protocol_scope" in rfw_source
    assert "strict_official=True" in rfw_source
    assert "PCA/PQ를 fit" in rfw_source
    assert "source_identities.txt" in rfw_source

    assert "RFW_SUCCESS_PATH" in balanced_source
    assert "build_balancedface_index_bundle" in balanced_source
    assert "VERIFY_RECORDIO_ARCHIVE = True" in balanced_source
    assert "JPG archive는 절단" in balanced_source
    assert "RecordIO decoder" in balanced_source
