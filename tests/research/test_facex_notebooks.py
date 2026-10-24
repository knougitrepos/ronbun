from pathlib import Path

import nbformat
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["01_data_preparation.ipynb", "02_inputs.ipynb", "03_experiments.ipynb", "04_results.ipynb"])
def test_active_notebooks_safe_and_five_dataset_configuration(name, monkeypatch):
    from research.experiments import facex_workflow as workflow
    monkeypatch.chdir(ROOT)
    calls = []
    monkeypatch.setattr(workflow, "inspect_workflow", lambda *a, **kw: pd.DataFrame({"dataset": kw["datasets"]}))
    def forbidden(*a, **kw):
        calls.append(True)
        raise AssertionError("Default notebook must not extract/train/run")
    monkeypatch.setattr(workflow, "prepare_inputs", forbidden)
    monkeypatch.setattr(workflow, "run_workflow", forbidden)
    monkeypatch.setattr(workflow, "read_results", forbidden)
    nb = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    nbformat.validate(nb)
    scope = {"display": lambda *a: None}
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert not cell.outputs and cell.execution_count is None
            exec(compile(cell.source, name, "exec"), scope)
    assert scope["DATASETS"] == ["survface", "tinyface", "scface", "lfw", "rfw"]
    assert scope["EXECUTE"] is False
    assert not calls
