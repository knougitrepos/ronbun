"""Fresh-kernel entry points and explicit input/report handoff without GPU runs."""
import ast
from pathlib import Path
from unittest.mock import Mock

import nbformat
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks" / "legacy" / "lfw_sequential_v2"
NAMES = (
    "01_data_preparation.ipynb", "02_inputs.ipynb",
    "03_calibration.ipynb", "04_results.ipynb",
)


def execute(name, overrides=None):
    namespace = {}
    for i, cell in enumerate(c for c in nbformat.read(NOTEBOOKS / name, as_version=4).cells
                             if c.cell_type == "code"):
        exec(compile(cell.source, name, "exec"), namespace)
        if i == 0:
            namespace.update(overrides or {})
    return namespace


def plan(ready=True):
    return dict(ready=ready, expected_jobs=800, coverage=pd.DataFrame(),
                inventory=pd.DataFrame())


def test_only_four_current_notebooks_and_one_legacy_directory():
    assert {p.name for p in NOTEBOOKS.glob("*.ipynb")} == set(NAMES)
    current = ROOT / "notebooks"
    assert {p.name for p in current.glob("*.ipynb")} == {
        "01_data_preparation.ipynb", "02_inputs.ipynb", "03_experiments.ipynb", "04_results.ipynb"}
    assert {p.name for p in current.iterdir() if p.is_dir()
            and p.name != ".ipynb_checkpoints"} == {"legacy"}
    assert len(list((current / "legacy").rglob("*.ipynb"))) >= 56
    for name in NAMES:
        notebook = nbformat.read(NOTEBOOKS / name, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert not cell.outputs and cell.execution_count is None
                assert "LFW_PROTOCOL_MODE" not in cell.source


def test_prepare_pairs_reuses_existing_manifest(tmp_path, monkeypatch):
    from research.datasets import lfw_pairs
    import research.datasets as datasets
    (tmp_path / "research").mkdir()
    (tmp_path / "configs/experiments").mkdir(parents=True)
    config = tmp_path / "configs/experiments/lfw_pair_verification.yaml"
    config.write_text(yaml.safe_dump(dict(image_manifest="population.csv", pairs_file="pairs.txt")))
    pd.DataFrame(dict(image_id=["x"], identity_id=["p"])).to_csv(tmp_path / "population.csv", index=False)
    load = Mock(return_value=(pd.DataFrame(dict(fold=[1], is_genuine=[True])),
                              pd.DataFrame(dict(identity_id=["unused"]))))
    build = Mock(side_effect=AssertionError("existing input must not be rebuilt"))
    monkeypatch.setattr(lfw_pairs, "load_pairs", load)
    monkeypatch.setattr(datasets, "build_lfw_manifest", build)
    monkeypatch.chdir(tmp_path)
    execute(NAMES[0])
    build.assert_not_called()
    assert load.call_args.kwargs["download"] is False


def test_input_preflight_has_no_extraction_and_blocks_missing_coverage(monkeypatch):
    from research.experiments import lfw_pair_verification as pairs, lfw_resize_inputs as inputs
    extraction = Mock(side_effect=AssertionError("preflight invoked extraction"))
    inspect = Mock(return_value=plan())
    monkeypatch.setattr(inputs, "prepare_resize_inputs", extraction)
    monkeypatch.setattr(pairs, "run_verification", inspect)
    execute(NAMES[1])
    assert inspect.call_args.kwargs["execute"] is False
    extraction.assert_not_called()
    inspect.return_value = plan(False)
    with pytest.raises(RuntimeError, match="입력 누락"):
        execute(NAMES[1])


def test_preparation_rejects_mismatched_source_before_extraction(tmp_path, monkeypatch):
    from research.experiments import lfw_resize_inputs as inputs
    (tmp_path / "research").mkdir()
    (tmp_path / "configs").mkdir()
    config = dict(image_manifest="images.csv", fiqa_root="q", fiqa_variant="L",
                  source_runs={"arcface": "source-a"})
    (tmp_path / "pair.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "input.yaml").write_text(yaml.safe_dump(dict(config, source_runs={"arcface": "source-b"})))
    extraction = Mock(side_effect=AssertionError("mismatched paths must not extract"))
    monkeypatch.setattr(inputs, "prepare_resize_inputs", extraction)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="source 불일치"):
        execute(NAMES[1], dict(CONFIG_PATH="pair.yaml", INPUT_CONFIG_PATH="input.yaml", PREPARE_INPUTS=True))
    extraction.assert_not_called()


def test_calibration_flags_and_explicit_report_handoff(tmp_path, monkeypatch, capsys):
    from research.experiments import lfw_pair_verification as pairs, lfw_pair_storage as storage
    run = Mock(return_value=plan())
    read = Mock()
    monkeypatch.setattr(pairs, "run_verification", run)
    monkeypatch.setattr(storage, "read_report", read)
    execute(NAMES[2])
    assert run.call_args.kwargs["execute"] is False
    assert run.call_args.kwargs["keep_raw_results"] is False
    execute(NAMES[3])
    read.assert_not_called()
    report = tmp_path / "explicit-partial-report"
    run.return_value = dict(plan(), report_dir=report)
    namespace = execute(NAMES[2], dict(EXECUTE=True, KEEP_RAW_RESULTS=True, MAX_NEW_JOBS_PER_RUN=1))
    assert run.call_args.kwargs["execute"] is True
    assert run.call_args.kwargs["keep_raw_results"] is True
    assert run.call_args.kwargs["max_new_jobs"] == 1
    read.return_value = (dict(operating=pd.DataFrame(), accuracy_cv=pd.DataFrame()),
                         dict(status="partial", completed_jobs=1, expected_jobs=800, report_uid="explicit-id"))
    execute(NAMES[3], dict(REPORT_DIR=str(namespace["REPORT_DIR"])))
    read.assert_called_once_with(report)
    assert "부분 실행" in capsys.readouterr().out


def test_legacy_report_runner_selects_1toN_route_in_memory(monkeypatch):
    import nbclient
    from scripts import run_integrated_postprocessing as runner
    monkeypatch.setattr(runner, "build_report_parameter_source", Mock(return_value=("INJECTED = True", {})))
    captured = {}
    class Client:
        def __init__(self, notebook, **kwargs):
            captured["notebook"] = notebook
        def execute(self):
            pass
    monkeypatch.setattr(nbclient, "NotebookClient", Client)
    result = runner.run_cross_dataset_report_notebook(ROOT, model_name="arcface", selected_runs={}, write_outputs=False)
    source = "\n".join(c.source for c in captured["notebook"].cells if c.cell_type == "code")
    assert 'LFW_PROTOCOL_MODE = "legacy"' in source
    assert "notebooks/legacy/pipeline/04_results.ipynb" in result["notebook"].replace("\\", "/")
    saved = nbformat.read(ROOT / "notebooks/legacy/pipeline/04_results.ipynb", as_version=4)
    assert 'LFW_PROTOCOL_MODE = "pair_verification_report"' in "\n".join(c.source for c in saved.cells)
