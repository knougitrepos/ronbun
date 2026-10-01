"""Exercise notebook routing without GPU work or touching completed experiments."""
import ast
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PATHS = {
    "prepare": "notebooks/lfw/00_data_preparation/00_data_preparation.ipynb",
    "benchmark": "notebooks/common/orchestration/00_batch_experiment_runner.ipynb",
    "inputs": "notebooks/common/orchestration/01_batch_fiqa_saliency_calibration.ipynb",
    "calibrate": "notebooks/calibration/03_origin_vs_pq_fiqa_calibration.ipynb",
    "report": "notebooks/common/reports/00_cross_dataset_results.ipynb",
}


def test_notebook_source_cells_are_syntactically_valid_and_defaults_preserved():
    for relative in PATHS.values():
        notebook = json.loads((ROOT / relative).read_text(encoding="utf8"))
        code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        assert 'LFW_PROTOCOL_MODE = "legacy"' in code[0]
        for cell in code:
            ast.parse(cell)


@pytest.mark.parametrize("kind", ["benchmark", "inputs", "calibrate", "report"])
def test_new_notebook_mode_never_calls_legacy_pipeline(kind, monkeypatch):
    from research.experiments import lfw_protocol_experiments as module
    from research.experiments import pipeline_runner
    legacy = Mock(side_effect=AssertionError("new route called legacy GPU preparation"))
    monkeypatch.setattr(pipeline_runner, "prepare_common_model_checkpoint", legacy)
    inspection = dict(coverage=pd.DataFrame(), inventory=pd.DataFrame(), missing=pd.DataFrame())
    inspect = Mock(return_value=inspection)
    benchmark = Mock(return_value={"completed": True})
    matched = Mock(return_value={"plan": pd.DataFrame(), "execute": False})
    report = Mock(return_value=({}, {"spec": {"protocol_uid": module.MATCHED_PROTOCOL}}))
    monkeypatch.setattr(module, "inspect_blufr", inspect)
    monkeypatch.setattr(module, "run_blufr_benchmark", benchmark)
    monkeypatch.setattr(module, "run_matched_lfw_calibration", matched)
    monkeypatch.setattr(module, "read_lfw_protocol_report", report)
    notebook = json.loads((ROOT / PATHS[kind]).read_text(encoding="utf8"))
    namespace = {}
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    mode = "blufr_benchmark" if kind == "benchmark" else "protocol_report" if kind == "report" else "matched_calibration"
    for i, cell in enumerate(cells):
        if i == 0:
            cell = cell.replace('LFW_PROTOCOL_MODE = "legacy"', f'LFW_PROTOCOL_MODE = "{mode}"')
            cell += '\nEXECUTE = False\nLFW_PROTOCOL_REPORT_DIR = "explicit-report"\n'
        exec(compile(cell, PATHS[kind], "exec"), namespace)
    legacy.assert_not_called()
    if kind == "benchmark":
        inspect.assert_called_once()
        benchmark.assert_not_called()
    elif kind == "report":
        report.assert_called_once()
    else:
        matched.assert_called_once()
        assert matched.call_args.kwargs["execute"] is False
        if kind == "calibrate":
            assert matched.call_args.kwargs["partition_seeds"] == namespace["PARTITION_SEEDS"]
            assert matched.call_args.kwargs["maximum_process_gb"] == namespace["MAX_PROCESS_RAM_GB"]
