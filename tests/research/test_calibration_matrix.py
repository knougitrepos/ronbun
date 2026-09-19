import copy
import json

import pandas as pd
import pytest

from research.experiments import calibration_matrix as matrix
from research.experiments import saliency_calibration_inputs as producer
from research.experiments.saliency_condition_binding import bind_saliency_condition
from research.experiments.saliency_incremental_calibration import (
    assess_incremental_gate, load_saliency_incremental_inputs, run_saliency_incremental_calibration,
)
from research.experiments.fiqa_continuous_calibration import run_continuous_calibration
from research.runtime.hashing import canonical_sha256, sha256_file
from test_saliency_calibration_inputs import _mock_producer
from test_saliency_incremental_calibration import _sources


@pytest.mark.parametrize("dataset", ["lfw", "rfw_custom", "survface"])
def test_same_baseline_and_no_test_fitting_across_datasets(tmp_path, dataset):
    c, fiqa, sr, fr = _sources(tmp_path)
    c.manifest["dataset_id"] = dataset
    fiqa.manifest["dataset_id"] = dataset
    for rows in (c.calibration, c.test):
        rows["dataset_id"] = dataset
    for directory in (sr, fr):
        path = directory / "manifest.json"
        m = json.loads(path.read_text())
        m.update(dataset_id=dataset, condition_manifest_sha256=canonical_sha256(c.manifest))
        path.write_text(json.dumps(m))
    inputs = load_saliency_incremental_inputs(c, sr, fr)
    options = dict(target_fpirs=(.1,), partition_seeds=(8972,), resamples=100)
    control = run_continuous_calibration(c, fiqa, **options)
    sal = run_saliency_incremental_calibration(c, fiqa, inputs, **options)
    cols = ["target_fpir", "realized_fpir", "tpir_at_rank_k"]
    pd.testing.assert_frame_equal(control["method_summary"].query("method == 'continuous_fiqa'")[cols].reset_index(drop=True),
                                  sal["method_summary"].query("method == 'baseline'")[cols].reset_index(drop=True))
    changed = copy.deepcopy(c)
    changed.test.loc[~changed.test.is_mated, "score"] += .5
    after = run_saliency_incremental_calibration(changed, fiqa, inputs, **options)
    assert after["models"].model_json.equals(sal["models"].model_json)


@pytest.mark.parametrize("publish_failure", [None, "transient", "persistent"])
def test_cross_budget_reuse_rejects_gallery_or_query_changes(tmp_path, monkeypatch, publish_failure):
    c, calls, _ = _mock_producer(tmp_path, monkeypatch)
    for key in ("source_run_manifest_sha256", "source_freeze_manifest_sha256", "selected_manifest_sha256",
                "prepared_population_manifest_sha256", "calibration_protocol"):
        c.manifest[key] = key
    c.manifest["calibration_seed"] = 8972
    source = producer.build_saliency_calibration_inputs("source", c, tmp_path / "input",
        reuse_test_saliency=False, chunk_size=20, bootstrap_repeats=100)
    target = copy.deepcopy(c)
    target.manifest.update(compression_profile="pq_512_m32_b8", condition_uid="different-budget")
    for frame in (target.calibration, target.test):
        frame["compression_profile"] = "pq_512_m32_b8"
        frame["score"] -= .2
    before = {str(p): sha256_file(p) for p in source["directory"].rglob("*") if p.is_file()}
    kwargs = dict(saliency_directory=source["saliency_directory"], faithfulness_directory=source["faithfulness_directory"],
                  output_root=tmp_path / "bindings")
    from research.explainability.gradcam import artifacts
    replace = artifacts.os.replace
    attempts, delays = [], []

    def locked_replace(staging, destination):
        attempts.append((staging, destination))
        if publish_failure == "persistent" or (publish_failure == "transient" and len(attempts) <= 2):
            raise PermissionError(13, "simulated Windows directory lock")
        return replace(staging, destination)

    monkeypatch.setattr(artifacts.os, "replace", locked_replace)
    monkeypatch.setattr(artifacts.time, "sleep", delays.append)
    if publish_failure == "persistent":
        with pytest.raises(PermissionError, match="directory lock"):
            bind_saliency_condition(c, target, **kwargs)
        assert len(attempts) == artifacts._ATOMIC_REPLACE_ATTEMPTS
        assert len(delays) == artifacts._ATOMIC_REPLACE_ATTEMPTS - 1
        staging, destination = attempts[-1]
        assert not destination.exists()
        retained = load_saliency_incremental_inputs(target, staging / "saliency", staging / "faithfulness")
        assert assess_incremental_gate(target, retained)["comparison_enabled"]
        assert before == {str(p): sha256_file(p) for p in source["directory"].rglob("*") if p.is_file()}
        monkeypatch.setattr(artifacts.os, "replace", replace)
    result = bind_saliency_condition(c, target, **kwargs)
    if publish_failure == "transient":
        assert len(attempts) == 3 and len(delays) == 2
    inputs = load_saliency_incremental_inputs(target, result["saliency_directory"], result["faithfulness_directory"])
    assert assess_incremental_gate(target, inputs)["comparison_enabled"]
    assert len(calls) == 5
    assert before == {str(p): sha256_file(p) for p in source["directory"].rglob("*") if p.is_file()}
    assert bind_saliency_condition(c, target, **kwargs)["directory"] == result["directory"]
    # Reusing a completed binding must not try to publish it again.
    assert len(attempts) == (artifacts._ATOMIC_REPLACE_ATTEMPTS if publish_failure == "persistent"
                             else 3 if publish_failure == "transient" else 1)
    changed = copy.deepcopy(target)
    changed.manifest["calibration_seed"] = 7
    with pytest.raises(ValueError, match="calibration_seed"):
        bind_saliency_condition(c, changed, **kwargs)
    changed = copy.deepcopy(target)
    changed.test.loc[0, "identity_id"] = "other"
    with pytest.raises(AssertionError):
        bind_saliency_condition(c, changed, **kwargs)


def test_matrix_checkpoint_resumes_before_fit_and_detects_corruption(tmp_path, monkeypatch):
    c, fiqa, sr, fr = _sources(tmp_path)
    monkeypatch.setattr(matrix, "load_condition_score_artifact", lambda _: c)
    monkeypatch.setattr(matrix, "load_fiqa_score_artifact", lambda _: fiqa)
    inputs = pd.DataFrame([dict(dataset_id="survface", model="arcface", model_uid=c.manifest["model_uid"],
        compression_profile="pq_512_m128_b8", source_run_id="source", condition_dir="condition",
        condition_sha256=canonical_sha256(c.manifest), fiqa_dir="fiqa", saliency_dir=str(sr), faithfulness_dir=str(fr))])
    kwargs = dict(partition_seeds=(8972,), target_fpirs=(.1,), resamples=100, progress=lambda _: None)
    receipts = matrix.run_calibration_matrix(inputs, tmp_path / "jobs", **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("completed fit must be reused")
    monkeypatch.setattr(matrix, "run_continuous_calibration", forbidden)
    monkeypatch.setattr(matrix, "run_saliency_incremental_calibration", forbidden)
    again = matrix.run_calibration_matrix(inputs, tmp_path / "jobs", **kwargs)
    pd.testing.assert_frame_equal(receipts, again)
    report = matrix.summarize_calibration_matrix(receipts, inputs, expected_seeds=(8972,), output_root=tmp_path / "reports")
    assert report["manifest"]["status"] == "completed"
    assert len(report["method_summary"]) == 7
    assert set(report["method_summary"].method) == {"global_safe", "fiqa_2bin", "fiqa_5bin", "continuous_fiqa",
                                                   "plus_outside", "plus_entropy", "plus_both"}
    matrix.summarize_calibration_matrix(receipts, inputs, expected_seeds=(8972,), output_root=tmp_path / "reports")
    partial = matrix.summarize_calibration_matrix(receipts, inputs, expected_seeds=(0, 8972))
    assert partial["manifest"]["status"] == "partial" and len(partial["manifest"]["missing_jobs"]) == 2
    relabelled = receipts.copy()
    relabelled["compression_profile"] = "pq_512_m32_b8"
    with pytest.raises(ValueError, match="lineage mismatch"):
        matrix.summarize_calibration_matrix(relabelled, inputs, expected_seeds=(8972,))
    empty = matrix.summarize_calibration_matrix(receipts.iloc[:0], inputs, expected_seeds=(8972,), output_root=tmp_path / "reports")
    assert empty["manifest"]["status"] == "partial"
    matrix.summarize_calibration_matrix(receipts.iloc[:0], inputs, expected_seeds=(8972,), output_root=tmp_path / "reports")
    with pytest.raises(ValueError, match="duplicate"):
        matrix.summarize_calibration_matrix(pd.concat([receipts, receipts]), inputs, expected_seeds=(8972,))
    from pathlib import Path
    (Path(receipts.iloc[0].result_dir) / "method_summary.csv").write_text("tampered")
    with pytest.raises(ValueError, match="hash/path"):
        matrix.run_calibration_matrix(inputs, tmp_path / "jobs", **kwargs)


def test_default_matrix_is_36_distinct_conditions():
    assert len(matrix.DEFAULT_RUN_MATRIX) == 4
    assert all(set(runs) == {"lfw", "rfw_custom", "survface"} for runs in matrix.DEFAULT_RUN_MATRIX.values())
    assert len(matrix.PQ_PROFILES) == 3
    assert len({path for runs in matrix.DEFAULT_RUN_MATRIX.values() for path in runs.values()}) == 12


def test_notebook_configuration_and_all_conditions():
    import ast
    import nbformat
    from pathlib import Path
    nb = nbformat.read(Path(__file__).parents[2] / "notebooks/common/orchestration/01_batch_fiqa_saliency_calibration.ipynb", as_version=4)
    nbformat.validate(nb)
    config = next(c.source for c in nb.cells if c.id == "user-configuration")
    # EXECUTE is intentionally user-editable for production runs.
    execute = next(node.value for node in ast.parse(config).body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "EXECUTE" for t in node.targets))
    assert isinstance(execute, ast.Constant) and type(execute.value) is bool
    assert "RUN_SPLIT_STABILITY = True" in config
    assert all(p in config for p in matrix.PQ_PROFILES)
    for cell in nb.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
