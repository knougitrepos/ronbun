import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from research.experiments import calibration_matrix as matrix
from research.experiments import fiqa_continuous_calibration as fiqa_writer
from research.experiments import saliency_incremental_calibration as saliency_writer
from research.explainability.gradcam import artifacts
from research.runtime.hashing import canonical_sha256, sha256_file
from scripts import recover_calibration_matrix_publish as recovery
from test_saliency_incremental_calibration import _sources


@pytest.mark.parametrize("module,writer", [
    (fiqa_writer, fiqa_writer.write_continuous_calibration),
    (saliency_writer, saliency_writer.write_saliency_incremental_result),
])
@pytest.mark.parametrize("persistent", [False, True])
def test_result_publish_lock_retries_and_retains_staging(tmp_path, monkeypatch, module, writer, persistent):
    result = {name: pd.DataFrame({"value": [1.]}) for name in module.TABLES}
    result["manifest"] = {"result_uid": "result-test"}
    replace = artifacts.os.replace
    calls, delays = [], []

    def locked(source, destination):
        calls.append((source, destination))
        if persistent or len(calls) <= 2:
            raise PermissionError(5, "simulated lock")
        return replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", locked)
    monkeypatch.setattr(artifacts.time, "sleep", delays.append)
    if persistent:
        with pytest.raises(PermissionError, match="simulated lock"):
            writer(tmp_path, result)
        assert len(calls) == artifacts._ATOMIC_REPLACE_ATTEMPTS
        assert not (tmp_path / "result-test").exists()
        staging = calls[-1][0]
        manifest = json.loads((staging / "manifest.json").read_text())
        assert all(sha256_file(staging / name) == digest for name, digest in manifest["files"].items())
    else:
        destination = writer(tmp_path, result)
        assert len(calls) == 3 and delays == [.05, .1]
        before = {p.name: sha256_file(p) for p in destination.iterdir()}
        assert writer(tmp_path, result) == destination
        assert len(calls) == 3
        assert before == {p.name: sha256_file(p) for p in destination.iterdir()}


def test_audit_rejects_additional_scientific_change(tmp_path, monkeypatch):
    old = "import os\nfrom research.runtime.hashing import canonical_sha256, sha256_file\nVALUE = 1\n"
    for name in recovery.CHANGED:
        directory = "directory" if name == "calibration_matrix.py" else "destination"
        source = old + f"os.rename(staging, {directory})\n"
        path = tmp_path / "research/experiments" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source.replace("import os\n", "").replace("VALUE", recovery.IMPORT + "VALUE", 1).replace(
            f"os.rename(staging, {directory})", f"_publish_atomic_directory(staging, {directory}, overwrite=False)"))
    for name in ("cluster_bootstrap.py", "metrics.py"):
        path = tmp_path / "research/evaluation" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def historical(command, **_):
        directory = "directory" if command[-1].endswith("/calibration_matrix.py") else "destination"
        return (old + f"os.rename(staging, {directory})\n").encode()

    monkeypatch.setattr(recovery.subprocess, "check_output", historical)
    recovery.audit_sources(tmp_path)
    path = tmp_path / "research/experiments/fiqa_continuous_calibration.py"
    path.write_text(path.read_text().replace("VALUE = 1", "VALUE = 2"))
    with pytest.raises(ValueError, match="not the audited publish-only"):
        recovery.audit_sources(tmp_path)


def test_matrix_report_publish_retries_directory_lock(tmp_path, monkeypatch):
    replace = artifacts.os.replace
    attempts = []

    def locked(source, destination):
        if Path(source).is_dir():
            attempts.append(source)
            if len(attempts) <= 2:
                raise PermissionError(5, "simulated report lock")
        return replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", locked)
    monkeypatch.setattr(artifacts.time, "sleep", lambda _: None)
    plan = pd.DataFrame([dict(dataset_id="lfw", model="arcface", compression_profile="pq_512_m128_b8")])
    output = matrix.summarize_calibration_matrix(pd.DataFrame(), plan, expected_seeds=(8972,), output_root=tmp_path)
    assert output["manifest"]["status"] == "partial" and len(attempts) == 3
    again = matrix.summarize_calibration_matrix(pd.DataFrame(), plan, expected_seeds=(8972,), output_root=tmp_path)
    assert again["directory"] == output["directory"] and len(attempts) == 3


def test_recovery_keeps_results_and_resumes_without_refit(tmp_path, monkeypatch):
    condition, fiqa, saliency, faithfulness = _sources(tmp_path)
    monkeypatch.setattr(matrix, "load_condition_score_artifact", lambda _: condition)
    monkeypatch.setattr(matrix, "load_fiqa_score_artifact", lambda _: fiqa)
    inputs = pd.DataFrame([dict(dataset_id="survface", model="arcface", model_uid=condition.manifest["model_uid"],
        compression_profile="pq_512_m128_b8", source_run_id="source", condition_dir="condition",
        condition_sha256=canonical_sha256(condition.manifest), fiqa_dir="fiqa", saliency_dir=str(saliency),
        faithfulness_dir=str(faithfulness))])
    options = dict(partition_seeds=(8972,), target_fpirs=(.1,), resamples=100, progress=lambda _: None)
    root = tmp_path / "jobs"
    receipts = matrix.run_calibration_matrix(inputs, root, **options)
    source = Path(receipts.iloc[1].result_dir)
    staging = source.with_name(".staging-failed-publish")
    assert source.resolve().is_relative_to(tmp_path.resolve()) and staging.resolve().is_relative_to(tmp_path.resolve())
    source.rename(staging)
    (source.parent / "receipt.json").unlink()
    existing = next(root.glob("*/receipt.json"))
    receipt_bytes = existing.read_bytes()
    spec = json.loads(receipt_bytes)["spec"]
    original = {key.replace("\\", "/"): value for key, value in spec["implementation"].items()}
    current = copy.deepcopy(original)
    approved = {}
    for name in recovery.CHANGED:
        key = f"experiments/{name}"
        current[key] = "changed-" + name
        approved[key] = dict(before={original[key]}, after=current[key])
    monkeypatch.setattr(recovery, "audit_sources", lambda _: (approved, current))
    preview = recovery.recover(root)
    assert preview["completed_receipts"] == 1 and preview["recovered_staging"] == 1 and preview["aliases"] == 2
    assert staging.exists() and not source.exists()
    outcome = recovery.recover(root, apply=True)
    assert outcome["aliases"] == 2 and existing.read_bytes() == receipt_bytes
    assert source.exists() and not staging.exists()
    recovery.recover(root, apply=True)  # Idempotent; retain original implementation provenance.

    def forbidden(*args, **kwargs):
        raise AssertionError("completed fit must not be recomputed")

    monkeypatch.setattr(matrix, "run_continuous_calibration", forbidden)
    monkeypatch.setattr(matrix, "run_saliency_incremental_calibration", forbidden)
    monkeypatch.setattr(matrix, "sha256_file", lambda p: current.get("experiments/" + Path(p).name, sha256_file(p)))
    again = matrix.run_calibration_matrix(inputs, root, **options)
    pd.testing.assert_frame_equal(receipts, again)
    corrupted = copy.deepcopy(spec)
    corrupted["implementation"][next(key for key in corrupted["implementation"] if "continuous.py" in key)] = "wrong"
    with pytest.raises(ValueError, match="unaudited implementation"):
        recovery.revised_spec(corrupted, approved, current)
