"""Public workflow regressions for the November audit (F4, F6, F7)."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from research.experiments import facex_workflow as flow
from research.runtime.hashing import sha256_file
from test_facex_review_regressions import rank_workflow, stub_extraction


@pytest.mark.parametrize("entry", [flow.inspect_workflow, flow.prepare_inputs, flow.run_workflow])
@pytest.mark.parametrize("selection", [["scface", "scface"], [], "scface", ["unknown"]])
def test_invalid_override_rejected_before_work(tmp_path, monkeypatch, entry, selection):
    path, root = rank_workflow(tmp_path, monkeypatch, "scface")
    monkeypatch.setattr(flow, "_workload", lambda *a: pytest.fail("Invalid selection reached work"))
    with pytest.raises(ValueError, match="datasets"):
        entry(path, output_root=root, datasets=selection) if entry != flow.inspect_workflow else entry(path, datasets=selection)


def test_duplicate_config_rejected(tmp_path, monkeypatch):
    path, _ = rank_workflow(tmp_path, monkeypatch, "scface")
    config = yaml.safe_load(path.read_text())
    config["datasets"] = ["scface", "scface"]
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="unique datasets"):
        flow.inspect_workflow(path)


def test_wrong_report_rejected_and_genuine_report_reused(tmp_path, monkeypatch):
    path, root = rank_workflow(tmp_path, monkeypatch, "scface")
    first = Path(flow.run_workflow(path, output_root=root, keep_raw_results=False)["report_dir"])
    second = Path(flow.run_workflow(path, output_root=root, keep_raw_results=True)["report_dir"])
    intact_hash = sha256_file(second / "run_manifest.json")
    # Move our temporary fixture, retaining the genuine report for the control.
    backup = first.with_name("backup")
    first.rename(backup)
    shutil.copytree(second, first)
    with pytest.raises(ValueError, match="UID, directory and contract"):
        flow.run_workflow(path, output_root=root, keep_raw_results=False)
    assert sha256_file(second / "run_manifest.json") == intact_hash
    assert flow.run_workflow(path, output_root=root, keep_raw_results=True)["reused"]
    contract = flow.read_results(second)["manifest"]["contract"]
    with pytest.raises(ValueError, match="requested experiment"):
        flow.read_results(second, expected_contract={**contract, "keep_raw_results": False})


def test_manifest_contract_mutation_is_rejected(tmp_path, monkeypatch):
    path, root = rank_workflow(tmp_path, monkeypatch, "scface")
    report = Path(flow.run_workflow(path, output_root=root)["report_dir"])
    manifest_path = report / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["contract"]["datasets"] = ["tinyface"]
    flow.write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="UID, directory and contract"):
        flow.read_results(report)


def test_initial_manifest_error_releases_only_owned_lock_and_can_retry(tmp_path, monkeypatch):
    path, root = rank_workflow(tmp_path, monkeypatch, "scface")
    real = flow.write_json
    def fail_manifest(path, value):
        if Path(path).name == "run_manifest.json" and value["status"] == "running":
            raise OSError("injected initial manifest error")
        return real(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(flow, "write_json", fail_manifest)
        with pytest.raises(OSError, match="injected"):
            flow.run_workflow(path, output_root=root)
    assert not list(root.rglob("RUNNING.lock"))
    work = next((root / "reports").glob("*.partial"))
    lock = work / "RUNNING.lock"
    lock.write_text("another process owns this")
    with pytest.raises(FileExistsError):
        flow.run_workflow(path, output_root=root)
    assert lock.read_text() == "another process owns this"
    lock.unlink()  # Only the test-owned lock in tmp_path.
    report = Path(flow.run_workflow(path, output_root=root)["report_dir"])
    assert flow.read_results(report)["manifest"]["completed_jobs"] == 1
    assert not list(root.rglob("RUNNING.lock"))


def test_prepare_inputs_reuses_valid_cache_with_less_than_rebuild_space(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from research.experiments import facex_inputs as inputs
    population, spec = stub_extraction(tmp_path, monkeypatch)
    root = tmp_path / "outputs"
    cached = inputs.materialize_inputs(tmp_path, population, spec, "scface", root,
                                      dict(source="raw_norm"), quality_required=False)
    config = dict(project_root=str(tmp_path), embedding_dim=1024, datasets=["scface"],
                  target_rates=[.1], quality=dict(source="raw_norm"), output_root=str(root))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(flow, "_workload", lambda *a: ({}, {}, {"scface": (population, None)},
                        [("model", spec, "scface", "scface")], {"fixture": True}))
    monkeypatch.setattr(inputs.shutil, "disk_usage", lambda _: SimpleNamespace(free=25000))
    result = flow.prepare_inputs(path)
    record = json.loads(Path(result["input_index"]).read_text())
    assert Path(record["inputs"]["model/scface"]["path"]) == cached
    assert result["input_count"] == 1 and result["estimated_input_bytes"] > 25000
