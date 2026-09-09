import json
from pathlib import Path

import pandas as pd
import pytest

from research.experiments.calibration_evidence_report import (
    load_calibration_evidence_report, render_calibration_evidence_markdown,
    write_calibration_evidence_report,
)
from research.experiments.fiqa_split_stability import write_fiqa_split_stability
from research.runtime.hashing import sha256_file
from test_fiqa_split_stability import _inputs, _run


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def sources(tmp_path):
    condition, artifacts = _inputs(tmp_path)
    cmroot, fmroot, ledgerroot = [tmp_path / p for p in ("common", "condition", "ledger")]
    for root in (cmroot, fmroot, ledgerroot):
        root.mkdir()
    core = ledgerroot / "core.bin"
    core.write_bytes(b"immutable synthetic test-score core")
    condition.manifest.update(source_run_id="run-test", status="completed",
                              artifact_type="compressed_calibration_test_score_tables",
                              persisted_test_core_sha256=sha256_file(core), files={})
    for split in ("calibration", "test"):
        path = fmroot / f"{split}_scores.parquet"
        getattr(condition, split).to_parquet(path)
        condition.manifest["files"][path.name] = {"sha256": sha256_file(path)}
    _json(fmroot / "manifest.json", condition.manifest)
    result = _run(condition, artifacts)
    smroot = write_fiqa_split_stability(tmp_path / "splits", result)
    control = result["seed_metrics"].query("method == 'global_empirical'").iloc[0]
    row = {"target_fpir": .1, "tpir_rank": 20, "threshold_policy": "recalibrated_compressed",
           "compressed_score_space": condition.manifest["score_space"],
           "tpir_cluster_ci_status": "ok", "tpir_cluster_ci_identity_count": control.mated_identity_count,
           "compressed_minus_origin_tpir_at_rank_k": 0,
           "compressed_minus_origin_tpir_at_rank_k_identity_cluster95_low": 0,
           "compressed_minus_origin_tpir_at_rank_k_identity_cluster95_high": 0}
    for prefix in ("origin", "compressed"):
        row.update({f"{prefix}_tpir20_count": control.true_identification_at_rank_k_count,
                    f"{prefix}_tpir20_denominator": control.test_mated_count,
                    f"{prefix}_tpir20": control.tpir_at_rank_k,
                    f"{prefix}_false_accept_count": control.false_accept_count,
                    f"{prefix}_fpir_denominator": control.test_non_mated_count,
                    f"{prefix}_fpir": control.realized_fpir,
                    f"{prefix}_tpir_at_rank_k_identity_cluster95_low": control.tpir_cluster95_low,
                    f"{prefix}_tpir_at_rank_k_identity_cluster95_high": control.tpir_cluster95_high})
    pd.DataFrame([row]).to_csv(cmroot / "retrieval_summary.csv", index=False)
    _json(ledgerroot / "manifest.json", {"conditions": [{"condition_id": "core-test", "core": {
        "path": core.name, "sha256": sha256_file(core)}}]})
    cm = {"status": "completed", "artifact_type": "common_tpir_identity_cluster_ci",
          "contract": "query-weighted-mated-identity-percentile-v1",
          "condition": {k: condition.manifest[k] for k in ("dataset_id", "model_uid", "extraction_uid",
              "origin_embedding_artifact_uid", "protocol_uid", "compression_profile", "search_mode")},
          "condition_id": "core-test", "source_manifest": str(ledgerroot / "manifest.json"),
          "source_manifest_sha256": sha256_file(ledgerroot / "manifest.json"),
          "summary_sha256": sha256_file(cmroot / "retrieval_summary.csv")}
    _json(cmroot / "manifest.json", cm)
    return dict(common_ci=cmroot, split_stability=smroot, condition=fmroot,
                expected_run_id="run-test", expected_model_uid=condition.manifest["model_uid"],
                expected_hashes={"common_ci": sha256_file(cmroot / "manifest.json"),
                                 "split_stability": sha256_file(smroot / "manifest.json"),
                                 "condition": sha256_file(fmroot / "manifest.json")})


def test_report_separates_comparisons_scales_and_preserves_sources(sources, tmp_path):
    report = load_calibration_evidence_report(**sources)
    text = render_calibration_evidence_markdown(report)
    assert "독립 반복 성공 확률이 아니" in text
    assert "%p" in text and "고정 기준 seed 8972" in text
    assert len(report["tables"]["compression_operating_points"]) == 2
    assert len(report["tables"]["fiqa_fixed_split"]) == 4
    out = write_calibration_evidence_report(tmp_path / "reports", report)
    manifest = json.loads((out / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert sha256_file(out / name) == digest
    with pytest.raises(FileExistsError):
        write_calibration_evidence_report(tmp_path / "reports", report)


@pytest.mark.parametrize("override", [
    {"expected_run_id": "other-run"}, {"expected_model_uid": "other-model"},
    {"reference_seed": 999}, {"reference_seed": True},
])
def test_rejects_wrong_selection(sources, override):
    with pytest.raises(ValueError):
        load_calibration_evidence_report(**{**sources, **override})


def test_rejects_corruption_and_semantic_drift_even_with_updated_hash(sources):
    path = sources["common_ci"] / "retrieval_summary.csv"
    frame = pd.read_csv(path)
    frame["compressed_tpir20"] = .123456
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="hash"):
        load_calibration_evidence_report(**sources)
    mpath = sources["common_ci"] / "manifest.json"
    m = json.loads(mpath.read_text())
    m["summary_sha256"] = sha256_file(path)
    _json(mpath, m)
    sources["expected_hashes"]["common_ci"] = sha256_file(mpath)
    with pytest.raises(ValueError, match="rate/count"):
        load_calibration_evidence_report(**sources)


def test_common_notebook_skips_nonmatching_model_without_reading_pilot():
    root = Path(__file__).resolve().parents[2]
    n = json.loads((root / "notebooks/common/reports/00_cross_dataset_results.ipynb").read_text(encoding="utf-8"))
    cell = next(c for c in n["cells"] if c["id"] == "calibration-evidence-report")
    context = dict(PROJECT_ROOT=root, SELECTED_RUN_IDS={"survface": "another-run"},
                   MODEL_UIDS={"survface": "adaface-test"})
    exec("".join(cell["source"]), context)
    assert context["CALIBRATION_EVIDENCE"] is None
    assert "mismatch" in context["CALIBRATION_EVIDENCE_STATUS"]
