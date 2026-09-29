import copy
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from research.experiments.fiqa_continuous_calibration import run_continuous_calibration, write_continuous_calibration
from research.experiments.origin_pq_inputs import (
    ORIGIN_MODE, ORIGIN_PROFILE, ORIGIN_SPACE, exact_origin_rows, origin_rows_from_ledger,
)
from research.experiments.origin_vs_pq_calibration import (
    OriginPQSettings, VerifiedPQModels, failure_decomposition, frozen_score_diagnostic,
    run_origin_pq_split, run_origin_pq_campaign,
    _validate_split_tables,
)
from research.runtime.hashing import canonical_sha256
from test_fiqa_split_stability import _inputs as base_inputs


def inputs(tmp_path):
    pq, artifacts = base_inputs(tmp_path)
    pq.manifest.update(source_run_id="test-run", calibration_seed=42)
    for col in ("source_run_manifest_sha256", "source_freeze_manifest_sha256",
                "selected_manifest_sha256", "prepared_population_manifest_sha256"):
        pq.manifest[col] = col
    origin = copy.deepcopy(pq)
    origin.manifest.update(artifact_type="origin_calibration_test_score_tables", condition_uid="origin-test",
                           compression_profile=ORIGIN_PROFILE, search_mode=ORIGIN_MODE, score_space=ORIGIN_SPACE)
    for frame in (origin.calibration, origin.test):
        frame["compression_profile"] = ORIGIN_PROFILE
        frame["search_mode"] = ORIGIN_MODE
        frame["score_space"] = ORIGIN_SPACE
    return {ORIGIN_PROFILE: origin, "pq_512_m128_b8": pq}, artifacts[1]


def options():
    return OriginPQSettings(target_fpirs=(.1,), resamples=100, minimum_group_non_mated=5,
                             diagnostic_fpir_grid=(0., .01, .1, 1.))


def test_exact_origin_search_keeps_genuine_score_and_rank_exit():
    gallery = np.array([[1., 0.], [.8, .6], [0., 1.]])
    rows = exact_origin_rows(np.array([[1., 0.], [1., 0.]]), gallery, ["a", "b"],
                             ["second", "unknown"], ["first", "second", "third"], top_k=2)
    assert rows.loc[0, "origin_top1_score"] == 1.
    assert rows.loc[0, "origin_true_identity_rank"] == 2
    assert rows.loc[0, "origin_true_identity_score"] == pytest.approx(.8)
    assert not rows.loc[1, "is_mated"]
    assert np.isnan(rows.loc[1, "origin_true_identity_rank"])
    exit_rows = exact_origin_rows(np.array([[1., 0.]]), gallery, ["a"], ["second"],
                                  ["first", "second", "third"], top_k=1)
    assert exit_rows.loc[0, "is_mated"]
    assert not exit_rows.loc[0, "origin_top_k_correct"]
    assert np.isnan(exit_rows.loc[0, "origin_true_identity_score"])


def test_origin_adapter_does_not_use_compressed_genuine_score(tmp_path):
    conditions, _ = inputs(tmp_path)
    template = conditions["pq_512_m128_b8"].test.iloc[:2].copy()
    raw = pd.DataFrame({
        "query_id": template.sample_id, "query_identity_id": template.identity_id,
        "is_mated": template.is_mated, "top_k": 20, "origin_score_space": ORIGIN_SPACE,
        "origin_top1_score": [.9, .7], "origin_true_identity_rank": [2., np.nan],
        "origin_true_identity_score": [.2, np.nan], "origin_rank1_correct": False,
        "origin_top_k_correct": [True, False],
    })
    out = origin_rows_from_ledger(raw, template)
    assert out.loc[0, "score"] == .9 and out.loc[0, "true_identity_score"] == .2
    raw.loc[0, "query_identity_id"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        origin_rows_from_ledger(raw, template)


def test_joint_comparison_has_zero_interaction_for_equal_scores(tmp_path):
    conditions, fiqa = inputs(tmp_path)
    result = run_origin_pq_split(conditions, fiqa, settings=options())
    assert len(result["method_summary"]) == 6
    assert result["interactions"].interaction.eq(0).all()
    assert result["interactions"].paired_bootstrap95_low.eq(0).all()
    assert result["diagnostic_interactions"].interaction.eq(0).all()
    summary = result["method_summary"]
    assert (summary.rank_failure_count + summary.threshold_failure_count
            + summary.true_identification_at_rank_k_count).equals(summary.test_mated_count)
    assert result["partition_inventory"].assignment_sha256.nunique() == 1
    paired = result["paired_comparisons"].query("comparison == 'compression_vs_origin'")
    assert paired.candidate_minus_reference.eq(0).all()
    assert paired.paired_bootstrap95_low.eq(0).all()
    assert result["diagnostic_curves"].deployment_threshold_selected.eq(False).all()


def test_test_scores_never_change_fitted_models(tmp_path):
    conditions, fiqa = inputs(tmp_path)
    before = run_origin_pq_split(conditions, fiqa, settings=options())
    changed = copy.deepcopy(conditions)
    for condition in changed.values():
        condition.test.loc[~condition.test.is_mated, "score"] += .5
    after = run_origin_pq_split(changed, fiqa, settings=options())
    assert before["models"].model_json.equals(after["models"].model_json)
    assert after["method_summary"].realized_fpir.min() > before["method_summary"].realized_fpir.max()
    assert before["partition_inventory"].equals(after["partition_inventory"])


def test_legacy_pq_calibration_maxima_do_not_require_genuine_scores(tmp_path):
    conditions, fiqa = inputs(tmp_path)
    from dataclasses import replace
    pq = conditions["pq_512_m128_b8"]
    conditions["pq_512_m128_b8"] = replace(pq, calibration=pq.calibration.drop(
        columns=["true_identity_rank", "true_identity_score"]))
    result = run_origin_pq_split(conditions, fiqa, settings=options())
    assert len(result["method_summary"]) == 6
    assert result["interactions"].interaction.eq(0).all()


@pytest.mark.parametrize("corruption", ["missing_method", "wrong_denominator", "wrong_failure_count", "wrong_model"])
def test_split_semantics_reject_invalid_results_even_with_valid_file_hashes(tmp_path, corruption):
    conditions, fiqa = inputs(tmp_path)
    result = run_origin_pq_split(conditions, fiqa, settings=options())
    if corruption == "missing_method":
        result["method_summary"] = result["method_summary"].iloc[1:]
    elif corruption == "wrong_denominator":
        result["method_summary"].loc[0, "test_mated_count"] += 1
    elif corruption == "wrong_failure_count":
        result["method_summary"].loc[0, "rank_failure_count"] += 1
    else:
        result["models"].loc[0, "model_uid"] = "wrong"
    with pytest.raises(ValueError, match="grid|denominator|counts|identity"):
        _validate_split_tables(result, conditions, 8972, options())


@pytest.mark.parametrize("column", ["sample_id", "identity_id", "is_mated", "aligned_content_sha256"])
def test_cohort_drift_rejected(tmp_path, column):
    conditions, fiqa = inputs(tmp_path)
    changed = conditions["pq_512_m128_b8"].test
    changed.loc[0, column] = False if column == "is_mated" else "drift"
    with pytest.raises(ValueError, match="cohort"):
        run_origin_pq_split(conditions, fiqa, settings=options())


def test_calibration_test_identity_overlap_rejected(tmp_path):
    conditions, fiqa = inputs(tmp_path)
    for condition in conditions.values():
        condition.calibration.loc[0, "identity_id"] = condition.test.loc[0, "identity_id"]
    with pytest.raises(ValueError, match="overlap"):
        run_origin_pq_split(conditions, fiqa, settings=options())


def test_failure_split_and_curve_use_genuine_not_top1():
    from types import SimpleNamespace
    rows = pd.DataFrame(dict(is_mated=[True, True, True, False, False], top_k_correct=[True, True, False, False, False],
                             score=[.9, .8, .9, .7, .4], true_identity_score=[.2, .8, np.nan, np.nan, np.nan],
                             applied_threshold=.5, true_identification_at_rank_k=[False, True, False, False, False]))
    evaluation = SimpleNamespace(decisions=rows)
    counts = failure_decomposition(evaluation)
    assert counts["rank_failure_count"] == 1 and counts["threshold_failure_count"] == 1
    curve = frozen_score_diagnostic(evaluation, (0., .25, .5, 1.))
    assert curve.diagnostic_tpir.iloc[0] == pytest.approx(1/3)
    assert curve.diagnostic_tpir.iloc[-1] == pytest.approx(2/3)
    assert curve.diagnostic_tpir.max() < 1


def test_existing_pq_models_reused_only_for_matching_fit_settings(tmp_path):
    conditions, fiqa = inputs(tmp_path)
    pq = conditions["pq_512_m128_b8"]
    old = run_continuous_calibration(pq, fiqa, partition_seeds=(8972,), target_fpirs=(.1,),
                                     resamples=100, minimum_group_non_mated=5)
    directory = write_continuous_calibration(tmp_path / "existing", old)
    manifest = json.loads((directory / "manifest.json").read_text())
    report = tmp_path / "report"
    report.mkdir()
    (report / "manifest.json").write_text(json.dumps(dict(status="completed", artifact_type="calibration_matrix_report",
        files={}, missing_jobs=[], sources=[dict(family="fiqa", source_run_id="test-run",
            compression_profile="pq_512_m128_b8", partition_seed=8972, result_dir=str(directory),
            result_manifest_sha256=canonical_sha256(manifest))])))
    cache = VerifiedPQModels(report)
    reuse = cache.load(conditions, fiqa, 8972, options())
    result = run_origin_pq_split(conditions, fiqa, settings=options(), reused_models=reuse)
    assert result["models"].groupby("compression_profile").fitting_reused.all().to_dict() == {
        ORIGIN_PROFILE: False, "pq_512_m128_b8": True}
    for method in ("global_safe", "fiqa_5bin", "continuous_fiqa"):
        a = old["method_summary"].query("method == @method").iloc[0]
        b = result["method_summary"].query("method == @method and compression_profile == 'pq_512_m128_b8'").iloc[0]
        assert a.realized_fpir == b.realized_fpir and a.tpir_at_rank_k == b.tpir_at_rank_k
    with pytest.raises(ValueError, match="setting"):
        cache.load(conditions, fiqa, 8972, OriginPQSettings(target_fpirs=(.1,), resamples=100))
    (directory / "models.csv").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        cache.load(conditions, fiqa, 8972, options())


def test_campaign_resume_and_output_tampering(tmp_path, monkeypatch):
    import research.experiments.origin_vs_pq_calibration as module
    conditions, fiqa = inputs(tmp_path)
    monkeypatch.setattr(module, "load_run_inputs", lambda *a, **k: (conditions, fiqa))
    plan = pd.DataFrame([dict(source_run_dir=str(tmp_path/"source"), condition_dir=str(tmp_path/"condition"),
                             fiqa_dir=str(tmp_path/"fiqa"), source_run_id="test-run", dataset_id="survface",
                             model="arcface", compression_profile="pq_512_m128_b8")])
    output = tmp_path / "output"
    first = run_origin_pq_campaign(plan, output, partition_seeds=(8972,), settings=options(), progress=None)
    monkeypatch.setattr(module, "run_origin_pq_split", lambda *a, **k: pytest.fail("completed fit repeated"))
    second = run_origin_pq_campaign(plan, output, partition_seeds=(8972,), settings=options(), progress=None)
    assert first["report_dir"] == second["report_dir"]
    assert first["method_summary"].equals(second["method_summary"])
    job = Path(first["source_receipts"].iloc[0].result_dir)
    (job / "method_summary.csv").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_origin_pq_campaign(plan, output, partition_seeds=(8972,), settings=options(), progress=None)


def test_notebook_is_thin_valid_and_readonly_by_default():
    path = Path(__file__).parents[2] / "notebooks/calibration/03_origin_vs_pq_fiqa_calibration.ipynb"
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    code = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    assert "EXECUTE = False" in code[0]
    assert "SOURCE_REPORT_DIR" in code[0] and "PARTITION_SEEDS" in code[0]
    assert any("inspect_origin_pq_experiment" in s for s in code)
    assert any("run_origin_pq_campaign" in s for s in code)
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(path), "exec")
            assert not cell.outputs and cell.execution_count is None
