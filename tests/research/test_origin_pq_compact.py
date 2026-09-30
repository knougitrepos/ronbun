"""Compact exports preserve conditions while keeping physical file counts fixed."""
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest


# This exporter needs no model/search runtime. Direct loading deliberately avoids
# experiments/__init__.py eagerly importing optional sklearn/GPU dependencies.
_SPEC = importlib.util.spec_from_file_location("origin_pq_compact_under_test",
    Path(__file__).parents[2] / "research/experiments/origin_pq_compact.py")
compact = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compact)


def _tables(seeds=(0, 1), targets=(.1, .2), models=("arcface",)):
    methods, paired, interactions, curves, diagnostic = [], [], [], [], []
    for model in models:
        for seed in seeds:
            context = dict(dataset_id="lfw", model=model, source_run_id="run-" + model, partition_seed=seed)
            for target in targets:
                for profile in ("origin_512", "pq_512_m32_b8"):
                    for method in ("global_safe", "fiqa_5bin", "continuous_fiqa"):
                        success, false_accept = 70 + seed, 10 + seed
                        methods.append(dict(**context, compression_profile=profile, method=method, target_fpir=target,
                            test_probe_count=200, test_mated_count=100, test_non_mated_count=100, rank_k=20,
                            score_space="cosine_similarity" if profile == "origin_512" else "negative_squared_l2_adc",
                            false_accept_count=false_accept, true_identification_at_rank_k_count=success,
                            rank_failure_count=10, threshold_failure_count=100-success-10,
                            rank_failure_rate=.1, threshold_failure_rate=(100-success-10)/100,
                            rank_k_ceiling=.9, realized_fpir=false_accept/100, tpir_at_rank_k=success/100,
                            target_met_on_test=false_accept/100 <= target, tpir_cluster95_low=.6+seed*.01,
                            tpir_cluster95_high=.8+seed*.01))
                        for fpir in (0., .1, 1.):
                            curves.append(dict(**context, compression_profile=profile, method=method,
                                fitted_target_fpir=target, diagnostic_fpir=fpir, diagnostic_tpir=min(fpir+.1,.9),
                                diagnostic_only=True, interpolated_test_curve=True,
                                deployment_threshold_selected=False, diagnostic_ci_available=False))
                pair_context = dict(**context, target_fpir=target)
                paired.append(dict(**pair_context, comparison="compression_vs_origin", reference_profile="origin_512",
                    reference_method="continuous_fiqa", candidate_profile="pq_512_m32_b8",
                    candidate_method="continuous_fiqa", metric="tpir_at_rank_k", reference_successes=70,
                    candidate_successes=70+seed, both_successes=60, total=100, candidate_minus_reference=seed/100,
                    paired_bootstrap95_low=-.01, paired_bootstrap95_high=.02, resamples=100,
                    resampling_unit="mated_identity_cluster", reference_realized_fpir=.1, candidate_realized_fpir=.11,
                    both_target_met=False, comparison_basis="calibration_fixed_same_target_not_matched_actual_fpir"))
                interactions.append(dict(**pair_context, compression_profile="pq_512_m32_b8", method="continuous_fiqa",
                    pq_gain=.03, origin_gain=.02, interaction=.01, paired_bootstrap95_low=-.01,
                    paired_bootstrap95_high=.03, all_four_target_met=False,
                    comparison_basis="calibration_fixed_same_target_not_matched_actual_fpir",
                    pq_method_fpir=.11, pq_global_fpir=.1, origin_method_fpir=.09, origin_global_fpir=.1))
                for fpir in (0., .1, 1.):
                    diagnostic.append(dict(**context, compression_profile="pq_512_m32_b8", method="continuous_fiqa",
                        fitted_target_fpir=target, diagnostic_fpir=fpir, pq_gain=.03, origin_gain=.02, interaction=.01,
                        diagnostic_only=True, interpolated_test_curve=True, diagnostic_ci_available=False))
    return {"method_summary": pd.DataFrame(methods), "paired_comparisons": pd.DataFrame(paired),
            "interactions": pd.DataFrame(interactions), "diagnostic_curves": pd.DataFrame(curves),
            "diagnostic_interactions": pd.DataFrame(diagnostic)}


def test_summary_preserves_all_conditions_and_shared_test_counts():
    result = compact.summarize_for_chat(_tables())
    frame = result["operating_performance"]
    assert len(frame) == 2 * 3 * 2
    assert frame.seed_count.eq(2).all()
    assert frame.test_mated_count.eq(100).all()  # Not 200 after two seeds.
    assert frame.true_identification_at_rank_k_count_min.eq(70).all()
    assert frame.true_identification_at_rank_k_count_median.eq(70.5).all()
    assert frame.true_identification_at_rank_k_count_max.eq(71).all()
    assert frame.loc[frame.target_fpir.eq(.1), "target_met_on_test_seed_count"].eq(1).all()
    assert frame.loc[frame.target_fpir.eq(.2), "target_met_on_test_seed_count"].eq(2).all()
    assert result["operating_ci_endpoints"].tpir_cluster95_low_min.eq(.6).all()
    assert result["operating_ci_endpoints"].tpir_cluster95_low_max.eq(.61).all()
    assert len(result["diagnostic_curves"]) == 2 * 3 * 2 * 3
    assert set(result["diagnostic_curves"].diagnostic_fpir) == {0., .1, 1.}
    assert result["paired_effects"].total.eq(100).all()
    assert result["paired_effects"].candidate_minus_reference_max.eq(.01).all()


def test_only_three_physical_files_all_archive_pages_and_deterministic_reuse(tmp_path):
    tables = _tables(models=("arcface", "adaface"))
    root = compact.write_chat_bundle(tmp_path, tables, {"checkpoint": "test.sqlite"}, 4, 4,
                                     max_page_rows=3, page_bytes_guidance=3000)
    assert sorted(p.name for p in root.iterdir()) == ["START_HERE.md", "analysis.zip", "manifest.json"]
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in root.iterdir()}
    reused = compact.write_chat_bundle(tmp_path, tables, {"checkpoint": "test.sqlite"}, 4, 4,
                                       max_page_rows=3, page_bytes_guidance=3000)
    assert reused == root
    assert before == {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in root.iterdir()}
    manifest = compact.validate_chat_bundle(root)
    assert manifest["experiment_status"] == "completed"
    assert manifest["ci_endpoint_ranges_are_merged_intervals"] is False
    with zipfile.ZipFile(root / "analysis.zip") as archive:
        index = pd.read_csv(io.BytesIO(archive.read("INDEX.csv")))
        assert set(index.model) == {"arcface", "adaface"}
        assert index.rows.le(3).all()
        for name, expected in manifest["summarized_table_rows"].items():
            assert index.loc[index.table.eq(name), "rows"].sum() == expected
        frame = pd.concat([pd.read_csv(io.BytesIO(archive.read(p))) for p in
                           index.loc[index.table.eq("operating_performance"), "entry"]])
        assert len(frame) == 24
        assert set(frame.target_fpir) == {.1, .2}
        assert set(frame.method) == {"global_safe", "fiqa_5bin", "continuous_fiqa"}


def test_partial_coverage_names_missing_jobs_without_fabricated_zero_rows(tmp_path):
    expected = [("run-arcface", 0), ("run-arcface", 1)]
    root = compact.write_chat_bundle(tmp_path, _tables(seeds=(0,)), {}, expected, expected[:1])
    manifest = compact.validate_chat_bundle(root)
    assert manifest["experiment_status"] == "partial"
    with zipfile.ZipFile(root / "analysis.zip") as archive:
        coverage = json.loads(archive.read("coverage.json"))
        assert coverage["missing_job_keys"] == [["run-arcface", 1]]
        assert coverage["completed_jobs"] == 1
    assert "**partial**" in (root / "START_HERE.md").read_text(encoding="utf8")
    # Tuple job keys are normalized before immutable-spec comparison.
    assert compact.write_chat_bundle(tmp_path, _tables(seeds=(0,)), {}, expected, expected[:1]) == root


def test_empty_interrupted_campaign_export_is_explicitly_partial(tmp_path):
    root = compact.write_chat_bundle(tmp_path, {}, {}, 240, 0)
    manifest = compact.validate_chat_bundle(root)
    assert manifest["experiment_status"] == "partial"
    assert manifest["spec"]["coverage"]["missing_job_count"] == 240
    assert manifest["summarized_table_rows"] == {}


def test_archive_tampering_rejected_without_overwrite(tmp_path):
    root = compact.write_chat_bundle(tmp_path, _tables(), {}, 2, 2)
    with (root / "analysis.zip").open("ab") as handle:
        handle.write(b"tampered")
    corrupted = (root / "analysis.zip").read_bytes()
    with pytest.raises(ValueError, match="hash mismatch"):
        compact.write_chat_bundle(tmp_path, _tables(), {}, 2, 2)
    assert (root / "analysis.zip").read_bytes() == corrupted


@pytest.mark.parametrize("mutation,error", [
    (lambda t: t["method_summary"].loc.__setitem__((0, "rank_failure_count"), 9), "counts/rates"),
    (lambda t: t["method_summary"].loc.__setitem__((0, "target_met_on_test"), False), "target attainment"),
    (lambda t: t.__setitem__("method_summary", pd.concat([t["method_summary"], t["method_summary"].iloc[:1]])), "duplicate"),
])
def test_malformed_evidence_rejected(mutation, error):
    tables = _tables()
    mutation(tables)
    with pytest.raises(ValueError, match=error):
        compact.summarize_for_chat(tables)


def test_actual_completed_job_count_must_match_method_table(tmp_path):
    with pytest.raises(ValueError, match="coverage differs"):
        compact.write_chat_bundle(tmp_path, _tables(), {}, 3, 3)


def test_single_large_row_is_preserved_and_flagged(tmp_path):
    root = compact.write_chat_bundle(tmp_path, _tables(seeds=(0,)), {}, 1, 1, page_bytes_guidance=50)
    with zipfile.ZipFile(root / "analysis.zip") as archive:
        index = pd.read_csv(io.BytesIO(archive.read("INDEX.csv")))
        assert index.rows.eq(1).all()
        assert index.exceeds_byte_guidance.all()
