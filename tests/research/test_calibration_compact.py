import io
import json
from itertools import product

import pandas as pd
import pytest

from scripts.compact_calibration_results import (
    CONDITION, METHODS, PAIRS, aggregate, build_compact, csv_pages, load_report, preview_file, verify_compact,
)
from research.runtime.hashing import sha256_file


def _report(root):
    root.mkdir(parents=True)
    rows, pairs, sources = [], [], []
    for dataset, model, pq, seed in product(("lfw", "rfw_custom", "survface"),
            ("arcface", "adaface", "magface", "edgeface"),
            ("pq_512_m128_b8", "pq_512_m64_b8", "pq_512_m32_b8"), (0, 8972)):
        common = dict(dataset_id=dataset, model=model, compression_profile=pq, partition_seed=seed,
                      model_uid=model + "-uid", source_run_id=dataset + model)
        for family in ("fiqa", "saliency"):
            source = dict(common, family=family, result_dir=f"/evidence/{dataset}/{model}/{pq}/{seed}/{family}")
            sources.append(source)
            for target in (.01, .05, .1, .2, .3):
                for method in (METHODS[:4] if family == "fiqa" else METHODS[4:]):
                    fp = 10 if seed == 0 else 20
                    rows.append(dict(source, target_fpir=target, method=method, rank_k=20,
                        test_non_mated_count=1000, test_mated_count=100, false_accept_count=fp,
                        true_identification_at_rank_k_count=80, realized_fpir=fp / 1000, tpir_at_rank_k=.8,
                        target_met_on_test=fp / 1000 <= target, threshold_fit_on_test=False,
                        fpir_wilson95_low=0., fpir_wilson95_high=.04, tpir_cluster95_low=.7, tpir_cluster95_high=.9,
                        fallback_query_count=0, fallback_query_fraction=0., faithfulness_status="failed" if family == "saliency" else ""))
                for reference, candidate in PAIRS:
                    if (reference == "baseline") != (family == "saliency"):
                        continue
                    for metric in ("fpir", "tpir_at_rank_k"):
                        count, total = ((10 if seed == 0 else 20), 1000) if metric == "fpir" else (80, 100)
                        pairs.append(dict(source, target_fpir=target, reference_method=reference,
                            candidate_method=candidate, metric=metric, reference_successes=count, candidate_successes=count,
                            total=total, candidate_minus_reference=0., paired_bootstrap95_low=0., paired_bootstrap95_high=0.))
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(CONDITION + ["target_fpir", "method"]).agg(
        split_count=("partition_seed", "nunique"), target_met_split_count=("target_met_on_test", "sum"),
        fpir_min=("realized_fpir", "min"), fpir_median=("realized_fpir", "median"), fpir_max=("realized_fpir", "max"),
        tpir_min=("tpir_at_rank_k", "min"), tpir_median=("tpir_at_rank_k", "median"), tpir_max=("tpir_at_rank_k", "max")).reset_index()
    for name, table in (("method_summary", metrics), ("paired_comparisons", pd.DataFrame(pairs)), ("split_summary", summary)):
        table.to_csv(root / f"{name}.csv", index=False)
    manifest = dict(status="completed", observed_jobs=len(sources), expected_jobs=len(sources), sources=sources,
                    files={p.name: sha256_file(p) for p in root.glob("*.csv")})
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_full_grid_summary_preserves_misses_and_source_bytes(tmp_path):
    source = _report(tmp_path / "matrix/reports/report")
    before = {p.name: sha256_file(p) for p in source.iterdir()}
    report = load_report(source)
    metrics, overview, pairs = aggregate(report)
    assert len(metrics) == 1260 and len(overview) == 105 and len(pairs) == 3240
    row = metrics.query("target_fpir == .01").iloc[0]
    assert row.target_met_split_count == 1 and row.split_count == 2 and row.fpir_median == .015
    assert pairs.query("target_fpir == .01").both_target_met_seeds.eq(1).all()
    assert pairs.positive_ci_seeds.eq(0).all()
    output, manifest = build_compact([source], tmp_path / "compact", preferred_file_bytes=1000)
    assert manifest["above_preferred_files"] > 0  # Guidance never deletes content.
    assert verify_compact(output) == manifest
    assert build_compact([source], tmp_path / "compact", preferred_file_bytes=1000)[0] == output
    assert before == {p.name: sha256_file(p) for p in source.iterdir()}
    assert "미리보기 생략" in preview_file(output, "METRICS.md", max_bytes=1)
    (output / "START_HERE.md").write_text("corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_compact(output)
    with pytest.raises(ValueError, match="separate"):
        build_compact([source], source / "output")


@pytest.mark.parametrize("filename,column", [("split_summary.csv", "fpir_median"),
                                             ("paired_comparisons.csv", "candidate_minus_reference")])
def test_recomputed_values_reject_inconsistent_report_even_with_updated_hash(tmp_path, filename, column):
    source = _report(tmp_path / "matrix/reports/report")
    table = pd.read_csv(source / filename)
    table.loc[0, column] += .1
    table.to_csv(source / filename, index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_report(source)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["files"][filename] = sha256_file(source / filename)
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inconsistent"):
        load_report(source)


def test_pagination_keeps_wide_information_without_size_cutoff():
    original = pd.DataFrame({"description": ["정보" * 200] * 301, "value": range(301)})
    pages = csv_pages(original, max_rows=150)
    assert len(pages) == 3 and len(pages[0]) > 12000
    restored = pd.concat([pd.read_csv(io.BytesIO(p)) for p in pages], ignore_index=True)
    pd.testing.assert_frame_equal(original, restored)
