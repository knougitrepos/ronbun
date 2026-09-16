"""Continuous FIQA and retrieval-feature ablations on frozen score artifacts."""

import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import scipy

from research.calibration.conditional import (
    fit_global_threshold, fit_conditional_threshold, apply_threshold_model,
    paired_method_comparison,
)
from research.calibration.continuous import fit_continuous_threshold, apply_continuous_threshold
from research.evaluation.cluster_bootstrap import cluster_rate_draws
from research.experiments.fiqa_threshold_calibration import (
    join_fiqa_score_artifacts, _validate_comparison_contract,
)
from research.experiments.fiqa_priority_diagnostics import _reuse_completed_result
from research.experiments.fiqa_retrieval_features import join_retrieval_features
from research.experiments.fiqa_split_stability import _frame_hash
from research.runtime.hashing import canonical_sha256, sha256_file

CONTINUOUS_FEATURES = {
    "continuous_fiqa": ("fiqa_score",),
    "continuous_fiqa_margin": ("fiqa_score", "adc_margin"),
    "continuous_fiqa_margin_distortion": ("fiqa_score", "adc_margin", "top1_gallery_pq_distortion"),
    "continuous_fiqa_runnerup": ("fiqa_score", "adc_s2"),
}
TABLES = ("method_summary", "paired_comparisons", "models", "split_summary", "ablation_summary")


def summarize_continuous_ablations(method_summary, paired_comparisons):
    """Expose planned incremental contrasts without ranking/selecting a method.

    Same target FPIR is not the same realized operating point. Carry both rates
    and target checks alongside the paired TPIR evidence, including target misses.
    """
    keys = ["partition_seed", "target_fpir"]
    if method_summary.duplicated([*keys, "method"]).any():
        raise ValueError("duplicate method summary keys")
    if paired_comparisons.duplicated([*keys, "reference_method", "candidate_method", "metric"]).any():
        raise ValueError("duplicate paired comparison keys")
    contrasts = (
        ("continuous_vs_2bin", "fiqa_2bin", "continuous_fiqa"),
        ("continuous_vs_5bin", "fiqa_5bin", "continuous_fiqa"),
        ("add_margin", "continuous_fiqa", "continuous_fiqa_margin"),
        ("add_gallery_distortion", "continuous_fiqa_margin", "continuous_fiqa_margin_distortion"),
        ("runnerup_control", "continuous_fiqa", "continuous_fiqa_runnerup"),
    )
    records = []
    for (seed, target), group in method_summary.groupby(keys, sort=False):
        methods = group.set_index("method")
        for stage, reference, candidate in contrasts:
            if reference not in methods.index or candidate not in methods.index:
                continue
            evidence = paired_comparisons.loc[
                paired_comparisons.partition_seed.eq(seed) & paired_comparisons.target_fpir.eq(target)
                & paired_comparisons.reference_method.eq(reference)
                & paired_comparisons.candidate_method.eq(candidate)
            ].set_index("metric")
            if set(evidence.index) != {"fpir", "tpir_at_rank_k"}:
                raise ValueError("incomplete incremental paired evidence")
            ref, cand = methods.loc[reference], methods.loc[candidate]
            row = dict(partition_seed=seed, target_fpir=target, stage=stage,
                       reference_method=reference, candidate_method=candidate)
            for prefix, item in (("reference", ref), ("candidate", cand)):
                row.update({f"{prefix}_{col}": item[col] for col in (
                    "realized_fpir", "tpir_at_rank_k", "target_met_on_test", "target_met_by_wilson_upper")})
            for metric, rate in (("fpir", "realized_fpir"), ("tpir_at_rank_k", "tpir_at_rank_k")):
                e = evidence.loc[metric]
                if not np.isclose(e.candidate_minus_reference, cand[rate]-ref[rate], rtol=0, atol=1e-12):
                    raise ValueError("paired delta disagrees with method rates")
                row.update({f"delta_{metric}": e.candidate_minus_reference,
                            f"delta_{metric}_ci_low": e.paired_bootstrap95_low,
                            f"delta_{metric}_ci_high": e.paired_bootstrap95_high,
                            f"{metric}_resampling_unit": e.resampling_unit})
            low, high = row["delta_tpir_at_rank_k_ci_low"], row["delta_tpir_at_rank_k_ci_high"]
            row["tpir_ci_direction"] = "increase" if low > 0 else "decrease" if high < 0 else "includes_zero"
            row["operating_point_status"] = (
                "candidate_misses_target" if not cand.target_met_on_test else
                "reference_misses_target" if not ref.target_met_on_test else "both_point_targets_met")
            row["same_realized_fpir_enforced"] = False
            row["test_based_selection"] = False
            records.append(row)
    return pd.DataFrame(records)


def run_continuous_calibration(
    condition, fiqa, *, retrieval=None, methods=("continuous_fiqa",),
    partition_seeds=(8972,), target_fpirs=(.01, .05, .1, .2, .3),
    safety_fraction=.3, shrinkage_strength=200., minimum_group_non_mated=100,
    knot_quantiles=(1/3, 2/3), smoothing=.01, ridge=.001, max_iterations=2000,
    margin_slope_cap=.95, resamples=2000, bootstrap_seed=8972, progress=None,
):
    """Include Global-safe and 2/5-bin controls; no test-based model selection."""
    methods, seeds, targets = tuple(methods), tuple(partition_seeds), tuple(target_fpirs)
    if (not methods or len(set(methods)) != len(methods) or not set(methods) <= set(CONTINUOUS_FEATURES)
            or methods[0] != "continuous_fiqa"):
        raise ValueError("explicit methods must begin with continuous_fiqa")
    if "continuous_fiqa_margin_distortion" in methods and "continuous_fiqa_margin" not in methods:
        raise ValueError("distortion ablation requires the margin-only control")
    if (isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100
            or isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0):
        raise ValueError("at least 100 integer resamples and non-negative integer bootstrap_seed required")
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)):
        raise ValueError("unique non-negative integer seeds required")
    if not targets or targets != tuple(sorted(set(targets))) or any(not 0 < t < 1 for t in targets):
        raise ValueError("unique increasing FPIR targets inside (0,1) required")
    if condition.manifest.get("dataset_id") != "survface":
        raise ValueError("identity CI semantics are currently audited for SurvFace only")
    cal, test = join_fiqa_score_artifacts(condition, fiqa)
    _validate_comparison_contract(cal, test, condition_manifest=condition.manifest, fiqa_manifest=fiqa.manifest)
    if condition.manifest["score_space"] != "negative_squared_l2_adc":
        raise ValueError("this experiment requires the ADC score space")
    needs_retrieval = any(len(CONTINUOUS_FEATURES[name]) > 1 for name in methods)
    if needs_retrieval:
        if retrieval is None:
            raise ValueError("selected methods require a completed retrieval feature artifact")
        if (retrieval["manifest"].get("status") != "completed" or
                retrieval["manifest"]["spec"]["condition_manifest_sha256"] != canonical_sha256(condition.manifest)):
            raise ValueError("retrieval feature lineage mismatch")
        cal = join_retrieval_features(cal, retrieval["calibration"])
        test = join_retrieval_features(test, retrieval["test"])
    summary, paired, fitted = [], [], []
    for seed in seeds:
        for target in targets:
            common = dict(target_fpir=target, safety_fraction=safety_fraction,
                          partition_seed=seed, score_space=condition.manifest["score_space"])
            models = {"global_safe": fit_global_threshold(cal, partition_column="identity_id", **common)}
            for bins in (2, 5):
                models[f"fiqa_{bins}bin"] = fit_conditional_threshold(
                    cal, bin_count=bins, partition_column="identity_id",
                    shrinkage_strength=shrinkage_strength,
                    minimum_group_non_mated=minimum_group_non_mated, **common)
            timings = {}
            for name in methods:
                if progress:
                    progress({"stage": "fit", "seed": seed, "target_fpir": target, "method": name})
                start = perf_counter()
                models[name] = fit_continuous_threshold(
                    cal, features=CONTINUOUS_FEATURES[name], method=name,
                    knot_quantiles=knot_quantiles, smoothing=smoothing, ridge=ridge,
                    max_iterations=max_iterations, margin_slope_cap=margin_slope_cap, **common)
                timings[name] = perf_counter() - start
            evaluations = {}
            for name, model in models.items():
                start = perf_counter()
                evaluations[name] = (apply_continuous_threshold(test, model) if name in methods
                                     else apply_threshold_model(test, model))
                fitted.append({"partition_seed": seed, "target_fpir": target, "method": name,
                               "model_uid": model.model_uid, "fit_seconds": timings.get(name, np.nan),
                               "evaluation_seconds": perf_counter() - start,
                               "model_json": json.dumps(model.as_dict(), ensure_ascii=False, allow_nan=False)})
            names = list(models)
            mated = test.is_mated.astype(bool).to_numpy()
            events = np.column_stack([evaluations[n].decisions.true_identification_at_rank_k.to_numpy()[mated]
                                      for n in names])
            draws = cluster_rate_draws(test.loc[mated, "identity_id"], events,
                                       resamples=resamples, seed=bootstrap_seed)
            for i, name in enumerate(names):
                low, high = np.quantile(draws[:, i], [.025, .975])
                summary.append({**evaluations[name].summary, "method": name, "partition_seed": seed,
                                "tpir_cluster95_low": low, "tpir_cluster95_high": high})
            pairs = [("global_safe", n) for n in names[1:]]
            pairs += [("fiqa_2bin", "fiqa_5bin"), ("fiqa_2bin", "continuous_fiqa"),
                      ("fiqa_5bin", "continuous_fiqa")]
            pairs += [("continuous_fiqa", n) for n in methods[1:]]
            if "continuous_fiqa_margin" in methods and "continuous_fiqa_margin_distortion" in methods:
                pairs.append(("continuous_fiqa_margin", "continuous_fiqa_margin_distortion"))
            for ref, cand in pairs:
                evidence = paired_method_comparison(evaluations[ref], evaluations[cand],
                                                    resamples=resamples, random_seed=bootstrap_seed)
                for metric in ("fpir", "tpir_at_rank_k"):
                    item = dict(evidence[metric])
                    unit = "query"
                    if metric == "tpir_at_rank_k":
                        low, high = np.quantile(draws[:, names.index(cand)]-draws[:, names.index(ref)], [.025, .975])
                        item.update(paired_bootstrap95_low=low, paired_bootstrap95_high=high)
                        unit = "mated_identity_cluster"
                    paired.append({"partition_seed": seed, "target_fpir": target,
                                   "reference_method": ref, "candidate_method": cand, "metric": metric,
                                   **item, "resampling_unit": unit,
                                   "resamples": resamples if metric != "fpir" else evidence["resamples"],
                                   "bootstrap_seed": bootstrap_seed if metric != "fpir" else evidence["random_seed"]})
    metrics = pd.DataFrame(summary)
    split_summary = metrics.groupby(["method", "target_fpir"]).agg(
        split_count=("partition_seed", "count"), target_met_split_count=("target_met_on_test", "sum"),
        fpir_min=("realized_fpir", "min"), fpir_median=("realized_fpir", "median"),
        fpir_max=("realized_fpir", "max"), tpir_min=("tpir_at_rank_k", "min"),
        tpir_median=("tpir_at_rank_k", "median"), tpir_max=("tpir_at_rank_k", "max"),
    ).reset_index()
    settings = dict(methods=methods, partition_seeds=seeds, target_fpirs=targets,
                    safety_fraction=safety_fraction, shrinkage_strength=shrinkage_strength,
                    minimum_group_non_mated=minimum_group_non_mated, knot_quantiles=knot_quantiles,
                    smoothing=smoothing, ridge=ridge, max_iterations=max_iterations,
                    margin_slope_cap=margin_slope_cap, resamples=resamples, bootstrap_seed=bootstrap_seed)
    paths = [Path(__file__), Path(__file__).with_name("fiqa_retrieval_features.py"),
             Path(__file__).with_name("fiqa_threshold_calibration.py"),
             Path(__file__).parents[1]/"calibration/continuous.py",
             Path(__file__).parents[1]/"calibration/conditional.py",
             Path(__file__).parents[1]/"calibration/rejection.py",
             Path(__file__).parents[1]/"evaluation/cluster_bootstrap.py",
             Path(__file__).parents[1]/"evaluation/metrics.py"]
    manifest = {"artifact_type": "fiqa_continuous_calibration", "schema_version": 2,
                "metric_contract": condition.manifest["metric_contract"],
                "score_space": condition.manifest["score_space"],
                "source_run_id": condition.manifest["source_run_id"],
                "model_uid": condition.manifest["model_uid"],
                "condition_manifest_sha256": canonical_sha256(condition.manifest),
                "condition_uid": condition.condition_uid,
                "fiqa_manifest_sha256": canonical_sha256(fiqa.manifest),
                "retrieval_manifest_sha256": canonical_sha256(retrieval["manifest"]) if needs_retrieval else None,
                "input_frame_sha256": {"calibration": _frame_hash(cal), "test": _frame_hash(test)},
                "settings": json.loads(json.dumps(settings)),
                "implementation_sha256": {p.name: sha256_file(p) for p in paths},
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__},
                "uncertainty": {"tpir": "query_weighted_mated_identity_cluster",
                                "fpir": "query_only_unknown_identity_unavailable",
                                "threshold_uncertainty_included": False, "formal_fpir_guarantee": False,
                                "multiple_comparison_adjustment": "none", "test_based_selection": False,
                                "between_splits": "descriptive_not_ci_not_independent_replications"}}
    manifest["result_uid"] = "fiqa-continuous-" + canonical_sha256(manifest)[:24]
    paired_frame = pd.DataFrame(paired)
    ablations = summarize_continuous_ablations(metrics, paired_frame)
    return {"method_summary": metrics, "paired_comparisons": paired_frame, "ablation_summary": ablations,
            "models": pd.DataFrame(fitted), "split_summary": split_summary, "manifest": manifest}


def write_continuous_calibration(root, result):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / result["manifest"]["result_uid"]
    if destination.exists():
        return _reuse_completed_result(destination, result, TABLES)
    staging = root / (".staging-" + uuid4().hex)
    staging.mkdir()
    files = {}
    for name in TABLES:
        path = staging / f"{name}.csv"
        result[name].to_csv(path, index=False)
        files[path.name] = sha256_file(path)
    manifest = {**result["manifest"], "status": "completed", "files": files}
    (staging/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    os.rename(staging, destination)
    return destination
