"""Gate-first, offline saliency incremental calibration; no Grad-CAM inference.

High/Low/Random occlusion drops are faithfulness evidence, NEVER predictors.
Test-only saliency/faithfulness cannot authorize fitting. Missing features are
not imputed, and no subset of easier queries silently replaces the 01 cohort.
"""
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import scipy

from research.calibration.conditional import paired_method_comparison
from research.calibration.continuous import fit_continuous_threshold, apply_continuous_threshold
from research.evaluation.cluster_bootstrap import cluster_rate_draws
from research.evaluation.saliency_faithfulness import (
    assess_saliency_faithfulness_reliability, summarize_faithfulness, FAITHFULNESS_METRICS,
)
from research.experiments.fiqa_continuous_calibration import CONTINUOUS_FEATURES
from research.experiments.fiqa_priority_diagnostics import _reuse_completed_result
from research.experiments.fiqa_retrieval_features import join_retrieval_features
from research.experiments.fiqa_split_stability import _frame_hash
from research.experiments.fiqa_threshold_calibration import (
    assess_saliency_incremental_readiness, join_fiqa_score_artifacts, _validate_comparison_contract,
)
from research.runtime.hashing import canonical_sha256, sha256_file

SALIENCY_FEATURES = ("outside_face_attention", "saliency_entropy")
TABLES = ("method_summary", "paired_comparisons", "models", "split_summary")


def _verified_file(root, entry):
    root = Path(root).resolve()
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("artifact path escapes its root")
    if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise ValueError(f"artifact size/hash mismatch: {path}")
    return path


def load_saliency_incremental_inputs(condition, saliency_directory, faithfulness_directory):
    """Read explicit, hash-checked sources; old population artifacts are diagnostic-only.

    A calibration-ready saliency manifest uses artifact_type
    saliency_calibration_features, schema_version=1, status=completed,
    condition_manifest_sha256 and gallery_contract=split_matched_origin_top1.
    It must include the same lineage keys and saliency_features file receipt as
    population_gradcam_saliency. CSV includes aligned_content_sha256 and split.
    Faithfulness uses the existing v2 format plus evaluation_split=calibration
    with verified calibration membership; unbound development evidence is diagnostic-only.
    """
    sal_root, faith_root = Path(saliency_directory), Path(faithfulness_directory)
    sm = json.loads((sal_root / "manifest.json").read_text(encoding="utf8"))
    fm = json.loads((faith_root / "manifest.json").read_text(encoding="utf8"))
    kind = (sm.get("artifact_type"), sm.get("schema_version"))
    if kind not in (("population_gradcam_saliency", 3), ("saliency_calibration_features", 1)):
        raise ValueError("unsupported saliency artifact")
    if fm.get("artifact_type") != "open_set_gradcam_faithfulness" or fm.get("schema_version") != 2:
        raise ValueError("requires v2 High/Low/Random faithfulness")
    cm = condition.manifest
    for key in ("dataset_id", "model_uid", "origin_embedding_artifact_uid"):
        if sm.get(key) != cm[key] or fm.get(key) != cm[key]:
            raise ValueError(f"saliency/faithfulness lineage mismatch: {key}")
    if sm.get("extraction_uid") != cm["extraction_uid"] or fm.get("source_run_id") != cm["source_run_id"]:
        raise ValueError("saliency extraction/source run mismatch")
    if not sm.get("saliency_spec_uid") or sm["saliency_spec_uid"] != fm.get("saliency_spec_uid"):
        raise ValueError("saliency specification mismatch")
    if sm.get("target_name") != "origin_top1_gallery_cosine" or fm.get("saliency_target_name") != sm["target_name"]:
        raise ValueError("only prespecified label-free origin top1 saliency is supported")
    path = _verified_file(sal_root, sm["saliency_features"])
    wanted = {"sample_id", "saliency_target_name", "heatmap_available", "gradcam_valid_heatmap",
              "aligned_content_sha256", "split", *SALIENCY_FEATURES}
    sal = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    receipts = {e["path"]: e for e in fm["outputs"]}
    summary = pd.read_csv(_verified_file(faith_root, receipts["faithfulness_summary.csv"]))
    wanted_faith = {"sample_id", "identity_id", "aligned_content_sha256", "split", *FAITHFULNESS_METRICS}
    rows = pd.read_csv(_verified_file(faith_root, receipts["faithfulness_rows.csv"]),
                       usecols=lambda c: c in wanted_faith)
    if rows.sample_id.isna().any() or rows.sample_id.duplicated().any() or rows.empty:
        raise ValueError("invalid faithfulness sample IDs")
    # Historical test evidence remains readable for diagnostics. New calibration
    # evidence must also reproduce the statistical contract and both paired CIs.
    for metric in FAITHFULNESS_METRICS:
        selected = summary.loc[summary.group.eq("all") & summary.metric.eq(metric)]
        if (len(selected) != 1 or selected.iloc[0].sample_count != len(rows)
                or not np.isfinite(rows[metric]).all()
                or not np.isclose(selected.iloc[0]["mean"], rows[metric].mean(), atol=1e-9, rtol=0)):
            raise ValueError("faithfulness summary/row evidence mismatch")
    if fm.get("evaluation_split") == "calibration":
        missing = wanted_faith - set(rows)
        if missing:
            raise ValueError(f"calibration faithfulness columns missing: {sorted(missing)}")
        stats = fm.get("statistics", {})
        repeats, seed = stats.get("bootstrap_repeats"), stats.get("seed")
        if (stats.get("bootstrap_method") != "identity_cluster" or stats.get("confidence_level") != .95
                or type(repeats) is not int or repeats < 100 or type(seed) is not int or seed < 0):
            raise ValueError("invalid calibration faithfulness CI settings")
        if rows.identity_id.isna().any() or rows.identity_id.astype(str).eq("").any() or rows.identity_id.nunique() < 2:
            raise ValueError("faithfulness CI requires at least two identified clusters")
        for gain, control in (("faithfulness_gain_over_low_saliency", "low_saliency_occlusion_score_drop"),
                              ("faithfulness_gain_over_random", "random_occlusion_score_drop")):
            if not np.allclose(rows[gain], rows.high_saliency_occlusion_score_drop-rows[control], atol=1e-9, rtol=0):
                raise ValueError("faithfulness paired rows are inconsistent")
        recalculated = summarize_faithfulness(rows, group_columns=(), bootstrap_repeats=repeats, seed=seed)
        recorded = summary.loc[summary.group.eq("all")].set_index("metric").loc[recalculated.metric]
        for col in ("bootstrap_method", "confidence_level", "bootstrap_repeats", "identity_count"):
            if col not in recorded or not np.array_equal(recorded[col].to_numpy(), recalculated[col].to_numpy()):
                raise ValueError(f"faithfulness CI metadata mismatch: {col}")
        for col in ("mean_ci_lower", "mean_ci_upper"):
            if not np.allclose(recorded[col], recalculated[col], atol=1e-9, rtol=0):
                raise ValueError("faithfulness CI differs from identity-cluster recomputation")
    return {"saliency": sal, "faithfulness_summary": summary, "faithfulness_ids": rows.sample_id.astype(str),
            "faithfulness_rows": rows,
            "saliency_manifest": sm, "faithfulness_manifest": fm,
            "verified_manifest_sha256": {"saliency": canonical_sha256(sm), "faithfulness": canonical_sha256(fm)},
            "condition_manifest_sha256": canonical_sha256(cm),
            "verified_frame_sha256": {"saliency": _frame_hash(sal), "faithfulness_summary": _frame_hash(summary),
                                      "faithfulness_ids": _frame_hash(rows[["sample_id"]].astype(str)),
                                      "faithfulness_rows": _frame_hash(rows)}}


def assess_incremental_gate(condition, inputs):
    """Recompute gates on every use; there is no force/ignore-gate switch."""
    if inputs["condition_manifest_sha256"] != canonical_sha256(condition.manifest):
        raise ValueError("gate condition lineage mismatch")
    if inputs["verified_manifest_sha256"] != {
        "saliency": canonical_sha256(inputs["saliency_manifest"]),
        "faithfulness": canonical_sha256(inputs["faithfulness_manifest"]),
    }:
        raise ValueError("verified saliency manifests were modified in memory")
    sal, summary = inputs["saliency"], inputs["faithfulness_summary"]
    current = {"saliency": _frame_hash(sal), "faithfulness_summary": _frame_hash(summary),
               "faithfulness_ids": _frame_hash(inputs["faithfulness_ids"].to_frame(name="sample_id")),
               "faithfulness_rows": _frame_hash(inputs["faithfulness_rows"])}
    if current != inputs["verified_frame_sha256"]:
        raise ValueError("verified saliency evidence was modified in memory")
    readiness = assess_saliency_incremental_readiness(condition.calibration, condition.test, sal,
                                                     requested_features=SALIENCY_FEATURES, minimum_coverage=1.)
    reliability = assess_saliency_faithfulness_reliability(summary, group="all")
    reasons = [*readiness.reasons, *reliability.reasons]
    sm, fm = inputs["saliency_manifest"], inputs["faithfulness_manifest"]
    if (sm.get("artifact_type") != "saliency_calibration_features" or sm.get("status") != "completed"
            or sm.get("condition_manifest_sha256") != canonical_sha256(condition.manifest)
            or sm.get("gallery_contract") != "split_matched_origin_top1"):
        reasons.append("split-matched calibration saliency provenance is not verified")
    if fm.get("evaluation_split") not in ("calibration", "development"):
        reasons.append("faithfulness must be pre-test calibration/development evidence")
    if sm.get("smoke_only", False):
        reasons.append("limited smoke inputs cannot authorize the full comparison")
    if "saliency_target_name" not in sal or not sal.saliency_target_name.eq("origin_top1_gallery_cosine").all():
        reasons.append("every saliency row must declare the label-free target")
    if fm.get("evaluation_split") in ("calibration", "development"):
        if (fm.get("evaluation_split") != "calibration" or
                fm.get("condition_manifest_sha256") != canonical_sha256(condition.manifest)):
            reasons.append("verified calibration faithfulness cohort required; unbound development evidence is diagnostic-only")
        else:
            evidence = inputs["faithfulness_rows"]
            columns = {"sample_id", "identity_id", "aligned_content_sha256", "split"}
            if not columns <= set(evidence) or not evidence.sample_id.isin(condition.calibration.sample_id).all():
                reasons.append("faithfulness samples are not members of the calibration cohort")
            else:
                expected = condition.calibration.set_index("sample_id").loc[evidence.sample_id]
                if (not evidence.split.eq("calibration").all()
                        or not np.array_equal(evidence.identity_id.astype(str), expected.identity_id.astype(str))
                        or not np.array_equal(evidence.aligned_content_sha256, expected.aligned_content_sha256)):
                    reasons.append("faithfulness calibration identity/alignment/split mismatch")
                if (evidence.identity_id.isin(condition.test.identity_id).any()
                        or evidence.aligned_content_sha256.isin(condition.test.aligned_content_sha256).any()):
                    reasons.append("faithfulness identity/content overlaps test")
    overlap = len(set(inputs["faithfulness_ids"]) & set(condition.test.sample_id.astype(str)))
    if overlap:
        reasons.append("faithfulness overlaps test: cannot use test to enable correction")
    if fm.get("statistics", {}).get("bootstrap_method") != "identity_cluster":
        reasons.append("faithfulness CI must use identity-cluster resampling")
    if fm.get("statistics", {}).get("confidence_level") != .95:
        reasons.append("faithfulness gate requires 95 percent CIs")
    if "aligned_content_sha256" not in sal or "split" not in sal:
        reasons.append("saliency per-image alignment hashes/splits are unavailable")
    elif not sal.sample_id.duplicated().any():
        index = sal.set_index("sample_id")
        for split in ("calibration", "test"):
            expected = getattr(condition, split).set_index("sample_id")
            matched = index.reindex(expected.index)
            if (not matched.split.eq(split).all() or
                    not matched.aligned_content_sha256.eq(expected.aligned_content_sha256).all()):
                reasons.append(f"{split} saliency alignment/split mismatch")
    return {"status": "ready" if not reasons else "blocked", "comparison_enabled": not reasons,
            "calibration_coverage": readiness.calibration_coverage, "test_coverage": readiness.test_coverage,
            "strong_faithfulness_pass": reliability.strong_faithfulness_pass,
            "faithfulness_test_overlap": overlap, "reasons": reasons,
            "faithfulness": reliability.as_dict(), "readiness": readiness.as_dict(),
            "requires_origin_gallery": True, "deployment_claim_supported": False,
            "random_occlusion_is_threshold_feature": False}


def run_saliency_incremental_calibration(
    condition, fiqa, inputs, *, baseline_method="continuous_fiqa", retrieval=None,
    partition_seeds=(8972,), target_fpirs=(.01, .05, .1, .2, .3),
    safety_fraction=.3, knot_quantiles=(1/3, 2/3), smoothing=.01, ridge=.001,
    max_iterations=2000, margin_slope_cap=.95, resamples=2000, bootstrap_seed=8972,
):
    """Prespecified FIQA baseline vs +outside / +entropy / +both on identical rows."""
    gate = assess_incremental_gate(condition, inputs)
    if not gate["comparison_enabled"]:
        raise ValueError("saliency incremental gate blocked: " + "; ".join(gate["reasons"]))
    if baseline_method not in CONTINUOUS_FEATURES or baseline_method == "continuous_fiqa_runnerup":
        raise ValueError("choose a prespecified FIQA/margin/margin+distortion baseline")
    seeds, targets = tuple(partition_seeds), tuple(target_fpirs)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("unique nonnegative partition seeds required")
    if not targets or targets != tuple(sorted(set(targets))) or any(not 0 < t < 1 for t in targets):
        raise ValueError("unique increasing target FPIRs in (0,1) required")
    if type(resamples) is not int or resamples < 100 or type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("invalid bootstrap settings")
    cal, test = join_fiqa_score_artifacts(condition, fiqa)
    _validate_comparison_contract(cal, test, condition_manifest=condition.manifest, fiqa_manifest=fiqa.manifest)
    if condition.manifest["dataset_id"] != "survface" or condition.manifest["score_space"] != "negative_squared_l2_adc":
        raise ValueError("02 currently supports SurvFace PQ-ADC only")
    base_features = CONTINUOUS_FEATURES[baseline_method]
    if len(base_features) > 1:
        if (retrieval is None or retrieval["manifest"].get("status") != "completed" or
                retrieval["manifest"]["spec"]["condition_manifest_sha256"] != canonical_sha256(condition.manifest)):
            raise ValueError("baseline requires matching completed retrieval features from 01")
        cal, test = (join_retrieval_features(rows, retrieval[split]) for split, rows in (("calibration", cal), ("test", test)))
    sal = inputs["saliency"].set_index("sample_id")
    for rows in (cal, test):
        for feature in SALIENCY_FEATURES:
            rows[feature] = sal.loc[rows.sample_id, feature].to_numpy()
    methods = {"baseline": base_features, "plus_outside": (*base_features, SALIENCY_FEATURES[0]),
               "plus_entropy": (*base_features, SALIENCY_FEATURES[1]), "plus_both": (*base_features, *SALIENCY_FEATURES)}
    metrics, paired, fitted = [], [], []
    for seed in seeds:
        for target in targets:
            evaluated = {}
            for method, features in methods.items():
                model = fit_continuous_threshold(cal, target_fpir=target, features=features, method=method,
                                                partition_seed=seed, safety_fraction=safety_fraction,
                                                knot_quantiles=knot_quantiles, smoothing=smoothing, ridge=ridge,
                                                max_iterations=max_iterations, margin_slope_cap=margin_slope_cap)
                evaluated[method] = apply_continuous_threshold(test, model)
                fitted.append(dict(partition_seed=seed, target_fpir=target, method=method,
                                   model_uid=model.model_uid, model_json=json.dumps(model.as_dict(), allow_nan=False)))
            mated = test.is_mated.to_numpy(bool)
            names = list(evaluated)
            events = np.column_stack([evaluated[n].decisions.true_identification_at_rank_k.to_numpy()[mated] for n in names])
            draws = cluster_rate_draws(test.loc[mated, "identity_id"], events, resamples=resamples, seed=bootstrap_seed)
            for i, name in enumerate(names):
                lo, hi = np.quantile(draws[:, i], [.025, .975])
                metrics.append({**evaluated[name].summary, "partition_seed": seed, "method": name,
                                "tpir_cluster95_low": lo, "tpir_cluster95_high": hi})
            for candidate in names[1:]:
                evidence = paired_method_comparison(evaluated["baseline"], evaluated[candidate],
                                                    resamples=resamples, random_seed=bootstrap_seed)
                for metric in ("fpir", "tpir_at_rank_k"):
                    row = dict(evidence[metric])
                    if metric == "tpir_at_rank_k":
                        lo, hi = np.quantile(draws[:, names.index(candidate)]-draws[:, 0], [.025, .975])
                        row.update(paired_bootstrap95_low=lo, paired_bootstrap95_high=hi)
                    paired.append({**row, "partition_seed": seed, "target_fpir": target,
                                   "reference_method": "baseline", "candidate_method": candidate, "metric": metric,
                                   "resampling_unit": "query" if metric == "fpir" else "mated_identity_cluster",
                                   "resamples": resamples, "bootstrap_seed": bootstrap_seed,
                                   "reference_realized_fpir": evaluated["baseline"].summary["realized_fpir"],
                                   "candidate_realized_fpir": evaluated[candidate].summary["realized_fpir"],
                                   "candidate_target_met": evaluated[candidate].summary["target_met_on_test"]})
    metrics = pd.DataFrame(metrics)
    split_summary = metrics.groupby(["method", "target_fpir"]).agg(
        split_count=("partition_seed", "count"), target_met_split_count=("target_met_on_test", "sum"),
        fpir_min=("realized_fpir", "min"), fpir_median=("realized_fpir", "median"), fpir_max=("realized_fpir", "max"),
        tpir_min=("tpir_at_rank_k", "min"), tpir_median=("tpir_at_rank_k", "median"), tpir_max=("tpir_at_rank_k", "max"),
    ).reset_index()
    paths = [Path(__file__), Path(__file__).parents[1]/"calibration/continuous.py",
             Path(__file__).parents[1]/"calibration/conditional.py", Path(__file__).parents[1]/"calibration/rejection.py",
             Path(__file__).parents[1]/"evaluation/cluster_bootstrap.py", Path(__file__).parents[1]/"evaluation/metrics.py",
             Path(__file__).parents[1]/"evaluation/saliency_faithfulness.py", Path(__file__).with_name("fiqa_threshold_calibration.py"),
             Path(__file__).with_name("fiqa_retrieval_features.py"), Path(__file__).with_name("fiqa_continuous_calibration.py")]
    manifest = dict(artifact_type="saliency_incremental_calibration", schema_version=1,
                    source_run_id=condition.manifest["source_run_id"], model_uid=condition.manifest["model_uid"],
                    metric_contract=condition.manifest["metric_contract"], score_space=condition.manifest["score_space"],
                    condition_uid=condition.condition_uid, gate=gate,
                    condition_manifest_sha256=canonical_sha256(condition.manifest),
                    fiqa_manifest_sha256=canonical_sha256(fiqa.manifest),
                    saliency_manifest_sha256=canonical_sha256(inputs["saliency_manifest"]),
                    faithfulness_manifest_sha256=canonical_sha256(inputs["faithfulness_manifest"]),
                    retrieval_manifest_sha256=canonical_sha256(retrieval["manifest"]) if len(base_features)>1 else None,
                    input_frame_sha256={"calibration": _frame_hash(cal), "test": _frame_hash(test)},
                    implementation_sha256={p.name: sha256_file(p) for p in paths},
                    versions={"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
                    settings=dict(baseline_method=baseline_method, partition_seeds=list(seeds), target_fpirs=list(targets),
                                  safety_fraction=safety_fraction, knot_quantiles=list(knot_quantiles), smoothing=smoothing,
                                  ridge=ridge, max_iterations=max_iterations, margin_slope_cap=margin_slope_cap,
                                  resamples=resamples, bootstrap_seed=bootstrap_seed),
                    uncertainty=dict(threshold_uncertainty_included=False, multiple_comparison_adjustment="none",
                                      formal_fpir_guarantee=False, automatic_test_based_selection=False,
                                      baseline_selection="01_test_informed_exploratory",
                                     between_splits="descriptive_not_ci", deployment_claim_supported=False))
    manifest["result_uid"] = "saliency-incremental-" + canonical_sha256(manifest)[:24]
    return dict(method_summary=metrics, paired_comparisons=pd.DataFrame(paired), models=pd.DataFrame(fitted),
                split_summary=split_summary, manifest=manifest)


def write_saliency_incremental_result(root, result):
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
    (staging/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf8")
    os.rename(staging, destination)
    return destination
