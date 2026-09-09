"""Paired S/L calibration and fixed-threshold diagnostics (no inference)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from research.calibration.conditional import (
    IDENTIFICATION_METRIC_CONTRACT,
    apply_threshold_model,
    assign_quality_groups,
    deterministic_calibration_partition,
    fit_conditional_threshold,
    fit_global_threshold,
    paired_method_comparison,
)
from research.experiments.fiqa_threshold_calibration import (
    _validate_comparison_contract,
    join_fiqa_score_artifacts,
)
from research.runtime.hashing import canonical_sha256, sha256_file
from research.evaluation.cluster_bootstrap import cluster_rate_draws


def quality_tail_transfer(calibration, test, model, *, safety_fraction, seed):
    """Describe non-mated tails with frozen fit cutpoints and thresholds."""
    partition = deterministic_calibration_partition(
        calibration, safety_fraction=safety_fraction, seed=seed,
        partition_column="identity_id",
    )
    rows = []
    for split, frame in (
        ("fit", calibration.loc[partition.eq("fit")]),
        ("safety", calibration.loc[partition.eq("safety")]),
        ("test", test),
    ):
        nm = frame.loc[~frame.is_mated.astype(bool)]
        groups = (
            assign_quality_groups(nm[model.quality_column],
                                  cutpoints=model.quality_cutpoints,
                                  labels=model.group_labels)
            if model.quality_column else np.full(len(nm), "all")
        )
        for group in model.group_labels:
            scores = nm.loc[np.asarray(groups) == group, model.score_column]
            threshold = model.thresholds[group]
            count = len(scores)
            exceedances = int((scores >= threshold).sum())
            rows.append({
                "split": split, "quality_group": group,
                "non_mated_count": count,
                "non_mated_fraction": count / len(nm) if len(nm) else np.nan,
                "false_accept_count": exceedances,
                "realized_fpir": exceedances / count if count else np.nan,
                "threshold": threshold,
                "score_q99": float(scores.quantile(.99)) if count else np.nan,
                "quality_cutpoints": json.dumps(list(model.quality_cutpoints)),
                "threshold_fit_on_test": False,
            })
    return pd.DataFrame(rows)


def run_fiqa_priority_diagnostics(
    condition, fiqa_s, fiqa_l, *, target_fpirs=(.01, .05, .10, .20, .30),
    partition_seed=8972, safety_fraction=.30, bin_count=2,
    shrinkage_strength=200., minimum_group_non_mated=100,
    resamples=2000, bootstrap_seed=8972,
):
    """Compare four methods under one split; test data never chooses thresholds.

    SurvFace mated identity labels support cluster intervals. Unknown probes
    have synthetic IDs, so FPIR retains explicitly query-level intervals.
    """
    if condition.manifest.get("dataset_id") != "survface":
        raise ValueError("identity semantics currently audited for SurvFace only")
    if fiqa_s.manifest.get("variant") != "S" or fiqa_l.manifest.get("variant") != "L":
        raise ValueError("explicit S then L variant manifests required")
    if fiqa_s.manifest["fiqa_model_uid"] == fiqa_l.manifest["fiqa_model_uid"]:
        raise ValueError("S and L must be distinct FIQA models")
    targets = tuple(float(x) for x in target_fpirs)
    if not targets or targets != tuple(sorted(set(targets))):
        raise ValueError("targets must be nonempty, unique and increasing")
    if not 0 < safety_fraction < 1:
        raise ValueError("a nonempty held-out safety partition is required")
    joined = {}
    for name, artifact in (("fiqa_s", fiqa_s), ("fiqa_l", fiqa_l)):
        cal, test = join_fiqa_score_artifacts(condition, artifact)
        _validate_comparison_contract(
            cal, test, condition_manifest=condition.manifest,
            fiqa_manifest=artifact.manifest,
        )
        joined[name] = (cal, test)
    cal, test = joined["fiqa_s"]
    for column in ("sample_id", "identity_id", "is_mated", "score"):
        for i in (0, 1):
            if not joined["fiqa_s"][i][column].equals(joined["fiqa_l"][i][column]):
                raise ValueError(f"unpaired S/L inputs: {column}")
    summaries, paired_rows, tails, thresholds = [], [], [], []
    names = ("global_empirical", "global_safe", "fiqa_s", "fiqa_l")
    for target in targets:
        common = dict(target_fpir=target, partition_seed=partition_seed,
                      partition_column="identity_id",
                      score_space=condition.manifest["score_space"])
        models = {
            "global_empirical": fit_global_threshold(cal, safety_fraction=0, **common),
            "global_safe": fit_global_threshold(cal, safety_fraction=safety_fraction, **common),
        }
        for name in names[2:]:
            models[name] = fit_conditional_threshold(
                joined[name][0], safety_fraction=safety_fraction,
                bin_count=bin_count, shrinkage_strength=shrinkage_strength,
                minimum_group_non_mated=minimum_group_non_mated, **common,
            )
        evaluations = {
            name: apply_threshold_model(joined.get(name, (cal, test))[1], models[name])
            for name in names
        }
        mated = test.is_mated.astype(bool).to_numpy()
        events = np.column_stack([
            evaluations[name].decisions.true_identification_at_rank_k.to_numpy()[mated]
            for name in names
        ])
        draws = cluster_rate_draws(test.loc[mated, "identity_id"], events,
                                   resamples=resamples, seed=bootstrap_seed)
        for i, name in enumerate(names):
            low, high = np.quantile(draws[:, i], [.025, .975])
            summaries.append({
                **evaluations[name].summary, "method": name,
                "tpir_cluster95_low": low, "tpir_cluster95_high": high,
                "mated_identity_count": test.loc[mated, "identity_id"].nunique(),
            })
            for group in models[name].groups:
                thresholds.append({"method": name, "target_fpir": target,
                                   "model_uid": models[name].model_uid,
                                   **group.as_dict()})
            if name != "global_empirical":
                c, t = joined.get(name, (cal, test))
                tail = quality_tail_transfer(c, t, models[name],
                                             safety_fraction=safety_fraction,
                                             seed=partition_seed)
                tails.append(tail.assign(method=name, target_fpir=target))
        for ref, cand in ((0, 2), (1, 2), (0, 3), (1, 3), (2, 3)):
            evidence = paired_method_comparison(evaluations[names[ref]], evaluations[names[cand]])
            for metric in ("fpir", "tpir_at_rank_k"):
                result = dict(evidence[metric])
                unit = "query"
                n_resamples, random_seed = evidence["resamples"], evidence["random_seed"]
                if metric == "tpir_at_rank_k":
                    low, high = np.quantile(draws[:, cand] - draws[:, ref], [.025, .975])
                    result.update(paired_bootstrap95_low=low, paired_bootstrap95_high=high)
                    unit = "mated_identity_cluster"
                    n_resamples, random_seed = resamples, bootstrap_seed
                paired_rows.append({
                    "reference_method": names[ref], "candidate_method": names[cand],
                    "target_fpir": target, "metric": metric, **result,
                    "resampling_unit": unit, "resamples": n_resamples,
                    "random_seed": random_seed, "threshold_uncertainty_included": False,
                    "multiple_comparison_adjustment": "none",
                })
    manifest = {
        "schema_version": 1, "artifact_type": "fiqa_priority_diagnostics",
        "metric_contract": IDENTIFICATION_METRIC_CONTRACT,
        "condition_manifest_sha256": canonical_sha256(condition.manifest),
        "fiqa_s_manifest_sha256": canonical_sha256(fiqa_s.manifest),
        "fiqa_l_manifest_sha256": canonical_sha256(fiqa_l.manifest),
        "condition_uid": condition.condition_uid,
        "partition_seed": partition_seed, "safety_fraction": safety_fraction,
        "bin_count": bin_count, "shrinkage_strength": shrinkage_strength,
        "minimum_group_non_mated": minimum_group_non_mated,
        "target_fpirs": list(targets), "resamples": resamples,
        "bootstrap_seed": bootstrap_seed,
        "implementation_sha256": sha256_file(Path(__file__)),
        "cluster_ci_implementation_sha256": sha256_file(
            Path(__file__).parents[1] / "evaluation" / "cluster_bootstrap.py"),
        "conditional_implementation_sha256": sha256_file(
            Path(__file__).parents[1] / "calibration" / "conditional.py"),
        "uncertainty": {"tpir": "query_weighted_mated_identity_cluster",
                        "fpir": "query_only_unknown_identity_labels_unavailable",
                        "threshold_uncertainty_included": False,
                        "gallery_uncertainty_included": False,
                        "multiple_comparison_adjustment": "none",
                        "interpretation": "exploratory_fixed_threshold"},
        "threshold_fit_on_test": False,
    }
    manifest["diagnostic_uid"] = "fiqa-priority-" + canonical_sha256(manifest)[:24]
    return {"method_summary": pd.DataFrame(summaries),
            "paired_comparisons": pd.DataFrame(paired_rows),
            "group_tail_transfer": pd.concat(tails, ignore_index=True),
            "thresholds": pd.DataFrame(thresholds), "manifest": manifest}


def write_fiqa_priority_diagnostics(root, result):
    """Publish a new content-addressed directory; never replace completed data."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / result["manifest"]["diagnostic_uid"]
    if destination.exists():
        raise FileExistsError(f"Completed diagnostic already exists: {destination}")
    staging = root / (".staging-" + uuid4().hex)
    staging.mkdir()
    files = {}
    for name in ("method_summary", "paired_comparisons", "group_tail_transfer", "thresholds"):
        path = staging / f"{name}.csv"
        result[name].to_csv(path, index=False)
        files[path.name] = sha256_file(path)
    manifest = {**result["manifest"], "status": "completed", "files": files}
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.rename(staging, destination)
    return destination
