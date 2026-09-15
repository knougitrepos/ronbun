"""Repeated fit/safety splits on a frozen cohort; not independent test trials."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from research.calibration.conditional import (
    _boolean_values,
    deterministic_calibration_partition,
)
from research.experiments.fiqa_priority_diagnostics import (
    _reuse_completed_result, run_fiqa_priority_diagnostics,
)
from research.runtime.hashing import canonical_sha256, sha256_file

DEFAULT_PARTITION_SEEDS = (*range(19), 8972)
TABLES = ("seed_metrics", "seed_paired_comparisons", "seed_thresholds",
          "seed_group_tail_transfer", "partition_inventory", "stability_summary",
          "paired_stability_summary", "threshold_stability_summary")


def _frame_hash(frame):
    # Include values, row order, column names and dtypes, including in-memory inputs.
    h = hashlib.sha256()
    h.update(str([(str(c), str(t)) for c, t in frame.dtypes.items()]).encode())
    h.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return h.hexdigest()


def _validated_seeds(seeds):
    seeds = tuple(seeds)
    if (len(seeds) < 2 or any(isinstance(s, (bool, np.bool_))
                            or not isinstance(s, (int, np.integer)) or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("at least two unique non-negative integer partition seeds required")
    return tuple(int(s) for s in seeds)


def _partition_inventory(calibration, seed, safety_fraction):
    ids = calibration.identity_id
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError("calibration requires non-null genuine identity labels")
    partition = deterministic_calibration_partition(
        calibration, seed=seed, safety_fraction=safety_fraction, partition_column="identity_id")
    assignments = pd.DataFrame({"identity_id": ids.astype(str), "partition": partition})
    assignments = assignments.drop_duplicates().sort_values("identity_id")
    if assignments.identity_id.duplicated().any():
        raise ValueError("identity leaked between fit and safety")
    digest = canonical_sha256(assignments.to_dict("records"))
    mated = _boolean_values(calibration.is_mated, column="is_mated")
    return [{
        "partition_seed": seed, "partition": part,
        "assignment_sha256": digest,
        "query_count": int(partition.eq(part).sum()),
        "identity_count": int(ids.loc[partition.eq(part)].nunique()),
        "mated_query_count": int((partition.eq(part) & mated).sum()),
        "non_mated_query_count": int((partition.eq(part) & ~mated).sum()),
        "shared_by": "global_safe,fiqa_s,fiqa_l",
        "global_empirical_uses_all_calibration": True,
    } for part in ("fit", "safety")]


def _distribution(values, prefix):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite split statistics: {prefix}")
    return {f"{prefix}_{name}": float(value) for name, value in zip(
        ("min", "q25", "median", "q75", "max"), np.quantile(values, [0, .25, .5, .75, 1]))}


def summarize_split_stability(metrics, paired, thresholds, seeds):
    """Describe the fixed seed panel. Quantiles are NOT confidence intervals."""
    records = []
    for (method, target), group in metrics.groupby(["method", "target_fpir"], sort=True):
        if len(group) != len(seeds) or set(group.partition_seed) != set(seeds):
            raise ValueError("incomplete or duplicate seed panel")
        passes = int(group.target_met_on_test.sum())
        records.append({
            "method": method, "target_fpir": target, "split_count": len(group),
            "target_met_split_count": passes,
            "target_met_split_fraction": passes / len(group),
            "wilson_upper_met_split_count": int(group.target_met_by_wilson_upper.sum()),
            "wilson_lower_exceeds_target_split_count": int((group.fpir_wilson95_low > target).sum()),
            "observed_pattern": ("all_observed_splits_exceed" if passes == 0 else
                                 "all_observed_splits_meet" if passes == len(group) else
                                 "split_sensitive_target_attainment"),
            "split_dependent_method": method != "global_empirical",
            "between_split_statistics_are_ci": False,
            **_distribution(group.realized_fpir, "fpir"),
            **_distribution(group.realized_fpir - target, "fpir_excess"),
            **_distribution(group.tpir_at_rank_k, "tpir"),
        })
    paired_records = []
    for keys, group in paired.groupby(
            ["reference_method", "candidate_method", "target_fpir", "metric"], sort=True):
        ref, cand, target, metric = keys
        if len(group) != len(seeds) or set(group.partition_seed) != set(seeds):
            raise ValueError("incomplete paired seed panel")
        paired_records.append({
            "reference_method": ref, "candidate_method": cand,
            "target_fpir": target, "metric": metric, "split_count": len(group),
            **_distribution(group.candidate_minus_reference, "delta"),
            "positive_delta_split_count": int((group.candidate_minus_reference > 0).sum()),
            "negative_delta_split_count": int((group.candidate_minus_reference < 0).sum()),
            "positive_conditional_ci_split_count": int((group.paired_bootstrap95_low > 0).sum()),
            "negative_conditional_ci_split_count": int((group.paired_bootstrap95_high < 0).sum()),
            "between_split_statistics_are_ci": False,
            "independent_replications": False,
            "multiple_comparison_adjustment": "none",
        })
    threshold_records = []
    for (method, target, name), group in thresholds.groupby(["method", "target_fpir", "name"]):
        threshold_records.append({
            "method": method, "target_fpir": target, "quality_group": name,
            "split_count": len(group),
            **_distribution(group.final_threshold, "threshold"),
            **_distribution(group.fit_non_mated_count, "fit_non_mated_count"),
            **_distribution(group.safety_non_mated_count, "safety_non_mated_count"),
            "global_fallback_split_count": int(group.used_global_fallback.sum()),
        })
    return (pd.DataFrame(records), pd.DataFrame(paired_records), pd.DataFrame(threshold_records))


def run_fiqa_split_stability(
    condition, fiqa_s, fiqa_l, *, partition_seeds=DEFAULT_PARTITION_SEEDS,
    target_fpirs=(.01, .05, .10, .20, .30), safety_fraction=.30,
    bin_count=2, shrinkage_strength=200., minimum_group_non_mated=100,
    resamples=2000, bootstrap_seed=8972, progress=None, bin_counts=None,
):
    """Refit on calibration only for each shared seed; keep test/gallery frozen.

    Per-seed CI is conditional on that fitted threshold, not a combined CI over
    split and test uncertainty. No best seed or model is selected from test.
    """
    seeds = _validated_seeds(partition_seeds)
    bin_counts = None if bin_counts is None else tuple(bin_counts)
    targets = tuple(float(t) for t in target_fpirs)
    inventory = []
    parts = {name: [] for name in ("method_summary", "paired_comparisons", "thresholds", "group_tail_transfer")}
    seed_manifests = []
    for index, seed in enumerate(seeds):
        inventory.extend(_partition_inventory(condition.calibration, seed, safety_fraction))
        result = run_fiqa_priority_diagnostics(
            condition, fiqa_s, fiqa_l, target_fpirs=targets, partition_seed=seed,
            safety_fraction=safety_fraction, bin_count=bin_count,
            shrinkage_strength=shrinkage_strength, minimum_group_non_mated=minimum_group_non_mated,
            resamples=resamples, bootstrap_seed=bootstrap_seed,
            bin_counts=bin_counts,
        )
        for name in parts:
            parts[name].append(result[name].assign(partition_seed=seed))
        seed_manifests.append(result["manifest"])
        if progress is not None:
            progress({"completed": index + 1, "total": len(seeds), "partition_seed": seed})
    merged = {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}
    if bin_counts is not None:
        shared_by = ",".join(merged["method_summary"].method.unique()[1:])
        for row in inventory:
            row["shared_by"] = shared_by
    distinct_partitions = len({row["assignment_sha256"] for row in inventory})
    if distinct_partitions < 2:
        raise ValueError("seed panel produced fewer than two distinct fit/safety partitions")
    summary, paired_summary, threshold_summary = summarize_split_stability(
        merged["method_summary"], merged["paired_comparisons"], merged["thresholds"], seeds)
    implementation_paths = [
        Path(__file__), Path(__file__).with_name("fiqa_priority_diagnostics.py"),
        Path(__file__).with_name("fiqa_threshold_calibration.py"),
        Path(__file__).parents[1] / "evaluation/cluster_bootstrap.py",
        Path(__file__).parents[1] / "evaluation/metrics.py",
        Path(__file__).parents[1] / "calibration/conditional.py",
        Path(__file__).parents[1] / "calibration/rejection.py",
    ]
    manifest = {
        "schema_version": 1, "artifact_type": "fiqa_split_stability",
        "condition_uid": condition.condition_uid,
        "partition_seeds": list(seeds), "target_fpirs": list(targets),
        "seed_plan_origin": ("existing_exploratory_panel_not_new_preregistration"
                             if seeds == DEFAULT_PARTITION_SEEDS else "explicit_exploratory_panel"),
        "distinct_partition_count": distinct_partitions,
        "seed_manifests": seed_manifests,
        "input_frame_sha256": {
            "calibration": _frame_hash(condition.calibration), "test": _frame_hash(condition.test),
            "fiqa_s": _frame_hash(fiqa_s.scores), "fiqa_l": _frame_hash(fiqa_l.scores),
        },
        "implementation_sha256": {str(p.relative_to(Path(__file__).parents[2])): sha256_file(p)
                                  for p in implementation_paths},
        "partition_assignment_sha256": {str(r["partition_seed"]): r["assignment_sha256"] for r in inventory},
        "uncertainty": {
            "per_seed_tpir_ci": "query_weighted_mated_identity_cluster_fixed_threshold",
            "per_seed_fpir_ci": "query_wilson_fixed_threshold",
            "between_splits": "descriptive_min_quartiles_median_max_not_ci",
            "joint_split_test_ci": False, "independent_replications": False,
            "gallery_or_codec_refitted": False, "calibration_cohort_resampled": False,
            "test_based_seed_selection": False, "multiple_comparison_adjustment": "none",
            "scope": "frozen_cohort_internal_fit_safety_sensitivity_only",
        },
    }
    manifest["stability_uid"] = "fiqa-split-" + canonical_sha256(manifest)[:24]
    return {
        "seed_metrics": merged["method_summary"], "seed_paired_comparisons": merged["paired_comparisons"],
        "seed_thresholds": merged["thresholds"], "seed_group_tail_transfer": merged["group_tail_transfer"],
        "partition_inventory": pd.DataFrame(inventory), "stability_summary": summary,
        "paired_stability_summary": paired_summary, "threshold_stability_summary": threshold_summary,
        "manifest": manifest,
    }


def write_fiqa_split_stability(root, result, *, reuse_existing=False):
    """Publish new derived results atomically; never overwrite completed results."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / result["manifest"]["stability_uid"]
    if destination.exists():
        if reuse_existing:
            return _reuse_completed_result(destination, result, TABLES)
        raise FileExistsError(destination)
    staging = root / (".staging-" + uuid4().hex)
    staging.mkdir()
    files = {}
    for name in TABLES:
        path = staging / f"{name}.csv"
        result[name].to_csv(path, index=False)
        files[path.name] = sha256_file(path)
    manifest = {**result["manifest"], "status": "completed", "files": files}
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.rename(staging, destination)
    return destination
