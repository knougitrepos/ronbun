"""Paired origin/PQ quality calibration, failure decomposition and frozen-score diagnostics.

No training or model selection uses test. Test curve interpolation is explicitly
descriptive and never supplies the deployed thresholds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import scipy

from research.calibration.conditional import (
    ConditionalThresholdModel, ThresholdGroup, apply_threshold_model,
    deterministic_calibration_partition, fit_conditional_threshold, fit_global_threshold,
    validate_identification_scores, _boolean_values,
)
from research.calibration.continuous import (
    ContinuousThresholdModel, apply_continuous_threshold, fit_continuous_threshold,
)
from research.evaluation.cluster_bootstrap import cluster_rate_draws
from research.evaluation.metrics import paired_binary_rate_difference_bootstrap_interval
from research.experiments.fiqa_split_stability import _frame_hash
from research.experiments.fiqa_threshold_calibration import join_fiqa_scores, join_fiqa_score_artifacts
from research.experiments.origin_pq_inputs import (
    ORIGIN_MODE, ORIGIN_PROFILE, ORIGIN_SPACE, assert_same_cohort,
    input_code_hashes, load_run_inputs,
)
from research.explainability.gradcam.artifacts import _publish_atomic_directory
from research.runtime.hashing import canonical_sha256, sha256_file

METHODS = ("global_safe", "fiqa_5bin", "continuous_fiqa")
TABLES = ("method_summary", "paired_comparisons", "interactions", "diagnostic_curves",
          "diagnostic_interactions", "models", "partition_inventory")


@dataclass(frozen=True)
class OriginPQSettings:
    target_fpirs: tuple = (.01, .05, .10, .20, .30)
    safety_fraction: float = .30
    shrinkage_strength: float = 200.
    minimum_group_non_mated: int = 100
    knot_quantiles: tuple = (1/3, 2/3)
    smoothing: float = .01
    ridge: float = .001
    max_iterations: int = 2000
    margin_slope_cap: float = .95
    resamples: int = 2000
    bootstrap_seed: int = 8972
    diagnostic_fpir_grid: tuple = (.0, .001, .005, .01, .02, .05, .1, .2, .3, .5, 1.)

    def validate(self):
        for values, allow_edges in ((self.target_fpirs, False), (self.diagnostic_fpir_grid, True)):
            if (not values or tuple(values) != tuple(sorted(set(values))) or not np.isfinite(values).all()
                    or any(not (0 <= x <= 1 if allow_edges else 0 < x < 1) for x in values)):
                raise ValueError("unique sorted FPIR values required")
        for value, minimum in ((self.resamples, 100), (self.bootstrap_seed, 0),
                               (self.minimum_group_non_mated, 1), (self.max_iterations, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError("invalid integer calibration/CI settings")
        if (not 0 < self.safety_fraction < 1 or self.shrinkage_strength < 0
                or not np.isfinite([self.smoothing, self.ridge, self.shrinkage_strength]).all()
                or self.smoothing <= 0 or self.ridge < 0 or not 0 <= self.margin_slope_cap < 1
                or tuple(self.knot_quantiles) != tuple(sorted(set(self.knot_quantiles)))
                or any(not 0 < k < 1 for k in self.knot_quantiles)):
            raise ValueError("invalid fit/safety settings")

    def as_dict(self):
        return json.loads(json.dumps(asdict(self)))


def _seed(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("non-negative integer seed required")
    return value


def science_hashes():
    root = Path(__file__).parents[1]
    names = ("experiments/origin_vs_pq_calibration.py", "calibration/continuous.py",
             "experiments/origin_pq_storage.py", "experiments/origin_pq_resources.py",
             "calibration/conditional.py", "calibration/rejection.py",
             "evaluation/cluster_bootstrap.py", "evaluation/metrics.py",
             "experiments/fiqa_split_stability.py")
    return {**input_code_hashes(), **{n: sha256_file(root / n) for n in names}}


def _joined_frames(conditions, fiqa):
    if ORIGIN_PROFILE not in conditions or len(conditions) < 2:
        raise ValueError("one origin plus at least one PQ condition required")
    anchor = conditions[ORIGIN_PROFILE]
    joined = {}
    fm = fiqa.manifest
    if fm.get("status") != "completed" or fm.get("artifact_type") != "fiqa_score_table":
        raise ValueError("completed FIQA required")
    for profile, condition in conditions.items():
        cm = condition.manifest
        assert_same_cohort(anchor, condition)
        if (cm.get("status") != "completed" or cm.get("schema_version") != 2
                or cm.get("metric_contract") != "genuine-score-topk-v2"
                or cm.get("compression_profile") != profile
                or cm.get("dataset_id") != fm.get("dataset_id")
                or not cm.get("aligned_bundle_manifest_sha256")
                or cm["aligned_bundle_manifest_sha256"] != fm.get("aligned_bundle_manifest_sha256")):
            raise ValueError("condition/FIQA contract mismatch")
        if profile == ORIGIN_PROFILE:
            if (cm.get("artifact_type") != "origin_calibration_test_score_tables"
                    or cm.get("score_space") != ORIGIN_SPACE or cm.get("search_mode") != ORIGIN_MODE):
                raise ValueError("origin representation contract mismatch")
            cal, test = (join_fiqa_scores(getattr(condition, split), fiqa.scores)
                         for split in ("calibration", "test"))
        else:
            if cm.get("score_space") != "negative_squared_l2_adc" or cm.get("search_mode") != "pq_adc_exhaustive":
                raise ValueError("PQ ADC representation required")
            cal, test = join_fiqa_score_artifacts(condition, fiqa)
        for split, frame in (("calibration", cal), ("test", test)):
            # Historical PQ calibration tables store the non-mated maxima
            # needed for fitting, without genuine scores. Only test evaluation
            # requires the full genuine-score contract; do not invent cal ranks.
            if split == "test":
                validate_identification_scores(frame)
            if not np.isfinite(frame.score.to_numpy(dtype=float)).all():
                raise ValueError("non-finite calibration/evaluation maxima")
            for name in ("is_mated", "top_k_correct"):
                frame[name] = _boolean_values(frame[name], column=name)
            for key in ("dataset_id", "model_uid", "compression_profile", "search_mode", "score_space",
                        "protocol_uid", "extraction_uid", "origin_embedding_artifact_uid"):
                if not cm.get(key) or not frame[key].astype(str).eq(str(cm[key])).all():
                    raise ValueError(f"{split} row/manifest mismatch: {key}")
            if (not frame.evaluation_split.eq(split).all() or frame.sample_id.duplicated().any()
                    or not frame.top_k.eq(cm["top_k"]).all() or not np.isfinite(frame.fiqa_score).all()):
                raise ValueError("invalid split/query/K/FIQA")
        if (set(cal.sample_id) & set(test.sample_id)) or (set(cal.identity_id) & set(test.identity_id)):
            raise ValueError("calibration/test query or identity overlap")
        joined[profile] = (cal, test)
    return joined


def failure_decomposition(evaluation):
    rows = evaluation.decisions
    mated = rows.is_mated.to_numpy(dtype=bool)
    eligible = rows.top_k_correct.to_numpy(dtype=bool)
    success = rows.true_identification_at_rank_k.to_numpy(dtype=bool)
    rank_failure = mated & ~eligible
    threshold_failure = mated & eligible & ~success
    if int(mated.sum()) != int(rank_failure.sum() + threshold_failure.sum() + success.sum()):
        raise ValueError("failure partition does not cover mated cohort")
    return dict(rank_failure_count=int(rank_failure.sum()), threshold_failure_count=int(threshold_failure.sum()),
                rank_failure_rate=float(rank_failure.sum() / mated.sum()),
                threshold_failure_rate=float(threshold_failure.sum() / mated.sum()),
                rank_k_ceiling=float((mated & eligible).sum() / mated.sum()))


def frozen_score_diagnostic(evaluation, grid):
    """Empirical curve interpolation of one already-fitted residual score.

    Vertical segments use their upper endpoint; linear interpolation across
    ties is a randomized-threshold interpretation, not a deployed decision.
    Rank failures can never become true positives on this curve.
    """
    rows = evaluation.decisions
    mated = rows.is_mated.to_numpy(dtype=bool)
    offsets = rows.applied_threshold.to_numpy(dtype=float)
    nonmated = np.sort(rows.score.to_numpy(dtype=float)[~mated] - offsets[~mated])
    keep = mated & rows.top_k_correct.to_numpy(dtype=bool)
    genuine = np.sort(rows.true_identity_score.to_numpy(dtype=float)[keep] - offsets[keep])
    if not len(nonmated) or not mated.sum():
        raise ValueError("both probe populations required")
    thresholds = np.unique(np.r_[nonmated, genuine])[::-1]
    fpir = np.r_[0., (len(nonmated) - np.searchsorted(nonmated, thresholds, side="left")) / len(nonmated)]
    tpir = np.r_[0., (len(genuine) - np.searchsorted(genuine, thresholds, side="left")) / mated.sum()]
    points = pd.DataFrame({"fpir": fpir, "tpir": tpir}).groupby("fpir", sort=True).tpir.max()
    return pd.DataFrame(dict(diagnostic_fpir=grid, diagnostic_tpir=np.interp(grid, points.index, points),
                             diagnostic_only=True, interpolated_test_curve=True,
                             deployment_threshold_selected=False, diagnostic_ci_available=False))


def _decode_model(payload, method):
    data = json.loads(payload)
    if method == "continuous_fiqa":
        return ContinuousThresholdModel(**data)
    data.pop("schema_version")
    data["groups"] = tuple(ThresholdGroup(**g) for g in data["groups"])
    return ConditionalThresholdModel(**data)


class VerifiedPQModels:
    """Optional reuse of explicitly pinned completed matrix models, never latest."""
    def __init__(self, report_dir):
        self.directory = Path(report_dir).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf8"))
        if (self.manifest.get("status") != "completed"
                or self.manifest.get("artifact_type") != "calibration_matrix_report"
                or self.manifest.get("missing_jobs")):
            raise ValueError("PQ model source report is not complete")
        self.report_sha256 = canonical_sha256(self.manifest)
        for name, expected in self.manifest["files"].items():
            if sha256_file(self.directory / name) != expected:
                raise ValueError("PQ source report hash mismatch")

    def load(self, conditions, fiqa, seed, settings, *, joined_frames=None):
        # Campaign callers may share the full, already-validated join across
        # seeds. Keep full frames here because saved input hashes include all
        # provenance columns, not only the numerical fitting columns.
        joined = _joined_frames(conditions, fiqa) if joined_frames is None else joined_frames
        result = {}
        for profile, condition in conditions.items():
            if profile == ORIGIN_PROFILE:
                continue
            cm = condition.manifest
            matches = [x for x in self.manifest["sources"] if x["family"] == "fiqa"
                       and x["source_run_id"] == cm["source_run_id"] and x["compression_profile"] == profile
                       and x["partition_seed"] == seed]
            if len(matches) != 1:
                raise ValueError(f"no unique pinned PQ model: {profile}/{seed}; disable reuse to refit")
            receipt = matches[0]
            directory = Path(receipt["result_dir"])
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
            if canonical_sha256(manifest) != receipt["result_manifest_sha256"] or manifest.get("status") != "completed":
                raise ValueError("PQ source result manifest mismatch")
            expected = dict(condition_manifest_sha256=canonical_sha256(cm),
                            fiqa_manifest_sha256=canonical_sha256(fiqa.manifest),
                            score_space=cm["score_space"], source_run_id=cm["source_run_id"], model_uid=cm["model_uid"])
            for k, v in expected.items():
                if manifest.get(k) != v:
                    raise ValueError(f"PQ reuse provenance mismatch: {k}")
            for k in ("safety_fraction", "shrinkage_strength", "minimum_group_non_mated", "knot_quantiles",
                      "smoothing", "ridge", "max_iterations", "margin_slope_cap"):
                if manifest["settings"].get(k) != settings.as_dict()[k]:
                    raise ValueError(f"PQ reuse fitting setting mismatch: {k}; disable reuse to refit")
            if (seed not in manifest["settings"]["partition_seeds"]
                    or not set(settings.target_fpirs) <= set(manifest["settings"]["target_fpirs"])
                    or manifest["versions"] != dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__)):
                raise ValueError("PQ reuse seed/targets/runtime mismatch")
            research = Path(__file__).parents[1]
            for name in ("continuous.py", "conditional.py", "rejection.py"):
                if manifest["implementation_sha256"].get(name) != sha256_file(research / "calibration" / name):
                    raise ValueError(f"PQ reuse scientific code changed: {name}")
            for split, frame in zip(("calibration", "test"), joined[profile]):
                if _frame_hash(frame) != manifest["input_frame_sha256"][split]:
                    raise ValueError("PQ reuse input frame mismatch")
            for name, digest in manifest["files"].items():
                if sha256_file(directory / name) != digest:
                    raise ValueError("PQ source output hash mismatch")
            frame = pd.read_csv(directory / "models.csv")
            for target in settings.target_fpirs:
                for method in METHODS:
                    selected = frame.loc[frame.partition_seed.eq(seed) & frame.target_fpir.eq(target) & frame.method.eq(method)]
                    if len(selected) != 1:
                        raise ValueError("missing/duplicate saved PQ model")
                    model = _decode_model(selected.iloc[0].model_json, method)
                    if (model.model_uid != selected.iloc[0].model_uid or model.target_fpir != target
                            or model.score_space != cm["score_space"]):
                        raise ValueError("saved PQ model UID/target/score space mismatch")
                    result[(profile, target, method)] = (model, str(directory), receipt["result_manifest_sha256"])
        return result


def _fit_models(cal, space, seed, target, settings):
    common = dict(target_fpir=target, score_space=space, partition_seed=seed, safety_fraction=settings.safety_fraction)
    return {
        "global_safe": fit_global_threshold(cal, partition_column="identity_id", **common),
        "fiqa_5bin": fit_conditional_threshold(cal, bin_count=5, partition_column="identity_id",
                        shrinkage_strength=settings.shrinkage_strength,
                        minimum_group_non_mated=settings.minimum_group_non_mated, **common),
        "continuous_fiqa": fit_continuous_threshold(cal, knot_quantiles=settings.knot_quantiles,
                        smoothing=settings.smoothing, ridge=settings.ridge,
                        max_iterations=settings.max_iterations, margin_slope_cap=settings.margin_slope_cap, **common),
    }


def _compact_calibration_frames(joined):
    """Drop repeated metadata only after the full source/FIQA validation.

    Neither row order nor numerical values are changed. The canonical fit and
    evaluation routines still perform their ordinary scientific validation.
    """
    calibration_columns = ["sample_id", "identity_id", "is_mated", "score", "fiqa_score"]
    test_columns = [*calibration_columns, "top_k", "top_k_correct", "true_identity_score",
                    "true_identity_rank"]
    return {profile: (cal.loc[:, calibration_columns], test.loc[:, test_columns])
            for profile, (cal, test) in joined.items()}


def run_origin_pq_split(conditions, fiqa, *, partition_seed=8972, settings=None, reused_models=None,
                        progress=None, joined_frames=None):
    """Evaluate all profiles jointly, including paired compression interactions."""
    settings = settings or OriginPQSettings()
    settings.validate()
    seed = _seed(partition_seed)
    full_joined = _joined_frames(conditions, fiqa) if joined_frames is None else joined_frames
    joined = _compact_calibration_frames(full_joined)
    del full_joined
    reused = reused_models or {}
    tables = {n: [] for n in TABLES}
    anchor_cal, anchor_test = joined[ORIGIN_PROFILE]
    mated = anchor_test.is_mated.to_numpy(dtype=bool)
    partitions = {}
    for profile, (cal, _) in joined.items():
        p = deterministic_calibration_partition(cal, safety_fraction=settings.safety_fraction,
                                                seed=seed, partition_column="identity_id")
        if partitions and not np.array_equal(p, next(iter(partitions.values()))):
            raise ValueError("fit/safety membership differs across representations")
        partitions[profile] = p
        assignment = cal[["sample_id", "identity_id", "is_mated"]].assign(partition=p.to_numpy())
        for label in ("fit", "safety"):
            mask = p.eq(label)
            tables["partition_inventory"].append(dict(compression_profile=profile, partition_seed=seed,
                partition=label, query_count=int(mask.sum()), non_mated_count=int((mask & ~cal.is_mated).sum()),
                identity_count=int(cal.loc[mask, "identity_id"].nunique()), assignment_sha256=_frame_hash(assignment)))
    for target in settings.target_fpirs:
        evaluations = {}
        diagnostics = {}
        for profile, (cal, test) in joined.items():
            if progress:
                progress(dict(stage="calibration", partition_seed=seed, target_fpir=target, profile=profile))
            keys = [(profile, target, method) for method in METHODS]
            if any(k in reused for k in keys) and not all(k in reused for k in keys):
                raise ValueError("partial model reuse is not allowed within a profile/target")
            start = perf_counter()
            fits = ({method: reused[(profile, target, method)][0] for method in METHODS}
                    if all(k in reused for k in keys) else _fit_models(cal, conditions[profile].manifest["score_space"],
                                                                     seed, target, settings))
            elapsed = perf_counter() - start
            for method, model in fits.items():
                key = (profile, method)
                evaluation = (apply_continuous_threshold(test, model) if method == "continuous_fiqa"
                              else apply_threshold_model(test, model))
                # Retain only paired binary events and summary statistics.
                # A full copied decision frame for every profile/method would
                # keep twelve object-heavy SurvFace tables alive per target.
                summary = {**evaluation.summary, **failure_decomposition(evaluation)}
                evaluations[key] = dict(summary=summary,
                    true_identification_at_rank_k=evaluation.decisions.true_identification_at_rank_k.to_numpy(
                        dtype=bool, copy=True),
                    false_accept=evaluation.decisions.false_accept.to_numpy(dtype=bool, copy=True))
                source = reused.get((profile, target, method))
                tables["models"].append(dict(compression_profile=profile, partition_seed=seed, target_fpir=target,
                    method=method, model_uid=model.model_uid, model_json=json.dumps(model.as_dict(), allow_nan=False),
                    fitting_reused=source is not None, source_result_dir=source[1] if source else "",
                    source_manifest_sha256=source[2] if source else "", profile_fit_or_load_seconds=elapsed))
                diagnostics[key] = frozen_score_diagnostic(evaluation, settings.diagnostic_fpir_grid)
                for item in diagnostics[key].to_dict("records"):
                    tables["diagnostic_curves"].append(dict(compression_profile=profile, method=method,
                        partition_seed=seed, fitted_target_fpir=target, **item))
                del evaluation
        keys = list(evaluations)
        events = np.column_stack([evaluations[k]["true_identification_at_rank_k"][mated] for k in keys])
        draws = cluster_rate_draws(anchor_test.loc[mated, "identity_id"], events,
                                   resamples=settings.resamples, seed=settings.bootstrap_seed)
        for i, key in enumerate(keys):
            low, high = np.quantile(draws[:, i], [.025, .975])
            evaluation = evaluations[key]
            tables["method_summary"].append({**evaluation["summary"],
                "compression_profile": key[0], "method": key[1], "partition_seed": seed,
                "tpir_cluster95_low": low, "tpir_cluster95_high": high,
                "same_test_across_splits": True})
        pairs = []
        for profile in joined:
            pairs.extend([((profile, "global_safe"), (profile, method), "within_profile")
                          for method in METHODS[1:]])
            pairs.append(((profile, "fiqa_5bin"), (profile, "continuous_fiqa"), "within_profile"))
            if profile != ORIGIN_PROFILE:
                pairs.extend([((ORIGIN_PROFILE, method), (profile, method), "compression_vs_origin")
                              for method in METHODS])
        for left, right, kind in pairs:
            li, ri = keys.index(left), keys.index(right)
            for metric, column, mask in (("tpir_at_rank_k", "true_identification_at_rank_k", mated),
                                         ("fpir", "false_accept", ~mated)):
                a = evaluations[left][column][mask]
                b = evaluations[right][column][mask]
                low, high = (np.quantile(draws[:, ri] - draws[:, li], [.025, .975]) if metric != "fpir"
                    else paired_binary_rate_difference_bootstrap_interval(int(a.sum()), int(b.sum()),
                        int((a & b).sum()), len(a), resamples=settings.resamples, random_seed=settings.bootstrap_seed))
                tables["paired_comparisons"].append(dict(partition_seed=seed, target_fpir=target, comparison=kind,
                    reference_profile=left[0], reference_method=left[1], candidate_profile=right[0],
                    candidate_method=right[1], metric=metric, reference_successes=int(a.sum()),
                    candidate_successes=int(b.sum()), both_successes=int((a & b).sum()), total=len(a),
                    candidate_minus_reference=float(b.mean() - a.mean()), paired_bootstrap95_low=float(low),
                    paired_bootstrap95_high=float(high), resamples=settings.resamples,
                    resampling_unit="mated_identity_cluster" if metric != "fpir" else "query",
                    reference_realized_fpir=evaluations[left]["summary"]["realized_fpir"],
                    candidate_realized_fpir=evaluations[right]["summary"]["realized_fpir"],
                    both_target_met=bool(evaluations[left]["summary"]["target_met_on_test"]
                                         and evaluations[right]["summary"]["target_met_on_test"]),
                    comparison_basis="calibration_fixed_same_target_not_matched_actual_fpir"))
        for profile in joined:
            if profile == ORIGIN_PROFILE:
                continue
            for method in METHODS[1:]:
                order = [(profile, method), (profile, "global_safe"),
                         (ORIGIN_PROFILE, method), (ORIGIN_PROFILE, "global_safe")]
                v = np.array([evaluations[k]["summary"]["tpir_at_rank_k"] for k in order])
                di = [draws[:, keys.index(k)] for k in order]
                low, high = np.quantile(di[0] - di[1] - di[2] + di[3], [.025, .975])
                tables["interactions"].append(dict(compression_profile=profile, method=method,
                    partition_seed=seed, target_fpir=target, pq_gain=v[0]-v[1], origin_gain=v[2]-v[3],
                    interaction=v[0]-v[1]-v[2]+v[3], paired_bootstrap95_low=low, paired_bootstrap95_high=high,
                    all_four_target_met=all(evaluations[k]["summary"]["target_met_on_test"] for k in order),
                    comparison_basis="calibration_fixed_same_target_not_matched_actual_fpir",
                    **{name: evaluations[k]["summary"]["realized_fpir"] for name, k in zip(
                        ("pq_method_fpir", "pq_global_fpir", "origin_method_fpir", "origin_global_fpir"), order)}))
                values = [diagnostics[k].diagnostic_tpir.to_numpy() for k in order]
                for j, fpir in enumerate(settings.diagnostic_fpir_grid):
                    tables["diagnostic_interactions"].append(dict(compression_profile=profile, method=method,
                        partition_seed=seed, fitted_target_fpir=target, diagnostic_fpir=fpir,
                        pq_gain=values[0][j]-values[1][j], origin_gain=values[2][j]-values[3][j],
                        interaction=values[0][j]-values[1][j]-values[2][j]+values[3][j],
                        diagnostic_only=True, interpolated_test_curve=True, diagnostic_ci_available=False))
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}


def _load_tables(directory, spec=None):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("status") != "completed" or (spec is not None and manifest.get("spec") != spec):
        raise ValueError("comparison checkpoint/spec mismatch")
    tables = {}
    for name, digest in manifest["files"].items():
        path = directory / name
        if directory.resolve() not in path.resolve().parents or sha256_file(path) != digest:
            raise ValueError("comparison checkpoint hash mismatch")
        tables[path.stem] = pd.read_csv(path)
    return tables, manifest


def _validate_split_tables(tables, conditions, seed, settings):
    """Check coverage and arithmetic even after a checkpoint's hashes pass."""
    keys = ["compression_profile", "method", "target_fpir", "partition_seed"]
    expected = {(p, m, t, seed) for p in conditions for m in METHODS for t in settings.target_fpirs}
    for name in ("method_summary", "models"):
        frame = tables[name]
        observed = set(frame[keys].itertuples(index=False, name=None))
        if len(frame) != len(expected) or observed != expected:
            raise ValueError(f"incomplete split grid: {name}")
    summaries = tables["method_summary"].set_index(keys)
    for row in tables["models"].itertuples(index=False):
        model = _decode_model(row.model_json, row.method)
        summary = summaries.loc[(row.compression_profile, row.method, row.target_fpir, seed)]
        condition = conditions[row.compression_profile]
        if (model.model_uid != row.model_uid or summary.model_uid != row.model_uid
                or model.target_fpir != row.target_fpir or model.score_space != condition.manifest["score_space"]):
            raise ValueError("split model identity mismatch")
        mated = int(condition.test.is_mated.sum())
        nonmated = len(condition.test) - mated
        if (summary.test_mated_count != mated or summary.test_non_mated_count != nonmated
                or summary.test_probe_count != mated + nonmated
                or summary.rank_k != condition.manifest["top_k"]):
            raise ValueError("split evaluation denominator/K mismatch")
        counts = [summary.false_accept_count, summary.true_identification_at_rank_k_count,
                  summary.rank_failure_count, summary.threshold_failure_count]
        if (any(not np.isfinite(c) or c < 0 or c != int(c) for c in counts)
                or counts[0] > nonmated or sum(counts[1:]) != mated):
            raise ValueError("split success/failure counts mismatch")
        fpir, tpir = counts[0] / nonmated, counts[1] / mated
        if (not np.isclose(summary.realized_fpir, fpir, rtol=0, atol=1e-15)
                or not np.isclose(summary.tpir_at_rank_k, tpir, rtol=0, atol=1e-15)
                or bool(summary.target_met_on_test) != (fpir <= row.target_fpir)):
            raise ValueError("split rates/target flag mismatch")
    curves = tables["diagnostic_curves"]
    curve_keys = ["compression_profile", "method", "fitted_target_fpir", "partition_seed", "diagnostic_fpir"]
    curve_expected = {(*key, x) for key in expected for x in settings.diagnostic_fpir_grid}
    if (len(curves) != len(curve_expected)
            or set(curves[curve_keys].itertuples(index=False, name=None)) != curve_expected
            or not curves.diagnostic_tpir.between(0, 1).all()):
        raise ValueError("incomplete diagnostic curve grid")
    expected_interactions = (len(conditions) - 1) * 2 * len(settings.target_fpirs)
    if (len(tables["interactions"]) != expected_interactions
            or len(tables["diagnostic_interactions"]) != expected_interactions * len(settings.diagnostic_fpir_grid)
            or len(tables["paired_comparisons"]) != (len(conditions) * 3 + (len(conditions) - 1) * 3)
                * len(settings.target_fpirs) * 2):
        raise ValueError("incomplete paired comparison grid")


def _write_tables(root, prefix, tables, spec):
    uid = prefix + canonical_sha256(spec)[:24]
    destination = Path(root) / uid
    if destination.exists():
        _load_tables(destination, spec)
        return destination
    staging = Path(root) / (".staging-" + uuid4().hex)
    staging.mkdir(parents=True)
    files = {}
    for name, frame in tables.items():
        path = staging / f"{name}.csv"
        frame.to_csv(path, index=False)
        files[path.name] = sha256_file(path)
    manifest = dict(artifact_type="origin_vs_pq_calibration", schema_version=1, status="completed",
                    result_uid=uid, spec=spec, files=files, formal_fpir_guarantee=False,
                    uncertainty=dict(tpir="query_weighted_mated_identity_cluster", fpir="query_bootstrap_and_wilson",
                        threshold_uncertainty_included=False, multiple_comparison_adjustment="none",
                        between_splits="descriptive_not_independent_replications",
                        diagnostic_curves="test_interpolation_only_no_ci_no_deployment_threshold"),
                    checkpoint_training_overlap_verified=False,
                    unseen_identity_claim_for_rfw_edgeface=False)
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    _publish_atomic_directory(staging, destination, overwrite=False)
    return destination


def run_origin_pq_campaign(plan, output_root, *, partition_seeds=(*range(19), 8972), settings=None,
                           pq_model_report=None, shard_size=4096, progress=print,
                           max_new_jobs=None, blas_threads=2, minimum_available_gb=8.,
                           maximum_process_gb=16., export_chat=True):
    """One transactional checkpoint DB, bounded invocations and compact reports.

    max_new_jobs limits newly computed run/seed jobs per invocation, never the
    declared experiment matrix. Re-running resumes the same campaign database.
    Memory checks occur between stages; they are not a hard allocation limit.
    """
    import gc
    from threadpoolctl import threadpool_limits
    from research.experiments.origin_pq_storage import SplitStore, write_report
    from research.experiments.origin_pq_resources import (
        ResourceBudgetExceeded, check_resources, prune_origin_replay_scratch,
    )

    settings = settings or OriginPQSettings()
    settings.validate()
    seeds = tuple(_seed(s) for s in partition_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("unique non-empty partition seeds required")
    if plan.empty or plan.duplicated(["source_run_id", "compression_profile"]).any():
        raise ValueError("non-empty unique experiment plan required")
    if max_new_jobs is not None and (isinstance(max_new_jobs, bool)
            or not isinstance(max_new_jobs, int) or max_new_jobs < 1):
        raise ValueError("max_new_jobs must be None or a positive integer")
    if isinstance(blas_threads, bool) or not isinstance(blas_threads, int) or blas_threads < 1:
        raise ValueError("positive integer blas_threads required")
    root = Path(output_root).resolve()
    # Derived output cannot be written into a source run or existing input artifact.
    protected = {Path(p).resolve() for c in ("source_run_dir", "condition_dir", "fiqa_dir") for p in plan[c]}
    if any(root == p or p in root.parents or root in p.parents for p in protected):
        raise ValueError("output must be separate from completed source inputs")
    cache = VerifiedPQModels(pq_model_report) if pq_model_report is not None else None
    receipts, accumulated = [], {n: [] for n in TABLES}
    implementation = science_hashes()
    campaign_spec = dict(plan=plan.to_dict("records"), partition_seeds=list(seeds),
                         settings=settings.as_dict(), implementation=implementation,
                         versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
                         pq_model_report_sha256=cache.report_sha256 if cache else None,
                         shard_size=shard_size, blas_threads=blas_threads)
    checkpoint = root / "checkpoints" / ("campaign-" + canonical_sha256(campaign_spec)[:24] + ".sqlite3")
    expected_jobs = plan.source_run_id.nunique() * len(seeds)
    new_jobs, stop_reason = 0, None

    def notify(event):
        check_resources(minimum_available_gb=minimum_available_gb, maximum_process_gb=maximum_process_gb)
        if progress:
            progress(event)

    with SplitStore(checkpoint) as store, threadpool_limits(limits=blas_threads, user_api="blas"):
        try:
            for run_dir, group in plan.groupby("source_run_dir", sort=False):
                if new_jobs and max_new_jobs is not None and new_jobs >= max_new_jobs:
                    stop_reason = "new_job_budget_reached"
                    break
                notify(dict(stage="inputs", source_run_dir=run_dir, completed_jobs=len(receipts),
                            expected_jobs=expected_jobs))
                conditions, fiqa = load_run_inputs(group, root / "origin_scores", shard_size=shard_size, progress=notify)
                origin_uid = conditions[ORIGIN_PROFILE].manifest.get("condition_uid")
                if origin_uid:
                    prune_origin_replay_scratch(root / "origin_scores" / origin_uid)
                # Hash and join once per source run, before any compact science frame.
                context = dict(dataset_id=group.iloc[0].dataset_id, model=group.iloc[0].model,
                               source_run_id=group.iloc[0].source_run_id)
                inputs = dict(
                    condition_manifests={p: canonical_sha256(c.manifest) for p, c in conditions.items()},
                    input_frames={p: {s: _frame_hash(getattr(c, s)) for s in ("calibration", "test")}
                                  for p, c in conditions.items()},
                    fiqa_manifest_sha256=canonical_sha256(fiqa.manifest))
                joined = None
                for seed in seeds:
                    spec = dict(context=context, partition_seed=seed, settings=settings.as_dict(),
                                **inputs, implementation=implementation, versions=campaign_spec["versions"],
                                pq_model_report_sha256=campaign_spec["pq_model_report_sha256"],
                                blas_threads=blas_threads)
                    saved = store.get(spec)
                    if saved is not None:
                        tables, manifest = saved
                        notify(dict(stage="reuse_split", partition_seed=seed, **context))
                    else:
                        if max_new_jobs is not None and new_jobs >= max_new_jobs:
                            stop_reason = "new_job_budget_reached"
                            break
                        notify(dict(stage="new_split", partition_seed=seed, **context))
                        if joined is None:
                            joined = _joined_frames(conditions, fiqa)
                        models = cache.load(conditions, fiqa, seed, settings, joined_frames=joined) if cache else {}
                        tables = run_origin_pq_split(conditions, fiqa, partition_seed=seed, settings=settings,
                                                     reused_models=models, progress=notify, joined_frames=joined)
                        _validate_split_tables(tables, conditions, seed, settings)
                        tables = {name: frame.assign(**context) for name, frame in tables.items()}
                        manifest = store.put(spec, tables)
                        new_jobs += 1
                        del models
                    _validate_split_tables(tables, conditions, seed, settings)
                    for name in TABLES:
                        accumulated[name].append(tables[name])
                    receipts.append(dict(**context, partition_seed=seed, result_dir=str(checkpoint),
                                         result_uid=manifest["result_uid"],
                                         result_manifest_sha256=canonical_sha256(manifest)))
                del conditions, fiqa, joined
                gc.collect()
                if stop_reason:
                    break
        except (ResourceBudgetExceeded, KeyboardInterrupt) as exc:
            stop_reason = str(exc) or "user_interrupted"
        finally:
            # Do not retain cohort-sized inputs while publishing a partial report
            # after a memory guard or user interruption.
            conditions = fiqa = joined = models = None
            del conditions, fiqa, joined, models
            gc.collect()
    completed = len(receipts) == expected_jobs
    state = dict(checkpoint_path=checkpoint, completed=completed, completed_jobs=len(receipts),
                 expected_jobs=int(expected_jobs), new_jobs=new_jobs, stop_reason=stop_reason,
                 report_dir=None, chat_dir=None)
    if not receipts:
        return state
    frames = {name: pd.concat(parts, ignore_index=True) for name, parts in accumulated.items()}
    expected = sum((len(group) + 1) * len(seeds) * len(settings.target_fpirs) * len(METHODS)
                   for _, group in plan.groupby("source_run_dir"))
    summary = frames["method_summary"]
    observed_expected = sum((len(plan.loc[plan.source_run_id.eq(r["source_run_id"])]) + 1)
                            * len(settings.target_fpirs) * len(METHODS) for r in receipts)
    if (len(summary) != observed_expected or (completed and len(summary) != expected) or summary.duplicated(
            ["source_run_id", "compression_profile", "partition_seed", "target_fpir", "method"]).any()):
        raise ValueError("incomplete or duplicate comparison grid")
    frames["split_summary"] = summary.groupby(
        ["dataset_id", "model", "compression_profile", "method", "target_fpir"], sort=False).agg(
            split_count=("partition_seed", "count"), target_met_split_count=("target_met_on_test", "sum"),
            fpir_min=("realized_fpir", "min"), fpir_median=("realized_fpir", "median"),
            fpir_max=("realized_fpir", "max"), tpir_min=("tpir_at_rank_k", "min"),
            tpir_median=("tpir_at_rank_k", "median"), tpir_max=("tpir_at_rank_k", "max"),
            rank_failure_rate=("rank_failure_rate", "first"),
            threshold_failure_median=("threshold_failure_rate", "median")).reset_index()
    frames["source_receipts"] = pd.DataFrame(receipts)
    report_spec = dict(**campaign_spec, expected_metric_rows=expected, source_receipts=receipts,
                       expected_jobs=int(expected_jobs), completed_jobs=len(receipts),
                       experiment_complete=completed,
                       execution_policy=dict(max_new_jobs=max_new_jobs, minimum_available_gb=minimum_available_gb,
                                             maximum_process_gb=maximum_process_gb))
    # Partial detail remains in SQLite. Avoid duplicating a growing full archive
    # after every batch; publish the complete detailed report only once finished.
    if completed:
        state["report_dir"] = write_report(root / "reports", frames, report_spec)
    if export_chat:
        from research.experiments.origin_pq_compact import write_chat_bundle
        state["chat_dir"] = write_chat_bundle(root / "chat", frames, report_spec,
            expected_jobs=[f"{r}:{s}" for r in plan.source_run_id.unique() for s in seeds],
            completed_jobs=[f"{r['source_run_id']}:{r['partition_seed']}" for r in receipts])
    return dict(**state, **frames)
