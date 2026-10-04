"""Pairwise FMR/TAR evaluation; no gallery maximum, rank, FPIR or TPIR metrics."""

import json

import numpy as np
import pandas as pd

from research.calibration.conditional import assign_quality_groups
from research.evaluation.metrics import wilson_score_interval
from research.experiments.origin_vs_pq_calibration import _fit_models

METHODS = ("global_safe", "fiqa_5bin", "continuous_fiqa")


def pair_scores(vectors, pairs, image_ids, *, codec=None, batch_size=128):
    """Left is the original query; right is the stored reference (8-bit PQ for ADC)."""
    lookup = pd.Index(image_ids)
    left = lookup.get_indexer(pairs.left_image_id)
    right = lookup.get_indexer(pairs.right_image_id)
    if min(left.min(), right.min()) < 0 or batch_size < 1:
        raise ValueError("missing pair embeddings or invalid batch size")
    if codec is None:
        return np.einsum("ij,ij->i", vectors[left], vectors[right]).astype(float)
    if codec.nbits != 8:
        raise ValueError("pair ADC currently requires 8-bit subquantizers")
    import faiss

    result = []
    for start in range(0, len(pairs), batch_size):
        a = np.ascontiguousarray(vectors[left[start:start + batch_size]], dtype=np.float32)
        codes = codec.encode(vectors[right[start:start + batch_size]])
        tables = np.empty((len(a), codec.m, 256), dtype=np.float32)
        codec.index.pq.compute_distance_tables(len(a), faiss.swig_ptr(a), faiss.swig_ptr(tables))
        distances = np.take_along_axis(tables, codes[..., None].astype(np.int64), axis=2)
        result.extend(-distances[..., 0].sum(axis=1, dtype=np.float64))
    return np.asarray(result)


def empirical_fmr_threshold(scores, target):
    """Largest acceptance set obeying the empirical FMR ceiling; ties stay intact."""
    scores = np.asarray(scores, dtype=float)
    if not len(scores) or not np.isfinite(scores).all() or not 0 <= target < 1:
        raise ValueError("finite impostor scores and FMR inside [0,1) required")
    unique, counts = np.unique(scores, return_counts=True)
    descending, cumulative = unique[::-1], np.cumsum(counts[::-1])
    feasible = np.flatnonzero(cumulative / len(scores) <= target)
    first_rejected = 0 if not len(feasible) else int(feasible[-1]) + 1
    # Moving just above the next rejected impostor also admits genuine scores in
    # the gap, giving the best achievable TAR under this FMR ceiling.
    return float(np.nextafter(descending[first_rejected], np.inf))


def accuracy_threshold(scores, genuine):
    """Maximize calibration accuracy, breaking ties toward a stricter threshold."""
    scores, genuine = np.asarray(scores), np.asarray(genuine, dtype=bool)
    if not len(scores) or not genuine.any() or genuine.all() or not np.isfinite(scores).all():
        raise ValueError("accuracy fitting requires finite genuine and impostor pairs")
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], genuine[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    correct = int((~y).sum()) + np.cumsum(np.where(y, 1, -1))[ends]
    options = np.r_[int((~y).sum()), correct]
    thresholds = np.r_[np.nextafter(s[0], np.inf), s[ends]]
    return float(thresholds[np.argmax(options)])


def thresholds(model, frame):
    if model.method == "continuous_fiqa":
        return model.predict(frame)
    if model.quality_column is None:
        return np.full(len(frame), model.global_final_threshold)
    groups = assign_quality_groups(frame.fiqa_score, cutpoints=model.quality_cutpoints,
                                   labels=model.group_labels)
    return np.asarray([model.thresholds[g] for g in groups])


def _pair_model_payload(model):
    # Shared quantile engine historically uses FPIR names. Stored pair models use
    # the actual pairwise estimand, without modifying the 1:N engine or artifacts.
    def rename(value):
        if isinstance(value, dict):
            return {k.replace("fpir", "fmr").replace("non_mated", "impostor"): rename(v)
                    for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [rename(v) for v in value]
        return value
    return json.dumps(rename(model.as_dict()), sort_keys=True, allow_nan=False)


def _frame(pairs, scores, quality):
    frame = pairs.copy()
    frame["sample_id"] = frame.pair_id
    frame["identity_id"] = frame.left_identity
    frame["is_mated"] = frame.is_genuine
    frame["score"] = scores.loc[frame.pair_id].to_numpy()
    frame["fiqa_score"] = quality.loc[frame.left_image_id].to_numpy()
    return frame


def evaluate_fold(scored, quality, calibration, test, *, seed, settings, keep_decisions=False):
    """Fit all methods on the same calibration pairs, then freeze before test."""
    labels = test.is_genuine.to_numpy(dtype=bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if not npos or not nneg:
        raise ValueError("both pair classes required")
    identities, cluster = np.unique(test.loc[test.is_genuine, "left_identity"], return_inverse=True)
    sizes = np.bincount(cluster, minlength=len(identities))
    rng = np.random.default_rng(settings.bootstrap_seed)
    weights = rng.multinomial(len(identities), np.full(len(identities), 1 / len(identities)),
                              size=settings.resamples)
    denominator = weights @ sizes
    tables = {k: [] for k in ("operating", "diagnostic", "accuracy_cv", "paired", "models", "decisions")}
    bootstrap, events, operational = {}, {}, {}
    for profile, scores in scored.groupby("compression_profile", sort=False):
        space = scores.score_space.iloc[0]
        series = scores.set_index("pair_id").pair_score
        cal = _frame(calibration, series, quality)
        tst = _frame(test, series, quality)
        # Standard pair-fold accuracy baseline uses all remaining nine folds.
        other = scores.loc[~scores.pair_id.isin(test.pair_id)]
        acc_tau = accuracy_threshold(other.pair_score, other.is_genuine)
        acc_correct = int(((tst.score.to_numpy() >= acc_tau) == labels).sum())
        tables["accuracy_cv"].append(dict(compression_profile=profile, score_space=space,
            threshold=acc_tau, calibration_pairs=len(other), test_pairs=len(test),
            correct_count=acc_correct, accuracy=acc_correct / len(test),
            threshold_policy="nine_fold_accuracy_max_global", threshold_fit_on_test=False))
        for target in settings.target_fpirs:
            models = _fit_models(cal, space, seed, target, settings)
            for method, model in models.items():
                key = (profile, method, target)
                base = dict(compression_profile=profile, score_space=space, method=method, target_fmr=target)
                tau = thresholds(model, tst)
                accepted = tst.score.to_numpy() >= tau
                tp, fp = int(accepted[labels].sum()), int(accepted[~labels].sum())
                cluster_success = np.bincount(cluster, weights=accepted[labels], minlength=len(identities))
                boot = (weights @ cluster_success) / denominator
                bootstrap[key], events[key] = boot, accepted
                fmr, tar = fp / nneg, tp / npos
                low, high = wilson_score_interval(fp, nneg)
                tar_low, tar_high = np.quantile(boot, [.025, .975])
                operational[key] = (fmr, tar)
                tables["operating"].append(dict(**base, test_pairs=len(test), genuine_pairs=npos,
                    impostor_pairs=nneg, true_accepts=tp, false_accepts=fp, false_rejects=npos-tp,
                    true_rejects=nneg-fp, realized_fmr=fmr, tar=tar, fnmr=1-tar,
                    operating_accuracy=(tp+nneg-fp)/len(test), target_met_on_test=fmr <= target,
                    fmr_wilson95_nominal_low=low, fmr_wilson95_nominal_high=high,
                    tar_identity_bootstrap95_low=float(tar_low), tar_identity_bootstrap95_high=float(tar_high),
                    threshold_min=float(tau.min()), threshold_max=float(tau.max()),
                    threshold_fit_on_test=False, formal_fmr_guarantee=False))
                residual = tst.score.to_numpy() - tau
                for coordinate in settings.diagnostic_fpir_grid:
                    if not 0 < coordinate < 1:
                        continue
                    cutoff = empirical_fmr_threshold(residual[~labels], coordinate)
                    diagnostic = residual >= cutoff
                    tables["diagnostic"].append(dict(compression_profile=profile, method=method,
                        fitted_target_fmr=target, requested_fmr=coordinate,
                        achieved_fmr=float(diagnostic[~labels].mean()), tar=float(diagnostic[labels].mean()),
                        false_accepts=int(diagnostic[~labels].sum()), true_accepts=int(diagnostic[labels].sum()),
                        impostor_pairs=nneg, genuine_pairs=npos, residual_cutoff=cutoff,
                        diagnostic_only=True, deployment_threshold=False, interpolated=False))
                tables["models"].append(dict(**base, model_json=_pair_model_payload(model)))
                for j, pair_id in enumerate(test.pair_id if keep_decisions else []):
                    tables["decisions"].append(dict(**base, pair_id=pair_id, is_genuine=bool(labels[j]),
                        pair_score=float(tst.score.iloc[j]), applied_threshold=float(tau[j]), accepted=bool(accepted[j])))
    profiles = list(scored.compression_profile.unique())
    for target in settings.target_fpirs:
        for profile in profiles:
            for method in METHODS[1:]:
                candidate, reference = (profile, method, target), (profile, "global_safe", target)
                diff = bootstrap[candidate] - bootstrap[reference]
                low, high = np.quantile(diff, [.025, .975])
                tables["paired"].append(dict(compression_profile=profile, method=method, target_fmr=target,
                    comparison="method_minus_global", tar_difference=operational[candidate][1]-operational[reference][1],
                    reference_fmr=operational[reference][0], candidate_fmr=operational[candidate][0],
                    paired_identity_bootstrap95_low=float(low), paired_identity_bootstrap95_high=float(high)))
            if profile == "origin":
                continue
            for method in METHODS:
                a, b = ("origin", method, target), (profile, method, target)
                low, high = np.quantile(bootstrap[b]-bootstrap[a], [.025, .975])
                tables["paired"].append(dict(compression_profile=profile, method=method, target_fmr=target,
                    comparison="compressed_minus_origin", tar_difference=operational[b][1]-operational[a][1],
                    reference_fmr=operational[a][0], candidate_fmr=operational[b][0],
                    paired_identity_bootstrap95_low=float(low), paired_identity_bootstrap95_high=float(high)))
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}
