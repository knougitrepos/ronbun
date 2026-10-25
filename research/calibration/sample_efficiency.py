"""Calibration-only diagnostics; never optimize against final safety or test."""
import hashlib

import numpy as np
import pandas as pd

from research.calibration.continuous import _fit_continuous_quantile


def fit_safety_learning_curve(calibration, partition_labels, *, target, score_space,
                             fit_fractions=(.25, .5, .75, 1.), safety_fractions=(.25, .5, .75, 1.),
                             folds=5, seed=8972, inner_safety_fraction=.3):
    """Vary BOTH fit and safety sizes inside the original FIT identity pool.

    The final safety cohort is never used. Identity-disjoint inner validation is
    fixed within each fold; nested subsets keep comparisons paired. Pair tasks
    require BOTH identities in a role and drop cross-role pairs. Fractions refer
    to inner pools, not the full FR/calibration split. No test TPIR/TAR selection.
    """
    from research.calibration.continuous import _heldout_safety_offset
    from research.calibration.rejection import choose_non_mated_fpir_threshold
    from research.evaluation.metrics import wilson_score_interval
    from research.runtime.hashing import canonical_sha256

    if len(partition_labels) != len(calibration) or not set(partition_labels) <= {"fit", "safety"}:
        raise ValueError("Explicit fit/safety roles required")
    for values in (fit_fractions, safety_fractions):
        if not values or len(set(values)) != len(values) or any(not np.isfinite(f) or not 0 < f <= 1 for f in values):
            raise ValueError("Unique finite fractions in (0,1] required")
    if type(folds) is not int or folds < 2 or not 0 < inner_safety_fraction < 1 or not 0 < target < 1:
        raise ValueError("Invalid inner fit/safety/validation split")
    pool = calibration.loc[(np.asarray(partition_labels) == "fit") & ~calibration.is_mated].copy()
    if (pool.sample_id.duplicated().any()
            or not np.isfinite(pool[["score", "fiqa_score"]].to_numpy(dtype=float)).all()):
        raise ValueError("Diagnostic fit rows require unique sample IDs and finite scores/quality")
    paired = {"left_identity", "right_identity"} <= set(pool)
    columns = ["left_identity", "right_identity"] if paired else ["identity_id"]
    if pool[columns].isna().any().any() or pool[columns].astype(str).eq("").any().any():
        raise ValueError("Calibration identities must be known")
    identities = sorted(set(pool[columns].astype(str).to_numpy().ravel()),
                        key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    def rows(ids):
        mask = np.ones(len(pool), dtype=bool)
        for column in columns:
            mask &= pool[column].astype(str).isin(ids).to_numpy()
        return pool.loc[mask].copy()
    records = []
    for fold in range(folds):
        valid_ids = set(identities[fold::folds])
        remaining = [i for i in identities if i not in valid_ids]
        ns = max(1, int(len(remaining) * inner_safety_fraction))
        safety_ids, fit_ids = remaining[:ns], remaining[ns:]
        valid = rows(valid_ids)
        for ff in fit_fractions:
            for sf in safety_fractions:
                fit = rows(fit_ids[:max(1, int(len(fit_ids) * ff))])
                safety = rows(safety_ids[:max(1, int(len(safety_ids) * sf))])
                common = dict(fold=fold, fit_fraction=ff, safety_fraction=sf, target_rate=target,
                    sample_unit="pair" if paired else "query",
                    fit_samples=len(fit), safety_samples=len(safety), validation_samples=len(valid),
                    fit_identities=len(set(fit[columns].astype(str).to_numpy().ravel())),
                    safety_identities=len(set(safety[columns].astype(str).to_numpy().ravel())),
                    validation_identities=len(set(valid[columns].astype(str).to_numpy().ravel())),
                    safety_expected_tail_count=len(safety)*target,
                    validation_expected_tail_count=len(valid)*target,
                    score_space=score_space, final_safety_used=False, final_test_used=False,
                    formal_fpir_guarantee=False,
                    assignment_sha256=canonical_sha256(dict(fit=sorted(fit.sample_id.astype(str)),
                        safety=sorted(safety.sample_id.astype(str)), validation=sorted(valid.sample_id.astype(str)))),
                    scope="inner original-fit identities only; correlated folds/subsets; no optimal FR split claim")
                if min(len(fit), len(safety)) < 20 or valid.empty:
                    records.extend(dict(**common, method=m, status="insufficient_rows")
                                   for m in ("global_safe", "continuous_fiqa"))
                    continue
                # Same non-mated fitting/offset engines as the main experiment.
                # Do not invent genuine rows to satisfy its evaluation validator.
                global_tau = max(choose_non_mated_fpir_threshold(part.score.to_numpy(),
                    np.zeros(len(part), dtype=bool), target_fpir=target) for part in (fit, safety))
                model, _ = _fit_continuous_quantile(fit, features=("fiqa_score",), target_fpir=target,
                    method="continuous_fiqa", score_space=score_space, knots=(1/3, 2/3),
                    smoothing=.01, ridge=.001, max_iterations=2000, margin_slope_cap=.95)
                offset = _heldout_safety_offset(safety.score.to_numpy(), model.predict(safety), target_fpir=target)
                predictions = {"global_safe": np.full(len(valid), global_tau),
                               "continuous_fiqa": model.predict(valid) + offset}
                for method, tau in predictions.items():
                    residual = valid.score.to_numpy() - tau
                    count = int((residual >= 0).sum())
                    low, high = wilson_score_interval(count, len(valid))
                    records.append(dict(**common, method=method, status="completed",
                        validation_false_accepts=count, validation_false_accept_rate=count/len(valid),
                        validation_target_met=bool(count/len(valid) <= target),
                        validation_wilson95_low=low, validation_wilson95_high=high,
                        interval_scope="descriptive row-level Wilson; identity dependence not modeled",
                        validation_pinball_loss=float(np.maximum((1-target)*residual, -target*residual).mean()),
                        threshold_mean=float(np.mean(tau))))
    return pd.DataFrame(records)


def quantile_learning_curve(calibration, partition_labels, *, target, score_space,
                            fractions=(.25, .5, .75, 1.), folds=5, seed=8972):
    """Final safety rows never enter model selection or this diagnostic.

    Each validation identity is absent from the quantile fit. No final safety
    offset or final-test TPIR is optimized by these results.
    """
    roles = np.asarray(partition_labels)
    fit = calibration.loc[(roles == "fit") & ~calibration.is_mated].copy()
    identities = sorted(fit.identity_id.unique(), key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    if folds < 2 or len(identities) < folds * 2 or any(not 0 < f <= 1 for f in fractions):
        raise ValueError("Insufficient fit identities or invalid learning-curve settings")
    split = {identity: index % folds for index, identity in enumerate(identities)}
    records = []
    for fold in range(folds):
        valid = fit.loc[fit.identity_id.map(split).eq(fold)]
        candidates = [identity for identity in identities if split[identity] != fold]
        for fraction in fractions:
            selected = candidates[:max(1, int(len(candidates) * fraction))]
            train = fit.loc[fit.identity_id.isin(selected)]
            if len(train) < 20:
                records.append(dict(fold=fold, fraction=fraction, status="insufficient_fit_rows", fit_images=len(train)))
                continue
            model, _ = _fit_continuous_quantile(train, features=("fiqa_score",), target_fpir=target,
                method="continuous_fiqa", score_space=score_space, knots=(1/3, 2/3),
                smoothing=.01, ridge=.001, max_iterations=2000, margin_slope_cap=.95)
            residual = valid.score.to_numpy() - model.predict(valid)
            loss = np.maximum((1-target) * residual, -target * residual)
            records.append(dict(fold=fold, fraction=fraction, status="completed", fit_images=len(train),
                fit_identities=len(selected), validation_images=len(valid), validation_identities=valid.identity_id.nunique(),
                target_rate=target, validation_pinball_loss=float(loss.mean()),
                validation_exceedance_rate=float((residual >= 0).mean()),
                scope="fit-only quantile diagnostic; no final safety offset or TPIR optimization"))
    return pd.DataFrame(records)
