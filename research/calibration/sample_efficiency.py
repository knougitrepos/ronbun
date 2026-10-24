"""Fit-only quantile learning curves; not joint FR/calibration optimization."""
import hashlib

import numpy as np
import pandas as pd

from research.calibration.continuous import _fit_continuous_quantile


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
