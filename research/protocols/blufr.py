"""BLUFR open-set score curves, separate from held-out operating thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd


def identification_rows(scores, gallery_ids, query_ids, identity_ids):
    """Preserve full genuine ranks; stable gallery order resolves exact ties."""
    values = np.asarray(scores)
    gallery_ids, identity_ids = np.asarray(gallery_ids), np.asarray(identity_ids)
    if (values.shape != (len(query_ids), len(gallery_ids)) or not np.isfinite(values).all()
            or len(set(gallery_ids)) != len(gallery_ids) or len(set(query_ids)) != len(query_ids)):
        raise ValueError("invalid open-set score matrix/identities")
    order = np.argsort(-values, axis=1, kind="stable")
    lookup = {identity: i for i, identity in enumerate(gallery_ids)}
    genuine = np.array([lookup.get(identity, -1) for identity in identity_ids])
    mated = genuine >= 0
    ranks = np.full(len(query_ids), np.nan)
    true_scores = np.full(len(query_ids), np.nan)
    rows = np.flatnonzero(mated)
    ranks[rows] = np.argmax(order[rows] == genuine[rows, None], axis=1) + 1
    true_scores[rows] = values[rows, genuine[rows]]
    return pd.DataFrame(dict(sample_id=query_ids, identity_id=identity_ids, is_mated=mated,
                             score=values.max(axis=1), true_identity_rank=ranks,
                             true_identity_score=true_scores))


def benchmark_curve(rows, target_fpirs=(.01, .05, .10), ranks=(1, 20)):
    """Port OpenSetROC.m's operating points, including MATLAB positive rounding.

    Toolkit FAR is the requested false-alarm count / N, which can differ from
    measured FPIR when scores tie. Report BOTH and never export these thresholds
    as calibration models. Test scores are used only to describe this curve.
    """
    unknown = np.sort(rows.loc[~rows.is_mated, "score"].to_numpy(dtype=float))[::-1]
    genuine = rows.loc[rows.is_mated]
    if not len(unknown) or genuine.empty:
        raise ValueError("benchmark needs mated and non-mated probes")
    result = []
    for target in target_fpirs:
        if not np.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("invalid benchmark FPIR")
        count = int(np.floor(target * len(unknown) + .5))
        if count == 0:
            high = genuine.loc[genuine.true_identity_score > unknown[0], "true_identity_score"]
            threshold = (unknown[0] + high.min()) / 2 if len(high) else unknown[0] + np.sqrt(np.finfo(float).eps)
        elif count == len(unknown):
            threshold = min(unknown[-1], genuine.true_identity_score.min()) - np.sqrt(np.finfo(float).eps)
        else:
            threshold = unknown[count - 1]
        false_accepts = int((unknown >= threshold).sum())
        for rank in ranks:
            success = int(((genuine.true_identity_score >= threshold)
                           & (genuine.true_identity_rank <= rank)).sum())
            result.append(dict(target_fpir=target, rank=rank, toolkit_far=count / len(unknown),
                realized_fpir=false_accepts / len(unknown), false_accept_count=false_accepts,
                non_mated_count=len(unknown), true_identification_count=success,
                mated_count=len(genuine), tpir=success / len(genuine), threshold=float(threshold),
                threshold_source="test_curve_only", deployment_threshold_selected=False))
    return pd.DataFrame(result)
