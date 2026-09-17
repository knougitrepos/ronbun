"""Low-dimensional smoothed quantile thresholds, fitted on calibration only."""

from dataclasses import dataclass, asdict, replace

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from research.calibration.conditional import (
    _validated_rows, deterministic_calibration_partition,
    apply_threshold_model,
)
from research.calibration.rejection import choose_non_mated_fpir_threshold
from research.runtime.hashing import canonical_sha256


@dataclass(frozen=True)
class ContinuousThresholdModel:
    method: str
    target_fpir: float
    score_space: str
    features: tuple
    centers: tuple
    scales: tuple
    quality_bounds: tuple
    quality_knots: tuple
    coefficients: tuple
    score_center: float
    score_scale: float
    safety_offset: float
    settings: dict

    def as_dict(self):
        return asdict(self)

    @property
    def model_uid(self):
        return "continuous-threshold-" + canonical_sha256(self.as_dict())[:24]

    def design(self, frame):
        values = frame[list(self.features)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("continuous features must be finite")
        z = (values - np.asarray(self.centers)) / np.asarray(self.scales)
        z[:, 0] = np.clip(z[:, 0], *self.quality_bounds)
        return np.column_stack([
            np.ones(len(z)), z[:, 0],
            *[np.maximum(z[:, 0] - knot, 0) for knot in self.quality_knots],
            *[z[:, i] for i in range(1, z.shape[1])],
        ])

    def predict(self, frame):
        return (self.score_center + self.score_scale *
                (self.design(frame) @ np.asarray(self.coefficients)) + self.safety_offset)


def _fit_continuous_quantile(fit, *, features, target_fpir, method, score_space,
                             knots, smoothing, ridge, max_iterations, margin_slope_cap):
    """Fit preprocessing and coefficients on explicit non-mated FIT rows only."""
    values = fit[list(features)].to_numpy(dtype=float)
    centers = np.median(values, axis=0)
    scales = np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)
    scales = np.where(scales > 1e-12, scales, 1.)
    q = (values[:, 0] - centers[0]) / scales[0]
    ycenter = float(np.median(fit.score))
    yscale = max(float(np.std(fit.score)), 1e-6)
    model = ContinuousThresholdModel(
        method, float(target_fpir), score_space, features, tuple(centers), tuple(scales),
        (float(q.min()), float(q.max())), tuple(np.quantile(q, knots)), (),
        ycenter, yscale, 0., {},
    )
    x = model.design(fit)
    y = (fit.score.to_numpy(dtype=float) - ycenter) / yscale
    quantile = 1 - target_fpir
    def objective(beta):
        residual = y - x @ beta
        loss = np.mean(smoothing * np.logaddexp(0., residual / smoothing)
                       + (quantile - 1) * residual) + ridge * (beta[1:] @ beta[1:]) / 2
        grad = -(x.T @ (expit(residual / smoothing) + quantile - 1)) / len(x)
        grad[1:] += ridge * beta[1:]
        return loss, grad
    initial = np.zeros(x.shape[1])
    initial[0] = np.quantile(y, quantile)
    bounds = [(None, None)] * len(initial)
    if "adc_margin" in features:
        j = features.index("adc_margin")
        bounds[1 + len(knots) + j] = (0., margin_slope_cap * scales[j] / yscale)
    # Tiny feature dimension: avoid oversubscribing BLAS for each optimizer step.
    with threadpool_limits(limits=1, user_api="blas"):
        result = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": int(max_iterations), "ftol": 1e-12, "gtol": 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"continuous fit did not converge: {result.message}")
    model = replace(model, coefficients=tuple(float(v) for v in result.x))
    return model, result


def _heldout_safety_offset(scores, safety_base, *, target_fpir):
    """One shared offset for the actual whole-cohort floating-point decisions."""
    residual = scores - safety_base
    offset = max(0., float(choose_non_mated_fpir_threshold(
        residual, np.zeros(len(residual), dtype=bool), target_fpir=target_fpir)))
    # Residual -> score addition can lose the nextafter used to reject tied maxima.
    # Check the actual floating-point decision and move conservatively if needed.
    adjustment = max(float(np.max(np.abs(np.spacing(safety_base)))), np.finfo(float).eps)
    for _ in range(8):
        if np.mean(scores >= safety_base + offset) <= target_fpir + 1e-15:
            break
        offset += adjustment
        adjustment *= 2
    else:
        raise RuntimeError("held-out safety rounding guard failed")
    return offset


def fit_continuous_threshold(
    calibration, *, target_fpir, features=("fiqa_score",),
    partition_seed=8972, safety_fraction=.3, knot_quantiles=(1/3, 2/3),
    smoothing=.01, ridge=.001, max_iterations=2000, margin_slope_cap=.95,
    score_space="negative_squared_l2_adc", method="continuous_fiqa",
):
    """Smooth pinball regression + one-sided held-out residual quantile guard.

    Quality uses two fixed fit-quantile hinge knots; added retrieval features
    enter linearly. Margin slope is constrained to [0, cap < 1] in raw score
    units so s1 - tau(q, s1-s2, d) increases with s1 at fixed q/s2/d.
    This does not guarantee FPIR after calibration-to-test distribution shift.
    """
    features = tuple(features)
    # Saliency covariates are used only by the separately gated 02 experiment.
    allowed = {"fiqa_score", "adc_margin", "adc_s2", "top1_gallery_pq_distortion",
               "outside_face_attention", "saliency_entropy"}
    if (not features or features[0] != "fiqa_score" or len(set(features)) != len(features)
            or not set(features) <= allowed):
        raise ValueError("explicit quality-first feature set required")
    if "adc_margin" in features and "adc_s2" in features:
        raise ValueError("margin and runner-up are separate ablations")
    if (not 0 < target_fpir < 1 or not 0 < safety_fraction < 1
            or not np.isfinite([smoothing, ridge, margin_slope_cap]).all()
            or smoothing <= 0 or ridge < 0 or not 0 <= margin_slope_cap < 1
            or max_iterations < 1):
        raise ValueError("invalid continuous calibration settings")
    knots = tuple(knot_quantiles)
    if knots != tuple(sorted(set(knots))) or any(not 0 < k < 1 for k in knots):
        raise ValueError("knot quantiles must be unique, increasing and inside (0,1)")
    rows = _validated_rows(calibration, score_column="score", quality_column="fiqa_score")
    if not np.isfinite(rows[list(features)].to_numpy(dtype=float)).all():
        raise ValueError("continuous features must be finite")
    partition = deterministic_calibration_partition(
        rows, safety_fraction=safety_fraction, seed=partition_seed, partition_column="identity_id")
    fit = rows.loc[partition.eq("fit") & ~rows.is_mated]
    safety = rows.loc[partition.eq("safety") & ~rows.is_mated]
    if len(fit) < 20 or len(safety) < 20:
        raise ValueError("at least 20 non-mated probes per fit/safety partition required")
    model, result = _fit_continuous_quantile(
        fit, features=features, target_fpir=target_fpir, method=method, score_space=score_space,
        knots=knots, smoothing=smoothing, ridge=ridge, max_iterations=max_iterations,
        margin_slope_cap=margin_slope_cap)
    offset = _heldout_safety_offset(safety.score.to_numpy(dtype=float), model.predict(safety),
                                    target_fpir=target_fpir)
    return replace(model, safety_offset=offset, settings={
        "partition_seed": partition_seed, "safety_fraction": safety_fraction,
        "fit_non_mated_count": len(fit), "safety_non_mated_count": len(safety),
        "knot_quantiles": list(knots), "smoothing": smoothing, "ridge": ridge,
        "max_iterations": max_iterations, "optimizer_iterations": int(result.nit),
        "optimizer_success": bool(result.success), "margin_slope_cap": margin_slope_cap,
        "optimizer_blas_threads": 1,
        "fit_objective": "smoothed_pinball_plus_ridge", "fit_features_source": "fit_non_mated_only",
        "safety_rule": "max(0,heldout_non_mated_residual_quantile)",
        "formal_fpir_guarantee": False, "threshold_fit_on_test": False,
    })


def apply_continuous_threshold(frame, model):
    """Reuse the canonical Rank-K/FPIR evaluator with a zero residual threshold."""
    thresholds = model.predict(frame)
    if not np.isfinite(thresholds).all():
        raise ValueError("non-finite continuous thresholds")
    # A constant template is used only to evaluate already-frozen thresholds.
    # No test distribution is used to fit the template or select a threshold.
    from research.calibration.conditional import ConditionalThresholdModel, ThresholdGroup
    group = ThresholdGroup("all", 0, 0, None, 0., 0., None, 0., False, None, None, None, None)
    zero = ConditionalThresholdModel(
        model.method, model.target_fpir, "score", None, ("all",), (), 0., None, 0.,
        (group,), {}, model.score_space)
    residual = frame.copy()
    residual["score"] = frame.score.to_numpy(dtype=float) - thresholds
    residual["true_identity_score"] = frame.true_identity_score.to_numpy(dtype=float) - thresholds
    evaluated = apply_threshold_model(residual, zero)
    evaluated.decisions["score"] = frame.score.to_numpy()
    evaluated.decisions["true_identity_score"] = frame.true_identity_score.to_numpy()
    evaluated.decisions["applied_threshold"] = thresholds
    evaluated.summary["model_uid"] = model.model_uid
    return replace(evaluated, model=model)
