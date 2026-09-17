"""Full-cohort saliency/FIQA routing with one held-out safety correction."""
from dataclasses import asdict, dataclass, replace

import numpy as np

from research.calibration.conditional import _boolean_values, deterministic_calibration_partition
from research.calibration.continuous import (
    ContinuousThresholdModel, _fit_continuous_quantile, _heldout_safety_offset,
    fit_continuous_threshold,
)
from research.runtime.hashing import canonical_sha256

SALIENCY_FEATURES = ("outside_face_attention", "saliency_entropy")
FALLBACK_POLICY = "invalid-saliency-fiqa-only-joint-safety-v1"


def saliency_valid_mask(frame):
    """Only explicitly unavailable/invalid maps may fall back; corruption fails."""
    required = {"heatmap_available", "gradcam_valid_heatmap", *SALIENCY_FEATURES}
    if not required <= set(frame):
        raise ValueError("saliency routing columns missing: " + str(sorted(required - set(frame))))
    available = _boolean_values(frame.heatmap_available, column="heatmap_available")
    valid = _boolean_values(frame.gradcam_valid_heatmap, column="gradcam_valid_heatmap")
    values = frame[list(SALIENCY_FEATURES)].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("infinite saliency features are not a recorded missing-map case")
    finite = np.isfinite(values).all(axis=1)
    if (available & valid & ~finite).any():
        raise ValueError("valid heatmap has non-finite saliency features")
    return available & valid & finite


@dataclass(frozen=True)
class SaliencyFallbackModel:
    method: str
    target_fpir: float
    score_space: str
    features: tuple
    fiqa_model: ContinuousThresholdModel
    saliency_model: ContinuousThresholdModel | None
    safety_offset: float
    settings: dict

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        values["features"] = tuple(values["features"])
        values["fiqa_model"] = ContinuousThresholdModel(**values["fiqa_model"])
        if values["saliency_model"] is not None:
            values["saliency_model"] = ContinuousThresholdModel(**values["saliency_model"])
        return cls(**values)

    @property
    def model_uid(self):
        return "saliency-fallback-" + canonical_sha256(self.as_dict())[:24]

    def fallback_mask(self, frame):
        valid = saliency_valid_mask(frame)
        return ~valid if self.saliency_model is not None else np.ones(len(frame), dtype=bool)

    def predict(self, frame):
        fallback = self.fallback_mask(frame)
        values = self.fiqa_model.predict(frame)
        if (~fallback).any():
            values[~fallback] = self.saliency_model.predict(frame.loc[~fallback])
        # Child models have zero offsets: safety is calibrated on the mixture,
        # not separately tuned for the two routes (nor selected on test).
        return values + self.safety_offset


def fit_saliency_fallback(calibration, *, features, method, target_fpir,
                          partition_seed=8972, safety_fraction=.3,
                          knot_quantiles=(1/3, 2/3), smoothing=.01, ridge=.001,
                          max_iterations=2000, margin_slope_cap=.95):
    features = tuple(features)
    allowed = {"fiqa_score", "adc_margin", "top1_gallery_pq_distortion", *SALIENCY_FEATURES}
    if (not features or features[0] != "fiqa_score" or len(set(features)) != len(features)
            or not set(features) <= allowed or not set(features).intersection(SALIENCY_FEATURES)):
        raise ValueError("quality-first saliency ablation features required")
    # Public fitter validates common settings/cohort. Only its FIT coefficients
    # are reused; remove its standalone safety offset before routing.
    fiqa = fit_continuous_threshold(
        calibration, target_fpir=target_fpir, partition_seed=partition_seed,
        safety_fraction=safety_fraction, knot_quantiles=knot_quantiles,
        smoothing=smoothing, ridge=ridge, max_iterations=max_iterations,
        margin_slope_cap=margin_slope_cap)
    fiqa = replace(fiqa, safety_offset=0., settings={
        **fiqa.settings, "safety_rule": "disabled_child_joint_safety_in_parent"})
    valid = saliency_valid_mask(calibration)
    partition = deterministic_calibration_partition(calibration, seed=partition_seed,
                                                     safety_fraction=safety_fraction,
                                                     partition_column="identity_id")
    non_mated = ~_boolean_values(calibration.is_mated, column="is_mated")
    fit_mask = partition.eq("fit").to_numpy() & non_mated
    safety_mask = partition.eq("safety").to_numpy() & non_mated
    fit = calibration.loc[fit_mask & valid]
    saliency = None
    optimizer_iterations = None
    if len(fit) >= 20:
        saliency, optimizer = _fit_continuous_quantile(
            fit, features=features, target_fpir=target_fpir, method=method,
            score_space=fiqa.score_space, knots=tuple(knot_quantiles), smoothing=smoothing,
            ridge=ridge, max_iterations=max_iterations, margin_slope_cap=margin_slope_cap)
        optimizer_iterations = int(optimizer.nit)
    model = SaliencyFallbackModel(method, target_fpir, fiqa.score_space, features, fiqa, saliency, 0., {
        "fallback_policy": FALLBACK_POLICY, "fallback_method": "continuous_fiqa",
        "partition_seed": partition_seed, "safety_fraction": safety_fraction,
        "fit_non_mated_count": int(fit_mask.sum()), "fit_valid_non_mated_count": len(fit),
        "safety_non_mated_count": int(safety_mask.sum()),
        "safety_invalid_non_mated_count": int((safety_mask & ~valid).sum()),
        "saliency_branch_enabled": saliency is not None,
        "branch_disabled_reason": None if saliency is not None else "fewer_than_20_valid_fit_non_mated",
        "minimum_valid_fit_non_mated": 20, "optimizer_iterations": optimizer_iterations,
        "fit_features_source": "valid_fit_non_mated_only", "validity_features": list(SALIENCY_FEATURES),
        "safety_rule": "full_cohort_joint_non_mated_residual_quantile",
        "threshold_fit_on_test": False, "formal_fpir_guarantee": False,
    })
    safety = calibration.loc[safety_mask]
    offset = _heldout_safety_offset(safety.score.to_numpy(dtype=float), model.predict(safety),
                                    target_fpir=target_fpir)
    return replace(model, safety_offset=offset)
