"""Rejection calibration models."""

from research.calibration.gallery_size import (
    EffectiveGalleryRatioEstimate,
    GallerySizeThresholdResult,
    choose_effective_gallery_threshold,
    choose_sidak_gallery_threshold,
    estimate_effective_gallery_ratio,
    sidak_gallery_fpir,
    sidak_pair_false_match_rate,
)
from research.calibration.conditional import (
    ConditionalThresholdModel,
    ThresholdEvaluation,
    ThresholdGroup,
    apply_threshold_model,
    assign_quality_groups,
    deterministic_calibration_partition,
    fit_conditional_threshold,
    fit_global_threshold,
    paired_method_comparison,
)

__all__ = [
    "ConditionalThresholdModel",
    "EffectiveGalleryRatioEstimate",
    "GallerySizeThresholdResult",
    "ThresholdEvaluation",
    "ThresholdGroup",
    "apply_threshold_model",
    "assign_quality_groups",
    "choose_effective_gallery_threshold",
    "choose_sidak_gallery_threshold",
    "deterministic_calibration_partition",
    "estimate_effective_gallery_ratio",
    "fit_conditional_threshold",
    "fit_global_threshold",
    "paired_method_comparison",
    "sidak_gallery_fpir",
    "sidak_pair_false_match_rate",
]
