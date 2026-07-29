"""Shared preprocessing used by model validation and quantitative experiments."""

from research.preprocessing.aligned_crops import (
    ALIGNMENT_TEMPLATE_ID,
    DETECT_AND_ALIGN,
    OFFICIAL_FACE_CROP_RESIZE,
    AlignmentResult,
    materialize_aligned_crops,
    validate_aligned_crop_bundle,
)

__all__ = [
    "ALIGNMENT_TEMPLATE_ID",
    "DETECT_AND_ALIGN",
    "OFFICIAL_FACE_CROP_RESIZE",
    "AlignmentResult",
    "materialize_aligned_crops",
    "validate_aligned_crop_bundle",
]
