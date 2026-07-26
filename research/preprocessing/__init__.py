"""Shared preprocessing used by model validation and quantitative experiments."""

from research.preprocessing.aligned_crops import (
    ALIGNMENT_TEMPLATE_ID,
    AlignmentResult,
    materialize_aligned_crops,
)

__all__ = [
    "ALIGNMENT_TEMPLATE_ID",
    "AlignmentResult",
    "materialize_aligned_crops",
]
