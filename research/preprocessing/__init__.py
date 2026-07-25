"""Preprocessing utilities for face recognition experiments.

This package provides dataset-agnostic aligned crop materializers that
produce the common 112×112 RGB uint8 NHWC arrays required by ArcFace,
AdaFace, and MagFace inference backbones.
"""

from research.preprocessing.aligned_crops import (
    AlignmentResult,
    materialize_aligned_crops,
)

__all__ = [
    "AlignmentResult",
    "materialize_aligned_crops",
]
