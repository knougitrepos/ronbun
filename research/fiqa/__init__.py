"""Pretrained face-image-quality inference and immutable score artifacts."""

from research.fiqa.artifacts import (
    FIQAScoreArtifact,
    infer_aligned_bundle_scores,
    load_fiqa_score_artifact,
    materialize_aligned_bundle_score_artifact,
    write_fiqa_score_artifact,
)
from research.fiqa.cr_fiqa import (
    CRFIQABackbone,
    CRFIQACheckpointSpec,
    CRFIQA_VARIANTS,
    infer_cr_fiqa_scores,
    load_cr_fiqa,
    preprocess_cr_fiqa_rgb,
)

__all__ = [
    "CRFIQABackbone",
    "CRFIQACheckpointSpec",
    "CRFIQA_VARIANTS",
    "FIQAScoreArtifact",
    "infer_aligned_bundle_scores",
    "infer_cr_fiqa_scores",
    "load_cr_fiqa",
    "load_fiqa_score_artifact",
    "materialize_aligned_bundle_score_artifact",
    "preprocess_cr_fiqa_rgb",
    "write_fiqa_score_artifact",
]
