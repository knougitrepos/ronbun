"""Face Recognition baseline training package."""

from research.training.fr_baselines.classifiers import (
    AdaFaceClassifier,
    ArcFaceClassifier,
    MagFaceClassifier,
)
from research.training.fr_baselines.models import (
    FRBaselineTrainingModule,
    build_fr_baseline_model,
    export_backbone_checkpoint,
)
from research.training.fr_baselines.protocol import (
    FRDataset,
    SurvFaceFinetuneBundle,
    build_survface_finetune_manifest,
)
from research.training.fr_baselines.trainer import (
    FRBaselineTrainer,
)

__all__ = [
    "AdaFaceClassifier",
    "ArcFaceClassifier",
    "FRBaselineTrainer",
    "FRBaselineTrainingModule",
    "FRDataset",
    "MagFaceClassifier",
    "SurvFaceFinetuneBundle",
    "build_fr_baseline_model",
    "build_survface_finetune_manifest",
    "export_backbone_checkpoint",
]
