"""PyTorch training module and export utilities for face recognition baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from research.embeddings.base import ModelSpec
from research.embeddings.pytorch.official_backbones import (
    MobileFaceNetBackbone,
    build_mobilefacenet_backbone,
)
from research.embeddings.pytorch.official_loaders import (
    load_mobilefacenet_checkpoint,
)
from research.training.fr_baselines.classifiers import (
    AdaFaceClassifier,
    ArcFaceClassifier,
    MagFaceClassifier,
)


class FRBaselineTrainingModule(nn.Module):
    """End-to-end training wrapper pairing a backbone with a margin loss head."""

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        head_type: str = "arcface",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.head_type = head_type.lower()
        if self.head_type not in {"arcface", "adaface", "magface"}:
            raise ValueError(f"unsupported head_type: {head_type!r}")

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass over backbone and classifier head.

        Args:
            images: Tensor of shape (B, 3, 112, 112).
            labels: Ground-truth class indices of shape (B,). If None, no loss
                is computed.

        Returns:
            Dictionary with loss, logits, features, and auxiliary metrics.
        """
        features = self.backbone(images)
        if labels is None:
            if self.head_type == "magface":
                logits, _ = self.classifier(features)
            else:
                logits = self.classifier(features)
            return {"logits": logits, "features": features}

        if self.head_type == "magface":
            logits, aux_loss = self.classifier(features, labels)
            ce_loss = F.cross_entropy(logits, labels)
            total_loss = ce_loss + aux_loss
            return {
                "loss": total_loss,
                "ce_loss": ce_loss,
                "aux_loss": aux_loss,
                "logits": logits,
                "features": features,
            }

        logits = self.classifier(features, labels)
        total_loss = F.cross_entropy(logits, labels)
        return {
            "loss": total_loss,
            "logits": logits,
            "features": features,
        }


def build_fr_baseline_model(
    head_type: str,
    num_classes: int,
    *,
    architecture: str = "mobilefacenet",
    embedding_dim: int = 512,
    dropout: float = 0.0,
    head_kwargs: dict[str, Any] | None = None,
    pretrained_backbone: nn.Module | str | Path | None = None,
) -> FRBaselineTrainingModule:
    """Build a complete training module for the specified architecture and head.

    Args:
        head_type: Classification loss head ('arcface', 'adaface', 'magface').
        num_classes: Number of training classes / identities.
        architecture: Backbone architecture ('mobilefacenet').
        embedding_dim: Feature embedding dimension (512).
        dropout: Backbone dropout probability (default 0.0; 0.6 in UCFace fine-tuning).
        head_kwargs: Hyperparameters passed to the loss head.
        pretrained_backbone: Optional pre-trained backbone module or checkpoint file path
            to clone initial weights from for zero-leakage multi-head branching.
    """
    norm_head = head_type.lower()
    head_kwargs = dict(head_kwargs or {})

    if architecture == "mobilefacenet":
        backbone = build_mobilefacenet_backbone(
            architecture, embedding_dim=embedding_dim, dropout=dropout
        )
    else:
        raise ValueError(f"unsupported backbone architecture: {architecture!r}")

    if pretrained_backbone is not None:
        if isinstance(pretrained_backbone, nn.Module):
            backbone.load_state_dict(pretrained_backbone.state_dict(), strict=True)
        elif isinstance(pretrained_backbone, (str, Path)):
            ckpt_path = Path(pretrained_backbone).resolve()
            payload = torch.load(ckpt_path, map_location="cpu")
            if isinstance(payload, dict):
                sd = (
                    payload.get("state_dict_backbone")
                    or payload.get("backbone_state_dict")
                    or payload.get("state_dict")
                    or payload
                )
            else:
                sd = payload
            # Strip any 'backbone.' or 'module.' prefixes if present
            cleaned_sd = {}
            for k, v in sd.items():
                clean_k = k
                for prefix in ("backbone.", "module.backbone.", "module."):
                    if clean_k.startswith(prefix):
                        clean_k = clean_k[len(prefix) :]
                cleaned_sd[clean_k] = v
            backbone.load_state_dict(cleaned_sd, strict=True)
        else:
            raise TypeError(
                f"pretrained_backbone must be nn.Module, str, or Path, got {type(pretrained_backbone)}"
            )

    if norm_head == "arcface":
        classifier = ArcFaceClassifier(
            in_features=embedding_dim,
            num_classes=num_classes,
            scale=head_kwargs.get("scale", 64.0),
            margin=head_kwargs.get("margin", 0.6),
            easy_margin=head_kwargs.get("easy_margin", False),
        )
    elif norm_head == "adaface":
        classifier = AdaFaceClassifier(
            in_features=embedding_dim,
            num_classes=num_classes,
            scale=head_kwargs.get("scale", 64.0),
            margin=head_kwargs.get("margin", 0.4),
            h=head_kwargs.get("h", 0.333),
            t_alpha=head_kwargs.get("t_alpha", 0.01),
        )
    elif norm_head == "magface":
        classifier = MagFaceClassifier(
            in_features=embedding_dim,
            num_classes=num_classes,
            scale=head_kwargs.get("scale", 64.0),
            l_a=head_kwargs.get("l_a", 10.0),
            u_a=head_kwargs.get("u_a", 110.0),
            l_m=head_kwargs.get("l_m", 0.45),
            u_m=head_kwargs.get("u_m", 0.8),
            lambda_g=head_kwargs.get("lambda_g", 20.0),
        )
    else:
        raise ValueError(f"unsupported head_type: {head_type!r}")

    return FRBaselineTrainingModule(
        backbone=backbone, classifier=classifier, head_type=norm_head
    )


def export_backbone_checkpoint(
    module_or_backbone: nn.Module,
    output_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    verify_load: bool = True,
    model_color_order: str = "bgr",
) -> Path:
    """Strip classification head and save the pure backbone state dict.

    The saved payload is structured so that `load_mobilefacenet_checkpoint`
    and `load_arcface_checkpoint`/`load_adaface_checkpoint`/`load_magface_checkpoint`
    can strictly deserialize it.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(module_or_backbone, "backbone"):
        backbone = module_or_backbone.backbone
    else:
        backbone = module_or_backbone

    raw_state_dict = {k: v.cpu().clone() for k, v in backbone.state_dict().items()}
    payload: dict[str, Any] = {
        "state_dict": raw_state_dict,
        "state_dict_backbone": raw_state_dict,
        "metadata": metadata or {},
    }
    torch.save(payload, out_file)

    if verify_load:
        from research.embeddings.base import CheckpointProvenance, PreprocessingSpec

        spec = ModelSpec(
            family="arcface",
            architecture="mobilefacenet",
            training_dataset="training_export",
            implementation_repository="research.training.fr_baselines",
            checkpoint=CheckpointProvenance.from_file(
                out_file,
                source_url="local://trained",
            ),
            preprocessing=PreprocessingSpec(
                input_height=112,
                input_width=112,
                source_color_order="rgb",
                model_color_order=model_color_order,
                channel_mean=(127.5, 127.5, 127.5),
                channel_std=(128.0, 128.0, 128.0),
            ),
            target_layer="conv_sep.conv",
            embedding_dim=512,
        )
        loaded = load_mobilefacenet_checkpoint(spec)
        if not isinstance(loaded, MobileFaceNetBackbone):
            raise RuntimeError(
                f"exported checkpoint failed validation load: got {type(loaded)}"
            )

    return out_file
