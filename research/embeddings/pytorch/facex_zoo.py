"""Strict loader for FaceX-Zoo's completed backbone-only export format."""

from __future__ import annotations

from research.embeddings.base import ModelSpec
from research.embeddings.facex_zoo import _read_validated_export, _spec_from_metadata
from research.embeddings.pytorch.facex_backbone import FaceXMobileFaceNet


def build_export_backbone(metadata):
    config = metadata["source_config"]["backbone"]
    if config["type"] == "MobileFaceNet":
        return FaceXMobileFaceNet(metadata["feat_dim"])
    from research.embeddings.pytorch.facex_resnet import Resnet
    depth, mode = int(config.get("num_layers", 50)), config.get("mode", "ir")
    if depth not in (50, 100, 152) or mode not in ("ir", "ir_se"):
        raise ValueError("Unsupported FaceX Resnet depth/mode")
    return Resnet(depth, 0.0, mode=mode, feat_dim=metadata["feat_dim"], out_h=7, out_w=7)


def load_facex_mobilefacenet_checkpoint(spec: ModelSpec):
    """Load the raw backbone; training heads never normalize its output here."""
    spec.checkpoint.verify_local_file()
    path, metadata, weights = _read_validated_export(spec.checkpoint.path)
    expected = _spec_from_metadata(path, metadata, spec.preprocessing.source_color_order)
    if spec.model_uid != expected.model_uid:
        raise ValueError("FaceX model spec does not match export family, dimension, preprocessing, or provenance")
    model = build_export_backbone(metadata)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model
