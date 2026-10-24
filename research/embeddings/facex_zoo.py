"""Register completed FaceX-Zoo exports without changing the source files.

The supported format is produced by ``fine-tune/modules/checkpoint_manager.py``:
``{state_dict, metadata}`` plus ``<export-stem>_meta.json``. Registration does not
run training or certify the dataset split; the embedded protocol record remains
part of the hashed checkpoint provenance and must be audited separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from research.embeddings.base import CheckpointProvenance, ModelSpec, PreprocessingSpec
from research.embeddings.manifests import model_spec_registry_stem, write_model_spec
from research.embeddings.pytorch._torch import require_torch
from research.runtime.hashing import sha256_file

FACEX_REPOSITORY = "https://github.com/JDAI-CV/FaceX-Zoo"
FACEX_FACTORY = "research.embeddings.pytorch.facex_zoo:load_facex_mobilefacenet_checkpoint"
FACEX_HEADS = frozenset({"arcface", "cosface", "magface", "adaface"})


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"FaceX export {field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"FaceX export {field} must be a SHA-256 digest") from exc
    return value.lower()


def _config_hash(config: Mapping[str, Any]) -> str:
    # Match FaceX-Zoo compute_config_hash, including JSON whitespace.
    clean = {k: v for k, v in config.items() if k not in ("config_hash", "sha256")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _read_validated_export(export_path: str | Path) -> tuple[Path, dict, dict]:
    path = Path(export_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FaceX export does not exist: {path}")
    sidecar = path.with_name(path.stem + "_meta.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("FaceX export sidecar must contain a JSON object")
    if _sha256(metadata.get("sha256"), "sha256") != sha256_file(path):
        raise ValueError("FaceX export sha256 does not match its sidecar")
    if metadata.get("exported_file") != path.name:
        raise ValueError("FaceX export sidecar exported_file does not match filename")
    if metadata.get("completed") is not True or metadata.get("is_smoke") is not False:
        raise ValueError("FaceX registration requires a completed, explicitly non-smoke export")

    torch = require_torch()
    # FaceX exports contain tensors and JSON-compatible metadata only. Never
    # fall back to unrestricted pickle when given an incompatible checkpoint.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"state_dict", "metadata"}:
        raise ValueError("Expected a FaceX backbone export with state_dict and metadata")
    embedded = payload["metadata"]
    # The exporter adds these two fields after torch.save to avoid hashing a
    # checkpoint containing its own hash. No other sidecar-only changes pass.
    before_publication = {k: v for k, v in metadata.items() if k not in {"sha256", "exported_file"}}
    if not isinstance(embedded, dict) or embedded != before_publication:
        raise ValueError("FaceX embedded metadata and sidecar disagree")

    config = metadata.get("source_config")
    if not isinstance(config, dict):
        raise ValueError("FaceX export requires source_config provenance")
    if _sha256(metadata.get("source_config_hash"), "source_config_hash") != _config_hash(config):
        raise ValueError("FaceX source_config_hash does not match source_config")
    _sha256(metadata.get("source_sha256"), "source_sha256")
    _sha256(metadata.get("parent_hash"), "parent_hash")
    if not str(metadata.get("run_id", "")).strip() or not isinstance(metadata.get("protocol_record"), dict):
        raise ValueError("FaceX export requires run_id and protocol_record provenance")
    if not isinstance(metadata.get("epochs"), int) or isinstance(metadata["epochs"], bool) or metadata["epochs"] < 1:
        raise ValueError("FaceX export requires a positive completed epochs count")

    backbone = config.get("backbone", {})
    dimension = metadata.get("feat_dim")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension not in (512, 1024):
        raise ValueError("FaceX export feat_dim must be 512 or 1024")
    if backbone.get("type") not in ("MobileFaceNet", "Resnet", "ResNet") or backbone.get("feat_dim") != dimension:
        raise ValueError("FaceX source backbone must be MobileFaceNet/Resnet with matching feat_dim")
    if any(backbone.get(k, 7) != 7 for k in ("out_h", "out_w")):
        raise ValueError("FaceX MobileFaceNet importer requires a 7x7 output kernel")
    family = str(metadata.get("head", "")).lower()
    head_type = str(config.get("head", {}).get("type", "")).lower()
    head_type = {"am_softmax": "cosface"}.get(head_type, head_type)
    if family not in FACEX_HEADS or head_type != family:
        raise ValueError("FaceX export head and source_config head must name the same supported family")
    dataset = metadata.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip() or config.get("dataset", {}).get("name") != dataset:
        raise ValueError("FaceX export dataset and source_config dataset must agree")
    prep = metadata.get("preprocessing")
    config_prep = config.get("preprocessing")
    if not isinstance(prep, dict) or not isinstance(config_prep, dict) or prep != {**config_prep, "color": "BGR"}:
        raise ValueError("FaceX preprocessing metadata must agree with source_config")
    if (prep.get("color") != "BGR" or config_prep.get("color", "BGR") != "BGR"
            or prep.get("input_size") != [112, 112]
            or prep.get("pixel_mean") != 127.5 or prep.get("pixel_std") != 128.0):
        raise ValueError("Supported FaceX preprocessing is 112x112 BGR with (pixel - 127.5) / 128")
    weights = payload["state_dict"]
    if not isinstance(weights, Mapping) or not weights or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in weights.items()):
        raise ValueError("FaceX state_dict must contain named tensors only")
    linear_key = "linear.weight" if backbone["type"] == "MobileFaceNet" else "output_layer.3.weight"
    linear_width = 512 if backbone["type"] == "MobileFaceNet" else 512 * 7 * 7
    if linear_key not in weights or tuple(weights[linear_key].shape) != (dimension, linear_width):
        raise ValueError("FaceX state_dict output layer does not match declared feat_dim")
    if not all(bool(torch.isfinite(v).all()) for v in weights.values() if v.is_floating_point()):
        raise ValueError("FaceX state_dict contains non-finite weights")
    return path, metadata, dict(weights)


def _spec_from_metadata(path: Path, metadata: dict, source_color_order: str) -> ModelSpec:
    return ModelSpec(
        family=str(metadata["head"]).lower(),
        architecture=("MobileFaceNet" if metadata["source_config"]["backbone"]["type"] == "MobileFaceNet"
                      else f"FaceX-Resnet-{metadata['source_config']['backbone'].get('num_layers', 50)}-{metadata['source_config']['backbone'].get('mode', 'ir')}"),
        training_dataset=f"FaceX-Zoo fine-tune:{metadata['dataset']}; parent_sha256:{metadata['parent_hash']}",
        implementation_repository=FACEX_REPOSITORY,
        checkpoint=CheckpointProvenance(path=str(path), sha256=metadata["sha256"].lower(), source_url=path.as_uri(), verified_local_file=True),
        preprocessing=PreprocessingSpec(
            input_height=112, input_width=112,
            source_color_order=source_color_order, model_color_order="bgr",
            channel_mean=(127.5, 127.5, 127.5), channel_std=(128.0, 128.0, 128.0),
        ),
        target_layer=("conv_6_sep" if metadata["source_config"]["backbone"]["type"] == "MobileFaceNet" else "body"),
        embedding_dim=int(metadata["feat_dim"]), module_factory=FACEX_FACTORY,
    )


def model_spec_from_facex_export(export_path: str | Path, *, source_color_order: str = "rgb") -> ModelSpec:
    """Verify a formal export and create its checkpoint-specific ModelSpec.

    ``rgb`` is appropriate for PIL-decoded notebook inputs; use ``bgr`` for
    already BGR crops. This changes preprocessing identity and model UID.
    Crop/alignment policy belongs to the input artifact, not this tensor step.
    """
    from research.embeddings.pytorch.facex_zoo import build_export_backbone

    path, metadata, weights = _read_validated_export(export_path)
    spec = _spec_from_metadata(path, metadata, source_color_order)
    model = build_export_backbone(metadata)
    model.load_state_dict(weights, strict=True)
    return spec


def register_facex_export(export_path: str | Path, registry_root: str | Path, *, source_color_order: str = "rgb") -> Path:
    """Write an immutable model spec; preserve the original export and sidecar."""
    spec = model_spec_from_facex_export(export_path, source_color_order=source_color_order)
    return write_model_spec(Path(registry_root) / f"{model_spec_registry_stem(spec)}.json", spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_path", type=Path)
    parser.add_argument("--registry-root", required=True, type=Path)
    parser.add_argument("--source-color-order", choices=("rgb", "bgr"), default="rgb")
    args = parser.parse_args(argv)
    destination = register_facex_export(args.export_path, args.registry_root, source_color_order=args.source_color_order)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
