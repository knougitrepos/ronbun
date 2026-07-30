from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from research.runtime.hashing import canonical_sha256, sha256_file

FRModelFamily = Literal["arcface", "adaface", "magface"]
ColorOrder = Literal["rgb", "bgr"]


def _require_text(value: str, name: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{name} must be non-empty")
    return resolved


def _validate_sha256(value: str) -> str:
    resolved = str(value).lower().strip()
    if len(resolved) != hashlib.sha256().digest_size * 2:
        raise ValueError("checkpoint sha256 must contain 64 hexadecimal characters")
    try:
        int(resolved, 16)
    except ValueError as exc:
        raise ValueError(
            "checkpoint sha256 must contain 64 hexadecimal characters"
        ) from exc
    return resolved


@dataclass(frozen=True)
class CheckpointProvenance:
    """Immutable identity of a locally verified model checkpoint.

    ``source_url`` records where the file was obtained; it does not claim that
    the repository is official or that the checkpoint reproduces a paper unless
    the caller has independently verified those facts.
    """

    path: str
    sha256: str
    source_url: str
    verified_local_file: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _require_text(self.path, "checkpoint path"))
        object.__setattr__(self, "sha256", _validate_sha256(self.sha256))
        object.__setattr__(
            self,
            "source_url",
            _require_text(self.source_url, "checkpoint source_url"),
        )
        if not isinstance(self.verified_local_file, bool):
            raise TypeError("verified_local_file must be a boolean")

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        source_url: str,
    ) -> CheckpointProvenance:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"checkpoint file does not exist: {resolved}")
        return cls(
            path=str(resolved),
            sha256=sha256_file(resolved),
            source_url=source_url,
            verified_local_file=True,
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "source_url": self.source_url,
            "verified_local_file": self.verified_local_file,
        }

    @classmethod
    def from_manifest(
        cls,
        payload: Mapping[str, Any],
        *,
        verify_local_file: bool = True,
    ) -> CheckpointProvenance:
        path = Path(str(payload["path"])).expanduser().resolve()
        declared_sha256 = _validate_sha256(str(payload["sha256"]))
        if verify_local_file:
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint file does not exist: {path}")
            actual_sha256 = sha256_file(path)
            if actual_sha256 != declared_sha256:
                raise ValueError(
                    "checkpoint sha256 no longer matches the registered local file"
                )
        return cls(
            path=str(path),
            sha256=declared_sha256,
            source_url=str(payload["source_url"]),
            verified_local_file=bool(verify_local_file),
        )

    def verify_local_file(self) -> None:
        if not self.verified_local_file:
            raise ValueError(
                "checkpoint provenance was not verified from a local file"
            )
        resolved = Path(self.path)
        if not resolved.is_file():
            raise FileNotFoundError(f"checkpoint file does not exist: {resolved}")
        actual = sha256_file(resolved)
        if actual != self.sha256:
            raise ValueError(
                "checkpoint file hash no longer matches recorded provenance: "
                f"{resolved}"
            )


@dataclass(frozen=True)
class PreprocessingSpec:
    """Explicit aligned-crop preprocessing owned by an FR checkpoint."""

    input_height: int
    input_width: int
    source_color_order: ColorOrder
    model_color_order: ColorOrder
    channel_mean: tuple[float, float, float]
    channel_std: tuple[float, float, float]
    input_dtype: str = "uint8"

    def __post_init__(self) -> None:
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("preprocessing input dimensions must be positive")
        if self.source_color_order not in {"rgb", "bgr"}:
            raise ValueError("source_color_order must be 'rgb' or 'bgr'")
        if self.model_color_order not in {"rgb", "bgr"}:
            raise ValueError("model_color_order must be 'rgb' or 'bgr'")
        if len(self.channel_mean) != 3 or len(self.channel_std) != 3:
            raise ValueError("channel_mean and channel_std must contain three values")
        if not np.isfinite(np.asarray(self.channel_mean, dtype=np.float64)).all():
            raise ValueError("channel_mean must be finite")
        std = np.asarray(self.channel_std, dtype=np.float64)
        if not np.isfinite(std).all() or np.any(std <= 0.0):
            raise ValueError("channel_std must contain finite positive values")
        if self.input_dtype != "uint8":
            raise ValueError(
                "Step 2 currently accepts frozen uint8 aligned crops only"
            )

    @property
    def preprocess_hash(self) -> str:
        return canonical_sha256(self.to_manifest())

    def to_manifest(self) -> dict[str, Any]:
        return {
            "input_height": self.input_height,
            "input_width": self.input_width,
            "source_color_order": self.source_color_order,
            "model_color_order": self.model_color_order,
            "channel_mean": list(self.channel_mean),
            "channel_std": list(self.channel_std),
            "input_dtype": self.input_dtype,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> PreprocessingSpec:
        return cls(
            input_height=int(payload["input_height"]),
            input_width=int(payload["input_width"]),
            source_color_order=str(payload["source_color_order"]),
            model_color_order=str(payload["model_color_order"]),
            channel_mean=tuple(float(value) for value in payload["channel_mean"]),
            channel_std=tuple(float(value) for value in payload["channel_std"]),
            input_dtype=str(payload["input_dtype"]),
        )


@dataclass(frozen=True)
class ModelSpec:
    """Checkpoint-level FR model identity used by Step 2.

    The fields describe a selected checkpoint, not a controlled causal
    comparison of the ArcFace/AdaFace/MagFace loss functions.
    """

    family: FRModelFamily
    architecture: str
    training_dataset: str
    implementation_repository: str
    checkpoint: CheckpointProvenance
    preprocessing: PreprocessingSpec
    target_layer: str
    embedding_dim: int = 512
    module_factory: str | None = None

    def __post_init__(self) -> None:
        if self.family not in {"arcface", "adaface", "magface"}:
            raise ValueError("family must be arcface, adaface, or magface")
        for value, name in (
            (self.architecture, "architecture"),
            (self.training_dataset, "training_dataset"),
            (self.implementation_repository, "implementation_repository"),
            (self.target_layer, "target_layer"),
        ):
            _require_text(value, name)
        if self.module_factory is not None:
            _require_text(self.module_factory, "module_factory")
        if self.embedding_dim != 512:
            raise ValueError("Step 2 FR model checkpoints must output 512D embeddings")

    @property
    def model_uid(self) -> str:
        digest = canonical_sha256(
            {
                "family": self.family,
                "architecture": self.architecture,
                "training_dataset": self.training_dataset,
                "implementation_repository": self.implementation_repository,
                "checkpoint_sha256": self.checkpoint.sha256,
                "preprocess_hash": self.preprocessing.preprocess_hash,
                "embedding_dim": self.embedding_dim,
                "module_factory": self.module_factory,
            }
        )
        return f"{self.family}-{digest[:20]}"

    def to_manifest(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "architecture": self.architecture,
            "training_dataset": self.training_dataset,
            "implementation_repository": self.implementation_repository,
            "checkpoint": self.checkpoint.to_manifest(),
            "preprocessing": self.preprocessing.to_manifest(),
            "preprocess_hash": self.preprocessing.preprocess_hash,
            "target_layer": self.target_layer,
            "embedding_dim": self.embedding_dim,
            "module_factory": self.module_factory,
            "model_uid": self.model_uid,
            "comparison_scope": "pretrained_checkpoint",
        }

    @classmethod
    def from_manifest(
        cls,
        payload: Mapping[str, Any],
        *,
        verify_checkpoint: bool = True,
    ) -> ModelSpec:
        if payload.get("comparison_scope") not in {None, "pretrained_checkpoint"}:
            raise ValueError("model manifest comparison_scope must be pretrained_checkpoint")
        spec = cls(
            family=str(payload["family"]),
            architecture=str(payload["architecture"]),
            training_dataset=str(payload["training_dataset"]),
            implementation_repository=str(payload["implementation_repository"]),
            checkpoint=CheckpointProvenance.from_manifest(
                payload["checkpoint"],
                verify_local_file=verify_checkpoint,
            ),
            preprocessing=PreprocessingSpec.from_manifest(
                payload["preprocessing"]
            ),
            target_layer=str(payload["target_layer"]),
            embedding_dim=int(payload["embedding_dim"]),
            module_factory=(
                None
                if payload.get("module_factory") is None
                else str(payload["module_factory"])
            ),
        )
        declared_preprocess_hash = payload.get("preprocess_hash")
        if (
            declared_preprocess_hash is not None
            and str(declared_preprocess_hash) != spec.preprocessing.preprocess_hash
        ):
            raise ValueError("preprocess_hash does not match preprocessing metadata")
        declared_model_uid = payload.get("model_uid")
        if declared_model_uid is not None and str(declared_model_uid) != spec.model_uid:
            raise ValueError("model_uid does not match model provenance metadata")
        return spec


@dataclass(frozen=True)
class EmbeddingOutput:
    """Raw and normalized embeddings produced without updating model weights."""

    raw_embedding: np.ndarray
    raw_norm: np.ndarray
    normalized_embedding: np.ndarray
    model_uid: str
    checkpoint_sha256: str
    preprocess_hash: str

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw_embedding, dtype=np.float32)
        norm = np.asarray(self.raw_norm, dtype=np.float32)
        normalized = np.asarray(self.normalized_embedding, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 512:
            raise ValueError(
                f"raw_embedding must have shape (N, 512), got {raw.shape}"
            )
        if raw.shape[0] == 0:
            raise ValueError("embedding output must contain at least one row")
        if norm.shape != (raw.shape[0],):
            raise ValueError(
                f"raw_norm must have shape ({raw.shape[0]},), got {norm.shape}"
            )
        if normalized.shape != raw.shape:
            raise ValueError(
                "normalized_embedding shape must match raw_embedding shape"
            )
        if not np.isfinite(raw).all() or not np.isfinite(norm).all():
            raise ValueError("raw embeddings and norms must be finite")
        if np.any(norm <= 0.0) or not np.isfinite(normalized).all():
            raise ValueError("embedding norms must be positive and normalized values finite")
        measured_norm = np.linalg.norm(normalized, axis=1)
        if not np.allclose(measured_norm, 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("normalized_embedding rows must have L2 norm 1")

        object.__setattr__(self, "raw_embedding", raw)
        object.__setattr__(self, "raw_norm", norm)
        object.__setattr__(self, "normalized_embedding", normalized)
        object.__setattr__(self, "model_uid", _require_text(self.model_uid, "model_uid"))
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _validate_sha256(self.checkpoint_sha256),
        )
        object.__setattr__(
            self,
            "preprocess_hash",
            _validate_sha256(self.preprocess_hash),
        )
