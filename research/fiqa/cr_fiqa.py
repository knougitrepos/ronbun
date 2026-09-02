"""Strict CR-FIQA inference adapter for the official CVPR 2023 checkpoints.

The upstream model returns an identity embedding and one raw quality logit.  The
quality value is deliberately left on the official raw scale: no sigmoid,
clipping, min-max normalization, or L2 normalization is applied.  Calibration
quantiles are invariant to this scale and avoid fitting a transform on test
data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from research.embeddings.pytorch._torch import require_torch
from research.runtime.hashing import sha256_file


@dataclass(frozen=True)
class CRFIQACheckpointSpec:
    variant: str
    architecture: str
    official_filename: str
    expected_sha256: str
    expected_bytes: int
    license_id: str = "CC-BY-NC-4.0"
    upstream_url: str = "https://github.com/fdbtrs/CR-FIQA"

    @property
    def model_uid(self) -> str:
        return f"cr-fiqa-{self.variant.lower()}-{self.expected_sha256[:20]}"


CRFIQA_VARIANTS: dict[str, CRFIQACheckpointSpec] = {
    "S": CRFIQACheckpointSpec(
        variant="S",
        architecture="iresnet50",
        official_filename="32572backbone.pth",
        expected_sha256=(
            "b9f457a6f00e0363a0cfb47ba4075e866d29e4e03c7989afe84faa61054dc8d4"
        ),
        expected_bytes=174_679_421,
    ),
    "L": CRFIQACheckpointSpec(
        variant="L",
        architecture="iresnet100",
        official_filename="181952backbone.pth",
        expected_sha256=(
            "5fca24736e4f8df5fbfc0f31ca533fbcca5eb119e2bad8fff751fe72c4c4d0fd"
        ),
        expected_bytes=261_219_071,
    ),
}


def _variant(value: str) -> CRFIQACheckpointSpec:
    normalized = str(value).strip().upper()
    try:
        return CRFIQA_VARIANTS[normalized]
    except KeyError as exc:
        raise ValueError("CR-FIQA variant must be 'S' or 'L'") from exc


class CRFIQABackbone:
    """Factory wrapper returning the exact upstream IResNet + quality head.

    ``__new__`` returns an ``nn.Module`` so importing this package does not make
    PyTorch mandatory for non-FIQA code paths.
    """

    def __new__(cls, variant: str) -> Any:
        torch = require_torch()
        from research.embeddings.pytorch.official_backbones import InsightIResNet

        spec = _variant(variant)
        layers = {
            "iresnet50": (3, 4, 14, 3),
            "iresnet100": (3, 13, 30, 3),
        }[spec.architecture]

        class _Model(InsightIResNet):
            def __init__(self) -> None:
                super().__init__(
                    layers,
                    embedding_dim=512,
                    eps=1e-5,
                    momentum=0.1,
                    dropout=0.0,
                    dropout_2d=False,
                    freeze_output_scale=False,
                )
                self.qs = torch.nn.Linear(512, 1)

            def forward(self, inputs: Any) -> tuple[Any, Any]:
                embedding = super().forward(inputs)
                return embedding, self.qs(embedding)

        return _Model()


def _load_payload(path: Path) -> Mapping[str, Any]:
    torch = require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("CR-FIQA checkpoint must be a non-empty state dict")
    if not all(isinstance(key, str) for key in payload):
        raise ValueError("CR-FIQA state-dict keys must be strings")
    return payload


def load_cr_fiqa(
    checkpoint_path: str | Path,
    *,
    variant: str = "S",
    device: str = "cuda",
    verify_official_hash: bool = True,
) -> tuple[Any, CRFIQACheckpointSpec]:
    """Strictly load one official pretrained CR-FIQA checkpoint."""

    torch = require_torch()
    spec = _variant(variant)
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CR-FIQA checkpoint is missing: {path}")
    if path.stat().st_size != spec.expected_bytes:
        raise ValueError(
            "CR-FIQA checkpoint byte size differs from the registered official "
            f"artifact: expected={spec.expected_bytes}, actual={path.stat().st_size}"
        )
    actual_sha256 = sha256_file(path)
    if verify_official_hash and actual_sha256 != spec.expected_sha256:
        raise ValueError(
            "CR-FIQA checkpoint SHA-256 differs from the locally registered "
            f"official artifact: {actual_sha256}"
        )
    requested_device = str(device).strip().lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CR-FIQA CUDA inference was requested but torch.cuda is unavailable; "
            "do not silently run the full experiment on CPU"
        )
    model = CRFIQABackbone(spec.variant)
    payload = _load_payload(path)
    model.load_state_dict(payload, strict=True)
    model.eval()
    model.to(requested_device)
    model.cr_fiqa_checkpoint_sha256 = actual_sha256
    model.cr_fiqa_model_uid = spec.model_uid
    return model, spec


def preprocess_cr_fiqa_rgb(images: np.ndarray) -> Any:
    """Convert uint8 NHWC RGB 112x112 faces to float32 NCHW in [-1, 1]."""

    torch = require_torch()
    values = np.asarray(images)
    if values.ndim != 4 or values.shape[1:] != (112, 112, 3):
        raise ValueError(
            "CR-FIQA inputs must have shape [N, 112, 112, 3] in RGB order"
        )
    if values.dtype != np.uint8:
        raise ValueError("CR-FIQA aligned inputs must use uint8 pixels")
    if len(values) == 0:
        raise ValueError("CR-FIQA input batch must not be empty")
    # Aligned bundles are commonly read-only memmaps. Copy before
    # ``torch.from_numpy`` so PyTorch never receives a non-writable buffer.
    contiguous = np.array(values, dtype=np.uint8, order="C", copy=True)
    tensor = torch.from_numpy(contiguous).permute(0, 3, 1, 2).float()
    return tensor.div_(127.5).sub_(1.0)


def infer_cr_fiqa_scores(
    model: Any,
    images: np.ndarray,
    *,
    batch_size: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    """Return one finite raw CR-FIQA score per aligned face."""

    torch = require_torch()
    batch_value = int(batch_size)
    if batch_value <= 0:
        raise ValueError("batch_size must be positive")
    values = np.asarray(images)
    if values.ndim != 4:
        raise ValueError("images must be a four-dimensional NHWC array")
    outputs: list[np.ndarray] = []
    requested_device = str(device).strip().lower()
    with torch.inference_mode():
        for start in range(0, len(values), batch_value):
            tensor = preprocess_cr_fiqa_rgb(values[start : start + batch_value])
            tensor = tensor.to(requested_device, non_blocking=True)
            _, raw_quality = model(tensor)
            batch_scores = raw_quality.detach().float().cpu().numpy().reshape(-1)
            outputs.append(batch_scores.astype(np.float32, copy=False))
    scores = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
    if len(scores) != len(values) or not np.isfinite(scores).all():
        raise RuntimeError("CR-FIQA inference returned missing or non-finite scores")
    return scores
