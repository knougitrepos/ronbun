from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from research.embeddings.base import EmbeddingOutput, ModelSpec
from research.embeddings.pytorch._torch import require_torch
from research.embeddings.pytorch.target_layers import resolve_target_layer

OutputSelector = Callable[[Any], Any]


class PyTorchFaceEmbeddingAdapter:
    """Common inference boundary around a caller-supplied ``torch.nn.Module``.

    This class intentionally does not guess how an official repository builds
    its network or loads its checkpoint. Callers must inject the verified module
    (or use a separately registered factory) and a matching :class:`ModelSpec`.
    """

    family: str

    def __init__(
        self,
        module: Any,
        spec: ModelSpec,
        *,
        device: str = "cpu",
        output_selector: OutputSelector | None = None,
    ) -> None:
        torch = require_torch()
        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be an instance of torch.nn.Module")
        if spec.family != self.family:
            raise ValueError(
                f"{type(self).__name__} requires family={self.family!r}, "
                f"got {spec.family!r}"
            )
        spec.checkpoint.verify_local_file()
        self.model = module
        self.spec = spec
        self.device = torch.device(device)
        self.output_selector = output_selector
        self.model.to(self.device)
        self.model.eval()

    @property
    def target_layer(self) -> Any:
        return resolve_target_layer(self.model, self.spec.target_layer)

    def _prepare_aligned_faces(self, aligned_faces: np.ndarray) -> np.ndarray:
        faces = np.asarray(aligned_faces)
        if faces.ndim == 3:
            faces = faces[np.newaxis, ...]
        preprocessing = self.spec.preprocessing
        if faces.ndim != 4:
            raise ValueError(
                "aligned_faces must have shape "
                f"(N, {preprocessing.input_height}, "
                f"{preprocessing.input_width}, 3), got {faces.shape}"
            )
        expected = (
            faces.shape[0],
            preprocessing.input_height,
            preprocessing.input_width,
            3,
        )
        if faces.shape != expected or faces.shape[0] == 0:
            raise ValueError(
                "aligned_faces must have shape "
                f"(N, {preprocessing.input_height}, "
                f"{preprocessing.input_width}, 3), got {faces.shape}"
            )
        if faces.dtype != np.uint8:
            raise ValueError("aligned_faces must use the frozen uint8 crop format")

        values = faces.astype(np.float32, copy=True)
        if preprocessing.source_color_order != preprocessing.model_color_order:
            values = values[..., ::-1].copy()
        mean = np.asarray(preprocessing.channel_mean, dtype=np.float32)
        std = np.asarray(preprocessing.channel_std, dtype=np.float32)
        values = (values - mean) / std
        return np.transpose(values, (0, 3, 1, 2)).copy()

    def preprocess(self, aligned_faces: np.ndarray) -> Any:
        torch = require_torch()
        channels_first = self._prepare_aligned_faces(aligned_faces)
        return torch.from_numpy(channels_first).to(self.device)

    def select_embedding_tensor(self, model_output: Any) -> Any:
        """Select the raw embedding tensor from a repository-specific output.

        Grad-CAM uses this same public selector so inference and explanation
        cannot silently choose different tensors from tuple/dict outputs.
        """

        torch = require_torch()
        if self.output_selector is not None:
            selected = self.output_selector(model_output)
        elif torch.is_tensor(model_output):
            selected = model_output
        elif isinstance(model_output, Mapping):
            candidates = [
                model_output[key]
                for key in ("embedding", "embeddings", "feature", "features")
                if key in model_output and torch.is_tensor(model_output[key])
            ]
            if len(candidates) != 1:
                raise TypeError(
                    "mapping model output must expose exactly one recognized "
                    "embedding tensor or use output_selector"
                )
            selected = candidates[0]
        elif isinstance(model_output, Sequence) and not isinstance(
            model_output, (str, bytes)
        ):
            candidates = [value for value in model_output if torch.is_tensor(value)]
            candidates = [
                value
                for value in candidates
                if getattr(value, "ndim", 0) == 2
                and value.shape[-1] == self.spec.embedding_dim
            ]
            if len(candidates) != 1:
                raise TypeError(
                    "sequence model output must contain exactly one (N, 512) "
                    "tensor or use output_selector"
                )
            selected = candidates[0]
        else:
            raise TypeError(
                "model output must be a tensor or expose an explicit embedding tensor"
            )
        if not torch.is_tensor(selected):
            raise TypeError("output_selector must return a torch.Tensor")
        return selected

    def forward_raw_tensor(
        self,
        aligned_faces: np.ndarray,
        *,
        require_grad: bool = False,
    ) -> Any:
        """Return the differentiable raw tensor when Grad-CAM requests it."""

        torch = require_torch()
        inputs = self.preprocess(aligned_faces)
        self.model.eval()
        context = torch.enable_grad() if require_grad else torch.inference_mode()
        with context:
            raw = self.select_embedding_tensor(self.model(inputs))
            if raw.ndim != 2 or raw.shape[-1] != self.spec.embedding_dim:
                raise ValueError(
                    "FR model output must have shape "
                    f"(N, {self.spec.embedding_dim}), got {tuple(raw.shape)}"
                )
            if raw.shape[0] != inputs.shape[0]:
                raise ValueError(
                    "FR model output batch size does not match aligned_faces"
                )
        return raw

    def embed(self, aligned_faces: np.ndarray) -> EmbeddingOutput:
        """Run eval/inference mode and retain raw norm before L2 normalization."""

        raw_tensor = self.forward_raw_tensor(aligned_faces, require_grad=False)
        raw = (
            raw_tensor.detach()
            .to("cpu")
            .float()
            .numpy()
            .astype(np.float32, copy=False)
        )
        if not np.isfinite(raw).all():
            raise ValueError("FR model returned a non-finite embedding")
        norms = np.linalg.norm(raw, axis=1).astype(np.float32)
        if not np.isfinite(norms).all() or np.any(norms <= 0.0):
            raise ValueError("FR model returned a zero or non-finite embedding")
        normalized = (raw / norms[:, np.newaxis]).astype(np.float32)
        return EmbeddingOutput(
            raw_embedding=raw,
            raw_norm=norms,
            normalized_embedding=normalized,
            model_uid=self.spec.model_uid,
            checkpoint_sha256=self.spec.checkpoint.sha256,
            preprocess_hash=self.spec.preprocessing.preprocess_hash,
        )
