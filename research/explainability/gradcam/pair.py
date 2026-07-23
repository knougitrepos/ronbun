from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from research.explainability.gradcam.optional import require_torch


BatchMode = Literal["single", "independent"]


@dataclass(frozen=True)
class PairGradCAMResult:
    """Pair-conditioned Grad-CAM output at the selected feature-map resolution."""

    heatmaps: np.ndarray
    target_scores: np.ndarray
    target_space: str
    batch_mode: BatchMode


def pair_cosine_target(
    query_embeddings: Any,
    gallery_templates: Any,
) -> Any:
    """Return one cosine target per query against a detached gallery template.

    Both arguments must be floating point embedding tensors.  Integer PQ codes
    are deliberately rejected: hard PQ is not part of the differentiable
    target graph.  The gallery side is detached inside this function even when
    the caller supplied a tensor with ``requires_grad=True``.
    """

    torch = require_torch()
    if not torch.is_tensor(query_embeddings):
        raise TypeError("query_embeddings must be a PyTorch tensor")
    if not torch.is_tensor(gallery_templates):
        raise TypeError("gallery_templates must be a PyTorch tensor")
    if not torch.is_floating_point(query_embeddings):
        raise TypeError("query_embeddings must use a floating point dtype")
    if not torch.is_floating_point(gallery_templates):
        raise TypeError(
            "gallery_templates must be floating embeddings, not hard PQ codes"
        )
    if query_embeddings.ndim != 2:
        raise ValueError("query_embeddings must have shape [batch, dimension]")

    gallery = gallery_templates.detach()
    if gallery.ndim == 1:
        gallery = gallery.unsqueeze(0)
    if gallery.ndim != 2:
        raise ValueError(
            "gallery_templates must have shape [dimension] or [batch, dimension]"
        )
    if gallery.shape[0] == 1 and query_embeddings.shape[0] > 1:
        gallery = gallery.expand(query_embeddings.shape[0], -1)
    if gallery.shape != query_embeddings.shape:
        raise ValueError(
            "gallery_templates must match the query batch and embedding dimension"
        )

    return torch.nn.functional.cosine_similarity(
        query_embeddings,
        gallery,
        dim=1,
        eps=1e-12,
    )


class PairCosineGradCAM:
    """Grad-CAM for a frozen query encoder and an origin gallery template.

    The selected layer must produce one ``[B, C, H, W]`` feature map per
    forward pass.  ``batch_mode='single'`` requires exactly one query.
    ``batch_mode='independent'`` treats each query/template row as an
    independent pair and differentiates the sum of their scalar cosine
    targets.

    This class intentionally contains no PCA/PQ operation.  It explains the
    original FR encoder response, which can later be associated with the
    compression sensitivity measured by Step 1/Step 2 artifacts.
    """

    def __init__(
        self,
        model: Any,
        target_layer: Any,
        *,
        embedding_extractor: Callable[[Any], Any] | None = None,
        require_eval_mode: bool = True,
    ) -> None:
        self._torch = require_torch()
        if not callable(model):
            raise TypeError("model must be callable")
        if not hasattr(target_layer, "register_forward_hook"):
            raise TypeError("target_layer must support register_forward_hook")
        self.model = model
        self.target_layer = target_layer
        self.embedding_extractor = embedding_extractor
        self.require_eval_mode = bool(require_eval_mode)

    def _extract_embedding(self, model_output: Any) -> Any:
        embedding = (
            self.embedding_extractor(model_output)
            if self.embedding_extractor is not None
            else model_output
        )
        if not self._torch.is_tensor(embedding):
            raise TypeError(
                "model output must be an embedding tensor or embedding_extractor "
                "must select one"
            )
        if embedding.ndim != 2:
            raise ValueError("model embeddings must have shape [batch, dimension]")
        if not self._torch.is_floating_point(embedding):
            raise TypeError("model embeddings must use a floating point dtype")
        return embedding

    def generate(
        self,
        query_images: Any,
        gallery_templates: Any,
        *,
        batch_mode: BatchMode,
        target_space: str = "origin_embedding",
    ) -> PairGradCAMResult:
        """Generate normalized ReLU Grad-CAM maps for query/template pairs."""

        torch = self._torch
        if batch_mode not in {"single", "independent"}:
            raise ValueError("batch_mode must be 'single' or 'independent'")
        if target_space != "origin_embedding":
            raise ValueError(
                "target_space must be 'origin_embedding'; compressed/PQ targets "
                "are outside this Grad-CAM analysis boundary"
            )
        if not torch.is_tensor(query_images):
            raise TypeError("query_images must be a PyTorch tensor")
        if not torch.is_floating_point(query_images):
            raise TypeError("query_images must use a floating point dtype")
        if query_images.ndim != 4 or query_images.shape[0] == 0:
            raise ValueError("query_images must have shape [batch, channels, H, W]")
        if batch_mode == "single" and query_images.shape[0] != 1:
            raise ValueError("batch_mode='single' requires exactly one query image")
        if self.require_eval_mode and bool(getattr(self.model, "training", False)):
            raise RuntimeError(
                "model must be in eval mode so batch rows remain independent"
            )

        captured: list[Any] = []

        def capture_activation(_module: Any, _inputs: Any, output: Any) -> None:
            captured.append(output)

        handle = self.target_layer.register_forward_hook(capture_activation)
        try:
            # Input gradients keep Grad-CAM available even when every model
            # parameter is frozen.  The caller's tensor is not mutated.
            query = query_images.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                embeddings = self._extract_embedding(self.model(query))
                if embeddings.shape[0] != query.shape[0]:
                    raise ValueError(
                        "model embedding batch must match the query image batch"
                    )
                if len(captured) != 1:
                    raise RuntimeError(
                        "target_layer must run exactly once per model forward pass"
                    )
                activations = captured[0]
                if not torch.is_tensor(activations) or activations.ndim != 4:
                    raise ValueError(
                        "target_layer output must be one [B, C, H, W] tensor"
                    )
                if activations.shape[0] != query.shape[0]:
                    raise ValueError(
                        "target-layer activation batch must match the query batch"
                    )
                scores = pair_cosine_target(embeddings, gallery_templates)
                gradients = torch.autograd.grad(
                    scores.sum(),
                    activations,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]

                weights = gradients.mean(dim=(2, 3), keepdim=True)
                heatmaps = torch.relu((weights * activations).sum(dim=1))
                flat = heatmaps.flatten(start_dim=1)
                minimum = flat.amin(dim=1, keepdim=True)
                shifted = flat - minimum
                maximum = shifted.amax(dim=1, keepdim=True)
                normalized = torch.where(
                    maximum > 1e-12,
                    shifted / maximum.clamp_min(1e-12),
                    torch.zeros_like(shifted),
                ).reshape_as(heatmaps)

            return PairGradCAMResult(
                heatmaps=normalized.detach().cpu().numpy().astype(
                    np.float32,
                    copy=False,
                ),
                target_scores=scores.detach().cpu().numpy().astype(
                    np.float32,
                    copy=False,
                ),
                target_space=target_space,
                batch_mode=batch_mode,
            )
        finally:
            # Hooks never survive success or any validation/autograd failure.
            handle.remove()

