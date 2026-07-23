from __future__ import annotations

import importlib
from typing import Any


class PyTorchUnavailableError(RuntimeError):
    """Raised only when a PyTorch-backed operation is requested."""


def require_torch() -> Any:
    """Import PyTorch lazily so the existing ONNX path stays importable."""

    try:
        return importlib.import_module("torch")
    except (ImportError, ModuleNotFoundError) as exc:
        raise PyTorchUnavailableError(
            "Step 2 PyTorch inference requires the optional 'torch' package. "
            "The existing InsightFace/ONNX ArcFaceFeatureExtractor remains available."
        ) from exc
