from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType


class TorchUnavailableError(RuntimeError):
    """Raised when an optional PyTorch-only analysis is requested."""


def is_torch_available() -> bool:
    """Return whether PyTorch can be imported without importing it eagerly."""

    return importlib.util.find_spec("torch") is not None


def require_torch() -> ModuleType:
    """Import PyTorch or raise an actionable optional-dependency error."""

    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise TorchUnavailableError(
            "Pair-conditioned Grad-CAM requires the optional PyTorch "
            "environment; install the Step 2 requirements before running it"
        ) from exc

