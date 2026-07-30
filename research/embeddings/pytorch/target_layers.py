from __future__ import annotations

from typing import Any


def resolve_target_layer(module: Any, layer_path: str) -> Any:
    """Resolve an exact ``named_modules`` path for later Grad-CAM hooks."""

    resolved_path = str(layer_path).strip()
    if not resolved_path:
        raise ValueError("Grad-CAM target layer path must be non-empty")
    named_modules = getattr(module, "named_modules", None)
    if not callable(named_modules):
        raise TypeError("injected model must provide nn.Module.named_modules()")
    available = dict(named_modules())
    if resolved_path not in available:
        examples = sorted(name for name in available if name)[:10]
        suffix = f"; available examples={examples}" if examples else ""
        raise ValueError(
            f"Grad-CAM target layer '{resolved_path}' was not found{suffix}"
        )
    return available[resolved_path]
