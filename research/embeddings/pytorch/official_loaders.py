"""Strict checkpoint factories for the selected official FR implementations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from research.embeddings.base import ModelSpec
from research.embeddings.pytorch._torch import require_torch


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot exactly populate its declared backbone."""


def _torch_load(path: str) -> Any:
    torch = require_torch()
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def _mapping(value: Any, *, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError(
            f"{description} must be a mapping of state-dict keys to tensors"
        )
    result = {str(key): tensor for key, tensor in value.items()}
    if not result:
        raise CheckpointCompatibilityError(f"{description} is empty")
    return result


def _state_dict_candidates(
    payload: Any,
    *,
    nested_keys: tuple[str, ...],
) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    if isinstance(payload, Mapping):
        for key in nested_keys:
            if key in payload and isinstance(payload[key], Mapping):
                candidates.append(
                    (key, _mapping(payload[key], description=f"checkpoint[{key!r}]"))
                )
        if all(isinstance(key, str) for key in payload):
            raw = _mapping(payload, description="checkpoint")
            if any("." in key for key in raw):
                candidates.append(("checkpoint", raw))
    if not candidates:
        raise CheckpointCompatibilityError(
            "checkpoint does not contain a recognized state dict"
        )
    return candidates


def _strip_prefix(
    state_dict: Mapping[str, Any], prefix: str
) -> dict[str, Any] | None:
    keys = tuple(state_dict)
    if not keys or not all(key.startswith(prefix) for key in keys):
        return None
    return {key[len(prefix) :]: value for key, value in state_dict.items()}


def _select_prefix(
    state_dict: Mapping[str, Any], prefix: str
) -> dict[str, Any] | None:
    selected = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    return selected or None


def _exact_state_dict(
    model: Any,
    payload: Any,
    *,
    nested_keys: tuple[str, ...],
    prefixes: tuple[str, ...],
    select_prefixes: tuple[str, ...],
    family: str,
) -> dict[str, Any]:
    expected = model.state_dict()
    expected_keys = set(expected)
    attempts: list[str] = []
    for container_name, candidate in _state_dict_candidates(
        payload, nested_keys=nested_keys
    ):
        transformed: list[tuple[str, dict[str, Any]]] = [("identity", candidate)]
        for prefix in prefixes:
            stripped = _strip_prefix(candidate, prefix)
            if stripped is not None:
                transformed.append((f"strip:{prefix}", stripped))
        for prefix in select_prefixes:
            selected = _select_prefix(candidate, prefix)
            if selected is not None:
                transformed.append((f"select:{prefix}", selected))
        for transform_name, state_dict in transformed:
            keys = set(state_dict)
            missing = expected_keys - keys
            unexpected = keys - expected_keys
            mismatched = {
                key
                for key in expected_keys & keys
                if getattr(state_dict[key], "shape", None)
                != getattr(expected[key], "shape", None)
            }
            if not missing and not unexpected and not mismatched:
                return state_dict
            attempts.append(
                f"{container_name}/{transform_name}: "
                f"missing={len(missing)}, unexpected={len(unexpected)}, "
                f"shape_mismatch={len(mismatched)}"
            )
    detail = "; ".join(attempts[:12])
    raise CheckpointCompatibilityError(
        f"{family} checkpoint is not an exact match for the declared "
        f"architecture ({detail})"
    )


def _load_exact(
    model: Any,
    spec: ModelSpec,
    *,
    nested_keys: tuple[str, ...],
    prefixes: tuple[str, ...],
    select_prefixes: tuple[str, ...] = (),
) -> Any:
    spec.checkpoint.verify_local_file()
    payload = _torch_load(spec.checkpoint.path)
    state_dict = _exact_state_dict(
        model,
        payload,
        nested_keys=nested_keys,
        prefixes=prefixes,
        select_prefixes=select_prefixes,
        family=spec.family,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_arcface_checkpoint(spec: ModelSpec) -> Any:
    if spec.family != "arcface":
        raise ValueError("ArcFace loader requires family='arcface'")
    from research.embeddings.pytorch.official_backbones import (
        build_arcface_backbone,
    )

    model = build_arcface_backbone(
        spec.architecture, embedding_dim=spec.embedding_dim
    )
    return _load_exact(
        model,
        spec,
        nested_keys=("state_dict_backbone", "state_dict", "model"),
        prefixes=("module.", "backbone.", "model."),
    )


def load_adaface_checkpoint(spec: ModelSpec) -> Any:
    if spec.family != "adaface":
        raise ValueError("AdaFace loader requires family='adaface'")
    from research.embeddings.pytorch.official_backbones import (
        build_adaface_backbone,
    )

    model = build_adaface_backbone(
        spec.architecture, embedding_dim=spec.embedding_dim
    )
    return _load_exact(
        model,
        spec,
        nested_keys=("state_dict", "model"),
        prefixes=("model.", "module.model.", "module.", "backbone."),
        select_prefixes=("model.", "module.model."),
    )


def load_magface_checkpoint(spec: ModelSpec) -> Any:
    if spec.family != "magface":
        raise ValueError("MagFace loader requires family='magface'")
    from research.embeddings.pytorch.official_backbones import (
        build_magface_backbone,
    )

    model = build_magface_backbone(
        spec.architecture, embedding_dim=spec.embedding_dim
    )
    return _load_exact(
        model,
        spec,
        nested_keys=("state_dict", "model"),
        prefixes=(
            "module.features.",
            "features.module.",
            "features.",
            "module.",
            "backbone.",
        ),
        select_prefixes=("module.features.", "features.module."),
    )


def load_edgeface_checkpoint(spec: ModelSpec) -> Any:
    if spec.family != "edgeface":
        raise ValueError("EdgeFace loader requires family='edgeface'")
    from research.embeddings.pytorch.official_backbones import (
        build_edgeface_backbone,
    )

    model = build_edgeface_backbone(
        spec.architecture, embedding_dim=spec.embedding_dim
    )
    return _load_exact(
        model,
        spec,
        nested_keys=("state_dict", "model"),
        prefixes=("module.",),
    )
