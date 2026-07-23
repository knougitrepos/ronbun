from __future__ import annotations

import json
import os
from pathlib import Path

from research.embeddings.base import ModelSpec


def read_model_spec(
    path: str | Path,
    *,
    verify_checkpoint: bool = True,
) -> ModelSpec:
    """Read a registered model spec and revalidate its local checkpoint."""

    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model spec manifest must contain a JSON object")
    return ModelSpec.from_manifest(
        payload,
        verify_checkpoint=verify_checkpoint,
    )


def write_model_spec(path: str | Path, spec: ModelSpec) -> Path:
    """Write an immutable model spec; existing registrations are never replaced."""

    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"model spec already exists and will not be overwritten: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(spec.to_manifest(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination
