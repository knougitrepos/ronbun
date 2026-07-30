from __future__ import annotations

import json
import os
from pathlib import Path

from research.embeddings.base import FRModelFamily, ModelSpec


class ModelSpecSelectionError(ValueError):
    """Raised when a model registry cannot select one spec unambiguously."""


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
    """Write an immutable model spec, treating an identical rerun as success."""

    destination = Path(path).expanduser().resolve()
    serialized = json.dumps(
        spec.to_manifest(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if destination.exists():
        if destination.read_text(encoding="utf-8") == serialized:
            return destination
        raise FileExistsError(
            f"different model spec already exists and will not be overwritten: "
            f"{destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        serialized,
        encoding="utf-8",
    )
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def select_model_spec(
    registry_root: str | Path,
    *,
    family: FRModelFamily,
    model_uid: str | None = None,
    verify_checkpoint: bool = True,
) -> tuple[Path, ModelSpec]:
    """Select a registered ModelSpec by exact UID or unique model family.

    Family-only selection is intentionally allowed only when exactly one
    registered manifest matches. Once a second checkpoint for the same family
    is registered, callers must pin ``model_uid`` to preserve reproducibility.
    """

    root = Path(registry_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"model registry directory does not exist: {root}")
    if family not in {"arcface", "adaface", "magface"}:
        raise ValueError(f"unsupported FR model family: {family}")

    if model_uid is not None:
        resolved_uid = str(model_uid).strip()
        if not resolved_uid:
            raise ValueError("model_uid must be non-empty when provided")
        uid_family, separator, uid_digest = resolved_uid.partition("-")
        if (
            separator != "-"
            or uid_family not in {"arcface", "adaface", "magface"}
            or len(uid_digest) != 20
        ):
            raise ValueError(
                "model_uid must use '<family>-<20 lowercase hex>' format"
            )
        try:
            int(uid_digest, 16)
        except ValueError as exc:
            raise ValueError(
                "model_uid must use '<family>-<20 lowercase hex>' format"
            ) from exc
        if uid_digest != uid_digest.lower():
            raise ValueError(
                "model_uid must use '<family>-<20 lowercase hex>' format"
            )
        if uid_family != family:
            raise ModelSpecSelectionError(
                f"model_uid {resolved_uid!r} does not belong to requested "
                f"family {family!r}"
            )
        manifest_path = root / f"{resolved_uid}.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"registered model_uid was not found: {resolved_uid}"
            )
        spec = read_model_spec(
            manifest_path, verify_checkpoint=verify_checkpoint
        )
        if spec.model_uid != resolved_uid:
            raise ModelSpecSelectionError(
                "model manifest filename does not match its computed model_uid"
            )
        if spec.family != family:
            raise ModelSpecSelectionError(
                f"model_uid {resolved_uid!r} belongs to {spec.family!r}, "
                f"not requested family {family!r}"
            )
        return manifest_path, spec

    matches: list[tuple[Path, ModelSpec]] = []
    for manifest_path in sorted(root.glob(f"{family}-*.json")):
        spec = read_model_spec(
            manifest_path, verify_checkpoint=verify_checkpoint
        )
        if manifest_path.stem != spec.model_uid:
            raise ModelSpecSelectionError(
                "model manifest filename does not match its computed model_uid: "
                f"{manifest_path}"
            )
        if spec.family == family:
            matches.append((manifest_path, spec))
    if not matches:
        raise ModelSpecSelectionError(
            f"no registered ModelSpec exists for family {family!r}"
        )
    if len(matches) > 1:
        available = ", ".join(spec.model_uid for _, spec in matches)
        raise ModelSpecSelectionError(
            f"multiple ModelSpec manifests exist for family {family!r}; "
            f"set MODEL_UID explicitly. Available: {available}"
        )
    return matches[0]


def select_model_spec_by_profile(
    registry_root: str | Path,
    *,
    profile_id: str,
    profile_config: dict,
    verify_checkpoint: bool = True,
) -> tuple[Path, ModelSpec]:
    """Select a registered ModelSpec matching a profile's expected metadata.

    Unlike ``select_model_spec`` which uses only the family, this function
    verifies that the found manifest's family, architecture, and training
    dataset match the profile configuration.  Optional ``model_uid`` and
    ``checkpoint_path`` profile fields pin one exact registered checkpoint.
    This prevents provenance mismatches when the same family has multiple
    checkpoints registered.
    """

    family = str(profile_config["family"])
    expected_arch = str(profile_config["architecture"])
    expected_dataset = str(profile_config["training_dataset"])
    pinned_model_uid = profile_config.get("model_uid")
    expected_checkpoint_path = profile_config.get("checkpoint_path")
    resolved_expected_checkpoint = (
        None
        if expected_checkpoint_path is None
        else Path(str(expected_checkpoint_path)).expanduser().resolve()
    )

    root = Path(registry_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"model registry directory does not exist: {root}")
    if family not in {"arcface", "adaface", "magface"}:
        raise ValueError(f"unsupported FR model family: {family}")

    if pinned_model_uid is not None:
        manifest_path, spec = select_model_spec(
            root,
            family=family,
            model_uid=str(pinned_model_uid),
            verify_checkpoint=verify_checkpoint,
        )
        if (
            spec.architecture != expected_arch
            or spec.training_dataset != expected_dataset
        ):
            raise ModelSpecSelectionError(
                f"profile '{profile_id}' pins model_uid {spec.model_uid!r}, "
                "but its metadata does not match the profile: "
                f"expected architecture={expected_arch!r}, "
                f"training_dataset={expected_dataset!r}; "
                f"actual architecture={spec.architecture!r}, "
                f"training_dataset={spec.training_dataset!r}"
            )
        if (
            resolved_expected_checkpoint is not None
            and Path(spec.checkpoint.path).resolve() != resolved_expected_checkpoint
        ):
            raise ModelSpecSelectionError(
                f"profile '{profile_id}' pins checkpoint "
                f"{resolved_expected_checkpoint}, but model_uid "
                f"{spec.model_uid!r} uses {spec.checkpoint.path}"
            )
        return manifest_path, spec

    matches: list[tuple[Path, ModelSpec]] = []
    for manifest_path in sorted(root.glob(f"{family}-*.json")):
        spec = read_model_spec(
            manifest_path, verify_checkpoint=verify_checkpoint
        )
        if manifest_path.stem != spec.model_uid:
            raise ModelSpecSelectionError(
                "model manifest filename does not match its computed model_uid: "
                f"{manifest_path}"
            )
        if (
            spec.family == family
            and spec.architecture == expected_arch
            and spec.training_dataset == expected_dataset
            and (
                resolved_expected_checkpoint is None
                or Path(spec.checkpoint.path).resolve() == resolved_expected_checkpoint
            )
        ):
            matches.append((manifest_path, spec))

    if not matches:
        raise ModelSpecSelectionError(
            f"profile '{profile_id}'에 해당하는 등록된 ModelSpec이 없습니다. "
            f"family={family!r}, architecture={expected_arch!r}, "
            f"training_dataset={expected_dataset!r}"
        )
    if len(matches) > 1:
        available = ", ".join(spec.model_uid for _, spec in matches)
        raise ModelSpecSelectionError(
            f"profile '{profile_id}'에 해당하는 ModelSpec이 여러 개입니다; "
            f"MODEL_UID를 명시적으로 지정하세요. Available: {available}"
        )
    return matches[0]
