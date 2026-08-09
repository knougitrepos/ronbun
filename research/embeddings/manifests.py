from __future__ import annotations

import json
import os
from pathlib import Path

from research.embeddings.base import (
    FRModelFamily,
    ModelSpec,
    SUPPORTED_FR_MODEL_FAMILIES,
)


class ModelSpecSelectionError(ValueError):
    """Raised when a model registry cannot select one spec unambiguously."""


def model_spec_registry_stem(spec: ModelSpec) -> str:
    """Return a stable registry key for one checkpoint plus analysis target.

    ``model_uid`` deliberately identifies the checkpoint/preprocessing contract
    and therefore does not change when only the Grad-CAM target changes.  A
    target-layer revision still needs its own immutable registry file, so the
    full manifest digest is appended only to the registry filename.
    """

    from research.runtime.hashing import canonical_sha256

    digest = canonical_sha256(spec.to_manifest())[:16]
    return f"{spec.model_uid}--spec-{digest}"


def _model_spec_filename_matches(path: Path, spec: ModelSpec) -> bool:
    return path.stem in {spec.model_uid, model_spec_registry_stem(spec)}


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
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"existing model spec is not valid JSON: {destination}"
            ) from exc
        if existing == spec.to_manifest():
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
    if family not in SUPPORTED_FR_MODEL_FAMILIES:
        raise ValueError(f"unsupported FR model family: {family}")

    if model_uid is not None:
        resolved_uid = str(model_uid).strip()
        if not resolved_uid:
            raise ValueError("model_uid must be non-empty when provided")
        uid_family, separator, uid_digest = resolved_uid.partition("-")
        if (
            separator != "-"
            or uid_family not in SUPPORTED_FR_MODEL_FAMILIES
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
        candidates = sorted(root.glob(f"{resolved_uid}*.json"))
        matches: list[tuple[Path, ModelSpec]] = []
        for manifest_path in candidates:
            spec = read_model_spec(
                manifest_path, verify_checkpoint=verify_checkpoint
            )
            if not _model_spec_filename_matches(manifest_path, spec):
                raise ModelSpecSelectionError(
                    "model manifest filename does not match its computed "
                    f"registry key: {manifest_path}"
                )
            if spec.model_uid == resolved_uid:
                matches.append((manifest_path, spec))
        if not matches:
            raise FileNotFoundError(
                f"registered model_uid was not found: {resolved_uid}"
            )
        if len(matches) > 1:
            raise ModelSpecSelectionError(
                f"multiple ModelSpec manifests share model_uid {resolved_uid!r}; "
                "select by an explicit model profile so the Grad-CAM target "
                "layer is unambiguous"
            )
        manifest_path, spec = matches[0]
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
        if not _model_spec_filename_matches(manifest_path, spec):
            raise ModelSpecSelectionError(
                "model manifest filename does not match its computed registry key: "
                f"{manifest_path}"
            )
        if spec.family == family:
            matches.append((manifest_path, spec))
    if not matches:
        raise ModelSpecSelectionError(
            f"no registered ModelSpec exists for family {family!r}"
        )
    if len(matches) > 1:
        available_uids = sorted({spec.model_uid for _, spec in matches})
        if len(available_uids) == 1:
            raise ModelSpecSelectionError(
                f"multiple ModelSpec manifests share model_uid "
                f"{available_uids[0]!r}; select by an explicit model profile "
                "so the Grad-CAM target layer is unambiguous"
            )
        available = ", ".join(available_uids)
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
    expected_target_layer = str(profile_config["target_layer"])
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
    if family not in SUPPORTED_FR_MODEL_FAMILIES:
        raise ValueError(f"unsupported FR model family: {family}")

    matches: list[tuple[Path, ModelSpec]] = []
    for manifest_path in sorted(root.glob(f"{family}-*.json")):
        spec = read_model_spec(
            manifest_path, verify_checkpoint=verify_checkpoint
        )
        if not _model_spec_filename_matches(manifest_path, spec):
            raise ModelSpecSelectionError(
                "model manifest filename does not match its computed registry key: "
                f"{manifest_path}"
            )
        if (
            spec.family == family
            and spec.architecture == expected_arch
            and spec.training_dataset == expected_dataset
            and spec.target_layer == expected_target_layer
            and (
                pinned_model_uid is None
                or spec.model_uid == str(pinned_model_uid)
            )
            and (
                resolved_expected_checkpoint is None
                or Path(spec.checkpoint.path).resolve() == resolved_expected_checkpoint
            )
        ):
            matches.append((manifest_path, spec))

    if not matches:
        if pinned_model_uid is not None and resolved_expected_checkpoint is not None:
            pinned_candidates: list[ModelSpec] = []
            for manifest_path in sorted(root.glob(f"{family}-*.json")):
                spec = read_model_spec(
                    manifest_path, verify_checkpoint=verify_checkpoint
                )
                if (
                    spec.model_uid == str(pinned_model_uid)
                    and spec.family == family
                    and spec.architecture == expected_arch
                    and spec.training_dataset == expected_dataset
                    and spec.target_layer == expected_target_layer
                ):
                    pinned_candidates.append(spec)
            if pinned_candidates:
                actual_paths = sorted(
                    {
                        str(Path(spec.checkpoint.path).resolve())
                        for spec in pinned_candidates
                    }
                )
                raise ModelSpecSelectionError(
                    f"profile '{profile_id}' pins checkpoint "
                    f"{resolved_expected_checkpoint}, but model_uid "
                    f"{str(pinned_model_uid)!r} uses {actual_paths}"
                )
        raise ModelSpecSelectionError(
            f"profile '{profile_id}'에 해당하는 등록된 ModelSpec이 없습니다. "
            f"family={family!r}, architecture={expected_arch!r}, "
            f"training_dataset={expected_dataset!r}, "
            f"target_layer={expected_target_layer!r}"
        )
    if len(matches) > 1:
        available = ", ".join(spec.model_uid for _, spec in matches)
        raise ModelSpecSelectionError(
            f"profile '{profile_id}'에 해당하는 ModelSpec이 여러 개입니다; "
            f"MODEL_UID를 명시적으로 지정하세요. Available: {available}"
        )
    return matches[0]
