from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any

import numpy as np
import pandas as pd

from research.datasets.rfw_aligned_bin import (
    inspect_rfw_aligned_bin_archive,
    iter_rfw_aligned_pair_batches,
)
from research.compression import PCACompressor, PQCompressor
from research.embeddings.manifests import read_model_spec
from research.embeddings.registry import create_pytorch_adapter_from_spec
from research.evaluation.rfw_verification import (
    RFWVerificationResult,
    evaluate_rfw_10fold,
    evaluate_rfw_pair_scores,
)
from research.runtime.hashing import canonical_sha256, sha256_file


ProgressCallback = Callable[[str, Mapping[str, Any]], None]
_ORIGIN_MANIFEST = "rfw_origin_embedding_manifest.json"
_ORIGIN_EMBEDDINGS = "origin_embeddings.npy"
_RAW_NORMS = "raw_norms.npy"
_OCCURRENCES = "pair_occurrences.csv"
_SUCCESS = "_SUCCESS"
_PAIR_CONTRACT_COLUMNS = (
    "pair_id",
    "rfw_group",
    "fold_index",
    "official_index",
    "is_genuine",
)


@dataclass(frozen=True)
class RFWOriginEmbeddingArtifact:
    root: Path
    manifest: dict[str, Any]
    occurrences: pd.DataFrame
    embeddings: np.ndarray
    raw_norms: np.ndarray


@dataclass(frozen=True)
class FrozenCodecSpec:
    profile_name: str
    family: str
    artifact_path: Path
    artifact_sha256: str
    fit_source_dataset: str
    fit_source_run_id: str
    fit_manifest_path: Path
    fit_manifest_sha256: str

    def __post_init__(self) -> None:
        family = str(self.family).strip().lower()
        if family not in {"pca", "pq"}:
            raise ValueError("frozen RFW codec family must be pca or pq")
        source = str(self.fit_source_dataset).strip().lower()
        if source not in {"lfw", "survface"}:
            raise ValueError("RFW frozen codecs must be fitted on LFW or SurvFace")
        if not str(self.profile_name).strip() or not str(self.fit_source_run_id).strip():
            raise ValueError("frozen RFW codec profile and source run must be non-empty")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "fit_source_dataset", source)
        object.__setattr__(
            self, "artifact_path", Path(self.artifact_path).expanduser().resolve()
        )
        object.__setattr__(
            self,
            "fit_manifest_path",
            Path(self.fit_manifest_path).expanduser().resolve(),
        )
        for value, label in (
            (self.artifact_sha256, "artifact_sha256"),
            (self.fit_manifest_sha256, "fit_manifest_sha256"),
        ):
            normalized = str(value).strip().lower()
            if len(normalized) != 64:
                raise ValueError(f"{label} must contain 64 hexadecimal characters")
            try:
                int(normalized, 16)
            except ValueError as exc:
                raise ValueError(f"{label} must be hexadecimal") from exc
            object.__setattr__(self, label, normalized)

    def verified_manifest(self) -> dict[str, Any]:
        if not self.artifact_path.is_file():
            raise FileNotFoundError(self.artifact_path)
        if sha256_file(self.artifact_path) != self.artifact_sha256:
            raise ValueError(
                f"frozen RFW codec artifact SHA mismatch: {self.artifact_path}"
            )
        if not self.fit_manifest_path.is_file():
            raise FileNotFoundError(self.fit_manifest_path)
        if sha256_file(self.fit_manifest_path) != self.fit_manifest_sha256:
            raise ValueError(
                "frozen RFW codec fit-manifest SHA mismatch: "
                f"{self.fit_manifest_path}"
            )
        fit_manifest = json.loads(
            self.fit_manifest_path.read_text(encoding="utf-8")
        )
        expected_fit_contract = {
            "status": "completed",
            "artifact_type": "frozen_compression_codec_bundle",
            "fit_source_dataset": self.fit_source_dataset,
            "fit_source_run_id": self.fit_source_run_id,
            "fit_on_rfw": False,
        }
        fit_mismatches = {
            key: {"expected": value, "actual": fit_manifest.get(key)}
            for key, value in expected_fit_contract.items()
            if fit_manifest.get(key) != value
        }
        if fit_mismatches:
            raise ValueError(
                "frozen RFW codec fit-manifest contract mismatch: "
                f"{fit_mismatches}"
            )
        matching_entries = [
            entry
            for entry in fit_manifest.get("codecs", [])
            if str(entry.get("profile_name")) == self.profile_name
            and str(entry.get("family")).lower() == self.family
            and str(entry.get("artifact_sha256")).lower()
            == self.artifact_sha256
        ]
        if len(matching_entries) != 1:
            raise ValueError(
                "frozen RFW codec artifact is not uniquely registered in its "
                "fit manifest"
            )
        registered = matching_entries[0]
        if int(registered.get("artifact_byte_count", -1)) != int(
            self.artifact_path.stat().st_size
        ):
            raise ValueError(
                "frozen RFW codec byte count differs from its fit manifest"
            )
        return {
            "profile_name": self.profile_name,
            "family": self.family,
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": self.artifact_sha256,
            "artifact_byte_count": int(self.artifact_path.stat().st_size),
            "fit_source_dataset": self.fit_source_dataset,
            "fit_source_run_id": self.fit_source_run_id,
            "fit_manifest_path": str(self.fit_manifest_path),
            "fit_manifest_sha256": self.fit_manifest_sha256,
            "fit_seed": int(registered["fit_seed"]),
            "fit_model_uid": str(fit_manifest["model_uid"]),
            "fit_on_rfw": False,
        }


@dataclass(frozen=True)
class RFWFrozenCodecEvaluation:
    root: Path
    manifest: dict[str, Any]
    profile_summary: pd.DataFrame
    group_summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    pair_scores: pd.DataFrame


def _emit(
    callback: ProgressCallback | None,
    message: str,
    **details: Any,
) -> None:
    if callback is not None:
        callback(message, details)


def _pairs_contract(pairs: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    missing = sorted(set(_PAIR_CONTRACT_COLUMNS) - set(pairs.columns))
    if missing:
        raise ValueError(f"RFW pairs are missing required columns: {missing}")
    contract = pairs.loc[:, list(_PAIR_CONTRACT_COLUMNS)].copy()
    contract["pair_id"] = contract["pair_id"].astype(str)
    contract["rfw_group"] = contract["rfw_group"].astype(str)
    contract["fold_index"] = contract["fold_index"].astype(int)
    contract["official_index"] = contract["official_index"].astype(int)
    contract["is_genuine"] = contract["is_genuine"].astype(bool)
    contract = contract.sort_values(
        ["rfw_group", "official_index"]
    ).reset_index(drop=True)
    if contract["pair_id"].duplicated().any():
        raise ValueError("RFW pair_id must be unique")
    return contract, canonical_sha256(contract.to_dict(orient="records"))


def _artifact_entry(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "byte_count": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or np.any(norms <= 0.0):
        raise ValueError("RFW embeddings must be finite with positive norm")
    return (values / norms[:, None]).astype(np.float32, copy=False)


def load_rfw_origin_embedding_artifact(
    root: str | Path,
) -> RFWOriginEmbeddingArtifact:
    directory = Path(root).expanduser().resolve()
    manifest_path = directory / _ORIGIN_MANIFEST
    success_path = directory / _SUCCESS
    if not success_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"completed RFW origin embedding artifact not found: {directory}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("RFW origin embedding manifest is not completed")
    if success_path.read_text(encoding="utf-8").strip() != sha256_file(
        manifest_path
    ):
        raise ValueError("RFW origin embedding _SUCCESS does not match manifest")
    paths: dict[str, Path] = {}
    for name, entry in manifest["artifacts"].items():
        path = directory / str(entry["path"])
        if not path.is_file() or sha256_file(path) != str(entry["sha256"]):
            raise ValueError(f"RFW origin embedding artifact hash mismatch: {name}")
        if path.stat().st_size != int(entry["byte_count"]):
            raise ValueError(f"RFW origin embedding artifact size mismatch: {name}")
        paths[name] = path
    occurrences = pd.read_csv(paths["pair_occurrences"])
    embeddings = np.load(paths["origin_embeddings"], mmap_mode="r")
    raw_norms = np.load(paths["raw_norms"], mmap_mode="r")
    expected_count = int(manifest["occurrence_count"])
    if embeddings.shape != (expected_count, 512):
        raise ValueError("RFW origin embedding array shape is invalid")
    if raw_norms.shape != (expected_count,) or len(occurrences) != expected_count:
        raise ValueError("RFW occurrence/raw-norm row counts are inconsistent")
    if occurrences["occurrence_id"].duplicated().any():
        raise ValueError("RFW occurrence IDs must be unique")
    return RFWOriginEmbeddingArtifact(
        root=directory,
        manifest=manifest,
        occurrences=occurrences,
        embeddings=embeddings,
        raw_norms=raw_norms,
    )


def frozen_codec_specs_from_completed_run(
    run_dir: str | Path,
    *,
    expected_model_uid: str | None = None,
    families: Sequence[str] = ("pca", "pq"),
    profile_names: Sequence[str] | None = None,
) -> tuple[FrozenCodecSpec, ...]:
    """Resolve SHA-pinned codecs from one immutable completed Step 4 run.

    The caller selects the run explicitly. This function never searches for a
    latest run and never fits a codec. Historical runs without a frozen codec
    manifest fail closed.
    """

    source = Path(run_dir).expanduser().resolve()
    run_manifest_path = source / "run_manifest.json"
    freeze_manifest_path = (
        source / "artifacts/step2_workflow/freeze_manifest.json"
    )
    codec_manifest_path = (
        source / "artifacts/step2_workflow/frozen_codec_manifest.json"
    )
    if not (source / "COMPLETED").is_file():
        raise FileNotFoundError(f"completed Step 4 marker is missing: {source}")
    for path in (run_manifest_path, freeze_manifest_path, codec_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    freeze_manifest = json.loads(
        freeze_manifest_path.read_text(encoding="utf-8")
    )
    codec_manifest = json.loads(
        codec_manifest_path.read_text(encoding="utf-8")
    )
    run_id = str(run_manifest.get("run_id", ""))
    model_uid = str(freeze_manifest.get("model_uid", ""))
    dataset_id = str(freeze_manifest.get("dataset_id", "")).lower()
    expected_contract = {
        "run_status": (run_manifest.get("status"), "completed"),
        "freeze_run_id": (str(freeze_manifest.get("run_id", "")), run_id),
        "codec_status": (codec_manifest.get("status"), "completed"),
        "codec_artifact_type": (
            codec_manifest.get("artifact_type"),
            "frozen_compression_codec_bundle",
        ),
        "codec_source_dataset": (
            str(codec_manifest.get("fit_source_dataset", "")).lower(),
            dataset_id,
        ),
        "codec_source_run_id": (
            str(codec_manifest.get("fit_source_run_id", "")),
            run_id,
        ),
        "codec_model_uid": (
            str(codec_manifest.get("model_uid", "")),
            model_uid,
        ),
        "fit_on_rfw": (codec_manifest.get("fit_on_rfw"), False),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in expected_contract.items()
        if actual != expected
    }
    if dataset_id not in {"lfw", "survface"}:
        mismatches["dataset_id"] = {
            "actual": dataset_id,
            "expected": "lfw or survface",
        }
    if expected_model_uid is not None and model_uid != str(expected_model_uid):
        mismatches["expected_model_uid"] = {
            "actual": model_uid,
            "expected": str(expected_model_uid),
        }
    if mismatches:
        raise ValueError(f"completed codec-run lineage mismatch: {mismatches}")

    selected_families = tuple(
        dict.fromkeys(str(value).strip().lower() for value in families)
    )
    if not selected_families or set(selected_families) - {"pca", "pq"}:
        raise ValueError("families must contain pca and/or pq")
    selected_profiles = (
        None
        if profile_names is None
        else {str(value).strip() for value in profile_names}
    )
    if selected_profiles is not None and not all(selected_profiles):
        raise ValueError("profile_names must be non-empty strings")

    manifest_sha256 = sha256_file(codec_manifest_path)
    specs: list[FrozenCodecSpec] = []
    for entry in codec_manifest.get("codecs", []):
        family = str(entry.get("family", "")).lower()
        profile_name = str(entry.get("profile_name", ""))
        if family not in selected_families:
            continue
        if selected_profiles is not None and profile_name not in selected_profiles:
            continue
        artifact_path = source / str(entry.get("artifact", ""))
        expected_sha256 = str(entry.get("artifact_sha256", "")).lower()
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if sha256_file(artifact_path) != expected_sha256:
            raise ValueError(f"frozen codec SHA mismatch: {artifact_path}")
        if artifact_path.stat().st_size != int(
            entry.get("artifact_byte_count", -1)
        ):
            raise ValueError(f"frozen codec byte-size mismatch: {artifact_path}")
        specs.append(
            FrozenCodecSpec(
                profile_name=profile_name,
                family=family,
                artifact_path=artifact_path,
                artifact_sha256=expected_sha256,
                fit_source_dataset=dataset_id,
                fit_source_run_id=run_id,
                fit_manifest_path=codec_manifest_path,
                fit_manifest_sha256=manifest_sha256,
            )
        )
    if selected_profiles is not None:
        observed_profiles = {spec.profile_name for spec in specs}
        missing_profiles = sorted(selected_profiles - observed_profiles)
        if missing_profiles:
            raise ValueError(
                f"requested frozen codec profiles are absent: {missing_profiles}"
            )
    if not specs:
        raise ValueError("selected completed run contains no requested codecs")
    return tuple(specs)


def extract_rfw_origin_embeddings(
    *,
    aligned_bin_archive_path: str | Path,
    pairs: pd.DataFrame,
    model_spec_path: str | Path,
    output_dir: str | Path,
    expected_archive_sha256: str | None = None,
    expected_model_uid: str | None = None,
    device: str = "cuda",
    batch_size: int = 128,
    horizontal_flip_tta: bool = False,
    strict_official: bool = True,
    reuse_completed: bool = True,
    progress: ProgressCallback | None = None,
) -> RFWOriginEmbeddingArtifact:
    """Extract normalized embeddings for RFW pair occurrences.

    The output is immutable: an existing completed directory is reused only
    after every artifact hash and the requested science contract match.
    """

    destination = Path(output_dir).expanduser().resolve()
    contract, pairs_sha256 = _pairs_contract(pairs)
    archive_summary = inspect_rfw_aligned_bin_archive(
        aligned_bin_archive_path,
        expected_sha256=expected_archive_sha256,
        strict_official=strict_official,
    )
    spec_path = Path(model_spec_path).expanduser().resolve()
    spec = read_model_spec(spec_path)
    if expected_model_uid is not None and spec.model_uid != str(expected_model_uid):
        raise ValueError(
            "RFW model UID differs from the requested model: "
            f"expected={expected_model_uid}, actual={spec.model_uid}"
        )
    requested_contract = {
        "pairs_contract_sha256": pairs_sha256,
        "aligned_bin_archive_sha256": archive_summary.archive_sha256,
        "model_uid": spec.model_uid,
        "checkpoint_sha256": spec.checkpoint.sha256,
        "preprocess_hash": spec.preprocessing.preprocess_hash,
        "horizontal_flip_tta": bool(horizontal_flip_tta),
    }
    if destination.exists():
        if not reuse_completed:
            raise FileExistsError(destination)
        existing = load_rfw_origin_embedding_artifact(destination)
        mismatches = {
            key: {"expected": value, "actual": existing.manifest.get(key)}
            for key, value in requested_contract.items()
            if existing.manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "existing RFW origin embedding artifact has different lineage: "
                f"{mismatches}"
            )
        return existing

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    )
    staging.mkdir(parents=False, exist_ok=False)
    occurrence_count = 2 * len(contract)
    embedding_path = staging / _ORIGIN_EMBEDDINGS
    raw_norm_path = staging / _RAW_NORMS
    occurrence_path = staging / _OCCURRENCES
    embeddings = np.lib.format.open_memmap(
        embedding_path,
        mode="w+",
        dtype=np.float32,
        shape=(occurrence_count, 512),
    )
    raw_norms = np.lib.format.open_memmap(
        raw_norm_path,
        mode="w+",
        dtype=np.float32,
        shape=(occurrence_count,),
    )
    adapter = create_pytorch_adapter_from_spec(spec, device=device)
    occurrence_frames: list[pd.DataFrame] = []
    offset = 0
    try:
        for batch in iter_rfw_aligned_pair_batches(
            archive_summary.archive_path,
            contract,
            batch_size=batch_size,
            expected_sha256=archive_summary.archive_sha256,
            strict_official=strict_official,
        ):
            output = adapter.embed(batch.faces)
            normalized = output.normalized_embedding
            if horizontal_flip_tta:
                flipped = adapter.embed(batch.faces[:, :, ::-1, :].copy())
                normalized = _normalize_rows(
                    normalized + flipped.normalized_embedding
                )
            end = offset + len(batch.faces)
            embeddings[offset:end] = normalized
            raw_norms[offset:end] = output.raw_norm
            occurrence_frames.append(batch.occurrences)
            offset = end
            _emit(
                progress,
                "RFW origin embedding extraction",
                processed=offset,
                total=occurrence_count,
                model_uid=spec.model_uid,
            )
        if offset != occurrence_count:
            raise RuntimeError(
                "RFW aligned BIN extraction count mismatch: "
                f"expected={occurrence_count}, actual={offset}"
            )
        embeddings.flush()
        raw_norms.flush()
        del embeddings, raw_norms
        occurrences = pd.concat(occurrence_frames, ignore_index=True)
        if len(occurrences) != occurrence_count:
            raise RuntimeError("RFW occurrence metadata count mismatch")
        occurrences.to_csv(occurrence_path, index=False)
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "dataset": "rfw-v1",
            "protocol": "official_groupwise_10fold_pair_verification",
            **requested_contract,
            "aligned_bin_archive_path": str(archive_summary.archive_path),
            "model_spec_path": str(spec_path),
            "model_spec_sha256": sha256_file(spec_path),
            "family": spec.family,
            "architecture": spec.architecture,
            "training_dataset": spec.training_dataset,
            "device": str(device),
            "batch_size": int(batch_size),
            "pair_count": int(len(contract)),
            "occurrence_count": int(occurrence_count),
            "embedding_dim": 512,
            "artifacts": {
                "origin_embeddings": _artifact_entry(embedding_path),
                "raw_norms": _artifact_entry(raw_norm_path),
                "pair_occurrences": _artifact_entry(occurrence_path),
            },
        }
        manifest_path = staging / _ORIGIN_MANIFEST
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / _SUCCESS).write_text(
            sha256_file(manifest_path) + "\n", encoding="utf-8"
        )
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_rfw_origin_embedding_artifact(destination)


def rfw_occurrence_pairs(
    artifact: RFWOriginEmbeddingArtifact,
) -> pd.DataFrame:
    occurrences = artifact.occurrences.copy()
    sides = occurrences.pivot(
        index="pair_id", columns="side", values="occurrence_id"
    )
    if set(sides.columns) != {"left", "right"} or sides.isna().any().any():
        raise ValueError("RFW occurrence artifact must contain one left/right row per pair")
    metadata_columns = [
        "pair_id",
        "rfw_group",
        "fold_index",
        "official_index",
        "is_genuine",
    ]
    metadata = occurrences.loc[:, metadata_columns].drop_duplicates()
    if metadata["pair_id"].duplicated().any():
        raise ValueError("RFW pair metadata differs between left/right occurrences")
    result = metadata.merge(
        sides.rename(
            columns={"left": "left_image_id", "right": "right_image_id"}
        ).reset_index(),
        on="pair_id",
        how="inner",
        validate="one_to_one",
    )
    return result.sort_values(
        ["rfw_group", "official_index"]
    ).reset_index(drop=True)


def _load_completed_evaluation(
    root: Path,
) -> RFWFrozenCodecEvaluation:
    manifest_path = root / "rfw_frozen_codec_manifest.json"
    success_path = root / _SUCCESS
    if not manifest_path.is_file() or not success_path.is_file():
        raise FileNotFoundError(f"completed RFW codec evaluation not found: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("RFW frozen-codec evaluation is not completed")
    if success_path.read_text(encoding="utf-8").strip() != sha256_file(
        manifest_path
    ):
        raise ValueError("RFW frozen-codec _SUCCESS does not match manifest")
    frames: dict[str, pd.DataFrame] = {}
    for name, entry in manifest["artifacts"].items():
        path = root / str(entry["path"])
        if not path.is_file() or sha256_file(path) != str(entry["sha256"]):
            raise ValueError(f"RFW frozen-codec artifact hash mismatch: {name}")
        frames[name] = pd.read_csv(path)
    return RFWFrozenCodecEvaluation(
        root=root,
        manifest=manifest,
        profile_summary=frames["profile_summary"],
        group_summary=frames["group_summary"],
        fold_metrics=frames["fold_metrics"],
        pair_scores=frames["pair_scores"],
    )


def load_rfw_frozen_codec_evaluation(
    root: str | Path,
) -> RFWFrozenCodecEvaluation:
    """Load and hash-verify a completed RFW supplementary evaluation."""

    return _load_completed_evaluation(Path(root).expanduser().resolve())


def rfw_frozen_codec_evaluation_uid(
    *,
    origin_artifact_dir: str | Path,
    codec_specs: Sequence[FrozenCodecSpec],
    strict_official: bool = True,
    bootstrap_seed: int = 42,
    bootstrap_repeats: int = 2000,
) -> str:
    """Return a stable UID for one origin/codec/evaluation contract."""

    origin = load_rfw_origin_embedding_artifact(origin_artifact_dir)
    verified_codecs = [spec.verified_manifest() for spec in codec_specs]
    wrong_models = sorted(
        {
            str(entry["fit_model_uid"])
            for entry in verified_codecs
            if str(entry["fit_model_uid"]) != str(origin.manifest["model_uid"])
        }
    )
    if wrong_models:
        raise ValueError(
            "RFW origin/model codec mismatch: "
            f"origin={origin.manifest['model_uid']}, codecs={wrong_models}"
        )
    payload = {
        "artifact_type": "rfw_frozen_codec_evaluation_contract",
        "origin_manifest_sha256": sha256_file(origin.root / _ORIGIN_MANIFEST),
        "model_uid": origin.manifest["model_uid"],
        "pairs_contract_sha256": origin.manifest["pairs_contract_sha256"],
        "strict_official": bool(strict_official),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_repeats": int(bootstrap_repeats),
        "codecs": verified_codecs,
    }
    return f"rfw-{canonical_sha256(payload)[:20]}"


def _pair_indexes(
    pairs: pd.DataFrame,
    occurrence_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {str(value): index for index, value in enumerate(occurrence_ids)}
    if len(lookup) != len(occurrence_ids):
        raise ValueError("RFW occurrence IDs must be unique")
    try:
        left = np.asarray(
            [lookup[str(value)] for value in pairs["left_image_id"]],
            dtype=np.int64,
        )
        right = np.asarray(
            [lookup[str(value)] for value in pairs["right_image_id"]],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(f"RFW evaluation pair has no occurrence embedding: {exc}") from exc
    return left, right


def _adc_thresholds(scores: np.ndarray) -> np.ndarray:
    low = float(np.min(scores))
    high = float(np.max(scores))
    if low == high:
        return np.asarray([low - 1.0, high + 1.0], dtype=np.float64)
    margin = max((high - low) * 1e-6, 1e-9)
    return np.linspace(low - margin, high + margin, 4001)


def _result_frames(
    result: RFWVerificationResult,
    *,
    profile_name: str,
    compression_family: str,
    search_mode: str,
    fit_source_dataset: str,
    fit_source_run_id: str,
    codec_sha256: str | None,
    embedding_payload_bytes: int,
    codec_artifact_bytes: int,
    occurrence_count: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = {
        "compression_profile": profile_name,
        "compression_family": compression_family,
        "search_mode": search_mode,
        "fit_source_dataset": fit_source_dataset,
        "fit_source_run_id": fit_source_run_id,
        "codec_sha256": codec_sha256,
        "fit_on_rfw": False,
    }
    summary = {
        **common,
        "embedding_payload_bytes": int(embedding_payload_bytes),
        "codec_artifact_bytes": int(codec_artifact_bytes),
        "evaluated_occurrence_count": int(occurrence_count),
        "evaluated_payload_storage_bytes": int(
            embedding_payload_bytes * occurrence_count
        ),
        "evaluated_total_storage_bytes": int(
            embedding_payload_bytes * occurrence_count + codec_artifact_bytes
        ),
        "macro_group_accuracy": float(result.summary["macro_group_accuracy"]),
        "macro_group_eer": float(result.summary["macro_group_eer"]),
        "group_accuracy_gap": float(result.summary["group_accuracy_gap"]),
        "group_eer_gap": float(result.summary["group_eer_gap"]),
        "eer_threshold_policy": str(result.summary["eer_threshold_policy"]),
        "eer_threshold_source": str(result.summary["eer_threshold_source"]),
        "eer_uses_internal_9fold_threshold": bool(
            result.summary["eer_uses_internal_9fold_threshold"]
        ),
        "open_set_protocol": False,
    }
    group = result.group_summary.assign(**common)
    folds = result.fold_metrics.assign(**common)
    scores = result.pair_scores.copy()
    score_column = (
        "cosine_score" if "cosine_score" in scores.columns else "pair_score"
    )
    scores = scores.rename(columns={score_column: "score"}).assign(**common)
    return summary, group, folds, scores


def evaluate_rfw_frozen_codecs(
    *,
    origin_artifact_dir: str | Path,
    codec_specs: Sequence[FrozenCodecSpec],
    output_dir: str | Path,
    strict_official: bool = True,
    bootstrap_seed: int = 42,
    bootstrap_repeats: int = 2000,
    reuse_completed: bool = True,
) -> RFWFrozenCodecEvaluation:
    """Apply externally fitted codecs and run RFW 1:1 verification.

    No codec ``fit`` method is called here. PCA is reported in both direct and
    reconstructed cosine spaces. PQ is reported as reconstructed cosine and
    symmetric ADC-like negative squared-L2, averaged across both pair sides.
    """

    origin = load_rfw_origin_embedding_artifact(origin_artifact_dir)
    pairs = rfw_occurrence_pairs(origin)
    occurrence_ids = origin.occurrences["occurrence_id"].astype(str).tolist()
    matrix = np.asarray(origin.embeddings, dtype=np.float32)
    left, right = _pair_indexes(pairs, occurrence_ids)
    verified_codecs = [spec.verified_manifest() for spec in codec_specs]
    wrong_models = sorted(
        {
            str(entry["fit_model_uid"])
            for entry in verified_codecs
            if str(entry["fit_model_uid"]) != str(origin.manifest["model_uid"])
        }
    )
    if wrong_models:
        raise ValueError(
            "RFW origin/model codec mismatch: "
            f"origin={origin.manifest['model_uid']}, codecs={wrong_models}"
        )
    profile_keys = [
        (
            entry["fit_source_dataset"],
            entry["fit_source_run_id"],
            entry["profile_name"],
        )
        for entry in verified_codecs
    ]
    if len(set(profile_keys)) != len(profile_keys):
        raise ValueError("RFW frozen codec specifications contain duplicate profiles")
    origin_manifest_path = origin.root / _ORIGIN_MANIFEST
    requested_contract = {
        "origin_manifest_sha256": sha256_file(origin_manifest_path),
        "model_uid": origin.manifest["model_uid"],
        "pairs_contract_sha256": origin.manifest["pairs_contract_sha256"],
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_repeats": int(bootstrap_repeats),
        "strict_official": bool(strict_official),
        "codecs": verified_codecs,
    }
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        if not reuse_completed:
            raise FileExistsError(destination)
        existing = _load_completed_evaluation(destination)
        mismatches = {
            key: {"expected": value, "actual": existing.manifest.get(key)}
            for key, value in requested_contract.items()
            if existing.manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "existing RFW codec evaluation has different lineage: "
                f"{mismatches}"
            )
        return existing

    summary_rows: list[dict[str, Any]] = []
    group_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []

    origin_result = evaluate_rfw_10fold(
        pairs,
        image_ids=occurrence_ids,
        embeddings=matrix,
        strict_official=strict_official,
        bootstrap_seed=bootstrap_seed,
        bootstrap_repeats=bootstrap_repeats,
    )
    frames = _result_frames(
        origin_result,
        profile_name="origin_512",
        compression_family="origin",
        search_mode="origin_cosine",
        fit_source_dataset="none",
        fit_source_run_id="none",
        codec_sha256=None,
        embedding_payload_bytes=512 * np.dtype(np.float32).itemsize,
        codec_artifact_bytes=0,
        occurrence_count=len(matrix),
    )
    summary_rows.append(frames[0])
    group_frames.append(frames[1])
    fold_frames.append(frames[2])
    score_frames.append(frames[3])

    for spec, codec_manifest in zip(codec_specs, verified_codecs, strict=True):
        codec_bytes = int(codec_manifest["artifact_byte_count"])
        if spec.family == "pca":
            compressor = PCACompressor.load(spec.artifact_path)
            direct = compressor.transform(matrix)
            reconstructed = compressor.inverse_transform(direct)
            modes = (
                ("pca_direct_cosine", direct),
                ("pca_reconstruction_cosine", reconstructed),
            )
            payload_bytes = int(direct.shape[1] * np.dtype(np.float32).itemsize)
            for search_mode, values in modes:
                result = evaluate_rfw_10fold(
                    pairs,
                    image_ids=occurrence_ids,
                    embeddings=values,
                    strict_official=strict_official,
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_repeats=bootstrap_repeats,
                )
                frames = _result_frames(
                    result,
                    profile_name=spec.profile_name,
                    compression_family="pca",
                    search_mode=search_mode,
                    fit_source_dataset=spec.fit_source_dataset,
                    fit_source_run_id=spec.fit_source_run_id,
                    codec_sha256=spec.artifact_sha256,
                    embedding_payload_bytes=payload_bytes,
                    codec_artifact_bytes=codec_bytes,
                    occurrence_count=len(matrix),
                )
                summary_rows.append(frames[0])
                group_frames.append(frames[1])
                fold_frames.append(frames[2])
                score_frames.append(frames[3])
        else:
            compressor = PQCompressor.load(spec.artifact_path)
            compressor.source_profile = "origin_512"
            codes = compressor.encode(matrix)
            reconstructed = compressor.decode(codes)
            cosine_result = evaluate_rfw_10fold(
                pairs,
                image_ids=occurrence_ids,
                embeddings=reconstructed,
                strict_official=strict_official,
                bootstrap_seed=bootstrap_seed,
                bootstrap_repeats=bootstrap_repeats,
            )
            modes: list[tuple[str, RFWVerificationResult]] = [
                ("pq_reconstruction_cosine", cosine_result)
            ]
            symmetric_adc = -0.5 * (
                np.sum((matrix[left] - reconstructed[right]) ** 2, axis=1)
                + np.sum((matrix[right] - reconstructed[left]) ** 2, axis=1)
            )
            adc_result = evaluate_rfw_pair_scores(
                pairs,
                scores=symmetric_adc,
                score_space="pq_symmetric_adc_negative_squared_l2",
                thresholds=_adc_thresholds(symmetric_adc),
                strict_official=strict_official,
                bootstrap_seed=bootstrap_seed,
                bootstrap_repeats=bootstrap_repeats,
            )
            modes.append(("pq_symmetric_adc", adc_result))
            for search_mode, result in modes:
                frames = _result_frames(
                    result,
                    profile_name=spec.profile_name,
                    compression_family="pq",
                    search_mode=search_mode,
                    fit_source_dataset=spec.fit_source_dataset,
                    fit_source_run_id=spec.fit_source_run_id,
                    codec_sha256=spec.artifact_sha256,
                    embedding_payload_bytes=int(codes.shape[1]),
                    codec_artifact_bytes=codec_bytes,
                    occurrence_count=len(matrix),
                )
                summary_rows.append(frames[0])
                group_frames.append(frames[1])
                fold_frames.append(frames[2])
                score_frames.append(frames[3])

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    )
    staging.mkdir(parents=False, exist_ok=False)
    try:
        outputs = {
            "profile_summary": staging / "rfw_profile_summary.csv",
            "group_summary": staging / "rfw_group_summary.csv",
            "fold_metrics": staging / "rfw_fold_metrics.csv",
            "pair_scores": staging / "rfw_pair_scores.csv",
        }
        pd.DataFrame.from_records(summary_rows).to_csv(
            outputs["profile_summary"], index=False
        )
        pd.concat(group_frames, ignore_index=True).to_csv(
            outputs["group_summary"], index=False
        )
        pd.concat(fold_frames, ignore_index=True).to_csv(
            outputs["fold_metrics"], index=False
        )
        pd.concat(score_frames, ignore_index=True).to_csv(
            outputs["pair_scores"], index=False
        )
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "dataset": "rfw-v1",
            "evaluation_role": "supplementary_1to1_verification",
            "open_set_protocol": False,
            "codec_fit_on_rfw": False,
            "metrics": ["accuracy", "tar", "far", "eer"],
            "eer_contract": {
                "threshold_source": "heldout_fold_scores_and_labels",
                "method": "heldout_scores_minimum_absolute_far_frr_v1",
                "uses_other_9_fold_accuracy_threshold": False,
            },
            **requested_contract,
            "profile_result_count": int(len(summary_rows)),
            "artifacts": {
                name: _artifact_entry(path) for name, path in outputs.items()
            },
        }
        manifest_path = staging / "rfw_frozen_codec_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / _SUCCESS).write_text(
            sha256_file(manifest_path) + "\n", encoding="utf-8"
        )
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return _load_completed_evaluation(destination)
