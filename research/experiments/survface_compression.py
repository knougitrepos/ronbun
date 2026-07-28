"""Canonical SurvFace compressor fit and PostgreSQL materialization stages.

The official SurvFace gallery/probe rows are evaluation-only. Every compressor
and reconstruction-error normalization statistic in this module is fitted from
the identity-disjoint ``training_manifest.csv`` development split.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from research.compression import (
    ORIGIN_512,
    PCA_256,
    PCACompressor,
    PQCompressor,
    pca_profile_name,
    pq_profile_name,
)
from research.database.connection import session_scope
from research.database.models import Embedding512, Image
from research.experiments.materialization import (
    materialize_compressed_embeddings,
    materialize_pca_sweep_embeddings,
)
from research.protocols import validate_identity_disjoint_splits
from research.runtime import RunStore
from research.runtime.hashing import sha256_file


ProgressCallback = Callable[[str, dict[str, object]], None]
SURVFACE_COMPRESSOR_PHASE = "02_survface_compressor_fit"
SURVFACE_MATERIALIZATION_PHASE = (
    "03_survface_compressed_materialization_and_index"
)


def _emit(progress: ProgressCallback | None, message: str, **details: object) -> None:
    if progress is not None:
        progress(message, details)


def _canonical_path(value: object, *, project_root: Path) -> str:
    path = Path(str(value))
    resolved = path if path.is_absolute() else project_root / path
    return os.path.normcase(str(resolved.resolve(strict=False)))


def validate_survface_training_manifest(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and return a copy of the fit/calibration-only manifest."""

    required = {
        "image_id",
        "identity_id",
        "split",
        "image_path",
        "protocol_role",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(
            f"SurvFace training manifest is missing columns: {missing}"
        )
    if manifest.empty:
        raise ValueError("SurvFace training manifest must not be empty")
    result = manifest.copy()
    observed_splits = set(result["split"].astype(str))
    if observed_splits != {"development", "calibration"}:
        raise ValueError(
            "SurvFace training manifest must contain only development and "
            "calibration splits"
        )
    if not result["protocol_role"].astype(str).eq("training").all():
        raise ValueError(
            "SurvFace compressor fit manifest must contain only training rows"
        )
    if result["image_id"].astype(str).duplicated().any():
        raise ValueError("SurvFace training image_id values must be unique")
    validate_identity_disjoint_splits(result)
    return result


def survface_development_image_paths(
    manifest: pd.DataFrame,
    *,
    project_root: str | Path,
) -> set[str]:
    validated = validate_survface_training_manifest(manifest)
    root = Path(project_root).resolve()
    paths = {
        _canonical_path(value, project_root=root)
        for value in validated.loc[
            validated["split"].astype(str).eq("development"),
            "image_path",
        ]
    }
    if not paths:
        raise ValueError("SurvFace development split must not be empty")
    return paths


def _load_development_matrix(
    engine: Engine,
    *,
    run_uid: str,
    training_manifest: pd.DataFrame,
    project_root: Path,
    batch_size: int,
    progress: ProgressCallback | None,
) -> tuple[np.ndarray, dict[str, int]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    required_paths = survface_development_image_paths(
        training_manifest,
        project_root=project_root,
    )
    vectors: list[np.ndarray] = []
    observed_paths: set[str] = set()
    with session_scope(engine) as session:
        query = (
            session.query(Embedding512, Image)
            .join(Image, Embedding512.image_id == Image.id)
            .filter(
                Embedding512.run_uid == str(run_uid),
                Embedding512.vector_type == ORIGIN_512,
            )
        )
        source_count = int(query.count())
        for scanned, (embedding, image) in enumerate(
            query.yield_per(int(batch_size)),
            start=1,
        ):
            canonical = _canonical_path(
                image.image_path,
                project_root=project_root,
            )
            if canonical in required_paths:
                if canonical in observed_paths:
                    raise ValueError(
                        f"duplicate origin embedding path in run: {canonical}"
                    )
                vector = np.asarray(embedding.embedding, dtype=np.float32)
                if vector.shape != (512,) or not np.isfinite(vector).all():
                    raise ValueError(
                        "SurvFace origin embedding must be finite 512D"
                    )
                vectors.append(vector)
                observed_paths.add(canonical)
            if scanned % int(batch_size) == 0 or scanned == source_count:
                _emit(
                    progress,
                    "SurvFace development embedding scan",
                    processed=scanned,
                    total=source_count,
                    matched=len(vectors),
                )
    missing = required_paths.difference(observed_paths)
    if missing:
        raise RuntimeError(
            "SurvFace development origin embedding coverage is incomplete: "
            f"missing={len(missing)}, examples={sorted(missing)[:5]}"
        )
    if not vectors:
        raise RuntimeError("no SurvFace development embeddings were loaded")
    matrix = np.stack(vectors).astype(np.float32, copy=False)
    return matrix, {
        "source_vectors_scanned": int(source_count),
        "development_vectors": int(len(matrix)),
    }


def _latest_completed_attempt(run: RunStore, phase_name: str) -> int:
    attempts = run.run_dir / "phases" / phase_name / "attempts"
    completed: list[int] = []
    for path in sorted(attempts.glob("A*/phase_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            completed.append(int(payload["attempt"]))
    if not completed:
        raise RuntimeError(f"no completed attempt for phase {phase_name}")
    return max(completed)


@dataclass(frozen=True)
class SurvFaceCompressorBundle:
    attempt: int
    fit_count: int
    pcas: dict[str, PCACompressor]
    pca_paths: dict[str, Path]
    pca_sha256: dict[str, str]
    pqs: dict[str, PQCompressor]
    pq_paths: dict[str, Path]
    pq_sha256: dict[str, str]
    summary_path: Path
    summary: dict[str, Any]


def load_survface_compressor_bundle(
    run: RunStore,
    *,
    pca_dimensions: Sequence[int] | None = None,
    pq_settings: Sequence[tuple[int, int]] | None = None,
) -> SurvFaceCompressorBundle:
    """Load and checksum-verify the latest completed same-dataset fit."""

    attempt = _latest_completed_attempt(run, SURVFACE_COMPRESSOR_PHASE)
    run.verify_phase_artifacts(SURVFACE_COMPRESSOR_PHASE, attempt=attempt)
    suffix = f"A{attempt:03d}"
    summary_path = (
        run.run_dir
        / "artifacts"
        / SURVFACE_COMPRESSOR_PHASE
        / f"survface_compressor_summary_{suffix}.json"
    )
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("fit_source") != "survface_training_development"
        or summary.get("official_test_fit") is not False
    ):
        raise ValueError("SurvFace compressor summary has an invalid fit boundary")

    requested_pca = (
        tuple(int(value) for value in pca_dimensions)
        if pca_dimensions is not None
        else tuple(
            int(entry["n_components"])
            for entry in summary["pca_profiles"].values()
        )
    )
    requested_pq = (
        tuple((int(m), int(nbits)) for m, nbits in pq_settings)
        if pq_settings is not None
        else tuple(
            (int(entry["m"]), int(entry["nbits"]))
            for entry in summary["pq_profiles"].values()
        )
    )

    pcas: dict[str, PCACompressor] = {}
    pca_paths: dict[str, Path] = {}
    pca_hashes: dict[str, str] = {}
    for dimension in requested_pca:
        profile = pca_profile_name(dimension)
        entry = summary["pca_profiles"].get(profile)
        if entry is None:
            raise ValueError(f"missing frozen SurvFace PCA profile: {profile}")
        path = run.run_dir / str(entry["artifact"])
        expected_hash = str(entry["sha256"])
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"SurvFace PCA artifact checksum mismatch: {path}")
        pcas[profile] = PCACompressor.load(path)
        pca_paths[profile] = path
        pca_hashes[profile] = expected_hash

    pqs: dict[str, PQCompressor] = {}
    pq_paths: dict[str, Path] = {}
    pq_hashes: dict[str, str] = {}
    for m, nbits in requested_pq:
        profile = pq_profile_name(m, nbits)
        entry = summary["pq_profiles"].get(profile)
        if entry is None:
            raise ValueError(f"missing frozen SurvFace PQ profile: {profile}")
        path = run.run_dir / str(entry["artifact"])
        expected_hash = str(entry["sha256"])
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"SurvFace PQ artifact checksum mismatch: {path}")
        compressor = PQCompressor.load(path)
        compressor.source_profile = ORIGIN_512
        compressor.fit_count = int(summary["fit_count"])
        pqs[profile] = compressor
        pq_paths[profile] = path
        pq_hashes[profile] = expected_hash

    return SurvFaceCompressorBundle(
        attempt=attempt,
        fit_count=int(summary["fit_count"]),
        pcas=pcas,
        pca_paths=pca_paths,
        pca_sha256=pca_hashes,
        pqs=pqs,
        pq_paths=pq_paths,
        pq_sha256=pq_hashes,
        summary_path=summary_path,
        summary=summary,
    )


def fit_survface_compressors(
    run: RunStore,
    engine: Engine,
    *,
    training_manifest: pd.DataFrame,
    training_manifest_path: str | Path,
    project_root: str | Path,
    pca_dimensions: Sequence[int],
    pq_settings: Sequence[tuple[int, int]],
    seed: int = 42,
    batch_size: int = 1024,
    progress: ProgressCallback | None = None,
) -> SurvFaceCompressorBundle:
    """Fit all PCA/PQ profiles once from SurvFace training development."""

    requested_pca = tuple(int(value) for value in pca_dimensions)
    requested_pq = tuple((int(m), int(bits)) for m, bits in pq_settings)
    if not requested_pca or not requested_pq:
        raise ValueError("both PCA and PQ profile sets are required")
    if len(set(requested_pca)) != len(requested_pca):
        raise ValueError("PCA dimensions must be unique")
    if len(set(requested_pq)) != len(requested_pq):
        raise ValueError("PQ settings must be unique")
    validate_survface_training_manifest(training_manifest)
    try:
        existing = load_survface_compressor_bundle(
            run,
            pca_dimensions=requested_pca,
            pq_settings=requested_pq,
        )
    except RuntimeError as exc:
        if "no completed attempt" not in str(exc):
            raise
    else:
        _emit(
            progress,
            "SurvFace compressor fit reused",
            fit_count=existing.fit_count,
            attempt=f"A{existing.attempt:03d}",
        )
        return existing

    run.verify_inputs()
    run.verify_phase_artifacts("01_official_arcface_embedding_extraction")
    root = Path(project_root).resolve()
    manifest_path = Path(training_manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    with run.phase(SURVFACE_COMPRESSOR_PHASE) as phase:
        matrix, scan_counts = _load_development_matrix(
            engine,
            run_uid=run.run_id,
            training_manifest=training_manifest,
            project_root=root,
            batch_size=int(batch_size),
            progress=progress,
        )
        if len(matrix) < max(requested_pca):
            raise ValueError(
                "SurvFace development embeddings are insufficient for the "
                f"largest PCA dimension: {len(matrix)} < {max(requested_pca)}"
            )
        total_models = len(requested_pca) + len(requested_pq)
        completed_models = 0
        pcas: dict[str, PCACompressor] = {}
        for dimension in requested_pca:
            profile = pca_profile_name(dimension)
            pcas[profile] = PCACompressor(
                n_components=dimension,
                random_state=int(seed),
            ).fit(matrix)
            completed_models += 1
            _emit(
                progress,
                "SurvFace compressor fit",
                processed=completed_models,
                total=total_models,
                profile=profile,
            )

        pqs: dict[str, PQCompressor] = {}
        for m, nbits in requested_pq:
            profile = pq_profile_name(m, nbits)
            pqs[profile] = PQCompressor(
                source_dim=512,
                m=m,
                nbits=nbits,
                source_profile=ORIGIN_512,
            ).fit(matrix)
            completed_models += 1
            _emit(
                progress,
                "SurvFace compressor fit",
                processed=completed_models,
                total=total_models,
                profile=profile,
            )

        suffix = f"A{phase.attempt:03d}"
        pca_entries: dict[str, dict[str, object]] = {}
        for profile, compressor in pcas.items():
            source = compressor.save(
                phase.attempt_dir / f"{profile}_{suffix}.joblib"
            )
            published = phase.publish_artifact(source)
            pca_entries[profile] = {
                "artifact": str(published.relative_to(run.run_dir)),
                "sha256": sha256_file(published),
                "n_components": int(compressor.n_components),
                "fit_count": int(compressor.fit_count or 0),
                "source_profile": ORIGIN_512,
                "pgvector_searchable": True,
            }

        pq_entries: dict[str, dict[str, object]] = {}
        for profile, compressor in pqs.items():
            source = compressor.save(
                phase.attempt_dir / f"{profile}_{suffix}.faiss"
            )
            published = phase.publish_artifact(source)
            pq_entries[profile] = {
                "artifact": str(published.relative_to(run.run_dir)),
                "sha256": sha256_file(published),
                "m": int(compressor.m),
                "nbits": int(compressor.nbits),
                "fit_count": int(compressor.fit_count or 0),
                "source_profile": ORIGIN_512,
                "pgvector_searchable": False,
            }

        summary = {
            "schema_version": 1,
            "fit_source": "survface_training_development",
            "fit_split": "development",
            "official_test_fit": False,
            "calibration_fit": False,
            "fit_count": int(len(matrix)),
            "source_dimension": 512,
            "seed": int(seed),
            "training_manifest": str(manifest_path),
            "training_manifest_sha256": sha256_file(manifest_path),
            "pca_profiles": pca_entries,
            "pq_profiles": pq_entries,
            "pca_to_pq": False,
            "scan_counts": scan_counts,
        }
        summary_source = (
            phase.attempt_dir
            / f"survface_compressor_summary_{suffix}.json"
        )
        summary_source.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        phase.publish_artifact(summary_source)
        phase.record_counts(
            development_vectors=len(matrix),
            pca_models=len(pcas),
            pq_models=len(pqs),
        )

    return load_survface_compressor_bundle(
        run,
        pca_dimensions=requested_pca,
        pq_settings=requested_pq,
    )


def _load_materialization_summary(run: RunStore) -> dict[str, Any]:
    attempt = _latest_completed_attempt(run, SURVFACE_MATERIALIZATION_PHASE)
    run.verify_phase_artifacts(SURVFACE_MATERIALIZATION_PHASE, attempt=attempt)
    path = (
        run.run_dir
        / "artifacts"
        / SURVFACE_MATERIALIZATION_PHASE
        / f"survface_materialization_summary_A{attempt:03d}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def materialize_survface_compressed_profiles(
    run: RunStore,
    engine: Engine,
    *,
    training_manifest: pd.DataFrame,
    project_root: str | Path,
    pca_dimensions: Sequence[int],
    pq_settings: Sequence[tuple[int, int]],
    pq_materialization_setting: tuple[int, int] = (16, 8),
    batch_size: int = 512,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Materialize PCA pgvector rows and one auxiliary PQ code profile."""

    try:
        existing = _load_materialization_summary(run)
    except RuntimeError as exc:
        if "no completed attempt" not in str(exc):
            raise
    else:
        _emit(
            progress,
            "SurvFace compressed materialization reused",
            source_vectors=existing["primary"]["counts"]["source_vectors"],
        )
        return existing

    run.verify_inputs()
    bundle = load_survface_compressor_bundle(
        run,
        pca_dimensions=pca_dimensions,
        pq_settings=pq_settings,
    )
    root = Path(project_root).resolve()
    development_paths = survface_development_image_paths(
        training_manifest,
        project_root=root,
    )
    if PCA_256 not in bundle.pcas:
        raise ValueError("pca_256 is required for the primary DB materializer")
    pq_profile = pq_profile_name(*pq_materialization_setting)
    if pq_profile not in bundle.pqs:
        raise ValueError(
            f"{pq_profile} is required for auxiliary PQ materialization"
        )

    with run.phase(SURVFACE_MATERIALIZATION_PHASE) as phase:
        suffix = f"A{phase.attempt:03d}"
        primary_measurements = (
            phase.attempt_dir
            / f"survface_primary_compression_measurements_{suffix}.csv"
        )
        primary = materialize_compressed_embeddings(
            engine,
            run_uid=run.run_id,
            pca=bundle.pcas[PCA_256],
            pq=bundle.pqs[pq_profile],
            pca_artifact_path=bundle.pca_paths[PCA_256],
            pca_artifact_sha256=bundle.pca_sha256[PCA_256],
            pq_artifact_path=bundle.pq_paths[pq_profile],
            pq_artifact_sha256=bundle.pq_sha256[pq_profile],
            development_image_paths=development_paths,
            measurements_path=primary_measurements,
            batch_size=int(batch_size),
            progress=progress,
        )

        sweep_profiles = {
            profile: compressor
            for profile, compressor in bundle.pcas.items()
            if profile != PCA_256
        }
        sweep: dict[str, Any] | None = None
        sweep_measurements: Path | None = None
        if sweep_profiles:
            sweep_measurements = (
                phase.attempt_dir
                / f"survface_pca_sweep_measurements_{suffix}.csv"
            )
            sweep = materialize_pca_sweep_embeddings(
                engine,
                run_uid=run.run_id,
                pcas=sweep_profiles,
                pca_artifact_paths={
                    profile: bundle.pca_paths[profile]
                    for profile in sweep_profiles
                },
                pca_artifact_sha256={
                    profile: bundle.pca_sha256[profile]
                    for profile in sweep_profiles
                },
                development_image_paths=development_paths,
                measurements_path=sweep_measurements,
                batch_size=int(batch_size),
                progress=progress,
            )

        summary = {
            "schema_version": 1,
            "fit_source": "survface_training_development",
            "official_test_fit": False,
            "compressor_attempt": f"A{bundle.attempt:03d}",
            "primary": primary,
            "pca_sweep": sweep,
            "pgvector_searchable_profiles": [
                ORIGIN_512,
                *bundle.pcas.keys(),
            ],
            "pq_auxiliary": {
                "profile": pq_profile,
                "stored_as": "pq_auxiliary",
                "pgvector_searchable": False,
            },
        }
        summary_source = (
            phase.attempt_dir
            / f"survface_materialization_summary_{suffix}.json"
        )
        summary_source.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        phase.publish_artifact(primary_measurements)
        if sweep_measurements is not None:
            phase.publish_artifact(sweep_measurements)
        phase.publish_artifact(summary_source)
        phase.record_counts(
            source_vectors=int(primary["counts"]["source_vectors"]),
            pca_profiles=len(bundle.pcas),
            pq_auxiliary_profiles=1,
        )
    return summary
