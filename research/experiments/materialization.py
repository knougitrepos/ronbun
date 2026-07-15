from __future__ import annotations

from collections.abc import Callable, Iterable
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from time import perf_counter
import uuid

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from research.compression import (
    ORIGIN_512,
    PCA_256,
    PQ_AUXILIARY,
    PCACompressor,
    PQCompressor,
    apply_reconstruction_error_stats,
    fit_reconstruction_error_stats,
    pca_profile_name,
)
from research.database.connection import (
    ensure_database_schema,
    ensure_vector_indexes,
    session_scope,
)
from research.database.models import (
    PCA_EMBEDDING_MODELS,
    Embedding256,
    Embedding512,
    EmbeddingPQ,
    Image,
)


ProgressCallback = Callable[[str, dict[str, object]], None]
ErrorNormalizationStats = dict[str, dict[str, float | int]]


def materialize_pca_sweep_embeddings(
    engine: Engine,
    *,
    run_uid: str,
    pcas: dict[str, PCACompressor],
    pca_artifact_paths: dict[str, str | Path],
    pca_artifact_sha256: dict[str, str],
    development_image_paths: set[str | Path],
    measurements_path: str | Path,
    batch_size: int = 512,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Materialize a reproducible multi-dimension PCA sweep into pgvector tables."""

    if not run_uid:
        raise ValueError("run_uid is required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not pcas:
        raise ValueError("pcas must not be empty")
    profiles: dict[str, PCACompressor] = {}
    for requested_profile, compressor in pcas.items():
        profile = pca_profile_name(compressor.n_components)
        if str(requested_profile) != profile:
            raise ValueError(
                f"PCA profile key {requested_profile} does not match compressor {profile}"
            )
        if profile not in pca_artifact_paths or profile not in pca_artifact_sha256:
            raise ValueError(f"missing PCA artifact provenance for {profile}")
        profiles[profile] = compressor

    ensure_database_schema(engine)
    development = {_canonical_path(path) for path in development_image_paths}
    if not development:
        raise ValueError("development_image_paths must not be empty")
    development_vectors: list[np.ndarray] = []
    source_count = 0
    for batch in _source_batches(engine, run_uid=run_uid, batch_size=batch_size):
        source_count += len(batch)
        development_vectors.extend(
            row.vector for row in batch if _canonical_path(row.image_path) in development
        )
        _emit(
            progress,
            "PCA sweep development normalization scan",
            scanned=source_count,
            development_vectors=len(development_vectors),
        )
    if not development_vectors:
        raise ValueError("no development embeddings matched the frozen manifest")
    development_matrix = np.stack(development_vectors)
    development_results = {
        profile: compressor.transform_profile(development_matrix)
        for profile, compressor in profiles.items()
    }
    error_stats = fit_reconstruction_error_stats(
        {
            profile: result.reconstruction_error
            for profile, result in development_results.items()
        }
    )
    del development_vectors, development_matrix, development_results

    destination = Path(measurements_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    fields = [
        "run_uid",
        "image_id",
        "source_embedding_id",
        "image_path",
        "identity_id",
        "compression_profile",
        "compressor_uid",
        "dimension",
        "reconstruction_error",
        "reconstruction_error_norm",
        "angular_error",
        "action",
    ]
    counts: dict[str, int] = {"source_vectors": source_count}
    for profile in profiles:
        counts[f"{profile}_inserted"] = 0
        counts[f"{profile}_skipped"] = 0
    model_uids = {
        profile: _artifact_uid(
            pca_artifact_paths[profile], pca_artifact_sha256[profile]
        )
        for profile in profiles
    }
    processed = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for batch in _source_batches(engine, run_uid=run_uid, batch_size=batch_size):
                matrix = np.stack([row.vector for row in batch])
                results = {
                    profile: compressor.transform_profile(matrix)
                    for profile, compressor in profiles.items()
                }
                normalized = apply_reconstruction_error_stats(
                    {
                        profile: result.reconstruction_error
                        for profile, result in results.items()
                    },
                    error_stats,
                )
                image_ids = [row.image_id for row in batch]
                actions_by_profile: dict[str, list[str]] = {}
                with session_scope(engine) as session:
                    for profile, compressor in profiles.items():
                        dimension = compressor.n_components
                        model = PCA_EMBEDDING_MODELS[dimension]
                        existing = {
                            int(row.image_id): row
                            for row in session.query(model).filter(
                                model.run_uid == run_uid,
                                model.vector_type == profile,
                                model.image_id.in_(image_ids),
                            )
                        }
                        profile_actions: list[str] = []
                        result = results[profile]
                        for index, source in enumerate(batch):
                            parameters = {
                                "run_uid": run_uid,
                                "compressor_uid": model_uids[profile],
                                "compressor_artifact_sha256": pca_artifact_sha256[profile],
                                "source_embedding_id": source.embedding_id,
                                "dimension": dimension,
                                "reconstruction_error": float(
                                    result.reconstruction_error[index]
                                ),
                                "reconstruction_error_norm": float(
                                    normalized[profile][index]
                                ),
                                "angular_error": float(result.angular_error[index]),
                                "error_normalization": error_stats[profile],
                            }
                            current = existing.get(source.image_id)
                            if current is None:
                                session.add(
                                    model(
                                        image_id=source.image_id,
                                        run_uid=run_uid,
                                        vector_type=profile,
                                        parameters=parameters,
                                        embedding=result.vectors[index],
                                        created_at=datetime.now(timezone.utc),
                                        log=(
                                            f"PCA {dimension}D retrieval vector; "
                                            "certification uses inverse-transformed 512D"
                                        ),
                                    )
                                )
                                action = "inserted"
                            elif current.parameters == parameters:
                                action = "skipped"
                            else:
                                raise ValueError(
                                    f"existing {profile} row has different compressor "
                                    "provenance; start a new run"
                                )
                            profile_actions.append(action)
                        actions_by_profile[profile] = profile_actions

                for profile, compressor in profiles.items():
                    result = results[profile]
                    for index, source in enumerate(batch):
                        action = actions_by_profile[profile][index]
                        counts[f"{profile}_{action}"] += 1
                        writer.writerow(
                            {
                                "run_uid": run_uid,
                                "image_id": source.image_id,
                                "source_embedding_id": source.embedding_id,
                                "image_path": source.image_path,
                                "identity_id": source.identity_id,
                                "compression_profile": profile,
                                "compressor_uid": model_uids[profile],
                                "dimension": compressor.n_components,
                                "reconstruction_error": float(
                                    result.reconstruction_error[index]
                                ),
                                "reconstruction_error_norm": float(
                                    normalized[profile][index]
                                ),
                                "angular_error": float(result.angular_error[index]),
                                "action": action,
                            }
                        )
                handle.flush()
                processed += len(batch)
                _emit(
                    progress,
                    "PCA sweep embedding batch committed",
                    processed=processed,
                    total=source_count,
                    **counts,
                )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    index_started = perf_counter()
    ensure_vector_indexes(engine)
    index_elapsed_seconds = perf_counter() - index_started
    storage_bytes: dict[str, int] = {}
    row_counts: dict[str, int] = {}
    with engine.connect() as connection:
        for profile, compressor in profiles.items():
            table_name = f"embedding_{compressor.n_components}"
            storage_bytes[profile] = int(
                connection.execute(
                    text("SELECT pg_total_relation_size(CAST(:table_name AS regclass))"),
                    {"table_name": table_name},
                ).scalar_one()
            )
            row_counts[profile] = int(
                connection.execute(
                    text(
                        f"SELECT count(1) FROM {table_name} "
                        "WHERE run_uid=:run_uid AND vector_type=:vector_type"
                    ),
                    {"run_uid": run_uid, "vector_type": profile},
                ).scalar_one()
            )
    return {
        "profiles": list(profiles),
        "counts": counts,
        "row_counts": row_counts,
        "error_normalization": error_stats,
        "measurements_path": str(destination),
        "index_ensure_elapsed_seconds": float(index_elapsed_seconds),
        "storage_bytes": storage_bytes,
        "vector_payload_bytes_per_row": {
            profile: int(compressor.n_components * np.dtype(np.float32).itemsize)
            for profile, compressor in profiles.items()
        },
        "pca_model_uids": model_uids,
    }


@dataclass(frozen=True)
class _SourceRow:
    embedding_id: int
    image_id: int
    image_path: str
    identity_id: str
    vector: np.ndarray


def _canonical_path(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).resolve()))


def _source_batches(
    engine: Engine,
    *,
    run_uid: str,
    batch_size: int,
) -> Iterable[list[_SourceRow]]:
    session_factory = sessionmaker(bind=engine, autoflush=False)
    last_id = 0
    while True:
        session = session_factory()
        try:
            rows = (
                session.query(Embedding512, Image)
                .join(Image, Embedding512.image_id == Image.id)
                .filter(
                    Embedding512.id > last_id,
                    Embedding512.run_uid == run_uid,
                    Embedding512.vector_type == ORIGIN_512,
                )
                .order_by(Embedding512.id)
                .limit(batch_size)
                .all()
            )
            batch = [
                _SourceRow(
                    embedding_id=int(embedding.id),
                    image_id=int(image.id),
                    image_path=str(image.image_path),
                    identity_id=str(image.label),
                    vector=np.asarray(embedding.embedding, dtype=np.float32),
                )
                for embedding, image in rows
            ]
        finally:
            session.close()
        if not batch:
            return
        yield batch
        last_id = batch[-1].embedding_id


def _emit(progress: ProgressCallback | None, message: str, **details: object) -> None:
    if progress is not None:
        progress(message, details)


def _artifact_uid(path: str | Path, sha256: str) -> str:
    return f"{Path(path).name}:{sha256}"


def materialize_compressed_embeddings(
    engine: Engine,
    *,
    run_uid: str,
    pca: PCACompressor,
    pq: PQCompressor,
    pca_artifact_path: str | Path,
    pca_artifact_sha256: str,
    pq_artifact_path: str | Path,
    pq_artifact_sha256: str,
    development_image_paths: set[str | Path],
    measurements_path: str | Path,
    batch_size: int = 512,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Fit error normalization on development rows, then materialize this run."""

    return _materialize_compressed_embeddings(
        engine,
        run_uid=run_uid,
        pca=pca,
        pq=pq,
        pca_artifact_path=pca_artifact_path,
        pca_artifact_sha256=pca_artifact_sha256,
        pq_artifact_path=pq_artifact_path,
        pq_artifact_sha256=pq_artifact_sha256,
        development_image_paths=development_image_paths,
        frozen_error_normalization=None,
        measurements_path=measurements_path,
        batch_size=batch_size,
        progress=progress,
    )


def materialize_compressed_embeddings_with_frozen_stats(
    engine: Engine,
    *,
    run_uid: str,
    pca: PCACompressor,
    pq: PQCompressor,
    pca_artifact_path: str | Path,
    pca_artifact_sha256: str,
    pq_artifact_path: str | Path,
    pq_artifact_sha256: str,
    error_normalization: ErrorNormalizationStats,
    measurements_path: str | Path,
    batch_size: int = 512,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Materialize a transfer/test run using frozen development-fitted statistics.

    No statistics are fit from the current run. This is the required entry point
    for SurvFace official-test transfer from LFW development artifacts.
    """

    return _materialize_compressed_embeddings(
        engine,
        run_uid=run_uid,
        pca=pca,
        pq=pq,
        pca_artifact_path=pca_artifact_path,
        pca_artifact_sha256=pca_artifact_sha256,
        pq_artifact_path=pq_artifact_path,
        pq_artifact_sha256=pq_artifact_sha256,
        development_image_paths=None,
        frozen_error_normalization=error_normalization,
        measurements_path=measurements_path,
        batch_size=batch_size,
        progress=progress,
    )


def _validate_error_normalization(
    stats: ErrorNormalizationStats,
) -> ErrorNormalizationStats:
    validated: ErrorNormalizationStats = {}
    for profile in (PCA_256, PQ_AUXILIARY):
        if profile not in stats:
            raise ValueError(f"missing frozen error normalization for {profile}")
        values = stats[profile]
        try:
            mean = float(values["mean"])
            std = float(values["std"])
            fit_count = int(values["fit_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid frozen error normalization for {profile}") from exc
        if not np.isfinite(mean) or not np.isfinite(std) or std < 0.0 or fit_count < 1:
            raise ValueError(f"invalid frozen error normalization for {profile}")
        validated[profile] = {"mean": mean, "std": std, "fit_count": fit_count}
    return validated


def _materialize_compressed_embeddings(
    engine: Engine,
    *,
    run_uid: str,
    pca: PCACompressor,
    pq: PQCompressor,
    pca_artifact_path: str | Path,
    pca_artifact_sha256: str,
    pq_artifact_path: str | Path,
    pq_artifact_sha256: str,
    development_image_paths: set[str | Path] | None,
    frozen_error_normalization: ErrorNormalizationStats | None,
    measurements_path: str | Path,
    batch_size: int,
    progress: ProgressCallback | None,
) -> dict[str, object]:
    """Materialize PCA vectors and PQ codes with development-frozen errors.

    Each batch is committed independently. A restart with identical model hashes
    skips committed rows. If a row exists with different provenance, the function
    stops instead of mixing compressor attempts inside one run.
    """

    if not run_uid:
        raise ValueError("run_uid is required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if pca.n_components != 256:
        raise ValueError("embedding_256 requires PCA n_components=256")
    ensure_database_schema(engine)

    if frozen_error_normalization is None:
        development = {
            _canonical_path(path) for path in (development_image_paths or set())
        }
        if not development:
            raise ValueError("development_image_paths must not be empty")
        development_vectors: list[np.ndarray] = []
        source_count = 0
        for batch in _source_batches(engine, run_uid=run_uid, batch_size=batch_size):
            source_count += len(batch)
            development_vectors.extend(
                row.vector
                for row in batch
                if _canonical_path(row.image_path) in development
            )
            _emit(
                progress,
                "development normalization scan",
                scanned=source_count,
                development_vectors=len(development_vectors),
            )
        if not development_vectors:
            raise ValueError("no development embeddings matched the frozen manifest")
        development_matrix = np.stack(development_vectors)
        pca_development = pca.transform_profile(development_matrix)
        pq_development = pq.transform_profile(development_matrix)
        error_stats = fit_reconstruction_error_stats(
            {
                PCA_256: pca_development.reconstruction_error,
                PQ_AUXILIARY: pq_development.reconstruction_error,
            }
        )
        del development_vectors, development_matrix, pca_development, pq_development
    else:
        error_stats = _validate_error_normalization(frozen_error_normalization)
        with engine.connect() as connection:
            source_count = int(
                connection.execute(
                    text(
                        "SELECT count(1) FROM embedding_512 "
                        "WHERE run_uid=:run_uid AND vector_type=:vector_type"
                    ),
                    {"run_uid": run_uid, "vector_type": ORIGIN_512},
                ).scalar_one()
            )
        _emit(
            progress,
            "frozen normalization accepted",
            source_vectors=source_count,
            fit_counts={
                profile: int(values["fit_count"])
                for profile, values in error_stats.items()
            },
        )
    if source_count == 0:
        raise ValueError(f"no {ORIGIN_512} source embeddings found for run {run_uid}")

    destination = Path(measurements_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    fields = [
        "run_uid",
        "image_id",
        "source_embedding_id",
        "image_path",
        "identity_id",
        "compression_profile",
        "compressor_uid",
        "reconstruction_error",
        "reconstruction_error_norm",
        "angular_error",
        "action",
    ]
    counts = {
        "source_vectors": source_count,
        "pca_inserted": 0,
        "pca_skipped": 0,
        "pq_inserted": 0,
        "pq_skipped": 0,
    }
    pca_uid = _artifact_uid(pca_artifact_path, pca_artifact_sha256)
    pq_uid = _artifact_uid(pq_artifact_path, pq_artifact_sha256)
    processed = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for batch in _source_batches(engine, run_uid=run_uid, batch_size=batch_size):
                matrix = np.stack([row.vector for row in batch])
                pca_profile = pca.transform_profile(matrix)
                pq_profile = pq.transform_profile(matrix)
                normalized = apply_reconstruction_error_stats(
                    {
                        PCA_256: pca_profile.reconstruction_error,
                        PQ_AUXILIARY: pq_profile.reconstruction_error,
                    },
                    error_stats,
                )
                image_ids = [row.image_id for row in batch]
                actions: list[tuple[str, str]] = []
                with session_scope(engine) as session:
                    pca_existing = {
                        int(row.image_id): row
                        for row in session.query(Embedding256).filter(
                            Embedding256.run_uid == run_uid,
                            Embedding256.vector_type == PCA_256,
                            Embedding256.image_id.in_(image_ids),
                        )
                    }
                    pq_existing = {
                        int(row.image_id): row
                        for row in session.query(EmbeddingPQ).filter(
                            EmbeddingPQ.run_uid == run_uid,
                            EmbeddingPQ.vector_type == PQ_AUXILIARY,
                            EmbeddingPQ.image_id.in_(image_ids),
                        )
                    }
                    for index, source in enumerate(batch):
                        pca_parameters = {
                            "run_uid": run_uid,
                            "compressor_uid": pca_uid,
                            "compressor_artifact_sha256": pca_artifact_sha256,
                            "source_embedding_id": source.embedding_id,
                            "reconstruction_error": float(
                                pca_profile.reconstruction_error[index]
                            ),
                            "reconstruction_error_norm": float(normalized[PCA_256][index]),
                            "angular_error": float(pca_profile.angular_error[index]),
                            "error_normalization": error_stats[PCA_256],
                        }
                        pq_parameters = {
                            "run_uid": run_uid,
                            "compressor_uid": pq_uid,
                            "compressor_artifact_sha256": pq_artifact_sha256,
                            "source_embedding_id": source.embedding_id,
                            "reconstruction_error": float(
                                pq_profile.reconstruction_error[index]
                            ),
                            "reconstruction_error_norm": float(
                                normalized[PQ_AUXILIARY][index]
                            ),
                            "angular_error": float(pq_profile.angular_error[index]),
                            "error_normalization": error_stats[PQ_AUXILIARY],
                        }

                        existing_pca = pca_existing.get(source.image_id)
                        if existing_pca is None:
                            session.add(
                                Embedding256(
                                    image_id=source.image_id,
                                    run_uid=run_uid,
                                    vector_type=PCA_256,
                                    parameters=pca_parameters,
                                    embedding=pca_profile.vectors[index],
                                    created_at=datetime.now(timezone.utc),
                                    log="PCA retrieval vector; certification uses inverse-transformed 512D",
                                )
                            )
                            pca_action = "inserted"
                        elif existing_pca.parameters == pca_parameters:
                            pca_action = "skipped"
                        else:
                            raise ValueError(
                                "existing PCA row has different compressor provenance; "
                                "start a new run"
                            )

                        existing_pq = pq_existing.get(source.image_id)
                        if existing_pq is None:
                            session.add(
                                EmbeddingPQ(
                                    image_id=source.image_id,
                                    run_uid=run_uid,
                                    vector_type=PQ_AUXILIARY,
                                    parameters=pq_parameters,
                                    codes=pq_profile.codes[index].tobytes(),
                                    created_at=datetime.now(timezone.utc),
                                    log="Faiss PQ auxiliary code; not pgvector-searchable",
                                )
                            )
                            pq_action = "inserted"
                        elif existing_pq.parameters == pq_parameters:
                            pq_action = "skipped"
                        else:
                            raise ValueError(
                                "existing PQ row has different compressor provenance; "
                                "start a new run"
                            )
                        actions.append((pca_action, pq_action))

                for index, source in enumerate(batch):
                    pca_action, pq_action = actions[index]
                    counts[f"pca_{pca_action}"] += 1
                    counts[f"pq_{pq_action}"] += 1
                    for profile, uid, result, error_norm, action in (
                        (
                            PCA_256,
                            pca_uid,
                            pca_profile,
                            normalized[PCA_256],
                            pca_action,
                        ),
                        (
                            PQ_AUXILIARY,
                            pq_uid,
                            pq_profile,
                            normalized[PQ_AUXILIARY],
                            pq_action,
                        ),
                    ):
                        writer.writerow(
                            {
                                "run_uid": run_uid,
                                "image_id": source.image_id,
                                "source_embedding_id": source.embedding_id,
                                "image_path": source.image_path,
                                "identity_id": source.identity_id,
                                "compression_profile": profile,
                                "compressor_uid": uid,
                                "reconstruction_error": float(
                                    result.reconstruction_error[index]
                                ),
                                "reconstruction_error_norm": float(error_norm[index]),
                                "angular_error": float(result.angular_error[index]),
                                "action": action,
                            }
                        )
                handle.flush()
                processed += len(batch)
                _emit(
                    progress,
                    "compressed embedding batch committed",
                    processed=processed,
                    total=source_count,
                    **counts,
                )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    index_started = perf_counter()
    ensure_vector_indexes(engine)
    index_ensure_elapsed_seconds = perf_counter() - index_started
    with engine.connect() as connection:
        storage_bytes = {
            table: int(
                connection.execute(
                    text(
                        "SELECT pg_total_relation_size(CAST(:table_name AS regclass))"
                    ),
                    {"table_name": table},
                ).scalar_one()
            )
            for table in ("embedding_512", "embedding_256", "embedding_pq")
        }
        row_counts = {
            "embedding_512": int(
                connection.execute(
                    text(
                        "SELECT count(1) FROM embedding_512 "
                        "WHERE run_uid=:run_uid AND vector_type=:vector_type"
                    ),
                    {"run_uid": run_uid, "vector_type": ORIGIN_512},
                ).scalar_one()
            ),
            "embedding_256": int(
                connection.execute(
                    text(
                        "SELECT count(1) FROM embedding_256 "
                        "WHERE run_uid=:run_uid AND vector_type=:vector_type"
                    ),
                    {"run_uid": run_uid, "vector_type": PCA_256},
                ).scalar_one()
            ),
            "embedding_pq": int(
                connection.execute(
                    text(
                        "SELECT count(1) FROM embedding_pq "
                        "WHERE run_uid=:run_uid AND vector_type=:vector_type"
                    ),
                    {"run_uid": run_uid, "vector_type": PQ_AUXILIARY},
                ).scalar_one()
            ),
        }
    return {
        "counts": counts,
        "row_counts": row_counts,
        "error_normalization": error_stats,
        "measurements_path": str(destination),
        "index_ensure_elapsed_seconds": float(index_ensure_elapsed_seconds),
        "index_measurement_note": (
            "Time spent ensuring indexes exist; this is not a clean HNSW build-time "
            "measurement when indexes were created before row insertion."
        ),
        "storage_bytes": storage_bytes,
        "pca_model_uid": pca_uid,
        "pq_model_uid": pq_uid,
    }
