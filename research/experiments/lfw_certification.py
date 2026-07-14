from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from research.compression import ORIGIN_512, PCA_256, PCACompressor
from research.database.connection import session_scope
from research.database.models import Embedding256, Embedding512, Image


PROTOCOL_ROLES = (
    "gallery",
    "registered_probes",
    "known_unknown_probes",
    "unknown_unknown_probes",
)


@dataclass(frozen=True)
class LFWCertificationInputs:
    probes: pd.DataFrame
    templates: pd.DataFrame
    coverage: dict[str, object]
    certificate_space: str


def _canonical_path(value: str | Path, project_root: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = Path(project_root) / path
    return os.path.normcase(str(path.resolve()))


def _l2(vector: object) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("embedding must be a finite one-dimensional vector")
    norm = float(np.linalg.norm(value))
    if norm == 0.0:
        raise ValueError("embedding norm must be positive")
    return (value / norm).astype(np.float32)


def _angle(first: object, second: object) -> float:
    cosine = float(np.dot(_l2(first), _l2(second)))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _validate_protocol_frames(protocol_frames: dict[str, pd.DataFrame]) -> None:
    missing_roles = set(PROTOCOL_ROLES).difference(protocol_frames)
    if missing_roles:
        raise ValueError(f"missing LFW protocol roles: {sorted(missing_roles)}")
    for role in PROTOCOL_ROLES:
        missing_columns = {"image_id", "identity_id", "image_path"}.difference(
            protocol_frames[role].columns
        )
        if missing_columns:
            raise ValueError(
                f"{role} is missing protocol columns: {sorted(missing_columns)}"
            )


def fetch_lfw_vector_records(
    engine: Engine,
    *,
    run_uid: str,
    image_paths: set[str],
    project_root: str | Path,
    compression_profile: str,
    pca: PCACompressor | None = None,
) -> pd.DataFrame:
    """Fetch exact vectors and construct certificate-space approximations.

    PCA retrieval remains stored and searched in 256D. Certification is a
    separate analysis in reconstructed 512D because its angular error must be
    measured against the original 512D ArcFace vector.
    """

    if compression_profile not in {ORIGIN_512, PCA_256}:
        raise ValueError("compression_profile must be origin_512 or pca_256")
    if compression_profile == PCA_256 and pca is None:
        raise ValueError("pca compressor is required for pca_256 certification")
    target_paths = {
        _canonical_path(path, project_root) for path in image_paths
    }

    with session_scope(engine) as session:
        origin_rows = (
            session.query(Embedding512, Image)
            .join(Image, Embedding512.image_id == Image.id)
            .filter(
                Embedding512.run_uid == run_uid,
                Embedding512.vector_type == ORIGIN_512,
            )
            .all()
        )
        origins = {
            _canonical_path(image.image_path, project_root): {
                "image_db_id": int(image.id),
                "origin_embedding": np.asarray(embedding.embedding, dtype=np.float32),
            }
            for embedding, image in origin_rows
            if _canonical_path(image.image_path, project_root) in target_paths
        }

        pca_rows: dict[str, dict[str, object]] = {}
        if compression_profile == PCA_256:
            rows = (
                session.query(Embedding256, Image)
                .join(Image, Embedding256.image_id == Image.id)
                .filter(
                    Embedding256.run_uid == run_uid,
                    Embedding256.vector_type == PCA_256,
                )
                .all()
            )
            pca_rows = {
                _canonical_path(image.image_path, project_root): {
                    "retrieval_embedding": np.asarray(
                        embedding.embedding, dtype=np.float32
                    ),
                    "parameters": dict(embedding.parameters or {}),
                }
                for embedding, image in rows
                if _canonical_path(image.image_path, project_root) in target_paths
            }

    available_paths = sorted(
        set(origins)
        if compression_profile == ORIGIN_512
        else set(origins).intersection(pca_rows)
    )
    reconstructed: dict[str, np.ndarray] = {}
    if compression_profile == PCA_256 and available_paths:
        retrieval_matrix = np.stack(
            [pca_rows[path]["retrieval_embedding"] for path in available_paths]
        )
        restored_matrix = pca.inverse_transform(retrieval_matrix)  # type: ignore[union-attr]
        reconstructed = {
            path: restored_matrix[index]
            for index, path in enumerate(available_paths)
        }

    records = []
    for path in available_paths:
        exact = _l2(origins[path]["origin_embedding"])
        if compression_profile == ORIGIN_512:
            approximate = exact.copy()
            retrieval = exact.copy()
            error_norm = 0.0
        else:
            approximate = _l2(reconstructed[path])
            retrieval = np.asarray(
                pca_rows[path]["retrieval_embedding"], dtype=np.float32
            )
            error_norm = float(
                pca_rows[path]["parameters"].get("reconstruction_error_norm", 0.0)
            )
        records.append(
            {
                "canonical_path": path,
                "image_db_id": origins[path]["image_db_id"],
                "origin_embedding": exact,
                "approximate_embedding": approximate,
                "retrieval_embedding": retrieval,
                "angular_error": _angle(exact, approximate),
                "reconstruction_error_norm": error_norm,
            }
        )
    return pd.DataFrame.from_records(records)


def assemble_lfw_certification_inputs(
    protocol_frames: dict[str, pd.DataFrame],
    vector_records: pd.DataFrame,
    *,
    project_root: str | Path,
    compression_profile: str,
    allow_empty_unknown_unknown: bool = False,
) -> LFWCertificationInputs:
    """Create leakage-safe probe/template frames after extraction filtering."""

    _validate_protocol_frames(protocol_frames)
    required_vectors = {
        "canonical_path",
        "origin_embedding",
        "approximate_embedding",
        "angular_error",
        "reconstruction_error_norm",
    }
    missing_vectors = required_vectors.difference(vector_records.columns)
    if missing_vectors:
        raise ValueError(f"vector records are missing columns: {sorted(missing_vectors)}")
    if vector_records["canonical_path"].duplicated().any():
        raise ValueError("vector records contain duplicate canonical paths")

    vectors = vector_records.set_index("canonical_path", drop=False)
    effective: dict[str, pd.DataFrame] = {}
    coverage: dict[str, object] = {"compression_profile": compression_profile}
    for role in PROTOCOL_ROLES:
        frame = protocol_frames[role].copy()
        frame["canonical_path"] = frame["image_path"].map(
            lambda value: _canonical_path(value, project_root)
        )
        available = frame["canonical_path"].isin(vectors.index)
        missing_rows = frame.loc[~available]
        effective[role] = frame.loc[available].reset_index(drop=True)
        coverage[role] = {
            "input_rows": int(len(frame)),
            "available_rows": int(available.sum()),
            "missing_rows": int((~available).sum()),
            "missing_image_ids": missing_rows["image_id"].astype(str).tolist(),
        }

    if effective["gallery"].empty:
        raise ValueError("no gallery embeddings are available")
    gallery_identities = set(effective["gallery"]["identity_id"].astype(str))
    registered = effective["registered_probes"]
    has_gallery = registered["identity_id"].astype(str).isin(gallery_identities)
    dropped_registered = registered.loc[~has_gallery]
    effective["registered_probes"] = registered.loc[has_gallery].reset_index(drop=True)
    coverage["registered_missing_gallery_identity"] = {
        "dropped_rows": int((~has_gallery).sum()),
        "dropped_identity_ids": sorted(
            set(dropped_registered["identity_id"].astype(str))
        ),
        "dropped_image_ids": dropped_registered["image_id"].astype(str).tolist(),
    }
    coverage["effective_gallery_identity_count"] = int(len(gallery_identities))
    if effective["registered_probes"].empty:
        raise ValueError("no registered probes remain for available gallery identities")
    for role in ("known_unknown_probes", "unknown_unknown_probes"):
        if role == "unknown_unknown_probes" and allow_empty_unknown_unknown:
            continue
        if effective[role].empty:
            raise ValueError(f"no {role} remain after extraction filtering")

    template_records = []
    for identity_id, group in effective["gallery"].groupby("identity_id", sort=True):
        rows = vectors.loc[group["canonical_path"].tolist()]
        exact_vectors = np.stack(rows["origin_embedding"].tolist())
        approximate_vectors = np.stack(rows["approximate_embedding"].tolist())
        retrieval_vectors = np.stack(
            rows[
                "retrieval_embedding"
                if "retrieval_embedding" in rows.columns
                else "approximate_embedding"
            ].tolist()
        )
        exact_template = _l2(exact_vectors.mean(axis=0))
        approximate_template = _l2(approximate_vectors.mean(axis=0))
        retrieval_template = _l2(retrieval_vectors.mean(axis=0))
        distances = 1.0 - np.clip(
            approximate_vectors @ approximate_template, -1.0, 1.0
        )
        template_records.append(
            {
                "identity_id": str(identity_id),
                "embedding": approximate_template,
                "retrieval_embedding": retrieval_template,
                "fallback_embedding": exact_template,
                "quality": 0.0,
                "variance": float(np.mean(distances**2)),
                "enrollment_count": int(len(group)),
                "angular_error": _angle(exact_template, approximate_template),
                "reconstruction_error_norm": float(
                    rows["reconstruction_error_norm"].astype(float).mean()
                ),
                "source_image_ids": group["image_id"].astype(str).tolist(),
            }
        )
    templates = pd.DataFrame.from_records(template_records).sort_values(
        "identity_id"
    ).reset_index(drop=True)

    probe_records = []
    for role, probe_type in (
        ("registered_probes", "registered"),
        ("known_unknown_probes", "known_unknown"),
        ("unknown_unknown_probes", "unknown_unknown"),
    ):
        for row in effective[role].itertuples(index=False):
            vector = vectors.loc[row.canonical_path]
            probe_records.append(
                {
                    "image_id": str(row.image_id),
                    "identity_id": str(row.identity_id),
                    "probe_type": probe_type,
                    "embedding": vector["approximate_embedding"],
                    "retrieval_embedding": vector.get(
                        "retrieval_embedding", vector["approximate_embedding"]
                    ),
                    "fallback_embedding": vector["origin_embedding"],
                    "quality": 0.0,
                    "angular_error": float(vector["angular_error"]),
                    "reconstruction_error_norm": float(
                        vector["reconstruction_error_norm"]
                    ),
                }
            )
    probes = pd.DataFrame.from_records(probe_records)
    coverage["effective_probe_counts"] = {
        str(probe_type): int(count)
        for probe_type, count in probes["probe_type"].value_counts().sort_index().items()
    }
    certificate_space = (
        ORIGIN_512 if compression_profile == ORIGIN_512 else "pca_reconstructed_512"
    )
    return LFWCertificationInputs(
        probes=probes,
        templates=templates,
        coverage=coverage,
        certificate_space=certificate_space,
    )


def build_lfw_certification_inputs(
    engine: Engine,
    *,
    run_uid: str,
    protocol_frames: dict[str, pd.DataFrame],
    project_root: str | Path,
    compression_profile: str,
    pca: PCACompressor | None = None,
    allow_empty_unknown_unknown: bool = False,
) -> LFWCertificationInputs:
    _validate_protocol_frames(protocol_frames)
    image_paths = {
        str(path)
        for role in PROTOCOL_ROLES
        for path in protocol_frames[role]["image_path"].tolist()
    }
    records = fetch_lfw_vector_records(
        engine,
        run_uid=run_uid,
        image_paths=image_paths,
        project_root=project_root,
        compression_profile=compression_profile,
        pca=pca,
    )
    return assemble_lfw_certification_inputs(
        protocol_frames,
        records,
        project_root=project_root,
        compression_profile=compression_profile,
        allow_empty_unknown_unknown=allow_empty_unknown_unknown,
    )


def write_vector_frame_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Atomically serialize vector/list columns for reproducible phase artifacts."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = frame.copy()
    for column in (
        "embedding",
        "retrieval_embedding",
        "fallback_embedding",
        "source_image_ids",
        "ranked_identities",
        "ranked_scores",
        "pca_exact_ranked_identities",
        "pca_hnsw_ranked_identities",
    ):
        if column in serialized.columns:
            serialized[column] = serialized[column].map(
                lambda value: json.dumps(
                    np.asarray(value).tolist() if isinstance(value, np.ndarray) else value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        serialized.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
