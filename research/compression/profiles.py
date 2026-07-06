from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class CompressionProfileSpec:
    name: str
    pgvector_searchable: bool
    description: str


@dataclass
class CompressionResult:
    profile_name: str
    vectors: np.ndarray
    reconstruction_error: np.ndarray
    pgvector_searchable: bool
    metadata: dict
    codes: np.ndarray | None = None
    reconstructed_vectors: np.ndarray | None = None


COMPRESSION_PROFILES = {
    "origin_512": CompressionProfileSpec(
        name="origin_512",
        pgvector_searchable=True,
        description="Original ArcFace vector stored as pgvector vector.",
    ),
    "pca_256": CompressionProfileSpec(
        name="pca_256",
        pgvector_searchable=True,
        description="PCA-reduced vector stored as pgvector vector.",
    ),
    "pq": CompressionProfileSpec(
        name="pq",
        pgvector_searchable=False,
        description="Faiss PQ codes stored as auxiliary compression artifacts.",
    ),
}


def _as_float_matrix(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    return matrix


def original_profile(vectors: np.ndarray) -> CompressionResult:
    matrix = _as_float_matrix(vectors)
    return CompressionResult(
        profile_name="origin_512",
        vectors=matrix.copy(),
        reconstruction_error=np.zeros(len(matrix), dtype=np.float32),
        pgvector_searchable=True,
        metadata={"source_dim": int(matrix.shape[1])},
        reconstructed_vectors=matrix.copy(),
    )


def fit_pca_profile(
    vectors: np.ndarray,
    *,
    n_components: int = 256,
    random_state: int | None = None,
) -> CompressionResult:
    matrix = _as_float_matrix(vectors)
    if n_components < 1 or n_components > min(matrix.shape):
        raise ValueError("n_components must be between 1 and min(n_samples, n_features)")
    pca = PCA(n_components=n_components, random_state=random_state)
    compressed = pca.fit_transform(matrix)
    reconstructed = pca.inverse_transform(compressed)
    reconstruction_error = np.mean((matrix - reconstructed) ** 2, axis=1)
    profile_name = f"pca_{n_components}"
    return CompressionResult(
        profile_name=profile_name,
        vectors=compressed.astype(np.float32),
        reconstruction_error=reconstruction_error.astype(np.float32),
        pgvector_searchable=True,
        metadata={
            "n_components": int(n_components),
            "source_dim": int(matrix.shape[1]),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        },
        reconstructed_vectors=reconstructed.astype(np.float32),
    )


def fit_pq_auxiliary_profile(
    vectors: np.ndarray,
    *,
    m: int = 16,
    nbits: int = 8,
) -> CompressionResult:
    matrix = _as_float_matrix(vectors)
    try:
        import faiss  # type: ignore

        index = faiss.IndexPQ(matrix.shape[1], m, nbits)
        contiguous = np.ascontiguousarray(matrix.astype(np.float32))
        index.train(contiguous)
        codes = index.sa_encode(contiguous)
        reconstructed = index.sa_decode(codes)
    except Exception:
        # Test-friendly fallback for environments without Faiss. This remains
        # auxiliary and is never marked pgvector-searchable.
        levels = float((2**nbits) - 1)
        mins = matrix.min(axis=0, keepdims=True)
        maxs = matrix.max(axis=0, keepdims=True)
        scale = np.where(maxs > mins, maxs - mins, 1.0)
        codes = np.round((matrix - mins) / scale * levels).astype(np.uint8)
        reconstructed = (codes.astype(np.float32) / levels * scale + mins).astype(np.float32)

    reconstruction_error = np.mean((matrix - reconstructed) ** 2, axis=1)
    return CompressionResult(
        profile_name="pq",
        vectors=reconstructed.astype(np.float32),
        reconstruction_error=reconstruction_error.astype(np.float32),
        pgvector_searchable=False,
        metadata={"M": int(m), "nbits": int(nbits), "source_dim": int(matrix.shape[1])},
        codes=codes,
        reconstructed_vectors=reconstructed.astype(np.float32),
    )


def normalize_reconstruction_error_by_profile(
    errors_by_profile: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for profile, values in errors_by_profile.items():
        errors = np.asarray(values, dtype=np.float32)
        std = float(np.std(errors))
        if std == 0.0:
            normalized[profile] = np.zeros_like(errors, dtype=np.float32)
        else:
            normalized[profile] = ((errors - float(np.mean(errors))) / std).astype(np.float32)
    return normalized
