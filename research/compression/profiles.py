from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import joblib
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
    angular_error: np.ndarray
    pgvector_searchable: bool
    metadata: dict
    codes: np.ndarray | None = None
    reconstructed_vectors: np.ndarray | None = None


ORIGIN_512 = "origin_512"
PCA_256 = "pca_256"
PQ_AUXILIARY = "pq_auxiliary"


COMPRESSION_PROFILES = {
    ORIGIN_512: CompressionProfileSpec(
        name=ORIGIN_512,
        pgvector_searchable=True,
        description="Original ArcFace vector stored as pgvector vector.",
    ),
    PCA_256: CompressionProfileSpec(
        name=PCA_256,
        pgvector_searchable=True,
        description="PCA retrieval vector stored as pgvector vector.",
    ),
    PQ_AUXILIARY: CompressionProfileSpec(
        name=PQ_AUXILIARY,
        pgvector_searchable=False,
        description="Faiss PQ codes stored as auxiliary artifacts, not pgvector vectors.",
    ),
}


def _as_float_matrix(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("vectors must contain only finite values")
    return matrix


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("vectors must be non-zero to compute angular error")
    return matrix / norms


def angular_error(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    source = _as_float_matrix(original)
    restored = _as_float_matrix(reconstructed)
    if source.shape != restored.shape:
        raise ValueError("original and reconstructed vectors must have the same shape")
    source_unit = _row_normalize(source)
    restored_unit = _row_normalize(restored)
    cosines = np.sum(source_unit * restored_unit, axis=1)
    return np.arccos(np.clip(cosines, -1.0, 1.0)).astype(np.float32)


def _atomic_joblib_dump(payload, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        joblib.dump(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class PCACompressor:
    def __init__(self, n_components: int = 256, random_state: int | None = None):
        self.n_components = int(n_components)
        self.random_state = random_state
        self.model: PCA | None = None
        self.fit_count: int | None = None
        self.source_dim: int | None = None

    def fit(self, development_vectors: np.ndarray) -> "PCACompressor":
        matrix = _as_float_matrix(development_vectors)
        if self.n_components < 1 or self.n_components > min(matrix.shape):
            raise ValueError("n_components must be between 1 and min(n_samples, n_features)")
        self.model = PCA(n_components=self.n_components, random_state=self.random_state)
        self.model.fit(matrix)
        self.fit_count = int(matrix.shape[0])
        self.source_dim = int(matrix.shape[1])
        return self

    def _require_fit(self) -> PCA:
        if self.model is None:
            raise ValueError("PCA compressor has not been fit")
        return self.model

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        matrix = _as_float_matrix(vectors)
        return self._require_fit().transform(matrix).astype(np.float32)

    def inverse_transform(self, retrieval_vectors: np.ndarray) -> np.ndarray:
        matrix = _as_float_matrix(retrieval_vectors)
        return self._require_fit().inverse_transform(matrix).astype(np.float32)

    def transform_profile(self, vectors: np.ndarray) -> CompressionResult:
        source = _as_float_matrix(vectors)
        retrieval = self.transform(source)
        reconstructed = self.inverse_transform(retrieval)
        reconstruction_error = np.mean((source - reconstructed) ** 2, axis=1)
        model = self._require_fit()
        return CompressionResult(
            profile_name=f"pca_{self.n_components}",
            vectors=retrieval,
            reconstruction_error=reconstruction_error.astype(np.float32),
            angular_error=angular_error(source, reconstructed),
            pgvector_searchable=True,
            metadata={
                "method": "pca",
                "n_components": self.n_components,
                "source_dim": int(source.shape[1]),
                "fit_count": self.fit_count,
                "explained_variance_ratio_sum": float(np.sum(model.explained_variance_ratio_)),
                "retrieval_space": f"pca_{self.n_components}",
                "certificate_space": f"reconstructed_{source.shape[1]}",
            },
            reconstructed_vectors=reconstructed,
        )

    def save(self, path: str | Path) -> Path:
        model = self._require_fit()
        return _atomic_joblib_dump(
            {
                "kind": "pca",
                "n_components": self.n_components,
                "random_state": self.random_state,
                "fit_count": self.fit_count,
                "source_dim": self.source_dim,
                "model": model,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PCACompressor":
        payload = joblib.load(path)
        if payload.get("kind") != "pca":
            raise ValueError("artifact is not a PCA compressor")
        compressor = cls(payload["n_components"], payload.get("random_state"))
        compressor.model = payload["model"]
        compressor.fit_count = payload.get("fit_count")
        compressor.source_dim = payload.get("source_dim")
        return compressor


class PQCompressor:
    def __init__(self, source_dim: int, m: int = 16, nbits: int = 8):
        self.source_dim = int(source_dim)
        self.m = int(m)
        self.nbits = int(nbits)
        self.index = None
        self.fit_count: int | None = None

    def fit(self, development_vectors: np.ndarray) -> "PQCompressor":
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Faiss is required for the PQ baseline") from exc
        matrix = _as_float_matrix(development_vectors)
        if matrix.shape[1] != self.source_dim:
            raise ValueError(f"PQ expected {self.source_dim} dimensions, got {matrix.shape[1]}")
        if self.source_dim % self.m != 0:
            raise ValueError("PQ source_dim must be divisible by m")
        index = faiss.IndexPQ(self.source_dim, self.m, self.nbits)
        try:
            index.train(np.ascontiguousarray(matrix))
        except Exception as exc:
            raise RuntimeError(
                f"Faiss PQ training failed for d={self.source_dim}, m={self.m}, nbits={self.nbits}"
            ) from exc
        if not index.is_trained:
            raise RuntimeError("Faiss PQ did not report a trained index")
        self.index = index
        self.fit_count = int(matrix.shape[0])
        return self

    def _require_fit(self):
        if self.index is None or not self.index.is_trained:
            raise ValueError("PQ compressor has not been fit")
        return self.index

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        matrix = _as_float_matrix(vectors)
        if matrix.shape[1] != self.source_dim:
            raise ValueError(f"PQ expected {self.source_dim} dimensions, got {matrix.shape[1]}")
        return self._require_fit().sa_encode(np.ascontiguousarray(matrix))

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return self._require_fit().sa_decode(np.ascontiguousarray(codes)).astype(np.float32)

    def transform_profile(self, vectors: np.ndarray) -> CompressionResult:
        source = _as_float_matrix(vectors)
        codes = self.encode(source)
        reconstructed = self.decode(codes)
        reconstruction_error = np.mean((source - reconstructed) ** 2, axis=1)
        return CompressionResult(
            profile_name=f"pq_m{self.m}_b{self.nbits}",
            vectors=reconstructed,
            reconstruction_error=reconstruction_error.astype(np.float32),
            angular_error=angular_error(source, reconstructed),
            pgvector_searchable=False,
            metadata={
                "method": "faiss_pq",
                "M": self.m,
                "nbits": self.nbits,
                "source_dim": self.source_dim,
                "fit_count": self.fit_count,
            },
            codes=codes,
            reconstructed_vectors=reconstructed,
        )

    def save(self, path: str | Path) -> Path:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Faiss is required for the PQ baseline") from exc
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            faiss.write_index(self._require_fit(), str(temporary))
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "PQCompressor":
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Faiss is required for the PQ baseline") from exc
        index = faiss.read_index(str(path))
        compressor = cls(index.d, index.pq.M, index.pq.nbits)
        compressor.index = index
        return compressor


def original_profile(vectors: np.ndarray) -> CompressionResult:
    matrix = _as_float_matrix(vectors)
    return CompressionResult(
        profile_name=ORIGIN_512,
        vectors=matrix.copy(),
        reconstruction_error=np.zeros(len(matrix), dtype=np.float32),
        angular_error=np.zeros(len(matrix), dtype=np.float32),
        pgvector_searchable=True,
        metadata={"method": "none", "source_dim": int(matrix.shape[1])},
        reconstructed_vectors=matrix.copy(),
    )


def fit_pca_profile(
    vectors: np.ndarray,
    *,
    n_components: int = 256,
    random_state: int | None = None,
) -> CompressionResult:
    compressor = PCACompressor(n_components=n_components, random_state=random_state).fit(vectors)
    return compressor.transform_profile(vectors)


def fit_pq_auxiliary_profile(
    vectors: np.ndarray,
    *,
    m: int = 16,
    nbits: int = 8,
) -> CompressionResult:
    matrix = _as_float_matrix(vectors)
    compressor = PQCompressor(source_dim=matrix.shape[1], m=m, nbits=nbits).fit(matrix)
    return compressor.transform_profile(matrix)


def normalize_reconstruction_error_by_profile(
    errors_by_profile: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    stats = fit_reconstruction_error_stats(errors_by_profile)
    return apply_reconstruction_error_stats(errors_by_profile, stats)


def fit_reconstruction_error_stats(
    development_errors_by_profile: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    """Fit profile-specific normalization statistics on development data only."""

    stats: dict[str, dict[str, float | int]] = {}
    for profile, values in development_errors_by_profile.items():
        errors = np.asarray(values, dtype=np.float32)
        if errors.ndim != 1 or len(errors) == 0 or not np.all(np.isfinite(errors)):
            raise ValueError(f"development reconstruction errors are invalid for {profile}")
        stats[str(profile)] = {
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),
            "fit_count": int(len(errors)),
        }
    return stats


def apply_reconstruction_error_stats(
    errors_by_profile: dict[str, np.ndarray],
    stats: dict[str, dict[str, float | int]],
) -> dict[str, np.ndarray]:
    """Apply frozen development statistics without refitting on calibration/test."""

    normalized: dict[str, np.ndarray] = {}
    for profile, values in errors_by_profile.items():
        if profile not in stats:
            raise ValueError(f"missing reconstruction error statistics for {profile}")
        errors = np.asarray(values, dtype=np.float32)
        if errors.ndim != 1 or not np.all(np.isfinite(errors)):
            raise ValueError(f"reconstruction errors are invalid for {profile}")
        mean = float(stats[profile]["mean"])
        std = float(stats[profile]["std"])
        if not np.isfinite(mean) or not np.isfinite(std) or std < 0.0:
            raise ValueError(f"invalid reconstruction error statistics for {profile}")
        if std == 0.0:
            normalized[profile] = np.zeros_like(errors, dtype=np.float32)
        else:
            normalized[profile] = ((errors - mean) / std).astype(np.float32)
    return normalized
