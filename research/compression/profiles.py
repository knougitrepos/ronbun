from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter
import tempfile

import joblib
import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class CompressionProfileSpec:
    name: str
    family: str
    source_profile: str | None
    source_dimension: int
    output_dimension: int | None
    pgvector_searchable: bool
    description: str
    active: bool = True


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


ORIGIN_EMBEDDING_DIMENSION = 512
ORIGIN_512 = "origin_512"
PCA_32 = "pca_32"
PCA_64 = "pca_64"
PCA_128 = "pca_128"
PCA_256 = "pca_256"
PCA_384 = "pca_384"
# Kept only so historical fallback/search artifacts can still be identified.
# It is intentionally excluded from the active Step-1 PCA family below.
PCA_448 = "pca_448"
# Step-1 PQ is fitted directly on the original 512D embedding.  The historical
# name remains available for old DB artifacts, but is not an active family.
PQ_ORIGIN_512 = "pq_origin_512"
PQ_AUXILIARY = "pq_auxiliary"

PCA_PROFILE_DIMENSIONS = {
    PCA_384: 384,
    PCA_256: 256,
    PCA_128: 128,
    PCA_64: 64,
    PCA_32: 32,
}
PCA_SWEEP_PROFILES = (PCA_384, PCA_256, PCA_128, PCA_64, PCA_32)
PCA_SWEEP_DIMENSIONS = tuple(
    PCA_PROFILE_DIMENSIONS[profile] for profile in PCA_SWEEP_PROFILES
)

# All active Step-1 PCA profiles are materializable and searchable in pgvector.
# PCA-448 remains available only for historical run replay.
CURRENT_DB_PCA_DIMENSIONS = frozenset({32, 64, 128, 256, 384, 448})

LEGACY_PCA_PROFILE_DIMENSIONS = {PCA_448: 448}


COMPRESSION_PROFILES = {
    ORIGIN_512: CompressionProfileSpec(
        name=ORIGIN_512,
        family="origin",
        source_profile=None,
        source_dimension=ORIGIN_EMBEDDING_DIMENSION,
        output_dimension=ORIGIN_EMBEDDING_DIMENSION,
        pgvector_searchable=True,
        description="Original full-precision 512D face embedding.",
    ),
    **{
        profile: CompressionProfileSpec(
            name=profile,
            family="pca",
            source_profile=ORIGIN_512,
            source_dimension=ORIGIN_EMBEDDING_DIMENSION,
            output_dimension=dimension,
            pgvector_searchable=dimension in CURRENT_DB_PCA_DIMENSIONS,
            description=(
                f"Independent PCA {dimension}D projection fitted directly on "
                "original 512D embeddings."
            ),
        )
        for profile, dimension in PCA_PROFILE_DIMENSIONS.items()
    },
    PQ_ORIGIN_512: CompressionProfileSpec(
        name=PQ_ORIGIN_512,
        family="pq",
        source_profile=ORIGIN_512,
        source_dimension=ORIGIN_EMBEDDING_DIMENSION,
        output_dimension=None,
        pgvector_searchable=False,
        description=(
            "Independent Faiss PQ codes fitted directly on original 512D "
            "embeddings; PCA-to-PQ chaining is not permitted."
        ),
    ),
    # Legacy descriptors are inactive and are not included in either Step-1
    # sweep.  They remain recognizable to the pre-existing DB/fallback path.
    PCA_448: CompressionProfileSpec(
        name=PCA_448,
        family="pca",
        source_profile=ORIGIN_512,
        source_dimension=ORIGIN_EMBEDDING_DIMENSION,
        output_dimension=448,
        pgvector_searchable=True,
        description="Legacy PCA 448D profile retained for historical artifacts.",
        active=False,
    ),
    PQ_AUXILIARY: CompressionProfileSpec(
        name=PQ_AUXILIARY,
        family="pq",
        source_profile=ORIGIN_512,
        source_dimension=ORIGIN_EMBEDDING_DIMENSION,
        output_dimension=None,
        pgvector_searchable=False,
        description=(
            "Legacy name for direct-origin Faiss PQ auxiliary artifacts; not "
            "part of the active Step-1 family."
        ),
        active=False,
    ),
}


def pca_profile_name(n_components: int, *, allow_legacy: bool = False) -> str:
    profile = f"pca_{int(n_components)}"
    supported = dict(PCA_PROFILE_DIMENSIONS)
    if allow_legacy:
        supported.update(LEGACY_PCA_PROFILE_DIMENSIONS)
    if profile not in supported:
        raise ValueError(
            f"unsupported PCA dimension {n_components}; "
            f"expected one of {sorted(supported.values())}"
        )
    return profile


def pca_profile_dimension(profile: str, *, allow_legacy: bool = False) -> int:
    supported = dict(PCA_PROFILE_DIMENSIONS)
    if allow_legacy:
        supported.update(LEGACY_PCA_PROFILE_DIMENSIONS)
    try:
        return supported[str(profile)]
    except KeyError as exc:
        raise ValueError(f"unsupported PCA profile: {profile}") from exc


def pq_profile_name(m: int, nbits: int) -> str:
    """Return the unique direct-origin PQ budget name used by run artifacts."""

    if isinstance(m, bool) or isinstance(nbits, bool):
        raise ValueError("PQ m and nbits must be positive integers")
    m_value = int(m)
    nbits_value = int(nbits)
    if m_value != m or nbits_value != nbits or m_value < 1 or nbits_value < 1:
        raise ValueError("PQ m and nbits must be positive integers")
    return f"pq_{ORIGIN_EMBEDDING_DIMENSION}_m{m_value}_b{nbits_value}"


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
            # Low-level PCA output remains a generic dense float32 vector.
            # Whether a named Step-1 profile has a physical PostgreSQL table is
            # enforced by the DB/materialization boundary, not this codec.
            pgvector_searchable=True,
            metadata={
                "method": "pca",
                "family": "pca",
                "source_profile": ORIGIN_512,
                "chained_from": None,
                "n_components": self.n_components,
                "source_dim": int(source.shape[1]),
                "output_dtype": "float32",
                "storage_bytes_per_vector": int(
                    self.n_components * np.dtype(np.float32).itemsize
                ),
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


def _pq_codebook_storage(index) -> tuple[int, str]:
    """Return PQ centroid storage without depending on a Faiss SWIG layout."""

    source_dim = int(index.d)
    nbits = int(index.pq.nbits)
    formula_bytes = int(
        source_dim * (1 << nbits) * np.dtype(np.float32).itemsize
    )
    try:
        import faiss  # type: ignore

        centroids = faiss.vector_to_array(index.pq.centroids)
        measured_bytes = int(np.asarray(centroids).nbytes)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return formula_bytes, "float32_formula"
    if measured_bytes <= 0:
        return formula_bytes, "float32_formula"
    return measured_bytes, "faiss_centroids"


class PQCompressor:
    """Low-level product quantizer retained for legacy and Step-1 artifacts.

    This class remains dimension-generic so historical indexes can be loaded.
    Step-1's direct-origin boundary is enforced by ``fit_pq_origin_profile`` and
    the experiment config, rather than by this low-level codec.
    """

    def __init__(
        self,
        source_dim: int = ORIGIN_EMBEDDING_DIMENSION,
        m: int = 16,
        nbits: int = 8,
        *,
        source_profile: str | None = None,
    ):
        self.source_dim = int(source_dim)
        self.m = int(m)
        self.nbits = int(nbits)
        self.source_profile = str(source_profile) if source_profile is not None else None
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

    def search_adc(
        self,
        queries: np.ndarray,
        gallery_codes: np.ndarray,
        *,
        top_k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run exhaustive asymmetric squared-L2 search over stored PQ codes."""

        distances, indices, _ = self.search_adc_with_metrics(
            queries,
            gallery_codes,
            top_k=top_k,
        )
        return distances, indices

    def search_adc_with_metrics(
        self,
        queries: np.ndarray,
        gallery_codes: np.ndarray,
        *,
        top_k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
        """Run ADC and report clone, add-code, and query-search wall time.

        Queries stay in the original float32 space. Gallery vectors remain PQ
        codes inside the cloned Faiss index; this method does not decode them
        before search and does not mutate the fitted codec index.
        """

        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Faiss is required for PQ ADC search") from exc
        query_matrix = _as_float_matrix(queries)
        if query_matrix.shape[1] != self.source_dim:
            raise ValueError(
                f"PQ ADC expected {self.source_dim} query dimensions, "
                f"got {query_matrix.shape[1]}"
            )
        codes = np.asarray(gallery_codes)
        index = self._require_fit()
        expected_code_size = int(index.sa_code_size())
        if codes.ndim != 2 or codes.shape[0] == 0:
            raise ValueError("gallery_codes must be a non-empty 2D array")
        if codes.shape[1] != expected_code_size:
            raise ValueError(
                f"PQ ADC expected code size {expected_code_size}, got {codes.shape[1]}"
            )
        if codes.dtype != np.uint8:
            raise ValueError("gallery_codes must use uint8 Faiss standalone codes")
        top_k_value = int(top_k)
        if isinstance(top_k, (bool, np.bool_)) or top_k_value <= 0:
            raise ValueError("top_k must be a positive integer")
        if top_k_value > len(codes):
            raise ValueError("top_k must not exceed the gallery code count")
        if int(index.metric_type) != int(faiss.METRIC_L2):
            raise ValueError("PQ ADC primary search requires a squared-L2 codebook")

        clone_started = perf_counter()
        search_index = faiss.clone_index(index)
        search_index.reset()
        clone_elapsed = perf_counter() - clone_started
        add_started = perf_counter()
        search_index.add_sa_codes(np.ascontiguousarray(codes))
        add_elapsed = perf_counter() - add_started
        if int(search_index.ntotal) != len(codes):
            raise RuntimeError("Faiss PQ ADC index did not retain every gallery code")
        search_started = perf_counter()
        distances, indices = search_index.search(
            np.ascontiguousarray(query_matrix),
            top_k_value,
        )
        search_elapsed = perf_counter() - search_started
        distances = np.asarray(distances, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int64)
        if distances.shape != indices.shape or distances.shape != (
            len(query_matrix),
            top_k_value,
        ):
            raise RuntimeError("Faiss PQ ADC search returned an invalid result shape")
        if not np.all(np.isfinite(distances)) or np.any(indices < 0):
            raise RuntimeError("Faiss PQ ADC search returned invalid distances or indices")
        query_count = int(len(query_matrix))
        metrics: dict[str, float | int | str] = {
            "latency_measurement_repeats": 1,
            "latency_timer": "time.perf_counter",
            "compressed_index_build_latency_ms": float(clone_elapsed * 1000.0),
            "compressed_gallery_add_latency_ms": float(add_elapsed * 1000.0),
            "compressed_search_latency_ms_total": float(search_elapsed * 1000.0),
            "compressed_search_latency_ms_per_query": float(
                search_elapsed * 1000.0 / query_count
            ),
            "compressed_search_queries_per_second": float(
                query_count / search_elapsed if search_elapsed > 0.0 else np.inf
            ),
        }
        return distances, indices, metrics

    def transform_profile(self, vectors: np.ndarray) -> CompressionResult:
        source = _as_float_matrix(vectors)
        codes = self.encode(source)
        reconstructed = self.decode(codes)
        reconstruction_error = np.mean((source - reconstructed) ** 2, axis=1)
        index = self._require_fit()
        code_bits = int(self.m * self.nbits)
        code_bytes = int((code_bits + 7) // 8)
        codebook_bytes, codebook_bytes_source = _pq_codebook_storage(index)
        direct_origin = (
            self.source_profile == ORIGIN_512
            and self.source_dim == ORIGIN_EMBEDDING_DIMENSION
        )
        return CompressionResult(
            profile_name=(
                pq_profile_name(self.m, self.nbits)
                if direct_origin
                else f"pq_m{self.m}_b{self.nbits}"
            ),
            vectors=reconstructed,
            reconstruction_error=reconstruction_error.astype(np.float32),
            angular_error=angular_error(source, reconstructed),
            pgvector_searchable=False,
            metadata={
                "method": "faiss_pq",
                "family": "pq",
                "family_profile": PQ_ORIGIN_512 if direct_origin else PQ_AUXILIARY,
                "source_profile": self.source_profile,
                "chained_from": None,
                "M": self.m,
                "nbits": self.nbits,
                "code_bits": code_bits,
                "code_bytes": code_bytes,
                "codebook_bytes": codebook_bytes,
                "codebook_bytes_source": codebook_bytes_source,
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
        metadata={
            "method": "none",
            "family": "origin",
            "source_dim": int(matrix.shape[1]),
            "output_dtype": "float32",
            "storage_bytes_per_vector": int(
                matrix.shape[1] * np.dtype(np.float32).itemsize
            ),
        },
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


def fit_pca_family(
    development_vectors: np.ndarray,
    *,
    dimensions: Iterable[int] = PCA_SWEEP_DIMENSIONS,
    random_state: int | None = None,
) -> dict[str, PCACompressor]:
    """Fit independent PCA models on the same original 512D development matrix."""

    matrix = _as_float_matrix(development_vectors)
    if matrix.shape[1] != ORIGIN_EMBEDDING_DIMENSION:
        raise ValueError(
            f"PCA family requires original {ORIGIN_EMBEDDING_DIMENSION}D embeddings, "
            f"got {matrix.shape[1]}"
        )
    requested = tuple(int(value) for value in dimensions)
    if not requested:
        raise ValueError("PCA family dimensions must not be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("PCA family dimensions must be unique")

    compressors: dict[str, PCACompressor] = {}
    for dimension in requested:
        profile = pca_profile_name(dimension)
        compressors[profile] = PCACompressor(
            n_components=dimension,
            random_state=random_state,
        ).fit(matrix)
    return compressors


def fit_pq_origin_profile(
    vectors: np.ndarray,
    *,
    m: int = 16,
    nbits: int = 8,
    source_profile: str = ORIGIN_512,
) -> CompressionResult:
    matrix = _as_float_matrix(vectors)
    if str(source_profile) != ORIGIN_512:
        raise ValueError(
            "Step-1 PQ must use origin_512 directly; PCA-to-PQ chaining is not supported"
        )
    if matrix.shape[1] != ORIGIN_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Step-1 direct-origin PQ requires {ORIGIN_EMBEDDING_DIMENSION} dimensions, "
            f"got {matrix.shape[1]}"
        )
    compressor = PQCompressor(
        source_dim=matrix.shape[1],
        m=m,
        nbits=nbits,
        source_profile=source_profile,
    ).fit(matrix)
    return compressor.transform_profile(matrix)


def fit_pq_auxiliary_profile(
    vectors: np.ndarray,
    *,
    m: int = 16,
    nbits: int = 8,
) -> CompressionResult:
    """Dimension-generic compatibility wrapper for historical experiments."""

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
