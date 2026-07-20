import numpy as np
import pytest

from research.compression.profiles import (
    COMPRESSION_PROFILES,
    ORIGIN_512,
    PCA_PROFILE_DIMENSIONS,
    PCA_SWEEP_DIMENSIONS,
    PCA_SWEEP_PROFILES,
    PQ_AUXILIARY,
    PQ_ORIGIN_512,
    PQCompressor,
    apply_reconstruction_error_stats,
    fit_reconstruction_error_stats,
    fit_pca_family,
    fit_pca_profile,
    fit_pq_auxiliary_profile,
    fit_pq_origin_profile,
    normalize_reconstruction_error_by_profile,
    original_profile,
    pca_profile_dimension,
    pca_profile_name,
    pq_profile_name,
)


def test_step1_pca_sweep_profiles_are_independent_origin_families():
    assert PCA_SWEEP_PROFILES == (
        "pca_384",
        "pca_256",
        "pca_128",
        "pca_64",
        "pca_32",
    )
    assert PCA_SWEEP_DIMENSIONS == (384, 256, 128, 64, 32)
    assert set(PCA_PROFILE_DIMENSIONS.values()) == {32, 64, 128, 256, 384}
    for profile, dimension in PCA_PROFILE_DIMENSIONS.items():
        spec = COMPRESSION_PROFILES[profile]
        assert spec.active is True
        assert spec.family == "pca"
        assert spec.source_profile == ORIGIN_512
        assert spec.source_dimension == 512
        assert spec.output_dimension == dimension
        assert spec.pgvector_searchable is (dimension in {128, 256, 384})
        assert pca_profile_name(dimension) == profile
        assert pca_profile_dimension(profile) == dimension

    assert COMPRESSION_PROFILES["pca_448"].active is False
    with pytest.raises(ValueError, match="unsupported PCA dimension"):
        pca_profile_name(448)
    assert pca_profile_name(448, allow_legacy=True) == "pca_448"


def test_pca_family_requires_original_512d_and_fits_each_requested_profile():
    vectors = np.random.default_rng(7).normal(size=(40, 512)).astype(np.float32)

    family = fit_pca_family(vectors, dimensions=(32,), random_state=0)

    assert set(family) == {"pca_32"}
    assert family["pca_32"].source_dim == 512
    with pytest.raises(ValueError, match="requires original 512D"):
        fit_pca_family(vectors[:, :256], dimensions=(32,), random_state=0)


def test_original_and_pca_profiles_record_searchability_and_reconstruction_error():
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )

    original = original_profile(vectors)
    pca = fit_pca_profile(vectors, n_components=2, random_state=0)

    assert original.profile_name == "origin_512"
    assert original.pgvector_searchable is True
    assert pca.profile_name == "pca_2"
    # Generic PCA remains a dense pgvector-compatible result. Step-1 PCA-64/32
    # table availability is rejected by the DB/materialization boundary.
    assert pca.pgvector_searchable is True
    assert pca.vectors.shape == (4, 2)
    assert pca.reconstruction_error.shape == (4,)
    assert np.all(pca.reconstruction_error >= 0.0)
    assert original.metadata["output_dtype"] == "float32"
    assert original.metadata["storage_bytes_per_vector"] == 12
    assert pca.metadata["output_dtype"] == "float32"
    assert pca.metadata["storage_bytes_per_vector"] == 8


def test_compression_profiles_report_angular_error():
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )

    original = original_profile(vectors)
    pca = fit_pca_profile(vectors, n_components=2, random_state=0)

    assert np.allclose(original.angular_error, np.zeros(4))
    assert pca.angular_error.shape == (4,)
    assert np.all(pca.angular_error >= 0.0)
    assert np.all(pca.angular_error <= np.pi)


def test_reconstruction_error_is_normalized_per_profile():
    errors = {
        "origin_512": np.array([0.0, 0.0, 0.0]),
        "pca_2": np.array([1.0, 2.0, 3.0]),
    }

    normalized = normalize_reconstruction_error_by_profile(errors)

    assert np.allclose(normalized["origin_512"], np.zeros(3))
    assert np.allclose(normalized["pca_2"], np.array([-1.22474487, 0.0, 1.22474487]))


def test_pq_profile_is_direct_origin_family_not_pgvector_searchable():
    vectors = np.random.default_rng(9).normal(size=(32, 512)).astype(np.float32)

    spec = COMPRESSION_PROFILES[PQ_ORIGIN_512]
    assert spec.active is True
    assert spec.family == "pq"
    assert spec.source_profile == ORIGIN_512
    assert spec.source_dimension == 512
    assert spec.pgvector_searchable is False
    assert COMPRESSION_PROFILES[PQ_AUXILIARY].active is False

    # The low-level codec stays generic for historical indexes; the Step-1
    # direct-origin helper owns the stricter experiment boundary.
    assert PQCompressor(source_dim=4, m=1, nbits=2).source_dim == 4
    with pytest.raises(ValueError, match="requires 512 dimensions"):
        fit_pq_origin_profile(vectors[:, :256], m=16, nbits=2)
    with pytest.raises(ValueError, match="PCA-to-PQ chaining"):
        fit_pq_origin_profile(
            vectors,
            m=16,
            nbits=2,
            source_profile="pca_256",
        )

    try:
        import faiss  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="Faiss is required for the PQ baseline"):
            fit_pq_origin_profile(vectors, m=16, nbits=2)
        return

    pq = fit_pq_origin_profile(vectors, m=16, nbits=2)

    assert pq.profile_name == "pq_512_m16_b2"
    assert pq.profile_name == pq_profile_name(16, 2)
    assert pq.pgvector_searchable is False
    assert pq.codes is not None
    assert pq.reconstruction_error.shape == (32,)
    assert pq.metadata["source_profile"] == ORIGIN_512
    assert pq.metadata["chained_from"] is None
    assert pq.metadata["code_bits"] == 32
    assert pq.metadata["code_bytes"] == 4
    assert pq.metadata["codebook_bytes"] == 512 * (2**2) * 4
    assert pq.metadata["codebook_bytes_source"] in {
        "faiss_centroids",
        "float32_formula",
    }

    legacy_vectors = np.random.default_rng(10).normal(size=(8, 4)).astype(np.float32)
    compatibility = fit_pq_auxiliary_profile(legacy_vectors, m=1, nbits=2)
    assert compatibility.profile_name == "pq_m1_b2"
    assert compatibility.vectors.shape == (8, 4)


def test_pq_budget_names_do_not_collide_across_m_or_bit_width():
    assert pq_profile_name(16, 8) == "pq_512_m16_b8"
    names = {
        pq_profile_name(16, 8),
        pq_profile_name(32, 8),
        pq_profile_name(16, 4),
    }
    assert len(names) == 3


def test_reconstruction_error_stats_are_fit_once_and_applied_without_test_refit():
    development = {"pca_256": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    test = {"pca_256": np.array([10.0, 11.0], dtype=np.float32)}

    stats = fit_reconstruction_error_stats(development)
    normalized = apply_reconstruction_error_stats(test, stats)

    assert stats["pca_256"]["fit_count"] == 3
    assert np.allclose(normalized["pca_256"], np.array([9.797959, 11.022704]))


def test_apply_reconstruction_error_stats_rejects_missing_or_nonfinite_inputs():
    stats = fit_reconstruction_error_stats(
        {"pca_256": np.array([1.0, 2.0], dtype=np.float32)}
    )

    with pytest.raises(ValueError, match="missing reconstruction error statistics"):
        apply_reconstruction_error_stats({"pq_auxiliary": np.array([1.0])}, stats)
    with pytest.raises(ValueError, match="reconstruction errors are invalid"):
        apply_reconstruction_error_stats(
            {"pca_256": np.array([np.nan], dtype=np.float32)}, stats
        )
