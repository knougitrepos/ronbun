import numpy as np
import pytest

from research.compression.profiles import (
    COMPRESSION_PROFILES,
    PCA_PROFILE_DIMENSIONS,
    PCA_SWEEP_PROFILES,
    PQ_AUXILIARY,
    apply_reconstruction_error_stats,
    fit_reconstruction_error_stats,
    fit_pca_profile,
    fit_pq_auxiliary_profile,
    normalize_reconstruction_error_by_profile,
    original_profile,
    pca_profile_dimension,
    pca_profile_name,
)


def test_pca_sweep_profiles_have_fixed_pgvector_dimensions():
    assert PCA_SWEEP_PROFILES == ("pca_448", "pca_384", "pca_256", "pca_128")
    assert set(PCA_PROFILE_DIMENSIONS.values()) == {128, 256, 384, 448}
    for profile, dimension in PCA_PROFILE_DIMENSIONS.items():
        assert COMPRESSION_PROFILES[profile].pgvector_searchable is True
        assert pca_profile_name(dimension) == profile
        assert pca_profile_dimension(profile) == dimension


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
    assert pca.pgvector_searchable is True
    assert pca.vectors.shape == (4, 2)
    assert pca.reconstruction_error.shape == (4,)
    assert np.all(pca.reconstruction_error >= 0.0)


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


def test_pq_profile_is_auxiliary_not_pgvector_searchable():
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.0, 1.0],
            [0.1, 0.0, 0.0, 0.9],
        ],
        dtype=np.float32,
    )

    try:
        import faiss  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="Faiss is required for the PQ baseline"):
            fit_pq_auxiliary_profile(vectors, m=1, nbits=2)
        return

    pq = fit_pq_auxiliary_profile(vectors, m=1, nbits=2)

    assert COMPRESSION_PROFILES[PQ_AUXILIARY].pgvector_searchable is False
    assert pq.pgvector_searchable is False
    assert pq.codes is not None
    assert pq.reconstruction_error.shape == (8,)


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
