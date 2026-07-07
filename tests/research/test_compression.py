import numpy as np

from research.compression.profiles import (
    COMPRESSION_PROFILES,
    fit_pca_profile,
    fit_pq_auxiliary_profile,
    normalize_reconstruction_error_by_profile,
    original_profile,
)


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

    pq = fit_pq_auxiliary_profile(vectors, m=1, nbits=2)

    assert COMPRESSION_PROFILES["pq"].pgvector_searchable is False
    assert pq.pgvector_searchable is False
    assert pq.codes is not None
    assert pq.reconstruction_error.shape == (8,)
