import numpy as np
import pandas as pd

from research.search.open_set import CALIBRATION_FEATURE_COLUMNS, build_search_features


def test_build_search_features_has_stable_schema_for_registered_and_unknown_probes():
    templates = pd.DataFrame(
        {
            "identity_id": ["a", "b"],
            "embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            "quality": [0.9, 0.8],
            "variance": [0.01, 0.02],
            "enrollment_count": [2, 1],
        }
    )
    probes = pd.DataFrame(
        {
            "image_id": ["qa", "qu"],
            "identity_id": ["a", "u"],
            "probe_type": ["registered", "known_unknown"],
            "embedding": [np.array([0.95, 0.05]), np.array([0.6, 0.4])],
            "quality": [0.7, 0.5],
            "reconstruction_error_norm": [0.0, 1.5],
        }
    )

    features = build_search_features(probes, templates, compression_profile="pca_2", top_k=2)

    assert list(features["query_id"]) == ["qa", "qu"]
    assert features.loc[0, "top1_identity"] == "a"
    assert features.loc[0, "y_true_accept"] == 1
    assert features.loc[1, "y_true_accept"] == 0
    assert features.loc[0, "score_margin"] > 0.0
    assert set(CALIBRATION_FEATURE_COLUMNS).issubset(features.columns)
    assert set(features["compression_profile"]) == {"pca_2"}
