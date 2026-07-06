import numpy as np
import pandas as pd

from research.templates.aggregation import aggregate_templates


def test_quality_weighted_templates_are_l2_normalized_and_keep_metadata():
    rows = pd.DataFrame(
        {
            "image_id": ["a1", "a2", "a3", "b1", "b2"],
            "identity_id": ["a", "a", "a", "b", "b"],
            "embedding": [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.9, 0.1, 0.0]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.9, 0.1]),
            ],
            "quality": [0.9, 0.8, 0.1, 0.7, 0.6],
        }
    )

    templates = aggregate_templates(rows, method="quality_weighted", outlier_threshold=0.5)

    assert list(templates["identity_id"]) == ["a", "b"]
    assert np.allclose(np.linalg.norm(templates.iloc[0]["embedding"]), 1.0)
    assert np.allclose(np.linalg.norm(templates.iloc[1]["embedding"]), 1.0)
    assert templates.iloc[0]["enrollment_count"] == 2
    assert templates.iloc[0]["quality"] > templates.iloc[1]["quality"]
    assert templates.iloc[0]["variance"] >= 0.0


def test_single_aggregation_selects_deterministic_first_image():
    rows = pd.DataFrame(
        {
            "image_id": ["b2", "b1"],
            "identity_id": ["b", "b"],
            "embedding": [np.array([0.0, 2.0]), np.array([1.0, 0.0])],
            "quality": [0.2, 0.9],
        }
    )

    templates = aggregate_templates(rows, method="single")

    assert templates.iloc[0]["source_image_ids"] == ["b1"]
    assert np.allclose(templates.iloc[0]["embedding"], np.array([1.0, 0.0]))
