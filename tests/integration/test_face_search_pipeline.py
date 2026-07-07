import numpy as np
import pandas as pd

from research.calibration.rejection import LogisticRegressionCalibrator
from research.compression.profiles import COMPRESSION_PROFILES, fit_pca_profile
from research.protocol import build_open_set_protocol
from research.search.open_set import build_certified_search_features
from research.templates.aggregation import aggregate_templates


def test_synthetic_pipeline_runs_protocol_to_calibration():
    manifest = pd.DataFrame(
        {
            "image_id": ["a1", "a2", "b1", "b2", "u1", "u2", "x1", "x2"],
            "identity_id": ["a", "a", "b", "b", "u", "u", "x", "x"],
            "split": ["test"] * 8,
            "image_path": [f"{image_id}.jpg" for image_id in ["a1", "a2", "b1", "b2", "u1", "u2", "x1", "x2"]],
        }
    )
    embeddings = {
        "a1": np.array([1.0, 0.0, 0.0]),
        "a2": np.array([0.9, 0.1, 0.0]),
        "b1": np.array([0.0, 1.0, 0.0]),
        "b2": np.array([0.1, 0.9, 0.0]),
        "u1": np.array([0.6, 0.4, 0.0]),
        "u2": np.array([0.4, 0.6, 0.0]),
        "x1": np.array([0.0, 0.0, 1.0]),
        "x2": np.array([0.1, 0.0, 0.9]),
    }

    protocol = build_open_set_protocol(
        manifest,
        gallery_identities=["a", "b"],
        unknown_unknown_identities=["x"],
        enrollment_count=1,
        seed=0,
    )
    gallery = protocol.gallery.assign(
        embedding=lambda frame: frame["image_id"].map(embeddings),
        quality=0.8,
    )
    probes = pd.concat(
        [
            protocol.registered_probes.assign(probe_type="registered"),
            protocol.known_unknown_probes.assign(probe_type="known_unknown"),
            protocol.unknown_unknown_probes.assign(probe_type="unknown_unknown"),
        ],
        ignore_index=True,
    ).assign(
        embedding=lambda frame: frame["image_id"].map(embeddings),
        quality=0.7,
        reconstruction_error_norm=0.0,
    )

    templates = aggregate_templates(gallery, method="mean").assign(angular_error=0.0)
    pca = fit_pca_profile(np.stack(templates["embedding"].to_numpy()), n_components=1, random_state=0)
    assert pca.pgvector_searchable is True
    assert COMPRESSION_PROFILES["pq"].pgvector_searchable is False

    features = build_certified_search_features(
        probes,
        templates,
        compression_profile="origin_512",
        threshold=0.5,
        top_k=2,
    )
    model = LogisticRegressionCalibrator().fit(features)
    probabilities = model.predict_proba(features)

    assert len(features) == 6
    assert set(features["probe_type"]) == {"registered", "known_unknown", "unknown_unknown"}
    assert set(features["certified_decision"]).issubset({"accept", "reject", "defer"})
    assert "certified_fallback_required" in features.columns
    assert len(probabilities) == 6
