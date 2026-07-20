from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.compression import PCA_256
from research.experiments.lfw_certification import (
    _stored_pca_dimension,
    assemble_lfw_certification_inputs,
    write_vector_frame_csv,
)


def test_stored_pca_dimension_preserves_legacy_and_rejects_step1_only_tables():
    assert _stored_pca_dimension("pca_448") == 448
    assert _stored_pca_dimension("pca_384") == 384
    with pytest.raises(ValueError, match="no current PostgreSQL embedding table"):
        _stored_pca_dimension("pca_64")


def _protocol(tmp_path):
    def frame(rows):
        return pd.DataFrame(
            [
                {
                    "image_id": image_id,
                    "identity_id": identity_id,
                    "image_path": str(tmp_path / f"{image_id}.jpg"),
                }
                for image_id, identity_id in rows
            ]
        )

    return {
        "gallery": frame([("ga", "a"), ("gb", "b"), ("gc", "c")]),
        "registered_probes": frame([("pa", "a"), ("pb", "b"), ("pc", "c")]),
        "known_unknown_probes": frame([("pk", "k")]),
        "unknown_unknown_probes": frame([("pu", "u")]),
    }


def _records(protocol, tmp_path):
    vectors = {
        "ga": ([1.0, 0.0], [0.98, 0.02]),
        "gb": ([0.0, 1.0], [0.02, 0.98]),
        # gc is intentionally missing; pc must not remain a registered probe.
        "pa": ([1.0, 0.0], [0.97, 0.03]),
        "pb": ([0.0, 1.0], [0.03, 0.97]),
        "pc": ([0.7, 0.7], [0.65, 0.75]),
        "pk": ([-1.0, 0.0], [-0.98, 0.02]),
        "pu": ([0.0, -1.0], [0.02, -0.98]),
    }
    return pd.DataFrame.from_records(
        [
            {
                "canonical_path": str((tmp_path / f"{image_id}.jpg").resolve()).lower(),
                "origin_embedding": np.asarray(exact, dtype=np.float32),
                "approximate_embedding": np.asarray(approximate, dtype=np.float32),
                "retrieval_embedding": np.asarray(approximate, dtype=np.float32),
                "angular_error": 0.02,
                "reconstruction_error_norm": 0.5,
            }
            for image_id, (exact, approximate) in vectors.items()
        ]
    )


def test_assemble_filters_registered_identity_without_gallery(tmp_path):
    protocol = _protocol(tmp_path)
    records = _records(protocol, tmp_path)

    bundle = assemble_lfw_certification_inputs(
        protocol,
        records,
        project_root=tmp_path,
        compression_profile=PCA_256,
    )

    assert bundle.certificate_space == "pca_reconstructed_512"
    assert bundle.templates["identity_id"].tolist() == ["a", "b"]
    assert "retrieval_embedding" in bundle.templates.columns
    assert "retrieval_embedding" in bundle.probes.columns
    assert set(bundle.probes["probe_type"]) == {
        "registered",
        "known_unknown",
        "unknown_unknown",
    }
    assert "pc" not in set(bundle.probes["image_id"])
    dropped = bundle.coverage["registered_missing_gallery_identity"]
    assert dropped["dropped_identity_ids"] == ["c"]
    assert dropped["dropped_image_ids"] == ["pc"]
    assert bundle.coverage["gallery"]["missing_image_ids"] == ["gc"]


def test_assemble_allows_empty_unknown_unknown_for_calibration(tmp_path):
    protocol = _protocol(tmp_path)
    protocol["unknown_unknown_probes"] = protocol["unknown_unknown_probes"].iloc[0:0]

    bundle = assemble_lfw_certification_inputs(
        protocol,
        _records(_protocol(tmp_path), tmp_path),
        project_root=tmp_path,
        compression_profile=PCA_256,
        allow_empty_unknown_unknown=True,
    )

    assert set(bundle.probes["probe_type"]) == {"registered", "known_unknown"}


def test_vector_csv_serializes_arrays_and_lists(tmp_path):
    frame = pd.DataFrame(
        {
            "image_id": ["q1"],
            "embedding": [np.asarray([1.0, 0.0], dtype=np.float32)],
            "fallback_embedding": [np.asarray([0.9, 0.1], dtype=np.float32)],
            "ranked_identities": [["a", "b"]],
        }
    )

    path = write_vector_frame_csv(frame, tmp_path / "vectors.csv")
    restored = pd.read_csv(path)

    assert json.loads(restored.loc[0, "embedding"]) == [1.0, 0.0]
    assert json.loads(restored.loc[0, "ranked_identities"]) == ["a", "b"]
