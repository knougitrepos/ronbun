import pandas as pd
import pytest

from research.protocols.open_set import build_open_set_protocol, validate_identity_disjoint_splits


def test_rejects_identity_leakage_between_splits():
    manifest = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "identity_id": ["id-1", "id-1", "id-2"],
            "split": ["development", "test", "calibration"],
            "image_path": ["a.jpg", "b.jpg", "c.jpg"],
        }
    )

    with pytest.raises(ValueError, match="identity leakage"):
        validate_identity_disjoint_splits(manifest)


def test_builds_gallery_registered_and_unknown_probes_without_overlap():
    manifest = pd.DataFrame(
        {
            "image_id": ["a1", "a2", "b1", "b2", "u1", "u2", "x1", "x2"],
            "identity_id": ["a", "a", "b", "b", "u", "u", "x", "x"],
            "split": ["test"] * 8,
            "image_path": [
                "a1.jpg",
                "a2.jpg",
                "b1.jpg",
                "b2.jpg",
                "u1.jpg",
                "u2.jpg",
                "x1.jpg",
                "x2.jpg",
            ],
        }
    )

    protocol = build_open_set_protocol(
        manifest,
        gallery_identities=["a", "b"],
        unknown_unknown_identities=["x"],
        enrollment_count=1,
        seed=7,
    )

    assert set(protocol.gallery["identity_id"]) == {"a", "b"}
    assert set(protocol.registered_probes["identity_id"]) == {"a", "b"}
    assert set(protocol.known_unknown_probes["identity_id"]) == {"u"}
    assert set(protocol.unknown_unknown_probes["identity_id"]) == {"x"}
    assert set(protocol.gallery["image_id"]).isdisjoint(protocol.registered_probes["image_id"])


def test_rejects_enrollment_count_that_consumes_all_registered_images():
    manifest = pd.DataFrame(
        {
            "image_id": ["a1", "b1", "u1", "x1"],
            "identity_id": ["a", "b", "u", "x"],
            "split": ["test"] * 4,
            "image_path": ["a1.jpg", "b1.jpg", "u1.jpg", "x1.jpg"],
        }
    )

    with pytest.raises(ValueError, match="registered probe set is empty"):
        build_open_set_protocol(
            manifest,
            gallery_identities=["a", "b"],
            unknown_unknown_identities=["x"],
            enrollment_count=1,
            seed=0,
        )
