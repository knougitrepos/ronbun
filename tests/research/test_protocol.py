import pandas as pd
import pytest

from research.protocols.open_set import (
    build_calibration_protocol,
    build_open_set_protocol,
    build_survface_official_protocol,
    rebase_survface_protocol_subset_indexes,
    validate_identity_disjoint_splits,
)


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


def test_calibration_protocol_is_deterministic_and_does_not_read_test_rows():
    rows = []
    for identity in ("a", "b", "c"):
        for index in range(3):
            rows.append(
                {
                    "image_id": f"{identity}{index}",
                    "identity_id": identity,
                    "split": "development",
                    "image_path": f"{identity}{index}.jpg",
                }
            )
    rows.append(
        {
            "image_id": "test0",
            "identity_id": "test-only",
            "split": "test",
            "image_path": "test0.jpg",
        }
    )
    manifest = pd.DataFrame(rows)

    first = build_calibration_protocol(
        manifest,
        split_name="development",
        gallery_identity_count=2,
        enrollment_count=1,
        seed=17,
    )
    second = build_calibration_protocol(
        manifest.sample(frac=1.0, random_state=3),
        split_name="development",
        gallery_identity_count=2,
        enrollment_count=1,
        seed=17,
    )

    assert first.gallery["image_id"].tolist() == second.gallery["image_id"].tolist()
    assert set(first.known_unknown_probes["identity_id"]).isdisjoint(
        first.gallery["identity_id"]
    )
    assert "test-only" not in set(
        pd.concat(
            [
                first.gallery,
                first.registered_probes,
                first.known_unknown_probes,
                first.unknown_unknown_probes,
            ]
        )["identity_id"]
    )


def _survface_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["g0", "g1", "p0", "p1", "u0", "u1"],
            "identity_id": ["a", "b", "a", "b", "u0", "u1"],
            "split": ["test"] * 6,
            "image_path": [
                f"{value}.jpg"
                for value in ("g0", "g1", "p0", "p1", "u0", "u1")
            ],
            "protocol_role": [
                "gallery",
                "gallery",
                "registered_probe",
                "registered_probe",
                "unknown_unknown_probe",
                "unknown_unknown_probe",
            ],
            "protocol_index": [0, 1, 0, 1, 0, 1],
        }
    )


def test_survface_protocol_preserves_each_official_role_order():
    manifest = _survface_manifest().iloc[[1, 3, 5, 0, 2, 4]].reset_index(drop=True)

    protocol = build_survface_official_protocol(manifest)

    assert protocol.gallery["image_id"].tolist() == ["g0", "g1"]
    assert protocol.registered_probes["image_id"].tolist() == ["p0", "p1"]
    assert protocol.known_unknown_probes.empty
    assert protocol.unknown_unknown_probes["image_id"].tolist() == ["u0", "u1"]


def test_survface_subset_rebase_preserves_source_indexes_and_local_order():
    manifest = _survface_manifest().assign(
        protocol_index=[4, 9, 7, 12, 3, 18]
    )

    rebased = rebase_survface_protocol_subset_indexes(manifest)
    repeated = rebase_survface_protocol_subset_indexes(rebased)
    protocol = build_survface_official_protocol(rebased)

    assert rebased["source_protocol_index"].tolist() == [4, 9, 7, 12, 3, 18]
    assert rebased["protocol_index"].tolist() == [0, 1, 0, 1, 0, 1]
    pd.testing.assert_frame_equal(rebased, repeated)
    assert protocol.gallery["image_id"].tolist() == ["g0", "g1"]
    assert protocol.registered_probes["image_id"].tolist() == ["p0", "p1"]
    assert protocol.unknown_unknown_probes["image_id"].tolist() == ["u0", "u1"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda frame: frame.assign(protocol_index=[0, 0, 0, 1, 0, 1]),
            "unique and contiguous",
        ),
        (
            lambda frame: frame.assign(
                identity_id=["a", "b", "a", "not-in-gallery", "u0", "u1"]
            ),
            "identity sets differ",
        ),
    ],
)
def test_survface_protocol_rejects_invalid_official_structure(mutation, message):
    with pytest.raises(ValueError, match=message):
        build_survface_official_protocol(mutation(_survface_manifest()))
