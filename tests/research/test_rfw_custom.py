from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from research.datasets.rfw import RFW_GROUPS
from research.datasets.rfw_custom import (
    RFW_CUSTOM_ARTIFACT_TYPE,
    RFW_CUSTOM_PROTOCOL_FAMILY_UID,
    RFW_OFFICIAL_PROTOCOL_UID,
    adapt_rfw_custom_manifest_to_open_set_protocol,
    build_rfw_custom_open_set_bundle,
    load_rfw_custom_open_set_bundle,
    select_rfw_custom_protocol_fraction,
    validate_rfw_custom_open_set_bundle,
    write_rfw_custom_open_set_bundle,
)


SOURCE_SHA256 = "A" * 64


def _source_manifest(*, include_cross_group_identity: bool = True) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in RFW_GROUPS:
        source_identities = [
            f"{group.lower()}-person-{index:02d}" for index in range(12)
        ]
        if include_cross_group_identity and group in {"African", "Asian"}:
            source_identities.append("shared-cross-group-person")
        for source_label, source_identity_id in enumerate(source_identities):
            for face_index in range(1, 4):
                filename = f"{source_identity_id}_{face_index:04d}.jpg"
                rows.append(
                    {
                        "image_id": (
                            f"rfw:{group.lower()}:{source_identity_id}_{face_index:04d}"
                        ),
                        "identity_id": f"rfw:{group.lower()}:{source_identity_id}",
                        "source_identity_id": source_identity_id,
                        "split": "test",
                        "image_path": (
                            "tar://data/raw/RFW/images/test.tar.gz#"
                            f"test/data/{group}/{source_identity_id}/{filename}"
                        ),
                        "dataset": "rfw-v1",
                        "dataset_role": "evaluation_test_only",
                        "protocol_role": "verification_image",
                        "rfw_group": group,
                        "group_label_source": "dataset_provided",
                        "source_label": source_label,
                        "face_index": face_index,
                        "protocol_index": len(
                            [row for row in rows if row["rfw_group"] == group]
                        ),
                        "source_archive_path": "data/raw/RFW/images/test.tar.gz",
                        "archive_member": (
                            f"test/data/{group}/{source_identity_id}/{filename}"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _build(source: pd.DataFrame | None = None, **overrides):
    options = {
        "source_archive_sha256": SOURCE_SHA256,
        "gallery_identity_count_per_group": 2,
        "enrollment_count": 1,
        "seed": 17,
        "development_fraction": 0.25,
        "calibration_fraction": 0.25,
        "unknown_unknown_fraction": 0.50,
    }
    options.update(overrides)
    return build_rfw_custom_open_set_bundle(
        _source_manifest() if source is None else source,
        **options,
    )


def test_rfw_custom_build_is_deterministic_identity_disjoint_and_balanced():
    source = _source_manifest()
    first = _build(source)
    repeated = _build(source.sample(frac=1.0, random_state=9))

    pd.testing.assert_frame_equal(first.manifest, repeated.manifest)
    assert first.summary["protocol_uid"] == repeated.summary["protocol_uid"]
    assert first.summary["protocol_uid"].startswith(
        f"{RFW_CUSTOM_PROTOCOL_FAMILY_UID}-"
    )
    assert first.summary["protocol_uid"] != RFW_OFFICIAL_PROTOCOL_UID
    assert first.summary["artifact_type"] == RFW_CUSTOM_ARTIFACT_TYPE
    assert first.summary["official_protocol"] is False
    assert first.summary["official_pair_protocol_used"] is False
    assert first.summary["open_set_protocol"] is True
    assert first.summary["checkpoint_overlap_status"] == "UNKNOWN"
    assert first.summary["strict_unseen_identity_evidence"] is False
    assert set(first.manifest["checkpoint_overlap_status"]) == {"UNKNOWN"}
    assert not first.manifest["strict_unseen_identity_evidence"].any()
    assert first.summary["source_archive_sha256"] == SOURCE_SHA256
    assert first.summary["compressor_fit_split"] == "development"
    assert first.summary["calibration_fit_split"] == "calibration"

    identity_split_counts = first.manifest.groupby("identity_id")["split"].nunique()
    assert identity_split_counts.eq(1).all()
    assert set(first.manifest["split"]) == {
        "development",
        "calibration",
        "test",
    }
    assert set(first.manifest["demographic_group"]) == set(RFW_GROUPS)
    assert first.manifest["demographic_group"].equals(first.manifest["rfw_group"])
    for group in RFW_GROUPS:
        assert first.summary["split_identity_counts_by_group"][group] == {
            "development": 3,
            "calibration": 3,
            "test": 6,
        }
        group_gallery = first.protocol.gallery.loc[
            first.protocol.gallery["rfw_group"].eq(group)
        ]
        assert group_gallery["identity_id"].nunique() == 2
        assert group_gallery.groupby("identity_id").size().eq(1).all()

    gallery_ids = set(first.protocol.gallery["identity_id"])
    registered_ids = set(first.protocol.registered_probes["identity_id"])
    known_ids = set(first.protocol.known_unknown_probes["identity_id"])
    unknown_ids = set(first.protocol.unknown_unknown_probes["identity_id"])
    assert gallery_ids == registered_ids
    assert gallery_ids.isdisjoint(known_ids)
    assert gallery_ids.isdisjoint(unknown_ids)
    assert known_ids.isdisjoint(unknown_ids)
    assert first.protocol.registered_probes["is_mated"].eq(True).all()
    assert first.protocol.known_unknown_probes["is_mated"].eq(False).all()
    assert first.protocol.unknown_unknown_probes["is_mated"].eq(False).all()
    validate_rfw_custom_open_set_bundle(first)


def test_cross_group_source_identity_is_excluded_before_all_splits():
    bundle = _build()

    assert bundle.summary["excluded_cross_group_source_identity_count"] == 1
    assert bundle.summary["excluded_cross_group_image_count"] == 6
    assert "shared-cross-group-person" not in set(bundle.manifest["source_identity_id"])
    assert bundle.summary["identity_split_unit"] == (
        "source_identity_id_with_group_ambiguity_excluded"
    )


def test_seed_changes_protocol_instance_but_not_contract_namespace():
    first = _build(seed=17)
    second = _build(seed=18)

    assert first.summary["protocol_uid"] != second.summary["protocol_uid"]
    assert first.summary["protocol_family_uid"] == RFW_CUSTOM_PROTOCOL_FAMILY_UID
    assert second.summary["protocol_family_uid"] == RFW_CUSTOM_PROTOCOL_FAMILY_UID


def test_custom_adapter_rejects_split_leakage_and_official_uid_collision():
    bundle = _build()
    leaked = bundle.manifest.copy()
    development_identity = leaked.loc[
        leaked["split"].eq("development"), "identity_id"
    ].iloc[0]
    row_index = leaked.index[leaked["identity_id"].eq(development_identity)][0]
    leaked.loc[row_index, "split"] = "calibration"
    leaked.loc[row_index, "protocol_role"] = "calibration_pool"
    leaked.loc[row_index, "dataset_role"] = "threshold_calibration"
    for role in ("development_pool", "calibration_pool"):
        role_indexes = leaked.index[leaked["protocol_role"].eq(role)]
        leaked.loc[role_indexes, "protocol_index"] = range(len(role_indexes))

    with pytest.raises(ValueError, match="identity leakage"):
        adapt_rfw_custom_manifest_to_open_set_protocol(leaked)

    official_collision = bundle.manifest.copy()
    official_collision["protocol_uid"] = RFW_OFFICIAL_PROTOCOL_UID
    with pytest.raises(ValueError, match="custom namespace"):
        adapt_rfw_custom_manifest_to_open_set_protocol(official_collision)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda frame: frame.drop(columns="source_archive_path"),
            "missing columns",
        ),
        (
            lambda frame: frame.assign(
                protocol_role=["not-official"]
                + ["verification_image"] * (len(frame) - 1)
            ),
            "protocol_role",
        ),
        (
            lambda frame: frame.assign(
                image_id=[frame.iloc[1]["image_id"]] + frame["image_id"].tolist()[1:]
            ),
            "image_id values must be unique",
        ),
    ],
)
def test_source_contract_fails_closed(mutation, message):
    with pytest.raises(ValueError, match=message):
        _build(mutation(_source_manifest()))


def test_source_sha_and_capacity_fail_closed():
    with pytest.raises(ValueError, match="64-character SHA-256"):
        _build(source_archive_sha256="not-a-sha")
    with pytest.raises(ValueError, match="requested 7"):
        _build(gallery_identity_count_per_group=7)


def test_atomic_write_and_load_verify_custom_artifact_lineage(tmp_path: Path):
    bundle = _build()
    output = tmp_path / "rfw-custom"

    paths = write_rfw_custom_open_set_bundle(bundle, output)
    loaded = load_rfw_custom_open_set_bundle(output)

    assert paths["_SUCCESS"].is_file()
    success = json.loads(paths["_SUCCESS"].read_text(encoding="utf-8"))
    assert success["artifact_type"] == RFW_CUSTOM_ARTIFACT_TYPE
    assert success["official_protocol"] is False
    assert loaded.summary["protocol_uid"] == bundle.summary["protocol_uid"]
    assert loaded.gallery_identities == bundle.gallery_identities
    assert loaded.known_unknown_identities == bundle.known_unknown_identities
    assert loaded.unknown_unknown_identities == bundle.unknown_unknown_identities
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_rfw_custom_open_set_bundle(bundle, output)

    gallery_path = paths["rfw_custom_gallery.csv"]
    gallery_path.write_text(
        gallery_path.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact SHA mismatch"):
        load_rfw_custom_open_set_bundle(output)


def test_real_source_role_names_are_not_reused_as_custom_official_claims():
    bundle = _build()

    assert set(bundle.manifest["source_protocol_role"]) == {"verification_image"}
    assert set(bundle.manifest["protocol_role"]) == {
        "development_pool",
        "calibration_pool",
        "gallery",
        "registered_probe",
        "known_unknown_probe",
        "unknown_unknown_probe",
    }
    assert not bundle.manifest["official_pair_protocol_used"].any()
    assert not bundle.manifest["official_result_eligible"].any()


def test_quick_fraction_is_group_role_stratified_nested_and_reindexed():
    bundle = _build()

    half = select_rfw_custom_protocol_fraction(bundle, 0.50, seed=31)
    repeated = select_rfw_custom_protocol_fraction(
        bundle.manifest.sample(frac=1.0, random_state=4),
        0.50,
        seed=31,
    )
    quarter = select_rfw_custom_protocol_fraction(bundle, 0.25, seed=31)

    assert set(half["image_id"]) == set(repeated["image_id"])
    assert set(quarter["image_id"]).issubset(set(half["image_id"]))
    half_protocol = adapt_rfw_custom_manifest_to_open_set_protocol(half)
    assert set(half_protocol.gallery["identity_id"]) == set(
        half_protocol.registered_probes["identity_id"]
    )
    for role in (
        "development_pool",
        "calibration_pool",
        "gallery",
        "registered_probe",
        "known_unknown_probe",
        "unknown_unknown_probe",
    ):
        role_rows = half.loc[half["protocol_role"].eq(role)]
        assert sorted(role_rows["protocol_index"].tolist()) == list(
            range(len(role_rows))
        )
        assert set(role_rows["rfw_group"]) == set(RFW_GROUPS)
    assert half["source_custom_protocol_index"].notna().all()
    assert set(half["scope_data_fraction"]) == {0.50}
    assert set(half["scope_seed"]) == {31}
    assert not half["scope_is_full"].any()


def test_quick_full_fraction_returns_an_unmodified_copy():
    bundle = _build()

    full = select_rfw_custom_protocol_fraction(bundle, 1.0, seed=999)

    pd.testing.assert_frame_equal(full, bundle.manifest)
    assert full is not bundle.manifest


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.1, float("nan"), True])
def test_quick_fraction_rejects_invalid_values(fraction):
    with pytest.raises(ValueError, match="data_fraction"):
        select_rfw_custom_protocol_fraction(_build(), fraction, seed=42)
