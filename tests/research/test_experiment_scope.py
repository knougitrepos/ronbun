from __future__ import annotations

import pandas as pd
import pytest

from research.experiments.scope import (
    ExperimentScope,
    select_manifest_fraction,
    select_open_set_protocol_fraction,
)
from research.protocols.open_set import OpenSetProtocol


def _manifest(identity_count: int = 20, rows_per_identity: int = 3) -> pd.DataFrame:
    rows = []
    for identity_index in range(identity_count):
        identity_id = f"id-{identity_index:02d}"
        for image_index in range(rows_per_identity):
            rows.append(
                {
                    "identity_id": identity_id,
                    "image_id": f"{identity_id}-{image_index}",
                    "split": "test",
                }
            )
    return pd.DataFrame(rows)


def test_experiment_scope_validates_and_marks_only_real_full_data_as_paper_run():
    dev = ExperimentScope(mode=" DEV ", data_fraction=0.25, seed=7)
    real = ExperimentScope.from_config(
        {"execution": {"mode": "real", "data_fraction": 1.0, "seed": 9}}
    )

    assert dev.mode == "dev"
    assert dev.is_full_dataset is False
    assert dev.is_paper_run is False
    assert real.is_paper_run is True
    assert real.as_dict() == {
        "mode": "real",
        "data_fraction": 1.0,
        "seed": 9,
        "is_full_dataset": True,
        "is_paper_run": True,
    }

    with pytest.raises(ValueError, match="mode must be"):
        ExperimentScope(mode="smoke", data_fraction=0.1)
    for invalid in (0.0, -0.1, 1.01, float("nan"), True):
        with pytest.raises(ValueError, match="data_fraction"):
            ExperimentScope(mode="dev", data_fraction=invalid)


def test_manifest_fraction_is_identity_atomic_order_invariant_and_nested():
    manifest = _manifest()
    small_scope = ExperimentScope(mode="dev", data_fraction=0.20, seed=17)
    large_scope = ExperimentScope(mode="dev", data_fraction=0.55, seed=17)

    small = select_manifest_fraction(manifest, small_scope, namespace="dataset-a")
    large = select_manifest_fraction(manifest, large_scope, namespace="dataset-a")
    shuffled = select_manifest_fraction(
        manifest.sample(frac=1.0, random_state=123),
        small_scope,
        namespace="dataset-a",
    )

    small_ids = set(small["identity_id"])
    large_ids = set(large["identity_id"])
    assert len(small_ids) == 4
    assert len(large_ids) == 11
    assert small_ids < large_ids
    assert small_ids == set(shuffled["identity_id"])
    assert small.groupby("identity_id").size().eq(3).all()
    assert small["image_id"].tolist() == manifest.loc[
        manifest["identity_id"].isin(small_ids), "image_id"
    ].tolist()


def test_manifest_fraction_honors_minimum_without_breaking_nested_prefix():
    manifest = _manifest(identity_count=5, rows_per_identity=2)
    tiny = select_manifest_fraction(
        manifest,
        ExperimentScope(mode="dev", data_fraction=0.01, seed=3),
        namespace="minimum",
        minimum_identities=3,
    )
    larger = select_manifest_fraction(
        manifest,
        ExperimentScope(mode="dev", data_fraction=0.80, seed=3),
        namespace="minimum",
        minimum_identities=3,
    )

    assert tiny["identity_id"].nunique() == 3
    assert set(tiny["identity_id"]).issubset(set(larger["identity_id"]))
    with pytest.raises(ValueError, match="exceeds available identities"):
        select_manifest_fraction(
            manifest,
            ExperimentScope(mode="dev", data_fraction=0.1),
            namespace="minimum",
            minimum_identities=6,
        )


def _role_frame(prefix: str, identity_count: int, rows_per_identity: int) -> pd.DataFrame:
    rows = []
    for identity_index in range(identity_count):
        identity_id = f"{prefix}-{identity_index:02d}"
        for row_index in range(rows_per_identity):
            rows.append(
                {
                    "identity_id": identity_id,
                    "image_id": f"{identity_id}-{row_index}",
                    "protocol_index": len(rows),
                }
            )
    return pd.DataFrame(rows)


def _protocol() -> OpenSetProtocol:
    gallery = _role_frame("registered", 8, 1)
    registered_probes = _role_frame("registered", 8, 2)
    return OpenSetProtocol(
        gallery=gallery,
        registered_probes=registered_probes,
        known_unknown_probes=_role_frame("known", 6, 2),
        unknown_unknown_probes=_role_frame("unknown", 10, 1),
    )


def test_open_set_fraction_keeps_roles_disjoint_and_registered_ids_aligned():
    protocol = _protocol()
    small = select_open_set_protocol_fraction(
        protocol,
        ExperimentScope(mode="dev", data_fraction=0.25, seed=31),
        namespace="open-set",
    )
    large = select_open_set_protocol_fraction(
        protocol,
        ExperimentScope(mode="dev", data_fraction=0.75, seed=31),
        namespace="open-set",
    )

    small_gallery_ids = set(small.gallery["identity_id"])
    assert small_gallery_ids == set(small.registered_probes["identity_id"])
    assert small_gallery_ids.issubset(set(large.gallery["identity_id"]))
    assert set(small.known_unknown_probes["identity_id"]).issubset(
        set(large.known_unknown_probes["identity_id"])
    )
    assert set(small.unknown_unknown_probes["identity_id"]).issubset(
        set(large.unknown_unknown_probes["identity_id"])
    )
    assert small_gallery_ids.isdisjoint(set(small.known_unknown_probes["identity_id"]))
    assert small_gallery_ids.isdisjoint(set(small.unknown_unknown_probes["identity_id"]))
    assert small.registered_probes.groupby("identity_id").size().eq(2).all()


def test_open_set_fraction_allows_an_official_protocol_without_known_unknowns():
    protocol = _protocol()
    protocol = OpenSetProtocol(
        gallery=protocol.gallery,
        registered_probes=protocol.registered_probes,
        known_unknown_probes=protocol.known_unknown_probes.iloc[0:0].copy(),
        unknown_unknown_probes=protocol.unknown_unknown_probes,
    )

    selected = select_open_set_protocol_fraction(
        protocol,
        ExperimentScope(mode="dev", data_fraction=0.5, seed=2),
        namespace="survface",
    )

    assert selected.known_unknown_probes.empty
    assert not selected.unknown_unknown_probes.empty


def test_full_scope_preserves_every_protocol_row_and_its_source_order():
    protocol = _protocol()

    selected = select_open_set_protocol_fraction(
        protocol,
        ExperimentScope(mode="real", data_fraction=1.0, seed=2),
        namespace="official-full",
    )

    for field in (
        "gallery",
        "registered_probes",
        "known_unknown_probes",
        "unknown_unknown_probes",
    ):
        pd.testing.assert_frame_equal(
            getattr(selected, field),
            getattr(protocol, field).reset_index(drop=True),
        )
