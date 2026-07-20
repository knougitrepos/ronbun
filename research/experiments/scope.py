from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

from research.protocols.open_set import OpenSetProtocol


EXPERIMENT_MODES = frozenset({"dev", "real"})


@dataclass(frozen=True)
class ExperimentScope:
    """Reproducible execution scope shared by dataset-specific experiments.

    ``mode`` labels the intent of a run, while ``data_fraction`` controls the
    actual identity fraction.  Keeping both explicit avoids silently treating a
    smoke run as a paper result.  A real full-data run is therefore exactly
    ``mode='real'`` and ``data_fraction=1.0``.
    """

    mode: str
    data_fraction: float
    seed: int = 42

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in EXPERIMENT_MODES:
            raise ValueError(f"mode must be one of {sorted(EXPERIMENT_MODES)}")
        if isinstance(self.data_fraction, bool):
            raise ValueError("data_fraction must be a number in (0, 1]")
        try:
            fraction = float(self.data_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError("data_fraction must be a number in (0, 1]") from exc
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("data_fraction must be in (0, 1]")
        if isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        try:
            seed = int(self.seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("seed must be an integer") from exc
        if seed != self.seed:
            raise ValueError("seed must be an integer")

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "data_fraction", fraction)
        object.__setattr__(self, "seed", seed)

    @property
    def is_full_dataset(self) -> bool:
        return self.data_fraction == 1.0

    @property
    def is_paper_run(self) -> bool:
        return self.mode == "real" and self.is_full_dataset

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "data_fraction": self.data_fraction,
            "seed": self.seed,
            "is_full_dataset": self.is_full_dataset,
            "is_paper_run": self.is_paper_run,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ExperimentScope":
        """Read either an ``execution`` section or an execution mapping itself."""

        values: Mapping[str, Any]
        nested = config.get("execution")
        if nested is None:
            values = config
        elif isinstance(nested, Mapping):
            values = nested
        else:
            raise ValueError("execution config must be a mapping")
        missing = {"mode", "data_fraction"}.difference(values)
        if missing:
            raise ValueError(f"missing execution settings: {sorted(missing)}")
        return cls(
            mode=values["mode"],
            data_fraction=values["data_fraction"],
            seed=values.get("seed", 42),
        )


def _namespace(value: str) -> str:
    namespace = str(value).strip()
    if not namespace:
        raise ValueError("namespace must not be empty")
    return namespace


def _minimum_identity_count(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if number != value or number < 1:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _identity_tokens(values: pd.Series, *, identity_column: str) -> dict[str, object]:
    if values.isna().any():
        raise ValueError(f"{identity_column} must not contain missing values")
    by_token: dict[str, object] = {}
    for value in pd.unique(values):
        token = str(value)
        if token in by_token and by_token[token] != value:
            raise ValueError(
                f"{identity_column} contains values with the same string representation"
            )
        by_token[token] = value
    return by_token


def _ordered_identity_tokens(
    values: pd.Series,
    scope: ExperimentScope,
    *,
    identity_column: str,
    namespace: str,
) -> list[str]:
    by_token = _identity_tokens(values, identity_column=identity_column)

    def key(token: str) -> tuple[str, str]:
        payload = f"{namespace}\0{scope.seed}\0{token}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), token

    return sorted(by_token, key=key)


def _selected_identity_tokens(
    values: pd.Series,
    scope: ExperimentScope,
    *,
    identity_column: str,
    namespace: str,
    minimum_identities: int,
) -> set[str]:
    ordered = _ordered_identity_tokens(
        values,
        scope,
        identity_column=identity_column,
        namespace=namespace,
    )
    if not ordered:
        raise ValueError("cannot select identities from an empty frame")
    minimum = _minimum_identity_count(
        minimum_identities,
        name="minimum_identities",
    )
    if minimum > len(ordered):
        raise ValueError(
            f"minimum_identities={minimum} exceeds available identities={len(ordered)}"
        )
    selected_count = max(minimum, math.ceil(len(ordered) * scope.data_fraction))
    return set(ordered[: min(selected_count, len(ordered))])


def select_manifest_fraction(
    frame: pd.DataFrame,
    scope: ExperimentScope,
    *,
    identity_column: str = "identity_id",
    namespace: str,
    minimum_identities: int = 1,
) -> pd.DataFrame:
    """Select a deterministic, nested fraction of complete identities.

    Selection depends only on ``namespace``, ``scope.seed`` and the identity
    value.  It is independent of input row order, and a smaller fraction is
    always an identity subset of a larger fraction for the same seed/namespace.
    Every row belonging to a selected identity is retained.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if identity_column not in frame.columns:
        raise ValueError(f"missing identity column: {identity_column}")
    if frame.empty:
        raise ValueError("manifest frame must not be empty")
    resolved_namespace = _namespace(namespace)
    selected = _selected_identity_tokens(
        frame[identity_column],
        scope,
        identity_column=identity_column,
        namespace=resolved_namespace,
        minimum_identities=minimum_identities,
    )
    mask = frame[identity_column].astype(str).isin(selected)
    return frame.loc[mask].copy()


def _select_optional_protocol_group(
    frame: pd.DataFrame,
    scope: ExperimentScope,
    *,
    identity_column: str,
    namespace: str,
    minimum_identities: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return select_manifest_fraction(
        frame,
        scope,
        identity_column=identity_column,
        namespace=namespace,
        minimum_identities=minimum_identities,
    ).reset_index(drop=True)


def select_open_set_protocol_fraction(
    protocol: OpenSetProtocol,
    scope: ExperimentScope,
    *,
    namespace: str,
    identity_column: str = "identity_id",
    minimum_registered_identities: int = 1,
    minimum_non_mated_identities: int = 1,
) -> OpenSetProtocol:
    """Apply one nested scope while preserving open-set role boundaries.

    Gallery and registered-probe rows are filtered by one shared set of enrolled
    identities.  Known-unknown and unknown-unknown identities are sampled in
    separate namespaces, so a reduced run cannot accidentally erase one role
    because another role is larger.
    """

    resolved_namespace = _namespace(namespace)
    for name, frame in {
        "gallery": protocol.gallery,
        "registered_probes": protocol.registered_probes,
        "known_unknown_probes": protocol.known_unknown_probes,
        "unknown_unknown_probes": protocol.unknown_unknown_probes,
    }.items():
        if identity_column not in frame.columns:
            raise ValueError(f"{name} is missing identity column: {identity_column}")
    if protocol.gallery.empty or protocol.registered_probes.empty:
        raise ValueError("gallery and registered probes must not be empty")

    gallery_ids = set(protocol.gallery[identity_column].astype(str))
    registered_probe_ids = set(protocol.registered_probes[identity_column].astype(str))
    missing_probe_ids = gallery_ids.difference(registered_probe_ids)
    if missing_probe_ids:
        raise ValueError(
            "gallery identities without registered probes: "
            f"{sorted(missing_probe_ids)[:10]}"
        )

    registered_tokens = _selected_identity_tokens(
        protocol.gallery[identity_column],
        scope,
        identity_column=identity_column,
        namespace=f"{resolved_namespace}:registered",
        minimum_identities=minimum_registered_identities,
    )

    def registered_rows(frame: pd.DataFrame) -> pd.DataFrame:
        mask = frame[identity_column].astype(str).isin(registered_tokens)
        return frame.loc[mask].copy().reset_index(drop=True)

    known_unknown = _select_optional_protocol_group(
        protocol.known_unknown_probes,
        scope,
        identity_column=identity_column,
        namespace=f"{resolved_namespace}:known_unknown",
        minimum_identities=minimum_non_mated_identities,
    )
    unknown_unknown = _select_optional_protocol_group(
        protocol.unknown_unknown_probes,
        scope,
        identity_column=identity_column,
        namespace=f"{resolved_namespace}:unknown_unknown",
        minimum_identities=minimum_non_mated_identities,
    )

    selected = OpenSetProtocol(
        gallery=registered_rows(protocol.gallery),
        registered_probes=registered_rows(protocol.registered_probes),
        known_unknown_probes=known_unknown,
        unknown_unknown_probes=unknown_unknown,
    )
    non_mated_ids = set(known_unknown[identity_column].astype(str)).union(
        unknown_unknown[identity_column].astype(str)
    )
    overlap = set(selected.gallery[identity_column].astype(str)).intersection(
        non_mated_ids
    )
    if overlap:
        raise ValueError(
            f"selected gallery and non-mated identities overlap: {sorted(overlap)[:10]}"
        )
    return selected
