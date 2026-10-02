"""Identity-disjoint calibration inside released BLUFR training pools.

This is a derived calibration protocol, not the BLUFR benchmark evaluator.
Released outer train/test membership is never regenerated.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.runtime.hashing import canonical_sha256


@dataclass(frozen=True)
class CalibrationSplit:
    assignment: pd.DataFrame
    inventory: dict


def _ordered(values, *, seed, namespace):
    return sorted(
        values, key=lambda value: canonical_sha256([namespace, seed, str(value)])
    )


def build_calibration_split(
    lists, bound, *, trial_id, development_fraction, calibration_gallery_count, seed
):
    """Reserve a fixed calibration watchlist, then vary training allocation.

    Counts include the reserved calibration watchlist. Released gallery and
    probe lists and their ordering are preserved, with no test subsampling.
    """
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(development_fraction, bool)
        or not np.isfinite(development_fraction)
        or not 0 < development_fraction < 1
    ):
        raise ValueError("valid seed and development_fraction in (0,1) required")
    for count in (calibration_gallery_count,):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("positive integer gallery counts required")
    if (
        len(bound) != len(lists.images)
        or bound.image_id.duplicated().any()
        or not bound.filename.reset_index(drop=True).equals(
            lists.images.filename.reset_index(drop=True)
        )
        or not bound.identity_id.astype(str)
        .reset_index(drop=True)
        .eq("lfw:" + lists.images.identity_name.reset_index(drop=True))
        .all()
    ):
        raise ValueError("bound images must match the public list and identity order")
    trial = lists.trial(trial_id)
    frame = bound[["image_id", "identity_id", "filename"]].reset_index(drop=True).copy()
    frame["outer_split"] = "test"
    frame.loc[trial.train, "outer_split"] = "train"
    frame["role"] = "unassigned"
    frame["protocol_order"] = np.arange(len(frame))
    training = frame.loc[trial.train]
    sizes = training.groupby("identity_id").size()
    eligible = _ordered(
        sizes[sizes >= 2].index, seed=seed, namespace=f"{trial_id}:cal-gallery"
    )
    if len(eligible) < calibration_gallery_count:
        raise ValueError(
            "not enough multi-image training identities for calibration gallery"
        )
    watchlist = set(eligible[:calibration_gallery_count])
    n_development = int(np.floor(len(sizes) * development_fraction))
    n_calibration = len(sizes) - n_development
    if n_development < 1 or n_calibration <= calibration_gallery_count:
        raise ValueError(
            "reserve compression training and non-mated calibration identities"
        )
    remaining = _ordered(
        set(sizes.index) - watchlist, seed=seed, namespace=f"{trial_id}:allocation"
    )
    calibration_ids = watchlist | set(
        remaining[: n_calibration - calibration_gallery_count]
    )
    development_ids = set(sizes.index) - calibration_ids
    frame.loc[trial.train, "role"] = "development"
    frame.loc[
        frame.outer_split.eq("train") & frame.identity_id.isin(calibration_ids), "role"
    ] = "calibration_non_mated"
    frame.loc[
        frame.outer_split.eq("train") & frame.identity_id.isin(watchlist), "role"
    ] = "calibration_mated"
    for identity in sorted(watchlist):
        indices = frame.index[
            frame.outer_split.eq("train") & frame.identity_id.eq(identity)
        ]
        chosen = _ordered(
            indices, seed=seed, namespace=f"{trial_id}:cal-image:{identity}"
        )[0]
        frame.loc[chosen, "role"] = "calibration_gallery"
    gallery = trial.gallery
    test_gallery_count = len(gallery)
    test_ids = set(frame.loc[gallery, "identity_id"])
    frame.loc[trial.probe, "role"] = "test_non_mated"
    mated = frame.loc[trial.probe, "identity_id"].isin(test_ids).to_numpy()
    frame.loc[trial.probe[mated], "role"] = "test_mated"
    frame.loc[gallery, "role"] = "test_gallery"
    frame.loc[gallery, "protocol_order"] = np.arange(len(gallery))
    frame.loc[trial.probe, "protocol_order"] = np.arange(len(trial.probe))
    if frame.role.eq("unassigned").any():
        raise ValueError("public images must have an explicit role")
    if not set(development_ids).isdisjoint(calibration_ids):
        raise ValueError("compression/calibration identity overlap")
    if (development_ids | calibration_ids) & set(frame.loc[trial.test, "identity_id"]):
        raise ValueError("outer train/test identity overlap")
    counts = frame.role.value_counts().to_dict()
    for role in (
        "calibration_mated",
        "calibration_non_mated",
        "test_mated",
        "test_non_mated",
    ):
        if not counts.get(role):
            raise ValueError(f"empty probe role: {role}")
    inventory = dict(
        trial_id=trial_id,
        development_fraction=float(development_fraction),
        development_identities=len(development_ids),
        calibration_identities=len(calibration_ids),
        calibration_gallery_identities=calibration_gallery_count,
        test_gallery_identities=test_gallery_count,
        enrollment_count=1,
        outer_train_test_preserved=True,
        public_probe_list_preserved=True,
        public_gallery_preserved=True,
        calibration_image_selection="seeded_one_image",
        test_image_selection="released_gallery_image",
        gallery_size_matched=calibration_gallery_count == test_gallery_count,
        assignment_sha256=canonical_sha256(frame.to_dict("records")),
        **{
            f"{role}_images": int(counts.get(role, 0))
            for role in (
                "development",
                "calibration_gallery",
                "calibration_mated",
                "calibration_non_mated",
                "test_gallery",
                "test_mated",
                "test_non_mated",
            )
        },
    )
    return CalibrationSplit(frame, inventory)
