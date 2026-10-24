"""Pinned LFW View-2 pairs and identity-disjoint compression/calibration roles."""

from pathlib import Path
from urllib.request import urlopen
import hashlib

import numpy as np
import pandas as pd

from research.runtime.hashing import sha256_file, canonical_sha256

PAIRS_URL = "https://raw.githubusercontent.com/davidsandberg/facenet/master/data/pairs.txt"
PAIRS_SHA256 = "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"
PROTOCOL_UID = "lfw-view2-pairs-development-unused-identities-v1"
OUTER_FOLD_PROTOCOL_UID = "lfw-view2-pairs-outer-fold-unused-identities-v2"


def outer_fold_development(people_path, population, pairs, development):
    """Exclude the whole held-out people fold, including people absent from pairs.

    The supplied CSV conversion preserves people.txt boundaries as empty rows.
    Its exact hash is recorded by the caller; this is not author-file byte authentication.
    """
    people = pd.read_csv(people_path, skip_blank_lines=False)
    if set(people.columns) != {"name", "images"}:
        raise ValueError("people CSV requires name/images and empty fold separators")
    separators = people.name.isna()
    if separators.sum() != 9 or people.loc[separators, "images"].notna().any():
        raise ValueError("people CSV must preserve exactly ten folds")
    people["fold"] = separators.cumsum() + 1
    people = people.loc[~separators].copy()
    people["identity_id"] = "lfw:" + people.name.astype(str)
    counts = population.groupby("identity_id").size()
    if (people.identity_id.duplicated().any() or set(people.identity_id) != set(counts.index)
            or not np.array_equal(counts.loc[people.identity_id].to_numpy(), people.images.to_numpy())):
        raise ValueError("people CSV/population identity or image-count mismatch")
    result = {}
    for fold in range(1, 11):
        test_people = set(people.loc[people.fold.eq(fold), "identity_id"])
        tests = pairs.loc[pairs.fold.eq(fold)]
        if not (set(tests.left_identity) | set(tests.right_identity)) <= test_people:
            raise ValueError("pair fold does not match people CSV fold")
        part = development.loc[~development.identity_id.isin(test_people)].copy()
        if len(part) < 256:
            raise ValueError("outer-fold PQ requires at least 256 development images")
        result[fold] = part
    return result


def ensure_pairs(path, *, download=False):
    """Download only on request, verify before publication, never replace a file."""
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"LFW pairs missing; enable DOWNLOAD_LFW_PAIRS: {path}")
        payload = urlopen(PAIRS_URL, timeout=30).read()
        if hashlib.sha256(payload).hexdigest() != PAIRS_SHA256:
            raise ValueError("downloaded LFW pairs hash mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    if sha256_file(path) != PAIRS_SHA256:
        raise ValueError("LFW pairs hash mismatch")
    return path


def parse_pairs(text):
    lines = text.splitlines()
    if not lines or lines[0].split() != ["10", "300"] or len(lines) != 6001:
        raise ValueError("expected official 10 x (300 genuine + 300 impostor) LFW pairs")
    records = []
    for i, line in enumerate(lines[1:]):
        tokens = line.split()
        genuine = i % 600 < 300
        if len(tokens) != (3 if genuine else 4):
            raise ValueError("LFW pair order/label structure differs from View-2")
        left, li = tokens[:2]
        right, ri = (left, tokens[2]) if genuine else tokens[2:]
        if (left == right) != genuine or int(li) < 1 or int(ri) < 1:
            raise ValueError("invalid LFW identity/label/index")
        a, b = f"{left}_{int(li):04d}", f"{right}_{int(ri):04d}"
        if a == b:
            raise ValueError("self-pair is not a verification trial")
        records.append(dict(pair_id=f"lfw-pair-{i:04d}", fold=i // 600 + 1,
                            left_image_id=f"lfw:{left}:{a}", right_image_id=f"lfw:{right}:{b}",
                            left_identity=f"lfw:{left}", right_identity=f"lfw:{right}",
                            is_genuine=genuine))
    result = pd.DataFrame(records)
    # Validate the actual file, not a generic assumption about pair cross-validation.
    seen = set()
    for _, fold in result.groupby("fold", sort=True):
        identities = set(fold.left_identity) | set(fold.right_identity)
        if seen & identities:
            raise ValueError("LFW fold identity overlap violates this protocol")
        seen |= identities
    return result


def load_pairs(path, population, *, download=False):
    pairs = parse_pairs(ensure_pairs(path, download=download).read_text(encoding="utf8"))
    if population.image_id.duplicated().any():
        raise ValueError("duplicate population image IDs")
    identities = population.set_index("image_id").identity_id.astype(str)
    for side in ("left", "right"):
        images = pairs[f"{side}_image_id"]
        if not images.isin(identities.index).all():
            raise ValueError("official pair image missing from population")
        if not np.array_equal(identities.loc[images].to_numpy(), pairs[f"{side}_identity"]):
            raise ValueError("pair/population identity mismatch")
    used = set(pairs.left_identity) | set(pairs.right_identity)
    development = population.loc[~population.identity_id.isin(used)].copy()
    if len(development) < 256:
        raise ValueError("at least 256 identity-disjoint development images required for 8-bit PQ")
    return pairs, development


def calibration_partition(pairs, fold, seed, safety_fraction=.3):
    """Split BOTH endpoints by identity; omit cross-fit/safety pairs, never test pairs."""
    if fold not in set(pairs.fold) or not 0 < safety_fraction < 1:
        raise ValueError("invalid fold/safety fraction")
    test = pairs.loc[pairs.fold.eq(fold)].copy()
    candidates = pairs.loc[pairs.fold.ne(fold)].copy()
    test_ids = set(test.left_identity) | set(test.right_identity)
    if test_ids & (set(candidates.left_identity) | set(candidates.right_identity)):
        raise ValueError("calibration/test identity overlap")
    def role(identity):
        h = hashlib.sha256(f"{int(seed)}:{identity}".encode()).digest()
        return "safety" if int.from_bytes(h[:8], "big") / float(2**64) < safety_fraction else "fit"
    left = candidates.left_identity.map(role)
    right = candidates.right_identity.map(role)
    retained = left.eq(right)
    calibration = candidates.loc[retained].copy()
    calibration["calibration_role"] = left[retained]
    for label in ("fit", "safety"):
        part = calibration.loc[calibration.calibration_role.eq(label)]
        if (~part.is_genuine).sum() < 20 or not part.is_genuine.any():
            raise ValueError("each calibration partition needs >=20 impostor pairs and genuine pairs")
    inventory = dict(fold=int(fold), partition_seed=int(seed), test_pairs=len(test),
                     test_genuine=int(test.is_genuine.sum()), test_impostor=int((~test.is_genuine).sum()),
                     calibration_candidate_pairs=len(candidates), calibration_pairs=len(calibration),
                     cross_partition_pairs_excluded=int((~retained).sum()), identity_overlap=0,
                     assignment_sha256=canonical_sha256(calibration[["pair_id", "calibration_role"]].to_dict("records")))
    for label in ("fit", "safety"):
        part = calibration.loc[calibration.calibration_role.eq(label)]
        inventory[f"{label}_genuine"] = int(part.is_genuine.sum())
        inventory[f"{label}_impostor"] = int((~part.is_genuine).sum())
        inventory[f"{label}_identities"] = len(set(part.left_identity) | set(part.right_identity))
    return calibration, test, inventory
