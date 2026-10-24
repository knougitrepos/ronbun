"""Validate FaceX-Zoo's explicit, immutable identity-role handoff."""
import json
from pathlib import Path

import pandas as pd

from research.runtime.hashing import sha256_file

TRAINING_DATASETS = ("survface", "tinyface", "scface")
EVALUATION_DATASETS = (*TRAINING_DATASETS, "lfw", "rfw")


def load_protocol(path, expected_sha256=None):
    path = Path(path).resolve()
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise ValueError("FaceX handoff protocol hash mismatch")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != "facex-ronbun-roles-v1" or record.get("name") not in TRAINING_DATASETS:
        raise ValueError("Unsupported training handoff; LFW/RFW cannot train FR")
    for key, hash_key in (("roles_path", "roles_sha256"), ("train_list", "train_sha256")):
        if sha256_file(record[key]) != record[hash_key]:
            raise ValueError(f"Changed FaceX handoff: {key}")
    frame = pd.read_csv(record["roles_path"], keep_default_na=False, dtype={"identity_id": str, "source_identity_id": str})
    required = {"image_id", "identity_id", "source_identity_id", "image_path", "role", "fr_train"}
    if not required <= set(frame) or frame.empty or frame.image_id.duplicated().any() or frame.image_path.duplicated().any():
        raise ValueError("Invalid or duplicate handoff rows")
    if not frame.fr_train.isin([0, 1]).all():
        raise ValueError("fr_train must be explicit 0/1")
    fr = frame.loc[frame.fr_train.eq(1)]
    held = frame.loc[frame.role.isin(["calibration_fit", "calibration_safety"]) | frame.role.str.startswith("test_")]
    if set(fr.identity_id) & (set(held.identity_id) - {""}):
        raise ValueError("FR training/calibration/test identity overlap")
    # Canonical train:/test: namespaces alone cannot establish native-ID
    # separation. Compare the original labels before trusting that namespace.
    if set(fr.source_identity_id) & (set(held.source_identity_id) - {""}):
        raise ValueError("Native FR training/calibration/test identity overlap")
    expected_ids = [f"{record['name']}:{'test' if r.role.startswith('test_') else 'train'}:{r.source_identity_id}"
                    if r.source_identity_id else "" for r in frame.itertuples()]
    if list(frame.identity_id) != expected_ids:
        raise ValueError("Canonical identity does not match native identity/role")
    if not frame.loc[frame.role.eq("pq_development"), "fr_train"].eq(1).all():
        raise ValueError("PQ-development must be a declared FR-training subset")
    fit_ids = set(frame.loc[frame.role.eq("calibration_fit"), "identity_id"])
    safety_ids = set(frame.loc[frame.role.eq("calibration_safety"), "identity_id"])
    if fit_ids & safety_ids:
        raise ValueError("Fit/safety identities overlap")
    if fr.identity_id.nunique() != record["num_classes"] or len(fr) != record["images"]:
        raise ValueError("FR counts differ from producer")
    if record["name"] == "survface":
        gallery_ids = set(frame.loc[frame.role.eq("calibration_gallery"), "identity_id"])
        dev_ids = set(frame.loc[frame.role.eq("pq_development"), "identity_id"])
        if (gallery_ids & dev_ids or (fit_ids | safety_ids) & (gallery_ids | dev_ids)
                or not fit_ids or not safety_ids):
            raise ValueError("SurvFace role contract violated")
    frame["image_path"] = frame.image_path.map(lambda v: str(Path(record["raw_root"]) / v))
    return record, frame


def load_handoff(path):
    index = json.loads(Path(path).read_text(encoding="utf-8"))
    if index.get("schema") != "facex-ronbun-handoff-v1":
        raise ValueError("Unsupported handoff schema")
    return {name: load_protocol(item["path"], item["sha256"])
            for name, item in index["protocols"].items()}


def load_pair_dataset(project_root, config, name):
    """Keep official test pairs; calibration is an additional research protocol."""
    from research.datasets.lfw_pairs import load_pairs
    root = Path(project_root)
    population = pd.read_csv(root / config["image_manifest"], keep_default_na=False)
    if name == "lfw":
        pairs, _ = load_pairs(root / config["pairs_file"], population)
        pairs["group"] = "all"
    elif name == "rfw":
        pairs = pd.read_csv(root / config["pairs_file"])
        pairs = pairs.rename(columns={"left_identity_id": "left_identity", "right_identity_id": "right_identity", "rfw_group": "group"})
        pairs["fold"] = pairs.fold_index + 1
        groups = set(pairs.group)
        if groups != {"African", "Asian", "Caucasian", "Indian"}:
            raise ValueError("RFW must preserve all four official groups")
    else:
        raise ValueError(name)
    if pairs.pair_id.duplicated().any() or population.image_id.duplicated().any():
        raise ValueError("Duplicate official pair/image ID")
    lookup = population.set_index("image_id").identity_id
    pairs["is_genuine"] = pairs.is_genuine.astype(str).str.lower().map({"true": True, "false": False})
    if pairs.is_genuine.isna().any():
        raise ValueError("Invalid pair labels")
    for (_, _), group in pairs.groupby(["group", "fold"]):
        if len(group) != 600 or group.is_genuine.sum() != 300:
            raise ValueError("Official pair fold requires 300 genuine/300 impostor")
    if set(pairs.fold) != set(range(1, 11)):
        raise ValueError("Ten official pair folds required")
    for side in ("left", "right"):
        ids = pairs[f"{side}_image_id"]
        if not ids.isin(lookup.index).all() or list(lookup.loc[ids]) != list(pairs[f"{side}_identity"]):
            raise ValueError("Pair image/identity mismatch")
    return population, pairs


def pair_partition(pairs, fold, *, seed=8972, safety_fraction=.3):
    """Purge outer-test identities for RFW; preserve original fold and group."""
    from research.datasets.lfw_pairs import calibration_partition
    test = pairs.loc[pairs.fold.eq(fold)]
    test_ids = set(test.left_identity) | set(test.right_identity)
    candidate = pairs.loc[pairs.fold.ne(fold)]
    retain = ~candidate.left_identity.isin(test_ids) & ~candidate.right_identity.isin(test_ids)
    clean = pd.concat([test, candidate.loc[retain]], ignore_index=True)
    cal, test, inventory = calibration_partition(clean, fold, seed, safety_fraction)
    inventory["test_identity_purged_pairs"] = int((~retain).sum())
    return cal, test, inventory
