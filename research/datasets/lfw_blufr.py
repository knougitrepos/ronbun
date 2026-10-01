"""Read the released BLUFR MATLAB lists; never regenerate their random trials."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from research.runtime.hashing import sha256_file

REFERENCE_SHA256 = "8f99875f50ee8a7b88659a15527f50ae9b8850a84df52b0b13ce32177895d51c"
PROJECT_URL = "https://shengcailiao.github.io/projects/Benchmark_of_Large-scale_Unconstrained_Face_Recognition.html"
MIRROR_URL = "https://raw.githubusercontent.com/zch-90/face_recognition_evalation/master/BLUFR/config/lfw/blufr_lfw_config.mat"


def fetch_blufr_config(destination, *, url=MIRROR_URL, expected_sha256=REFERENCE_SHA256):
    """Explicit, hash-pinned download; never overwrite an existing config."""
    from hashlib import sha256
    from urllib.request import urlopen
    from uuid import uuid4
    destination = Path(destination)
    if destination.exists():
        load_blufr_lists(destination, expected_sha256=expected_sha256)
        return destination
    if not url.startswith("https://"):
        raise ValueError("BLUFR download requires HTTPS")
    with urlopen(url, timeout=30) as response:
        payload = response.read(2_000_001)
    if len(payload) > 2_000_000 or sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("BLUFR download size/hash mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid4().hex + ".tmp")
    temporary.write_bytes(payload)
    load_blufr_lists(temporary, expected_sha256=expected_sha256)
    temporary.rename(destination)
    return destination


@dataclass(frozen=True)
class BLUFRTrial:
    trial_id: int
    train: np.ndarray
    test: np.ndarray
    gallery: np.ndarray
    probe: np.ndarray


@dataclass(frozen=True)
class BLUFRLists:
    images: pd.DataFrame
    trials: tuple[BLUFRTrial, ...]
    source_sha256: str

    def trial(self, trial_id: int) -> BLUFRTrial:
        if isinstance(trial_id, bool) or trial_id not in range(1, len(self.trials) + 1):
            raise ValueError("BLUFR trial_id must be a released, one-based trial number")
        return self.trials[trial_id - 1]


def _indices(values, size):
    raw = np.asarray(values).reshape(-1)
    if (not len(raw) or not np.issubdtype(raw.dtype, np.number)
            or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all()
            or (raw < 1).any() or (raw > size).any()):
        raise ValueError("BLUFR indices must be finite, one-based integers in imageList")
    indices = raw.astype(np.int64) - 1
    if len(np.unique(indices)) != len(indices):
        raise ValueError("duplicate BLUFR image index")
    return indices


def load_blufr_lists(path, *, expected_sha256=REFERENCE_SHA256, strict_reference=True):
    """Keep MATLAB list order and verify every split before exposing any trial.

    The pinned file was obtained from a toolkit mirror. Its origin and digest
    must remain visible; a hash pin alone is not author authentication.
    ``strict_reference=False`` is for synthetic tests, not paper evaluations.
    """
    path = Path(path)
    digest = sha256_file(path)
    if not expected_sha256 or digest != expected_sha256:
        raise ValueError("BLUFR config SHA-256 mismatch")
    data = loadmat(path, simplify_cells=True)
    required = {"imageList", "labels", "trainIndex", "testIndex", "galIndex", "probIndex"}
    if not required.issubset(data):
        raise ValueError(f"BLUFR config missing variables: {sorted(required - data.keys())}")
    names = np.asarray(data["imageList"]).reshape(-1).astype(str)
    identities = []
    for name in names:
        stem, separator, number = Path(name).stem.rpartition("_")
        if (Path(name).name != name or not separator or not stem or not number.isdigit()
                or Path(name).suffix.lower() != ".jpg"):
            raise ValueError(f"invalid BLUFR image filename: {name}")
        identities.append(stem)
    images = pd.DataFrame({"filename": names, "identity_name": identities})
    labels = np.asarray(data["labels"]).reshape(-1)
    if len(labels) != len(images) or images.filename.duplicated().any():
        raise ValueError("BLUFR image/label inventory mismatch")
    images["label"] = labels
    if ((images.groupby("identity_name").label.nunique() != 1).any()
            or (images.groupby("label").identity_name.nunique() != 1).any()):
        raise ValueError("BLUFR filename identity and labels disagree")
    arrays = {k: np.asarray(data[k], dtype=object).reshape(-1)
              for k in ("trainIndex", "testIndex", "galIndex", "probIndex")}
    counts = {len(v) for v in arrays.values()}
    if len(counts) != 1 or (strict_reference and counts != {10}):
        raise ValueError("BLUFR trial inventory mismatch")
    if strict_reference and (len(images) != 13233 or images.label.nunique() != 5749):
        raise ValueError("incomplete public LFW inventory")
    trials = []
    for offset in range(len(arrays["trainIndex"])):
        train, test, gallery, probe = (
            _indices(arrays[k][offset], len(images)) for k in arrays)
        if (set(train) & set(test) or set(train) | set(test) != set(range(len(images)))
                or set(gallery) & set(probe) or set(gallery) | set(probe) != set(test)):
            raise ValueError("BLUFR train/test or gallery/probe partition mismatch")
        train_ids, test_ids = set(labels[train]), set(labels[test])
        if train_ids & test_ids or len(set(labels[gallery])) != len(gallery):
            raise ValueError("BLUFR identity overlap or repeated gallery identity")
        if strict_reference and (len(train_ids) != 1500 or len(gallery) != 1000):
            raise ValueError("BLUFR released identity/gallery counts differ")
        mated = np.isin(labels[probe], labels[gallery])
        if not mated.any() or mated.all():
            raise ValueError("BLUFR requires both mated and non-mated probes")
        trials.append(BLUFRTrial(offset + 1, train, test, gallery, probe))
    return BLUFRLists(images, tuple(trials), digest)


def bind_image_manifest(lists, manifest):
    """Join by the released filename, never by incidental CSV ordering."""
    required = {"image_id", "identity_id", "image_path"}
    if not required.issubset(manifest):
        raise ValueError("LFW manifest lacks image_id, identity_id or image_path")
    frame = manifest.copy()
    frame["filename"] = frame.image_path.map(lambda p: Path(str(p).replace("\\", "/")).name)
    if frame.filename.duplicated().any() or frame.image_id.duplicated().any():
        raise ValueError("ambiguous LFW filenames/image IDs")
    result = lists.images.merge(frame, on="filename", how="left", validate="one_to_one", sort=False)
    if result.image_id.isna().any():
        raise ValueError(f"LFW manifest missing {result.image_id.isna().sum()} released images")
    if not result.identity_id.astype(str).eq("lfw:" + result.identity_name).all():
        raise ValueError("LFW manifest identity disagrees with released filename")
    return result


def trial_inventory(lists):
    labels = lists.images.label.to_numpy()
    return pd.DataFrame([dict(
        trial_id=t.trial_id, train_images=len(t.train), train_identities=len(set(labels[t.train])),
        gallery_images=len(t.gallery), gallery_identities=len(set(labels[t.gallery])),
        mated_probes=int(np.isin(labels[t.probe], labels[t.gallery]).sum()),
        non_mated_probes=int((~np.isin(labels[t.probe], labels[t.gallery])).sum()),
    ) for t in lists.trials])
