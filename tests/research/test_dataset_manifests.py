from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from research.datasets import (
    build_lfw_manifest,
    build_survface_official_manifest,
    write_lfw_manifest_bundle,
    write_survface_official_bundle,
)


def _touch_images(directory: Path, names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"test-image")


def test_build_lfw_manifest_is_deterministic_and_identity_disjoint(tmp_path: Path):
    lfw_root = tmp_path / "data" / "raw" / "LFW"
    for identity_index in range(15):
        identity = f"Person_{identity_index:02d}"
        _touch_images(
            lfw_root / identity,
            [f"{identity}_{image_index:04d}.jpg" for image_index in range(1, 5)],
        )

    first = build_lfw_manifest(
        lfw_root,
        tmp_path,
        seed=9,
        development_fraction=0.40,
        calibration_fraction=0.20,
        gallery_size=2,
        min_gallery_images=3,
    )
    second = build_lfw_manifest(
        lfw_root,
        tmp_path,
        seed=9,
        development_fraction=0.40,
        calibration_fraction=0.20,
        gallery_size=2,
        min_gallery_images=3,
    )

    pd.testing.assert_frame_equal(first.manifest, second.manifest)
    assert first.gallery_identities == second.gallery_identities
    assert first.unknown_unknown_identities == second.unknown_unknown_identities
    assert len(first.manifest) == 60
    assert first.summary["gallery_identity_count"] == 2
    assert first.summary["known_unknown_identity_count"] > 0
    assert first.summary["unknown_unknown_identity_count"] > 0
    assert (
        first.manifest.groupby("identity_id")["split"].nunique().max()
        == 1
    )
    assert all(path.startswith("data/raw/LFW/") for path in first.manifest["image_path"])


def test_lfw_writer_refuses_accidental_overwrite(tmp_path: Path):
    lfw_root = tmp_path / "raw"
    for identity_index in range(12):
        identity = f"Person_{identity_index:02d}"
        _touch_images(
            lfw_root / identity,
            [f"{identity}_{image_index}.jpg" for image_index in range(3)],
        )
    bundle = build_lfw_manifest(
        lfw_root,
        tmp_path,
        development_fraction=0.25,
        calibration_fraction=0.25,
        gallery_size=2,
        min_gallery_images=3,
    )
    output_dir = tmp_path / "out"

    paths = write_lfw_manifest_bundle(bundle, output_dir)

    assert set(paths) == {
        "face_manifest.csv",
        "gallery_identities.txt",
        "unknown_unknown_identities.txt",
        "summary.json",
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_lfw_manifest_bundle(bundle, output_dir)


def _mat_names(names: list[str]) -> np.ndarray:
    result = np.empty((len(names), 1), dtype=object)
    for index, name in enumerate(names):
        result[index, 0] = name
    return result


def _make_survface_fixture(root: Path) -> None:
    test_dir = root / "Face_Identification_Test_Set"
    evaluation_dir = root / "Face_Identification_Evaluation"
    gallery_names = ["person1_g1.jpg", "person1_g2.jpg", "person2_g1.jpg"]
    mated_names = ["person1_p1.jpg", "person2_p1.jpg"]
    unmated_names = ["unknown_a.jpg", "unknown_b.jpg"]
    _touch_images(test_dir / "gallery", gallery_names)
    _touch_images(test_dir / "mated_probe", mated_names)
    _touch_images(test_dir / "unmated_probe", unmated_names)
    evaluation_dir.mkdir(parents=True)
    savemat(
        evaluation_dir / "gallery_img_ID_pairs.mat",
        {
            "gallery_set": _mat_names(gallery_names),
            "gallery_ids": np.asarray([[1], [1], [2]], dtype=np.uint16),
        },
    )
    savemat(
        evaluation_dir / "mated_probe_img_ID_pairs.mat",
        {
            "mated_probe_set": _mat_names(mated_names),
            "mated_probe_ids": np.asarray([[1], [2]], dtype=np.uint16),
        },
    )


def test_build_survface_manifest_preserves_official_roles_and_order(tmp_path: Path):
    root = tmp_path / "data" / "raw" / "QMUL-SurvFace"
    _make_survface_fixture(root)

    bundle = build_survface_official_manifest(root, tmp_path)

    assert len(bundle.manifest) == 7
    assert bundle.gallery["protocol_index"].tolist() == [0, 1, 2]
    assert bundle.gallery["official_identity_id"].tolist() == [1, 1, 2]
    assert bundle.registered_probes["official_identity_id"].tolist() == [1, 2]
    assert set(bundle.manifest["protocol_role"]) == {
        "gallery",
        "registered_probe",
        "unknown_unknown_probe",
    }
    assert bundle.summary["registered_identity_count"] == 2
    assert bundle.summary["known_unknown_identity_count"] == 0
    assert len(set(bundle.unknown_unknown_identities)) == 2

    output_dir = tmp_path / "output"
    paths = write_survface_official_bundle(bundle, output_dir)
    assert pd.read_csv(paths["official_manifest.csv"]).shape[0] == 7
    assert paths["gallery.csv"].is_file()
    assert paths["registered_probes.csv"].is_file()
    assert paths["unknown_unknown_probes.csv"].is_file()


def test_survface_manifest_rejects_files_not_listed_by_official_mat(tmp_path: Path):
    root = tmp_path / "QMUL-SurvFace"
    _make_survface_fixture(root)
    (root / "Face_Identification_Test_Set" / "gallery" / "extra.jpg").write_bytes(b"x")

    with pytest.raises(ValueError, match="do not match the official protocol"):
        build_survface_official_manifest(root, tmp_path)
