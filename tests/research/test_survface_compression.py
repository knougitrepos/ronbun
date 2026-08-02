from __future__ import annotations

from pathlib import Path
import os

import nbformat
import pandas as pd
import pytest
import yaml

from research.experiments.survface_compression import (
    survface_development_image_paths,
    survface_watchlist_fit_image_paths,
    validate_survface_training_manifest,
)


def _training_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["d1", "d2", "c1", "c2"],
            "identity_id": ["dev-a", "dev-b", "cal-a", "cal-b"],
            "split": [
                "development",
                "development",
                "calibration",
                "calibration",
            ],
            "image_path": [
                "data/dev-a.jpg",
                "data/dev-b.jpg",
                "data/cal-a.jpg",
                "data/cal-b.jpg",
            ],
            "protocol_role": ["training"] * 4,
        }
    )


def test_survface_compressor_manifest_accepts_only_training_boundary():
    validated = validate_survface_training_manifest(_training_manifest())

    assert set(validated["split"]) == {"development", "calibration"}
    assert validated["protocol_role"].eq("training").all()

    official = _training_manifest()
    official.loc[0, ["split", "protocol_role"]] = ["test", "gallery"]
    with pytest.raises(ValueError, match="development and calibration"):
        validate_survface_training_manifest(official)


def test_survface_development_paths_exclude_calibration_and_test(tmp_path):
    paths = survface_development_image_paths(
        _training_manifest(),
        project_root=tmp_path,
    )

    assert paths == {
        os.path.normcase(str((Path(tmp_path) / "data/dev-a.jpg").resolve())),
        os.path.normcase(str((Path(tmp_path) / "data/dev-b.jpg").resolve())),
    }


def test_survface_watchlist_fit_paths_use_half_gallery(tmp_path):
    rows = []
    for identity, split in (
        ("dev-a", "development"),
        ("dev-b", "development"),
        ("cal-a", "calibration"),
        ("cal-b", "calibration"),
    ):
        for index in range(4):
            rows.append(
                {
                    "image_id": f"{identity}-{index}",
                    "identity_id": identity,
                    "split": split,
                    "image_path": f"data/{identity}-{index}.jpg",
                    "protocol_role": "training",
                }
            )
    manifest = pd.DataFrame(rows)

    paths = survface_watchlist_fit_image_paths(
        manifest,
        project_root=tmp_path,
        gallery_identity_count=2,
        seed=42,
    )

    assert len(paths) == 4


def test_survface_execution_config_uses_same_dataset_fit_and_full_search():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (
            root / "configs/experiments/survface_face_search.yaml"
        ).read_text(encoding="utf-8")
    )

    compression = config["compression"]
    assert compression["fit_policy"] == (
        "training_3000_watchlist_enrollment_only"
    )
    assert compression["fit_split"] == "training_3000_half_gallery_v2"
    assert compression["pca"]["dimensions"] == [384, 256, 128, 64, 32]
    assert compression["pq"]["pgvector_searchable"] is False
    assert config["calibration"]["fit_policy"] == (
        "training_3000_half_gallery_non_mated_only"
    )
    assert config["calibration"]["gallery_identity_count"] == 3000
    assert config["calibration"]["gallery_enrollment_policy"] == "half_floor"
    assert config["calibration"]["threshold_selection"] == "non_mated_only"
    assert config["search"]["compression_profiles"] == [
        "origin_512",
        "pca_256",
    ]
    assert config["search"]["modes"] == ["exact", "hnsw"]
    assert config["dataset"]["aligned_bundle_dir"] == (
        "data/interim/step4/survface/aligned_112"
    )
    assert config["embedding"] == {
        "framework": "pytorch",
        "model_uid": "arcface-7972a704552df378345f",
        "model_spec_path": (
            "runs/step2/model_registry/"
            "arcface-7972a704552df378345f.json"
        ),
        "source_color_order": "rgb",
        "input_size": [112, 112],
        "device": "cuda",
        "batch_size": 64,
        "detector_inference": False,
    }
    assert config["progress"] == {
        "milestone_percent": 10,
        "heartbeat_seconds": None,
    }


def test_survface_run_freeze_records_training_and_official_inputs():
    root = Path(__file__).resolve().parents[2]
    notebook = nbformat.read(
        root
        / "notebooks/survface/01_embeddings"
        / "00_official_protocol_and_run_freeze.ipynb",
        as_version=4,
    )
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )

    assert '"training_manifest": DATA_DIR / "training_manifest.csv"' in source
    assert '"training_summary": DATA_DIR / "training_summary.json"' in source
    assert '"official_manifest": DATA_DIR / "official_manifest.csv"' in source
    assert '"aligned_bundle_manifest": ALIGNED_DIR / "bundle_manifest.json"' in source
    assert '"aligned_index": ALIGNED_DIR / "aligned_index.csv"' in source
    assert 'role="arcface_model_spec"' in source
    assert 'role="arcface_checkpoint"' in source


def test_survface_embedding_uses_aligned_batch_without_detector():
    root = Path(__file__).resolve().parents[2]
    notebook = nbformat.read(
        root
        / "notebooks/survface/01_embeddings"
        / "01_official_arcface_embedding_extraction.ipynb",
        as_version=4,
    )
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )

    assert "create_pytorch_adapter_from_spec" in source
    assert "extractor.embed(" in source
    assert 'mmap_mode="r"' in source
    assert '"detector_inference": False' in source
    assert 'completed=counts["processed"]' in source
    assert '"model_uid": embedding_output.model_uid' in source
    assert "extract_with_metadata(" not in source
    assert "cv2.imread(" not in source
