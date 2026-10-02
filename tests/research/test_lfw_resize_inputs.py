from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from research.experiments import lfw_resize_inputs as module
from research.preprocessing import aligned_crops
from research.runtime.hashing import sha256_file


def source_images(root):
    rng = np.random.default_rng(42)
    paths = []
    for i in range(3):
        path = root / f"image-{i}.png"
        Image.fromarray(rng.integers(0, 256, (250, 250, 3), dtype="uint8")).save(path)
        paths.append(str(path))
    return pd.DataFrame(
        dict(
            image_id=["a", "b", "c"],
            identity_id=["id-a", "id-b", "id-c"],
            split="population",
            image_path=paths,
        )
    )


def test_resize_preserves_whole_image_and_never_loads_detector(tmp_path, monkeypatch):
    manifest = source_images(tmp_path)
    monkeypatch.setattr(
        aligned_crops, "_default_detector", lambda *a: pytest.fail("detector loaded")
    )
    result = aligned_crops.materialize_aligned_crops(
        manifest,
        project_root=tmp_path,
        output_dir=tmp_path / "bundle",
        dataset_id="lfw",
        preprocessing_mode=aligned_crops.LFW_DEEPFUNNELED_RESIZE,
        require_full_coverage=True,
        overwrite=False,
    )
    assert result.failed_index.empty
    assert result.bundle_manifest["detector"]["enabled"] is False
    assert result.bundle_manifest["preprocessing"]["alignment_recomputed"] is False
    assert result.aligned_index.landmark_5points_json.fillna("").eq("").all()
    assert set(result.aligned_index.alignment_template_id) == {
        aligned_crops.LFW_RESIZE_TEMPLATE_ID
    }
    for i, path in enumerate(manifest.image_path):
        expected = np.asarray(
            Image.open(path)
            .convert("RGB")
            .resize((112, 112), Image.Resampling.BILINEAR)
        )
        np.testing.assert_array_equal(result.aligned_faces[i], expected)
    result.aligned_faces._mmap.close()


@pytest.mark.parametrize("failure", ["missing", "wrong_size"])
def test_resize_never_silently_drops_samples(tmp_path, failure):
    manifest = source_images(tmp_path)
    if failure == "missing":
        manifest.loc[1, "image_path"] = str(tmp_path / "absent.png")
    else:
        Image.new("RGB", (112, 112)).save(manifest.iloc[1].image_path)
    with pytest.raises((ValueError, RuntimeError)):
        aligned_crops.materialize_aligned_crops(
            manifest,
            project_root=tmp_path,
            output_dir=tmp_path / "bundle",
            dataset_id="lfw",
            preprocessing_mode=aligned_crops.LFW_DEEPFUNNELED_RESIZE,
            require_full_coverage=True,
            overwrite=False,
        )
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_resize_source_shards_resume_with_validated_lineage(tmp_path, monkeypatch):
    manifest = source_images(tmp_path)
    bundle = tmp_path / "bundle"
    aligned = aligned_crops.materialize_aligned_crops(
        manifest,
        project_root=tmp_path,
        output_dir=bundle,
        dataset_id="lfw",
        preprocessing_mode=aligned_crops.LFW_DEEPFUNNELED_RESIZE,
        require_full_coverage=True,
        overwrite=False,
    )
    aligned.aligned_faces._mmap.close()
    spec = SimpleNamespace(
        model_uid="test-model",
        checkpoint=SimpleNamespace(sha256="checkpoint"),
        preprocessing=SimpleNamespace(preprocess_hash="normalization"),
        to_manifest=lambda: {"model_uid": "test-model"},
    )
    config = dict(
        source_runs={"arcface": "source"},
        resize_inputs=dict(embedding_shard_size=2, checkpoint_file="work.sqlite3"),
    )
    monkeypatch.setattr(module, "_implementation", lambda: {"code": "test"})
    import torch

    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda: "mock CUDA for routing test"
    )
    calls = []

    class Adapter:
        def embed(self, faces):
            calls.append(len(faces))
            vectors = np.zeros((len(faces), 512), dtype="float32")
            vectors[:, 0] = 1.0
            return SimpleNamespace(
                normalized_embedding=vectors,
                raw_norm=np.ones(len(faces), dtype="float32"),
            )

    monkeypatch.setattr(
        module, "create_pytorch_adapter_from_spec", lambda *a, **kw: Adapter()
    )
    publish = module._write_source
    monkeypatch.setattr(
        module, "_write_source", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        module._extract_source(
            tmp_path,
            config,
            "arcface",
            spec,
            bundle,
            "cuda",
            1,
            lambda x: None,
            lambda: None,
        )
    assert calls == [1, 1, 1]
    monkeypatch.setattr(module, "_write_source", publish)
    module._extract_source(
        tmp_path,
        config,
        "arcface",
        spec,
        bundle,
        "cuda",
        1,
        lambda x: None,
        lambda: None,
    )
    assert calls == [1, 1, 1]  # Completed shards are read; no second inference.
    _, meta, _, prepared, population, lineage = module.load_resize_source(
        tmp_path, "source"
    )
    assert prepared.sample_ids.tolist() == ["a", "b", "c"]
    assert len(population) == 3
    assert lineage["aligned_bundle_manifest_sha256"] == sha256_file(
        bundle / "bundle_manifest.json"
    )
    before = sha256_file(tmp_path / "source/manifest.json")
    module._extract_source(
        tmp_path,
        config,
        "arcface",
        spec,
        bundle,
        "cuda",
        1,
        lambda x: None,
        lambda: None,
    )
    assert sha256_file(tmp_path / "source/manifest.json") == before
    with pytest.raises(ValueError, match="contract differs"):
        module._extract_source(
            tmp_path,
            config,
            "arcface",
            spec,
            bundle,
            "cuda",
            2,
            lambda x: None,
            lambda: None,
        )
    with (tmp_path / "source/normalized_embeddings.npy").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.load_resize_source(tmp_path, "source")


def test_raw_input_config_does_not_reuse_old_detected_source():
    import yaml

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / module.CONFIG_PATH).read_text(encoding="utf8"))
    assert config["source_kind"] == module.SOURCE_TYPE
    assert len(config["source_runs"]) == 4
    assert config["output_root"] == "results/calibration/lfw_blufr_resize_based"
    assert all(
        "lfw_deepfunneled_resize_v1" in p for p in config["source_runs"].values()
    )
