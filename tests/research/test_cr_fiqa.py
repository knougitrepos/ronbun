import json
from pathlib import Path

import numpy as np

from research.fiqa import (
    CRFIQA_VARIANTS,
    infer_aligned_bundle_scores,
    infer_cr_fiqa_scores,
    load_fiqa_score_artifact,
    materialize_aligned_bundle_score_artifact,
    preprocess_cr_fiqa_rgb,
    write_fiqa_score_artifact,
)
from research.runtime.hashing import sha256_file


def test_registered_checkpoint_contract_matches_local_files():
    expected = {
        "S": (
            174_679_421,
            "b9f457a6f00e0363a0cfb47ba4075e866d29e4e03c7989afe84faa61054dc8d4",
        ),
        "L": (
            261_219_071,
            "5fca24736e4f8df5fbfc0f31ca533fbcca5eb119e2bad8fff751fe72c4c4d0fd",
        ),
    }
    for variant, (byte_count, digest) in expected.items():
        spec = CRFIQA_VARIANTS[variant]
        assert spec.expected_bytes == byte_count
        assert spec.expected_sha256 == digest
        assert spec.license_id == "CC-BY-NC-4.0"


def test_preprocessing_keeps_official_rgb_minus_one_to_one_contract():
    images = np.zeros((2, 112, 112, 3), dtype=np.uint8)
    images[0, :, :, 0] = 255
    tensor = preprocess_cr_fiqa_rgb(images)

    assert tuple(tensor.shape) == (2, 3, 112, 112)
    assert tensor.dtype.is_floating_point
    assert float(tensor[0, 0, 0, 0]) == 1.0
    assert float(tensor[0, 1, 0, 0]) == -1.0
    assert float(tensor[1, 2, 0, 0]) == -1.0


def test_raw_quality_inference_does_not_apply_sigmoid():
    import torch

    class FakeModel(torch.nn.Module):
        def forward(self, inputs):
            embedding = torch.zeros((len(inputs), 512), device=inputs.device)
            quality = inputs[:, 0].mean(dim=(1, 2), keepdim=False)[:, None] * 3.0
            return embedding, quality

    images = np.zeros((2, 112, 112, 3), dtype=np.uint8)
    images[0, :, :, 0] = 255
    scores = infer_cr_fiqa_scores(
        FakeModel(),
        images,
        batch_size=1,
        device="cpu",
    )

    assert scores.tolist() == [3.0, -3.0]


def _aligned_bundle(root: Path) -> Path:
    root.mkdir()
    faces = np.zeros((3, 112, 112, 3), dtype=np.uint8)
    faces[0, :, :, 0] = 255
    faces[1, :, :, :] = 127
    faces[2, :, :, 2] = 255
    faces_path = root / "aligned_faces.npy"
    np.save(faces_path, faces, allow_pickle=False)
    index_path = root / "aligned_index.csv"
    index_path.write_text(
        "sample_id,aligned_face_index,aligned_content_sha256\n"
        "a,0,hash-a\n"
        "b,1,hash-b\n"
        "c,2,hash-c\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "aligned_face_crops",
        "dataset_id": "synthetic",
        "array_contract": {
            "shape": [3, 112, 112, 3],
            "dtype": "uint8",
            "layout": "nhwc",
            "color_order": "rgb",
            "image_size": [112, 112],
        },
        "outputs": {
            "aligned_faces": {
                "path": faces_path.name,
                "sha256": sha256_file(faces_path),
            },
            "aligned_index": {
                "path": index_path.name,
                "sha256": sha256_file(index_path),
            },
        },
    }
    (root / "bundle_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return root


def test_bundle_inference_and_score_artifact_round_trip(tmp_path: Path):
    import torch

    class FakeModel(torch.nn.Module):
        def forward(self, inputs):
            embedding = torch.zeros((len(inputs), 512), device=inputs.device)
            return embedding, inputs.mean(dim=(1, 2, 3))[:, None]

    bundle = _aligned_bundle(tmp_path / "aligned")
    scores, manifest = infer_aligned_bundle_scores(
        bundle,
        model=FakeModel(),
        model_uid="fake-fiqa",
        checkpoint_sha256="f" * 64,
        variant="S",
        batch_size=2,
        device="cpu",
    )
    artifact = write_fiqa_score_artifact(
        tmp_path / "scores",
        scores,
        manifest,
    )
    loaded = load_fiqa_score_artifact(artifact.root)

    assert loaded.fiqa_uid == manifest["fiqa_uid"]
    assert loaded.scores["sample_id"].tolist() == ["a", "b", "c"]
    assert np.isfinite(loaded.scores["fiqa_score"]).all()
    assert loaded.manifest["preprocessing"]["score_transform"] == (
        "none_raw_quality_scalar"
    )


def test_full_bundle_materialization_resumes_completed_shards(tmp_path: Path):
    import torch

    class FlakyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, inputs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic interruption")
            embedding = torch.zeros((len(inputs), 512), device=inputs.device)
            return embedding, inputs.mean(dim=(1, 2, 3))[:, None]

    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, inputs):
            self.calls += 1
            embedding = torch.zeros((len(inputs), 512), device=inputs.device)
            return embedding, inputs.mean(dim=(1, 2, 3))[:, None]

    bundle = _aligned_bundle(tmp_path / "aligned-resume")
    output = tmp_path / "scores-resume"
    flaky = FlakyModel()
    try:
        materialize_aligned_bundle_score_artifact(
            bundle,
            output,
            model=flaky,
            model_uid="fake-fiqa",
            checkpoint_sha256="f" * 64,
            variant="S",
            batch_size=2,
            shard_size=2,
            device="cpu",
        )
    except RuntimeError as exc:
        assert "synthetic interruption" in str(exc)
    else:
        raise AssertionError("the synthetic first run must be interrupted")

    assert not output.exists()
    assert output.with_name(".scores-resume.inprogress").is_dir()

    resumed_model = CountingModel()
    artifact = materialize_aligned_bundle_score_artifact(
        bundle,
        output,
        model=resumed_model,
        model_uid="fake-fiqa",
        checkpoint_sha256="f" * 64,
        variant="S",
        batch_size=2,
        shard_size=2,
        device="cpu",
    )

    assert resumed_model.calls == 1
    assert len(artifact.scores) == 3
    assert artifact.manifest["execution"]["resumable_shards"] is True
    assert not output.with_name(".scores-resume.inprogress").exists()
