from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from research.embeddings.base import (  # noqa: E402
    CheckpointProvenance,
    ModelSpec,
    PreprocessingSpec,
)
from research.embeddings.pytorch import ArcFacePyTorchAdapter  # noqa: E402
from research.explainability.gradcam import (  # noqa: E402
    MISSING_IDENTITY_REASON,
    SINGLETON_IDENTITY_REASON,
    extract_population_gradcam,
    prepare_population_saliency_inputs,
    read_population_heatmaps,
    read_population_saliency_features,
    read_prepared_population_artifact,
    write_population_saliency_artifact,
    write_prepared_population_artifact,
)


class _DeterministicFaceEncoder(torch.nn.Module):
    """Small differentiable 512D encoder used only for pipeline integration."""

    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Conv2d(
            3,
            4,
            kernel_size=1,
            bias=True,
        )
        self.projection = torch.nn.Linear(4 * 4 * 4, 512, bias=False)
        with torch.no_grad():
            self.features.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0]], [[0.25]], [[0.10]]],
                        [[[0.15]], [[1.00]], [[0.20]]],
                        [[[0.20]], [[0.10]], [[1.00]]],
                        [[[0.40]], [[0.35]], [[0.25]]],
                    ],
                    dtype=torch.float32,
                )
            )
            self.features.bias.copy_(
                torch.tensor([0.05, 0.10, 0.15, 0.20], dtype=torch.float32)
            )
            coordinates = torch.arange(
                512 * 4 * 4 * 4,
                dtype=torch.float32,
            ).reshape(512, 4 * 4 * 4)
            weights = torch.sin(coordinates * 0.017) + 0.5 * torch.cos(
                coordinates * 0.031
            )
            self.projection.weight.copy_(weights)
        self.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        activations = torch.relu(self.features(images))
        return self.projection(activations.flatten(start_dim=1))


def _aligned_faces() -> np.ndarray:
    grid = np.arange(16, dtype=np.uint16).reshape(4, 4)
    rows = []
    for index in range(6):
        first = (grid * (index + 3) + 17 * index + 11) % 256
        second = np.rot90(
            (grid * (index + 5) + 9 * index + 23) % 256,
            k=index % 4,
        )
        third = (255 - first + 7 * index) % 256
        rows.append(np.stack((first, second, third), axis=-1))
    return np.stack(rows).astype(np.uint8)


def _adapter(tmp_path: Path) -> ArcFacePyTorchAdapter:
    checkpoint = tmp_path / "deterministic_arcface_fixture.pt"
    checkpoint.write_bytes(b"deterministic-test-checkpoint")
    spec = ModelSpec(
        family="arcface",
        architecture="deterministic_conv_fixture",
        training_dataset="synthetic_test_only",
        implementation_repository="https://example.invalid/test-only",
        checkpoint=CheckpointProvenance.from_file(
            checkpoint,
            source_url="https://example.invalid/test-only/checkpoint",
        ),
        preprocessing=PreprocessingSpec(
            input_height=4,
            input_width=4,
            source_color_order="rgb",
            model_color_order="rgb",
            channel_mean=(0.0, 0.0, 0.0),
            channel_std=(255.0, 255.0, 255.0),
        ),
        target_layer="features",
    )
    return ArcFacePyTorchAdapter(
        _DeterministicFaceEncoder(),
        spec,
        device="cpu",
    )


@pytest.fixture
def population_pipeline(tmp_path: Path):
    faces = _aligned_faces()
    adapter = _adapter(tmp_path)
    sample_ids = np.asarray(
        ["alpha-1", "alpha-2", "beta-1", "beta-2", "single-1", "missing-1"]
    )
    identity_ids = np.asarray(
        ["alpha", "alpha", "beta", "beta", "single", None],
        dtype=object,
    )
    scope_ids = np.asarray(["dev"] * len(sample_ids))

    prepared = prepare_population_saliency_inputs(
        adapter,
        faces,
        sample_ids=sample_ids,
        identity_ids=identity_ids,
        scope_ids=scope_ids,
        extraction_uid="synthetic-extraction-001",
        dataset_id="synthetic-faces",
        embedding_batch_size=2,
    )
    result = extract_population_gradcam(
        adapter,
        faces,
        prepared,
        gradcam_batch_size=3,
        capture_intermediates=False,
        minimum_pass_repeat_cosine=0.99999,
        faithfulness_fraction=None,
    )
    return faces, prepared, result


def test_population_gradcam_can_cap_only_the_backward_saliency_pass(tmp_path: Path):
    faces = _aligned_faces()
    adapter = _adapter(tmp_path)
    sample_ids = np.asarray(
        ["alpha-1", "alpha-2", "beta-1", "beta-2", "single-1", "missing-1"]
    )
    identity_ids = np.asarray(
        ["alpha", "alpha", "beta", "beta", "single", None],
        dtype=object,
    )
    prepared = prepare_population_saliency_inputs(
        adapter,
        faces,
        sample_ids=sample_ids,
        identity_ids=identity_ids,
        scope_ids=np.asarray(["dev"] * len(sample_ids)),
        extraction_uid="synthetic-extraction-capped",
        dataset_id="synthetic-faces",
        embedding_batch_size=2,
    )
    sample_mask = np.asarray([True, True, False, False, False, False])
    result = extract_population_gradcam(
        adapter,
        faces,
        prepared,
        gradcam_batch_size=2,
        minimum_pass_repeat_cosine=0.99999,
        faithfulness_fraction=None,
        saliency_sample_mask=sample_mask,
    )

    assert result.heatmap_sample_ids.tolist() == ["alpha-1", "alpha-2"]
    assert result.features["saliency_sample_selected"].tolist() == (
        sample_mask.tolist()
    )
    assert result.features["heatmap_available"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_population_pipeline_extracts_all_embeddings_and_all_eligible_gradcam(
    population_pipeline,
):
    faces, prepared, result = population_pipeline

    assert prepared.raw_embeddings.shape == (len(faces), 512)
    assert prepared.normalized_embeddings.shape == (len(faces), 512)
    assert np.linalg.norm(
        prepared.normalized_embeddings,
        axis=1,
    ) == pytest.approx(np.ones(len(faces)), abs=1e-6)

    expected_eligible = np.asarray([True, True, True, True, False, False])
    assert np.array_equal(prepared.loo_templates.eligible, expected_eligible)
    assert prepared.loo_templates.exclusion_reasons[4] == SINGLETON_IDENTITY_REASON
    assert prepared.loo_templates.exclusion_reasons[5] == MISSING_IDENTITY_REASON
    assert prepared.loo_templates.templates[0] == pytest.approx(
        prepared.normalized_embeddings[1],
        abs=1e-6,
    )
    assert prepared.loo_templates.templates[1] == pytest.approx(
        prepared.normalized_embeddings[0],
        abs=1e-6,
    )

    eligible_indices = np.flatnonzero(expected_eligible)
    assert (
        result.heatmap_sample_ids.tolist()
        == prepared.sample_ids[eligible_indices].tolist()
    )
    assert result.normalized_heatmaps.shape == (4, 4, 4)
    assert np.isfinite(result.normalized_heatmaps).all()
    assert result.normalized_heatmaps.min() >= 0.0
    assert result.normalized_heatmaps.max() <= 1.0
    assert result.activations is None
    assert result.gradients is None
    assert result.pass_b_normalized_embeddings == pytest.approx(
        prepared.normalized_embeddings[eligible_indices],
        abs=1e-6,
    )

    features = result.features.set_index("sample_id")
    assert len(features) == len(faces)
    assert features.index.is_unique
    assert features.loc["alpha-1", "heatmap_available"]
    assert features.loc["beta-2", "heatmap_available"]
    assert not features.loc["single-1", "heatmap_available"]
    assert not features.loc["missing-1", "heatmap_available"]
    assert features.loc["single-1", "heatmap_index"] == -1
    assert features.loc["missing-1", "heatmap_index"] == -1
    assert pd.isna(features.loc["single-1", "gradcam_target_score"])
    assert pd.isna(features.loc["missing-1", "gradcam_target_score"])
    assert (
        features.loc[
            expected_eligible,
            "pass_a_pass_b_embedding_cosine",
        ].min()
        >= 0.99999
    )
    assert (
        features.loc[
            expected_eligible,
            "pass_a_pass_b_target_score_abs_diff",
        ].max()
        <= 1e-5
    )
    assert "faithfulness_occlusion_fraction" not in features.columns
    assert "high_saliency_occlusion_score_drop" not in features.columns


def test_population_artifacts_are_roundtrippable_immutable_and_hash_checked(
    tmp_path: Path,
    population_pipeline,
):
    _faces, prepared, result = population_pipeline

    prepared_path = tmp_path / "prepared"
    assert (
        write_prepared_population_artifact(
            prepared,
            prepared_path,
            shard_size=2,
        )
        == prepared_path.resolve()
    )
    restored = read_prepared_population_artifact(prepared_path)
    assert restored.extraction_uid == prepared.extraction_uid
    assert restored.origin_embedding_artifact_uid == (
        prepared.origin_embedding_artifact_uid
    )
    assert restored.sample_ids.tolist() == prepared.sample_ids.tolist()
    assert restored.raw_embeddings == pytest.approx(
        prepared.raw_embeddings,
        abs=1e-7,
    )
    assert restored.loo_templates.templates == pytest.approx(
        prepared.loo_templates.templates,
        abs=1e-7,
        nan_ok=True,
    )
    assert np.array_equal(
        restored.loo_templates.eligible,
        prepared.loo_templates.eligible,
    )
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        write_prepared_population_artifact(prepared, prepared_path)

    prepared_manifest = json.loads(
        (prepared_path / "manifest.json").read_text(encoding="utf-8")
    )
    prepared_shard = prepared_path / prepared_manifest["embedding_shards"][0]["path"]
    prepared_shard.write_bytes(prepared_shard.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_prepared_population_artifact(prepared_path)

    saliency_path = tmp_path / "saliency"
    assert (
        write_population_saliency_artifact(
            result,
            saliency_path,
            shard_size=2,
            heatmap_dtype="float32",
        )
        == saliency_path.resolve()
    )
    restored_features = read_population_saliency_features(saliency_path)
    pd.testing.assert_frame_equal(
        restored_features,
        result.features,
        check_dtype=True,
    )
    restored_ids, restored_heatmaps = read_population_heatmaps(saliency_path)
    assert restored_ids.tolist() == result.heatmap_sample_ids.tolist()
    assert restored_heatmaps == pytest.approx(
        result.normalized_heatmaps,
        abs=1e-7,
    )
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        write_population_saliency_artifact(result, saliency_path)

    saliency_manifest = json.loads(
        (saliency_path / "manifest.json").read_text(encoding="utf-8")
    )
    saliency_shard = saliency_path / saliency_manifest["heatmap_shards"][0]["path"]
    saliency_shard.write_bytes(saliency_shard.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_population_heatmaps(saliency_path)
