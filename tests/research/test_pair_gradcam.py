import numpy as np
import pandas as pd
import pytest

from research.explainability.gradcam import (
    PairCosineGradCAM,
    TorchUnavailableError,
    central_region_concentration,
    is_torch_available,
    occlude_by_saliency,
    occlusion_faithfulness,
    pair_cosine_target,
    saliency_concentration,
    saliency_entropy,
    select_gradcam_cases,
    select_population_representative_cases,
)


def test_gradcam_package_keeps_missing_torch_isolated():
    if is_torch_available():
        return
    with pytest.raises(TorchUnavailableError, match="optional PyTorch"):
        PairCosineGradCAM(object(), object())


def test_pair_cosine_target_detaches_gallery_and_rejects_pq_codes():
    if not is_torch_available():
        pytest.skip("optional PyTorch environment is not installed")
    import torch

    query = torch.tensor([[1.0, 0.5]], requires_grad=True)
    gallery = torch.tensor([[1.0, 0.0]], requires_grad=True)
    score = pair_cosine_target(query, gallery)
    score.sum().backward()

    assert query.grad is not None
    assert gallery.grad is None
    with pytest.raises(TypeError, match="hard PQ codes"):
        pair_cosine_target(query, torch.tensor([[1, 2]], dtype=torch.uint8))


def test_pair_gradcam_toy_cnn_normalizes_maps_and_removes_hooks():
    if not is_torch_available():
        pytest.skip("optional PyTorch environment is not installed")
    import torch

    class ToyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
            with torch.no_grad():
                self.features.weight.fill_(1.0)

        def forward(self, images):
            activations = self.features(images)
            return torch.stack(
                (
                    activations.mean(dim=(1, 2, 3)),
                    activations.square().mean(dim=(1, 2, 3)),
                ),
                dim=1,
            )

    model = ToyEncoder().eval()
    analyzer = PairCosineGradCAM(model, model.features)
    before_hooks = len(model.features._forward_hooks)
    query = torch.tensor([[[[0.1, 0.8], [0.4, 1.0]]]])
    gallery = torch.tensor([1.0, 0.0], requires_grad=True)

    result = analyzer.generate(
        query,
        gallery,
        batch_mode="single",
        target_space="origin_embedding",
    )

    assert result.heatmaps.shape == (1, 2, 2)
    assert np.isfinite(result.heatmaps).all()
    assert result.heatmaps.min() >= 0.0
    assert result.heatmaps.max() <= 1.0
    assert result.target_scores.shape == (1,)
    assert gallery.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())
    assert len(model.features._forward_hooks) == before_hooks

    with pytest.raises(ValueError, match="embedding dimension"):
        analyzer.generate(
            query,
            torch.tensor([1.0, 0.0, 0.0]),
            batch_mode="single",
        )
    assert len(model.features._forward_hooks) == before_hooks

    batch = query.repeat(2, 1, 1, 1)
    with pytest.raises(ValueError, match="exactly one"):
        analyzer.generate(batch, gallery, batch_mode="single")
    independent = analyzer.generate(batch, gallery, batch_mode="independent")
    assert independent.heatmaps.shape == (2, 2, 2)


def test_select_gradcam_cases_is_deterministic_and_exclusive():
    paired = pd.DataFrame(
        {
            "sample_id": list("abcdef"),
            "compression_family": ["pca"] * 6,
            "compression_profile": ["pca_128"] * 6,
            "angular_error_rad": [0.01, 0.9, 0.4, 0.6, 0.2, 0.3],
            "origin_fallback_used": [False] * 6,
        }
    )
    retrieval = pd.DataFrame(
        {
            "query_id": list("abcdef"),
            "compression_family": ["pca"] * 6,
            "compression_profile": ["pca_128"] * 6,
            "top1_score_drift": [0.01, -0.4, -0.2, -0.3, 0.1, 0.12],
            "agreement_with_origin": [True, True, False, False, True, True],
            "threshold_crossing": [False, False, False, True, False, False],
            "origin_fallback_used": [False] * 6,
        }
    )

    selected = select_gradcam_cases(
        paired,
        retrieval,
        cases_per_group=1,
        seed=17,
    )
    shuffled = select_gradcam_cases(
        paired.sample(frac=1.0, random_state=2),
        retrieval.sample(frac=1.0, random_state=3),
        cases_per_group=1,
        seed=17,
    )

    pd.testing.assert_frame_equal(selected, shuffled)
    assert set(selected["case_group"]) == {
        "stable",
        "high_error",
        "rank_flip",
        "threshold_crossing",
    }
    assert selected["query_id"].is_unique
    assert selected["case_id"].is_unique
    assert selected["case_id"].str.startswith("gradcam-").all()
    by_group = selected.set_index("case_group")["query_id"]
    assert by_group["stable"] == "a"
    assert by_group["high_error"] == "b"
    assert by_group["rank_flip"] == "c"
    assert by_group["threshold_crossing"] == "d"


def test_select_gradcam_cases_rejects_fallback_and_ambiguous_rows():
    paired = pd.DataFrame(
        {
            "sample_id": ["a"],
            "compression_family": ["pca"],
            "compression_profile": ["pca_128"],
            "angular_error_rad": [0.1],
            "origin_fallback_used": [True],
        }
    )
    retrieval = pd.DataFrame(
        {
            "query_id": ["a"],
            "compression_family": ["pca"],
            "compression_profile": ["pca_128"],
            "top1_score_drift": [0.1],
            "agreement_with_origin": [True],
            "threshold_crossing": [False],
            "origin_fallback_used": [False],
        }
    )
    with pytest.raises(ValueError, match="fallback-free"):
        select_gradcam_cases(paired, retrieval)

    paired["origin_fallback_used"] = False
    duplicated = pd.concat([retrieval, retrieval], ignore_index=True)
    with pytest.raises(ValueError, match="evaluation policy"):
        select_gradcam_cases(paired, duplicated)


def test_population_case_selection_runs_only_on_joined_retrieval_rows():
    joined = pd.DataFrame(
        {
            "extraction_uid": ["extract"] * 5,
            "dataset_id": ["lfw"] * 5,
            "sample_id": [f"sample-{index}" for index in range(5)],
            "model_uid": ["arcface-a"] * 5,
            "compression_family": ["pca"] * 5,
            "compression_profile": ["pca_64"] * 5,
            "angular_error_rad": [0.01, 0.8, 0.5, 0.4, 0.2],
            "top1_score_drift": [0.0, -0.3, -0.2, 0.1, np.nan],
            "agreement_with_origin": [True, True, False, True, None],
            "threshold_crossing": [False, False, False, True, None],
            "origin_fallback_used": [False] * 5,
            "saliency_target_eligible": [True] * 5,
            "heatmap_available": [True] * 5,
            "retrieval_metrics_available": [True, True, True, True, False],
        }
    )

    selected = select_population_representative_cases(
        joined,
        cases_per_group=1,
        seed=13,
    )

    assert set(selected["case_group"]) == {
        "stable",
        "high_error",
        "rank_flip",
        "threshold_crossing",
    }
    assert "sample-4" not in set(selected["sample_id"])


def test_saliency_metrics_and_occlusion_faithfulness():
    heatmaps = np.array(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    entropy = saliency_entropy(heatmaps)
    assert entropy[0] == pytest.approx(0.0)
    assert entropy[1] == pytest.approx(1.0)
    assert np.isnan(entropy[2])

    mask = np.array([[False, False], [False, True]])
    concentration = saliency_concentration(heatmaps, mask)
    assert concentration[:2] == pytest.approx([1.0, 0.25])
    assert np.isnan(concentration[2])
    assert central_region_concentration(
        np.ones((1, 4, 4)),
        height_fraction=0.5,
        width_fraction=0.5,
    )[0] == pytest.approx(0.25)

    faithfulness = occlusion_faithfulness(
        np.array([0.9, 0.8]),
        np.array([0.4, 0.5]),
        control_occluded_scores=np.array([0.7, 0.6]),
    )
    assert faithfulness["saliency_score_drop"].to_numpy() == pytest.approx([0.5, 0.3])
    assert faithfulness["faithfulness_gain_over_control"].to_numpy() == pytest.approx(
        [0.3, 0.1]
    )


def test_saliency_occlusion_is_exact_and_deterministic():
    images = np.full((2, 4, 4, 3), 255, dtype=np.uint8)
    heatmaps = np.zeros((2, 2, 2), dtype=np.float32)
    heatmaps[:, 0, 0] = 1.0

    high = occlude_by_saliency(
        images,
        heatmaps,
        fraction=0.25,
        strategy="high_saliency",
        fill_value=(10, 20, 30),
        seed=7,
    )
    low = occlude_by_saliency(
        images,
        heatmaps,
        fraction=0.25,
        strategy="low_saliency",
        fill_value=(10, 20, 30),
        seed=7,
    )
    random_a = occlude_by_saliency(
        images,
        heatmaps,
        fraction=0.25,
        strategy="random",
        seed=7,
    )
    random_b = occlude_by_saliency(
        images,
        heatmaps,
        fraction=0.25,
        strategy="random",
        seed=7,
    )

    assert np.all(high[:, :2, :2] == np.array([10, 20, 30], dtype=np.uint8))
    assert np.all(low[:, :2, :2] == 255)
    assert np.count_nonzero(np.all(high == (10, 20, 30), axis=-1)) == 8
    assert np.array_equal(random_a, random_b)
