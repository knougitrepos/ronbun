from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.embeddings.base import (  # noqa: E402
    CheckpointProvenance,
    ModelSpec,
    PreprocessingSpec,
)
from research.embeddings.pytorch.official_backbones import (  # noqa: E402
    build_adaface_backbone,
    build_arcface_backbone,
    build_magface_backbone,
)
from research.embeddings.pytorch.official_loaders import (  # noqa: E402
    CheckpointCompatibilityError,
    load_adaface_checkpoint,
    load_arcface_checkpoint,
    load_magface_checkpoint,
)
from research.embeddings.registry import create_pytorch_adapter_from_spec  # noqa: E402


def _spec(
    checkpoint: Path,
    *,
    family: str,
    architecture: str,
    factory: str,
    target_layer: str,
    model_color_order: str,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> ModelSpec:
    return ModelSpec(
        family=family,
        architecture=architecture,
        training_dataset="synthetic_state_dict_fixture",
        implementation_repository="https://example.invalid/upstream",
        checkpoint=CheckpointProvenance.from_file(
            checkpoint,
            source_url="https://example.invalid/checkpoint",
        ),
        preprocessing=PreprocessingSpec(
            input_height=112,
            input_width=112,
            source_color_order="rgb",
            model_color_order=model_color_order,
            channel_mean=mean,
            channel_std=std,
        ),
        target_layer=target_layer,
        module_factory=factory,
    )


@pytest.mark.parametrize(
    (
        "family",
        "architecture",
        "builder",
        "loader",
        "factory",
        "target_layer",
        "model_color_order",
        "mean",
        "std",
        "wrapper",
    ),
    [
        (
            "arcface",
            "iresnet18",
            build_arcface_backbone,
            load_arcface_checkpoint,
            "research.embeddings.pytorch.official_loaders:load_arcface_checkpoint",
            "layer4.1.conv2",
            "rgb",
            (127.5, 127.5, 127.5),
            (127.5, 127.5, 127.5),
            "arcface",
        ),
        (
            "adaface",
            "ir_18",
            build_adaface_backbone,
            load_adaface_checkpoint,
            "research.embeddings.pytorch.official_loaders:load_adaface_checkpoint",
            "body.7.res_layer.4",
            "bgr",
            (127.5, 127.5, 127.5),
            (127.5, 127.5, 127.5),
            "adaface",
        ),
        (
            "magface",
            "iresnet18",
            build_magface_backbone,
            load_magface_checkpoint,
            "research.embeddings.pytorch.official_loaders:load_magface_checkpoint",
            "layer4.1.conv2",
            "bgr",
            (0.0, 0.0, 0.0),
            (255.0, 255.0, 255.0),
            "magface",
        ),
    ],
)
def test_official_loader_strictly_restores_checkpoint_and_runs_adapter(
    tmp_path,
    family,
    architecture,
    builder,
    loader,
    factory,
    target_layer,
    model_color_order,
    mean,
    std,
    wrapper,
):
    torch.manual_seed(17)
    source = builder(architecture)
    state_dict = source.state_dict()
    if wrapper == "arcface":
        payload = {"state_dict_backbone": state_dict}
    elif wrapper == "adaface":
        payload = {
            "state_dict": {
                **{f"model.{key}": value for key, value in state_dict.items()},
                "head.synthetic_weight": torch.ones(1),
            }
        }
    else:
        payload = {
            "state_dict": {
                **{
                    f"module.features.{key}": value
                    for key, value in state_dict.items()
                },
                "module.loss.synthetic_weight": torch.ones(1),
            }
        }
    checkpoint = tmp_path / f"{family}.pt"
    torch.save(payload, checkpoint)
    spec = _spec(
        checkpoint,
        family=family,
        architecture=architecture,
        factory=factory,
        target_layer=target_layer,
        model_color_order=model_color_order,
        mean=mean,
        std=std,
    )

    restored = loader(spec)
    for key, expected in state_dict.items():
        assert torch.equal(restored.state_dict()[key], expected)

    adapter = create_pytorch_adapter_from_spec(spec)
    faces = np.full((1, 112, 112, 3), 127, dtype=np.uint8)
    output = adapter.embed(faces)

    assert output.raw_embedding.shape == (1, 512)
    assert output.raw_norm.shape == (1,)
    assert output.raw_norm[0] > 0
    assert np.linalg.norm(output.normalized_embedding[0]) == pytest.approx(
        1.0, abs=1e-5
    )
    assert adapter.target_layer is not None


def test_official_loader_rejects_partial_state_dict(tmp_path):
    source = build_arcface_backbone("iresnet18")
    state_dict = dict(source.state_dict())
    state_dict.pop(next(iter(state_dict)))
    checkpoint = tmp_path / "partial.pt"
    torch.save(state_dict, checkpoint)
    spec = _spec(
        checkpoint,
        family="arcface",
        architecture="iresnet18",
        factory=(
            "research.embeddings.pytorch.official_loaders:"
            "load_arcface_checkpoint"
        ),
        target_layer="layer4.1.conv2",
        model_color_order="rgb",
        mean=(127.5, 127.5, 127.5),
        std=(127.5, 127.5, 127.5),
    )

    with pytest.raises(CheckpointCompatibilityError, match="not an exact match"):
        load_arcface_checkpoint(spec)
