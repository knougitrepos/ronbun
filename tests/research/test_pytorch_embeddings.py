from __future__ import annotations

import contextlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from research.embeddings.base import (
    CheckpointProvenance,
    ModelSpec,
    PreprocessingSpec,
)
from research.embeddings.pytorch import PyTorchUnavailableError, resolve_target_layer
from research.embeddings.manifests import (
    ModelSpecSelectionError,
    read_model_spec,
    select_model_spec,
    select_model_spec_by_profile,
    write_model_spec,
)
from research.embeddings.registry import (
    ModelFactoryUnavailableError,
    create_pytorch_adapter,
    create_pytorch_adapter_from_spec,
    load_pytorch_module_factory,
)


def _spec(tmp_path: Path, family: str = "arcface") -> ModelSpec:
    checkpoint = tmp_path / f"{family}.pt"
    checkpoint.write_bytes(f"test-{family}".encode())
    return ModelSpec(
        family=family,
        architecture="iresnet_test",
        training_dataset="test_fixture_only",
        implementation_repository="https://example.invalid/test-fixture",
        checkpoint=CheckpointProvenance.from_file(
            checkpoint,
            source_url="https://example.invalid/test-fixture/checkpoint",
        ),
        preprocessing=PreprocessingSpec(
            input_height=2,
            input_width=2,
            source_color_order="bgr",
            model_color_order="rgb",
            channel_mean=(127.5, 127.5, 127.5),
            channel_std=(128.0, 128.0, 128.0),
        ),
        target_layer="features",
    )


class _FakeTensor:
    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values)

    @property
    def ndim(self) -> int:
        return self.values.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def to(self, _device: object) -> _FakeTensor:
        return self

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def float(self) -> _FakeTensor:
        return _FakeTensor(self.values.astype(np.float32))

    def numpy(self) -> np.ndarray:
        return self.values

    def all(self) -> bool:
        return bool(self.values.all())

    def any(self) -> bool:
        return bool(self.values.any())

    def unsqueeze(self, dimension: int) -> _FakeTensor:
        return _FakeTensor(np.expand_dims(self.values, axis=dimension))

    def __le__(self, other: object) -> _FakeTensor:
        return _FakeTensor(self.values <= other)

    def __truediv__(self, other: _FakeTensor) -> _FakeTensor:
        return _FakeTensor(self.values / other.values)


class _FakeModule:
    def __init__(self) -> None:
        self.training = True
        self.features = object()
        self.seen: np.ndarray | None = None

    def to(self, _device: object) -> _FakeModule:
        return self

    def eval(self) -> _FakeModule:
        self.training = False
        return self

    def named_modules(self):
        return [("", self), ("features", self.features)]

    def __call__(self, inputs: _FakeTensor) -> _FakeTensor:
        self.seen = inputs.values
        batch = inputs.values.shape[0]
        values = np.tile(np.arange(1, 513, dtype=np.float32), (batch, 1))
        return _FakeTensor(values)


def _fake_torch_module() -> types.ModuleType:
    module = types.ModuleType("torch")
    module.nn = types.SimpleNamespace(Module=_FakeModule)
    module.device = lambda value: value
    module.from_numpy = _FakeTensor
    module.is_tensor = lambda value: isinstance(value, _FakeTensor)
    module.isfinite = lambda value: _FakeTensor(np.isfinite(value.values))
    module.linalg = types.SimpleNamespace(
        vector_norm=lambda value, ord, dim: _FakeTensor(
            np.linalg.norm(value.values, ord=ord, axis=dim)
        )
    )
    module.inference_mode = contextlib.nullcontext
    module.enable_grad = contextlib.nullcontext
    return module


def test_model_spec_records_checkpoint_and_preprocessing_provenance(tmp_path):
    spec = _spec(tmp_path)

    manifest = spec.to_manifest()

    assert manifest["comparison_scope"] == "pretrained_checkpoint"
    assert manifest["checkpoint"]["verified_local_file"] is True
    assert len(manifest["checkpoint"]["sha256"]) == 64
    assert len(manifest["preprocess_hash"]) == 64
    assert spec.model_uid.startswith("arcface-")


def test_model_spec_manifest_round_trip_revalidates_checkpoint(tmp_path):
    spec = _spec(tmp_path)
    path = write_model_spec(tmp_path / "registry" / "arcface.json", spec)

    loaded = read_model_spec(path)

    assert loaded == spec
    assert write_model_spec(path, spec) == path

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["preprocessing"]["channel_std"] = [64.0, 64.0, 64.0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="preprocess_hash"):
        read_model_spec(path)


def test_model_spec_manifest_detects_changed_checkpoint(tmp_path):
    spec = _spec(tmp_path)
    path = write_model_spec(tmp_path / "arcface.json", spec)
    Path(spec.checkpoint.path).write_bytes(b"changed-after-registration")

    with pytest.raises(ValueError, match="no longer matches"):
        read_model_spec(path)


def test_model_spec_registry_selects_unique_family_or_exact_uid(tmp_path):
    registry = tmp_path / "registry"
    arcface = _spec(tmp_path, "arcface")
    adaface = _spec(tmp_path, "adaface")
    arcface_path = write_model_spec(
        registry / f"{arcface.model_uid}.json", arcface
    )
    write_model_spec(registry / f"{adaface.model_uid}.json", adaface)

    selected_path, selected = select_model_spec(registry, family="arcface")
    exact_path, exact = select_model_spec(
        registry,
        family="arcface",
        model_uid=arcface.model_uid,
    )

    assert selected_path == arcface_path
    assert selected == arcface
    assert exact_path == arcface_path
    assert exact == arcface


def test_model_spec_registry_requires_uid_when_family_is_ambiguous(tmp_path):
    registry = tmp_path / "registry"
    first_root = tmp_path / "first"
    first_root.mkdir()
    second_root = tmp_path / "second"
    second_root.mkdir()
    first = _spec(first_root, "arcface")
    write_model_spec(registry / f"{first.model_uid}.json", first)
    second = _spec(second_root, "arcface")
    Path(second.checkpoint.path).write_bytes(b"different-arcface-checkpoint")
    second = ModelSpec(
        family=second.family,
        architecture=second.architecture,
        training_dataset=second.training_dataset,
        implementation_repository=second.implementation_repository,
        checkpoint=CheckpointProvenance.from_file(
            second.checkpoint.path,
            source_url=second.checkpoint.source_url,
        ),
        preprocessing=second.preprocessing,
        target_layer=second.target_layer,
    )
    write_model_spec(registry / f"{second.model_uid}.json", second)

    with pytest.raises(ModelSpecSelectionError, match="MODEL_UID explicitly"):
        select_model_spec(registry, family="arcface")


def test_profile_selection_pins_exact_uid_and_checkpoint(tmp_path):
    registry = tmp_path / "registry"
    first_root = tmp_path / "first"
    first_root.mkdir()
    second_root = tmp_path / "second"
    second_root.mkdir()
    first = _spec(first_root, "arcface")
    write_model_spec(registry / f"{first.model_uid}.json", first)
    second_base = _spec(second_root, "arcface")
    Path(second_base.checkpoint.path).write_bytes(b"second-arcface-checkpoint")
    second = ModelSpec(
        family=second_base.family,
        architecture=second_base.architecture,
        training_dataset=second_base.training_dataset,
        implementation_repository=second_base.implementation_repository,
        checkpoint=CheckpointProvenance.from_file(
            second_base.checkpoint.path,
            source_url=second_base.checkpoint.source_url,
        ),
        preprocessing=second_base.preprocessing,
        target_layer=second_base.target_layer,
    )
    second_path = write_model_spec(
        registry / f"{second.model_uid}.json",
        second,
    )
    profile = {
        "family": "arcface",
        "architecture": second.architecture,
        "training_dataset": second.training_dataset,
        "model_uid": second.model_uid,
        "checkpoint_path": second.checkpoint.path,
    }

    selected_path, selected = select_model_spec_by_profile(
        registry,
        profile_id="arcface_test",
        profile_config=profile,
    )

    assert selected_path == second_path
    assert selected == second

    wrong_checkpoint = dict(
        profile,
        checkpoint_path=first.checkpoint.path,
    )
    with pytest.raises(ModelSpecSelectionError, match="pins checkpoint"):
        select_model_spec_by_profile(
            registry,
            profile_id="arcface_test",
            profile_config=wrong_checkpoint,
        )


def test_model_spec_registry_rejects_unsafe_or_wrong_family_uid(tmp_path):
    registry = tmp_path / "registry"
    registry.mkdir()

    with pytest.raises(ValueError, match="20 lowercase hex"):
        select_model_spec(
            registry,
            family="arcface",
            model_uid="../summary",
        )
    with pytest.raises(ModelSpecSelectionError, match="requested family"):
        select_model_spec(
            registry,
            family="arcface",
            model_uid="adaface-" + "0" * 20,
        )


def test_factory_boundary_fails_before_guessing_an_official_loader(tmp_path):
    with pytest.raises(ModelFactoryUnavailableError, match="no verified"):
        create_pytorch_adapter(_spec(tmp_path))


def test_factory_path_requires_an_explicit_callable(monkeypatch):
    module = types.ModuleType("fixture_loader")
    module.build = lambda spec: spec
    module.not_callable = 3
    monkeypatch.setitem(sys.modules, "fixture_loader", module)

    assert load_pytorch_module_factory("fixture_loader:build") is module.build
    with pytest.raises(ValueError, match="package.module:function"):
        load_pytorch_module_factory("fixture_loader.build")
    with pytest.raises(TypeError, match="not callable"):
        load_pytorch_module_factory("fixture_loader:not_callable")


def test_adapter_can_be_created_from_factory_recorded_in_spec(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    module = types.ModuleType("fixture_loader")
    created = _FakeModule()
    module.build = lambda spec: created
    monkeypatch.setitem(sys.modules, "fixture_loader", module)
    base_spec = _spec(tmp_path)
    spec = ModelSpec(
        family=base_spec.family,
        architecture=base_spec.architecture,
        training_dataset=base_spec.training_dataset,
        implementation_repository=base_spec.implementation_repository,
        checkpoint=base_spec.checkpoint,
        preprocessing=base_spec.preprocessing,
        target_layer=base_spec.target_layer,
        module_factory="fixture_loader:build",
    )

    adapter = create_pytorch_adapter_from_spec(spec)

    assert adapter.model is created


def test_missing_torch_does_not_break_existing_package_import(tmp_path, monkeypatch):
    import research.embeddings as embeddings
    import research.embeddings.pytorch._torch as torch_boundary

    real_import_module = torch_boundary.importlib.import_module

    def missing_torch(name: str):
        if name == "torch":
            raise ModuleNotFoundError("torch is intentionally absent")
        return real_import_module(name)

    monkeypatch.setattr(torch_boundary.importlib, "import_module", missing_torch)

    assert embeddings.ArcFaceFeatureExtractor is not None
    with pytest.raises(PyTorchUnavailableError, match="optional 'torch'"):
        create_pytorch_adapter(_spec(tmp_path), module=object())


@pytest.mark.parametrize("family", ["arcface", "adaface", "magface"])
def test_injected_module_returns_raw_norm_and_normalized_embeddings(
    tmp_path,
    monkeypatch,
    family,
):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    model = _FakeModule()
    adapter = create_pytorch_adapter(_spec(tmp_path, family), module=model)
    bgr_faces = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    bgr_faces[..., 0] = 10
    bgr_faces[..., 2] = 30

    output = adapter.embed(bgr_faces)

    assert model.training is False
    assert output.raw_embedding.shape == (2, 512)
    assert output.raw_norm.shape == (2,)
    assert output.normalized_embedding.shape == (2, 512)
    assert np.allclose(np.linalg.norm(output.normalized_embedding, axis=1), 1.0)
    assert model.seen is not None
    assert model.seen[0, 0, 0, 0] == pytest.approx((30.0 - 127.5) / 128.0)
    assert adapter.target_layer is model.features
    assert adapter.select_embedding_tensor(_FakeTensor(np.ones((1, 512)))).shape == (
        1,
        512,
    )


def test_target_layer_resolution_is_exact_and_fail_fast():
    model = _FakeModule()

    assert resolve_target_layer(model, "features") is model.features
    with pytest.raises(ValueError, match="was not found"):
        resolve_target_layer(model, "last_conv")


def test_unverified_declared_checkpoint_is_rejected_by_adapter(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    spec = _spec(tmp_path)
    unverified = CheckpointProvenance(
        path=spec.checkpoint.path,
        sha256=spec.checkpoint.sha256,
        source_url=spec.checkpoint.source_url,
        verified_local_file=False,
    )
    spec = ModelSpec(
        family=spec.family,
        architecture=spec.architecture,
        training_dataset=spec.training_dataset,
        implementation_repository=spec.implementation_repository,
        checkpoint=unverified,
        preprocessing=spec.preprocessing,
        target_layer=spec.target_layer,
    )

    with pytest.raises(ValueError, match="not verified from a local file"):
        create_pytorch_adapter(spec, module=_FakeModule())
