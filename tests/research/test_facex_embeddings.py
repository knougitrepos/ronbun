from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

# Imports below must follow the optional dependency gate.
# ruff: noqa: E402
torch = pytest.importorskip("torch")

from research.embeddings.facex_zoo import (
    FACEX_FACTORY, main, model_spec_from_facex_export, register_facex_export,
)
from research.embeddings.manifests import read_model_spec
from research.embeddings.pytorch.facex_backbone import FaceXMobileFaceNet
from research.embeddings.pytorch.facex_zoo import load_facex_mobilefacenet_checkpoint
from research.embeddings.registry import create_pytorch_adapter_from_spec
from research.runtime.hashing import sha256_file


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def weights():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(734)
        return {dimension: FaceXMobileFaceNet(dimension).state_dict() for dimension in (512, 1024)}


def _metadata(dimension=1024, family="arcface"):
    config = {
        "backbone": {"type": "MobileFaceNet", "feat_dim": dimension, "dropout_rate": 0.6},
        "head": {"type": family.title(), "feat_dim": dimension},
        "dataset": {"name": "survface", "train_list_hash": "d" * 64},
        "preprocessing": {"input_size": [112, 112], "pixel_mean": 127.5, "pixel_std": 128.0, "color": "BGR", "alignment": "resize_only"},
    }
    return {
        "completed": True, "is_smoke": False, "head": family, "dataset": "survface",
        "feat_dim": dimension, "run_id": f"fixture_{family}_{dimension}", "epochs": 40,
        "source_sha256": "a" * 64, "parent_hash": "b" * 64,
        "source_config": config, "source_config_hash": _hash_config(config),
        "preprocessing": copy.deepcopy(config["preprocessing"]),
        "protocol_record": {"fixture_only": True, "training_identity_count": 2},
    }


def _hash_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _export(directory, weights, *, dimension=1024, family="arcface", metadata=None):
    path = directory / f"exported_mobilefacenet_{dimension}d_{family}.pt"
    metadata = _metadata(dimension, family) if metadata is None else metadata
    torch.save({"state_dict": weights[dimension], "metadata": metadata}, path)
    sidecar = path.with_name(path.stem + "_meta.json")
    sidecar.write_text(json.dumps({**metadata, "sha256": sha256_file(path), "exported_file": path.name}), encoding="utf-8")
    return path, sidecar


@pytest.mark.parametrize("dimension", [512, 1024])
@pytest.mark.parametrize("family", ["arcface", "cosface", "magface", "adaface"])
def test_formal_export_adapter_preserves_raw_norm_and_rgb_to_bgr(tmp_path, weights, dimension, family):
    path, _ = _export(tmp_path, weights, dimension=dimension, family=family)
    spec = model_spec_from_facex_export(path)
    assert spec.module_factory == FACEX_FACTORY
    assert spec.family == family and spec.embedding_dim == dimension
    adapter = create_pytorch_adapter_from_spec(spec, device="cpu")
    image = np.random.default_rng(42).integers(0, 256, size=(1, 112, 112, 3), dtype=np.uint8)
    output = adapter.embed(image)
    expected_model = FaceXMobileFaceNet(dimension).eval()
    expected_model.load_state_dict(weights[dimension], strict=True)
    expected_input = torch.from_numpy(((image[..., ::-1].astype(np.float32) - 127.5) / 128.0).transpose(0, 3, 1, 2).copy())
    with torch.inference_mode():
        expected = expected_model(expected_input).numpy()
    np.testing.assert_array_equal(output.raw_embedding, expected)
    np.testing.assert_allclose(output.raw_norm, np.linalg.norm(expected, axis=1), rtol=1e-5)
    np.testing.assert_allclose(output.normalized_embedding, expected / output.raw_norm[:, None], rtol=1e-5)
    assert output.raw_embedding.shape == (1, dimension)
    assert adapter.target_layer is adapter.model.conv_6_sep


def test_registration_is_immutable_and_color_contract_changes_uid(tmp_path, weights, capsys):
    path, sidecar = _export(tmp_path, weights)
    hashes = (sha256_file(path), sha256_file(sidecar))
    registry = tmp_path / "registry"
    registered = register_facex_export(path, registry)
    assert register_facex_export(path, registry) == registered
    spec = read_model_spec(registered)
    assert spec.model_uid == model_spec_from_facex_export(path).model_uid
    assert spec.model_uid != model_spec_from_facex_export(path, source_color_order="bgr").model_uid
    assert main([str(path), "--registry-root", str(registry)]) == 0
    assert str(registered) in capsys.readouterr().out
    assert (sha256_file(path), sha256_file(sidecar)) == hashes


@pytest.mark.parametrize("field,value,match", [
    ("completed", False, "completed"),
    ("is_smoke", True, "non-smoke"),
    ("is_smoke", None, "non-smoke"),
    ("head", "edgeface", "head"),
    ("feat_dim", 256, "feat_dim"),
    ("dataset", "tinyface", "dataset"),
    ("source_config_hash", "0" * 64, "source_config_hash"),
    ("parent_hash", "unknown", "parent_hash"),
    ("protocol_record", None, "protocol_record"),
])
def test_rejects_unsupported_or_incomplete_provenance(tmp_path, weights, field, value, match):
    metadata = _metadata()
    metadata[field] = value
    path, _ = _export(tmp_path, weights, metadata=metadata)
    with pytest.raises(ValueError, match=match):
        model_spec_from_facex_export(path)


def test_rejects_checkpoint_and_sidecar_tampering(tmp_path, weights):
    path, sidecar = _export(tmp_path, weights)
    metadata = json.loads(sidecar.read_text())
    metadata["run_id"] = "altered"
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata and sidecar disagree"):
        model_spec_from_facex_export(path)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="sha256"):
        model_spec_from_facex_export(path)


def test_rejects_validly_hashed_but_wrong_input_contract(tmp_path, weights):
    metadata = _metadata()
    metadata["preprocessing"]["pixel_std"] = 127.5
    metadata["source_config"]["preprocessing"]["pixel_std"] = 127.5
    metadata["source_config_hash"] = _hash_config(metadata["source_config"])
    path, _ = _export(tmp_path, weights, metadata=metadata)
    with pytest.raises(ValueError, match="112x112 BGR"):
        model_spec_from_facex_export(path)


@pytest.mark.parametrize("interpolation", ["bicubic", "bilinear"])
def test_input_policy_is_read_from_validated_export(tmp_path, weights, interpolation):
    from research.experiments.facex_inputs import resolve_input_transform
    metadata = _metadata()
    policy = dict(method="resize_only", interpolation=interpolation, input_size=[112, 112],
                  color="BGR", geometry="full_crop_stretch", face_detection="none")
    metadata["source_config"]["preprocessing"]["alignment_policy"] = policy
    metadata["preprocessing"]["alignment_policy"] = policy
    metadata["source_config_hash"] = _hash_config(metadata["source_config"])
    path, sidecar = _export(tmp_path, weights, metadata=metadata)
    spec = model_spec_from_facex_export(path)
    assert resolve_input_transform(spec)["interpolation"] == interpolation
    modified = json.loads(sidecar.read_text())
    modified["preprocessing"]["alignment_policy"]["interpolation"] = "nearest"
    sidecar.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="metadata and sidecar disagree"):
        resolve_input_transform(spec)


def test_rejects_spec_inconsistent_with_export(tmp_path, weights):
    path, _ = _export(tmp_path, weights)
    spec = model_spec_from_facex_export(path)
    with pytest.raises(ValueError, match="model spec does not match"):
        load_facex_mobilefacenet_checkpoint(replace(spec, embedding_dim=512))
    with pytest.raises(ValueError, match="model spec does not match"):
        load_facex_mobilefacenet_checkpoint(replace(spec, family="cosface"))


def test_rejects_classifier_weights_in_backbone_export(tmp_path, weights):
    extra = dict(weights[1024], **{"head.kernel": torch.ones(1024, 2)})
    path, _ = _export(tmp_path, {1024: extra})
    with pytest.raises(RuntimeError, match="Unexpected key"):
        model_spec_from_facex_export(path)


@pytest.mark.parametrize("dimension", [512, 1024])
def test_portable_backbone_matches_available_facex_source(weights, dimension):
    upstream_path = Path("F:/FaceX-Zoo/backbone/MobileFaceNets.py")
    if not upstream_path.is_file():
        pytest.skip("Optional local FaceX-Zoo checkout is absent")
    spec = importlib.util.spec_from_file_location("facex_upstream_test", upstream_path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    reference = upstream.MobileFaceNet(dimension, 7, 7, dropout_rate=0.6).eval()
    portable = FaceXMobileFaceNet(dimension).eval()
    reference.load_state_dict(weights[dimension], strict=True)
    portable.load_state_dict(weights[dimension], strict=True)
    sample = torch.linspace(-1, 1, 2 * 3 * 112 * 112).reshape(2, 3, 112, 112)
    with torch.inference_mode():
        torch.testing.assert_close(portable(sample), reference(sample), atol=0, rtol=0)
