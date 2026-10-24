from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from threadpoolctl import threadpool_limits

from research.datasets.lfw_pairs import outer_fold_development
from research.embeddings.base import CheckpointProvenance, ModelSpec, PreprocessingSpec
from research.embeddings.manifests import write_model_spec
from research.experiments.facex_lfw_config import generate_configs
from research.experiments.model_bindings import model_bindings
from research.experiments import lfw_resize_inputs, lfw_pair_verification

ROOT = Path(__file__).resolve().parents[2]


def test_model_bindings_never_infer_new_checkpoint_from_head_name():
    config = dict(source_runs={"surv_arc_s8972": "source"}, embedding_dim=1024)
    with pytest.raises(ValueError, match="explicit model_uids"):
        model_bindings(config)
    config["model_uids"] = {"surv_arc_s8972": "arcface-new-uid"}
    assert model_bindings(config) == (1024, config["model_uids"])


def test_people_fold_excludes_unused_test_people_from_pq(tmp_path):
    people, population, pairs = [], [], []
    for fold in range(1, 11):
        if people:
            people.append(dict(name=None, images=None))
        for i in range(31):
            name = f"f{fold}_person{i}"
            people.append(dict(name=name, images=1))
            population.append(dict(image_id=name, identity_id="lfw:" + name))
        pairs.append(dict(fold=fold, left_identity=f"lfw:f{fold}_person0",
                          right_identity=f"lfw:f{fold}_person1"))
    people_path = tmp_path / "people.csv"
    pd.DataFrame(people).to_csv(people_path, index=False)
    population = pd.DataFrame(population)
    pairs = pd.DataFrame(pairs)
    used = set(pairs.left_identity) | set(pairs.right_identity)
    development = population.loc[~population.identity_id.isin(used)]
    result = outer_fold_development(people_path, population, pairs, development)
    assert len(development) == 290
    assert all(len(frame) == 261 for frame in result.values())
    for fold, frame in result.items():
        assert not frame.identity_id.str.startswith(f"lfw:f{fold}_").any()
    pairs.loc[0, "left_identity"] = "lfw:f2_person0"
    with pytest.raises(ValueError, match="pair fold"):
        outer_fold_development(people_path, population, pairs, development)


def test_resize_artifact_1024_roundtrip_and_dimension_tamper(tmp_path):
    rng = np.random.default_rng(17)
    vectors = rng.normal(size=(3, 1024)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = pd.DataFrame(dict(image_id=["a", "b", "c"], identity_id=["a", "b", "c"]))
    contract = dict(model_uid="cosface-pinned", checkpoint_sha256="c", preprocess_hash="p",
                    model_spec=dict(embedding_dim=1024))
    lfw_resize_inputs._write_source(tmp_path / "source", index, vectors, np.ones(3), contract, {})
    _, _, _, prepared, _, _ = lfw_resize_inputs.load_resize_source(tmp_path, "source")
    np.testing.assert_array_equal(prepared.normalized_embeddings, vectors)
    assert prepared.embedding_dim == 1024
    prepared.normalized_embeddings._mmap.close()
    prepared.raw_norms._mmap.close()
    path = tmp_path / "source/manifest.json"
    metadata = json.loads(path.read_text())
    metadata["embedding_dimension"] = 512
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="embedding contract"):
        lfw_resize_inputs.load_resize_source(tmp_path, "source")


def test_config_generator_binds_exact_checkpoints_and_new_artifacts(tmp_path):
    checkpoint = tmp_path / "synthetic.pt"
    checkpoint.write_bytes(b"config-only fixture; not an inference checkpoint")
    spec = ModelSpec(family="cosface", architecture="MobileFaceNet", training_dataset="survface",
        implementation_repository="https://github.com/JDAI-CV/FaceX-Zoo",
        checkpoint=CheckpointProvenance.from_file(checkpoint, source_url="local-test"),
        preprocessing=PreprocessingSpec(112, 112, "rgb", "bgr", (127.5,)*3, (128.,)*3),
        target_layer="conv_6_sep", embedding_dim=1024,
        module_factory="research.embeddings.pytorch.facex_zoo:load_facex_mobilefacenet_checkpoint")
    registered = write_model_spec(tmp_path / "registered.json", spec)
    output = generate_configs(ROOT, {"surv_cos_s8972": str(registered)}, tmp_path / "configs",
                              artifact_root=tmp_path / "artifacts")
    pair = yaml.safe_load(output["pair_config"].read_text())
    inputs = yaml.safe_load(output["input_config"].read_text())
    assert model_bindings(pair) == model_bindings(inputs) == (1024, {"surv_cos_s8972": spec.model_uid})
    assert pair["input_config_path"] == output["input_config"].as_posix()
    assert pair["pq_profiles"] == [f"pq_1024_m{m}_b8" for m in (128, 64, 32)]
    assert pair["pq_training_policy"] == "outer_fold_unused_identities"
    assert pair["source_runs"] == inputs["source_runs"]
    assert spec.model_uid in pair["source_runs"]["surv_cos_s8972"]
    assert "sources/**/*.npy" in (tmp_path / "artifacts/.gitignore").read_text()
    with pytest.raises(FileExistsError):
        generate_configs(ROOT, {"surv_cos_s8972": str(registered)}, tmp_path / "configs")


def test_lfw_pq_uses_1024_width_and_outer_fold_assignment(monkeypatch):
    rng = np.random.default_rng(22)
    vectors = rng.normal(size=(260, 1024)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [str(i) for i in range(260)]
    prepared = SimpleNamespace(normalized_embeddings=vectors, sample_ids=ids)
    monkeypatch.setattr(lfw_pair_verification, "load_resize_source",
                        lambda *a: (None, None, None, prepared, None, None))
    plan = dict(config=dict(source_runs={"new": "source"}, embedding_dim=1024,
                           pq_profiles=["pq_1024_m32_b8"], codec_seed=8972),
                development=pd.DataFrame(dict(image_id=ids)),
                development_by_fold={1: pd.DataFrame(dict(image_id=ids[:256]))},
                pairs=pd.DataFrame(dict(left_image_id=["256", "257"], right_image_id=["258", "259"])))
    with pytest.raises(ValueError, match="outer fold"):
        lfw_pair_verification.prepare_model(ROOT, plan, "new", lambda: None)
    with threadpool_limits(limits=1):
        result = lfw_pair_verification.prepare_model(ROOT, plan, "new", lambda: None, fold=1)
    assert result["codecs"].development_images.tolist() == [256]
    assert result["codecs"].embedding_dimension.tolist() == [1024]
    assert result["codecs"].codebook_bytes.tolist() == [1024*256*4]
    assert result["codecs"].outer_fold.tolist() == [1]
    assert np.isfinite(result["scores"].pair_score).all()
