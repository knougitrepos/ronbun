from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest

from research.experiments.facex_workflow import (
    evaluate_rank, evaluate_survface, normalize, rank_scores, validate_codec_budget,
)
from research.protocols.facex_handoff import load_protocol, pair_partition


def test_small_dataset_codec_gate_and_byte_budgets():
    with pytest.raises(ValueError, match="undersampled"):
        validate_codec_budget(800, 1024, dict(family="pq", m=128, nbits=8))
    assert validate_codec_budget(800, 1024, dict(family="pq", m=128, nbits=4))["payload_bytes"] == 64
    assert validate_codec_budget(7804, 1024, dict(family="pq", m=128, nbits=7))["payload_bytes"] == 112
    with pytest.raises(ValueError, match="centered"):
        validate_codec_budget(800, 1024, dict(family="pca", dimensions=800))


@pytest.mark.parametrize("bits", [4, 7])
def test_real_low_bit_adc_matches_decoded_distance(bits):
    pytest.importorskip("faiss")
    from research.compression import PQCompressor
    from research.evaluation.lfw_verification import pair_scores
    import faiss
    faiss.omp_set_num_threads(1)
    x = normalize(np.random.default_rng(5).normal(size=(256, 16)))
    codec = PQCompressor(source_dim=16, m=4, nbits=bits).fit(x)
    pairs = pd.DataFrame(dict(left_image_id=["a", "b"], right_image_id=["c", "d"]))
    actual = pair_scores(x[:4], pairs, ["a", "b", "c", "d"], codec=codec)
    decoded = codec.decode(codec.encode(x[[2, 3]]))
    np.testing.assert_allclose(actual, -np.square(x[:2] - decoded).sum(axis=1), atol=1e-6)
    frame, _ = rank_scores(x[:2], x[2:4], ["a", "b"], ["c", "d"], family="pq", codec=codec)
    dense = -np.square(x[:2, None, :] - decoded[None, :, :]).sum(axis=2)
    np.testing.assert_allclose(frame.score, dense.max(axis=1), atol=1e-5)


def test_genuine_score_not_top_one_and_missing_genuine():
    q = normalize(np.array([[1., 0.], [0., 1.]], dtype=np.float32))
    g = normalize(np.array([[1., 0.], [.8, .6]], dtype=np.float32))
    frame, _ = rank_scores(q, g, ["b", "unknown"], ["a", "b"])
    assert frame.iloc[0].score > frame.iloc[0].true_identity_score
    assert frame.iloc[0].true_identity_rank == 2
    assert not frame.iloc[1].is_mated
    assert np.isnan(frame.iloc[1].true_identity_score)


def surv_fixture():
    rng = np.random.default_rng(42)
    gallery = normalize(rng.normal(size=(4, 32)))
    rows, vectors = [], []
    for role, offset in (("calibration_gallery", 0), ("test_gallery", 2)):
        for j in range(2):
            rows.append(dict(image_id=f"{role}{j}", identity_id=f"g{offset+j}", role=role, distance=""))
            vectors.append(gallery[offset+j])
    for j in range(12):
        rows.append(dict(image_id=f"calpos{j}", identity_id=f"g{j%2}", role="calibration_registered_probe", distance=""))
        vectors.append(gallery[j%2])
    for role in ("calibration_fit", "calibration_safety", "test_unknown_probe"):
        for j in range(60):
            rows.append(dict(image_id=f"{role}{j}", identity_id=f"{role}:id{j//3}", role=role, distance=""))
            vectors.append(normalize(rng.normal(size=(1, 32)))[0])
    for j in range(20):
        rows.append(dict(image_id=f"positive{j}", identity_id=f"g{2+j%2}", role="test_registered_probe", distance=str(j%3+1)))
        vectors.append(gallery[2+j%2])
    return pd.DataFrame(rows), np.array(vectors), rng.normal(size=len(rows))


def test_survface_frozen_roles_and_paired_realized_rates():
    pop, vectors, quality = surv_fixture()
    codecs = {"origin_32": ("origin", None, dict(payload_bytes=128))}
    config = dict(target_rates=[.1], bootstrap_resamples=100)
    metrics, paired, models = evaluate_survface(pop, vectors, quality, codecs, config)
    assert set(metrics.method) == {"global_safe", "continuous_fiqa"}
    assert metrics.non_mated_count.eq(60).all()
    assert metrics.mated_count.eq(20).all()
    assert (metrics.rank_failure_count + metrics.threshold_failure_count + metrics.true_positive_count).eq(20).all()
    assert len(paired) == 1
    assert len(models["models"]) == 2
    assert models["models"][1]["model"]["settings"]["fit_non_mated_count"] == 60
    assert models["models"][1]["model"]["settings"]["safety_non_mated_count"] == 60


def test_scface_rank_has_all_three_distance_cohorts():
    pop, vectors, _ = surv_fixture()
    metrics, _, _ = evaluate_rank(pop, vectors, {"origin": ("origin", None, dict(payload_bytes=128))}, "scface")
    assert set(metrics.distance) == {"all", "1", "2", "3"}
    assert set(metrics["rank"]) == {1, 5, 10, 20}


def test_identity_overlap_in_frozen_partition_is_rejected():
    from research.calibration.conditional import explicit_calibration_partition
    with pytest.raises(ValueError, match="overlaps"):
        explicit_calibration_partition(pd.DataFrame(dict(identity_id=["same", "same"])), ["fit", "safety"])


def test_pair_partition_purges_both_outer_test_identities():
    rows = []
    for fold in (1, 2):
        for j in range(800):
            left = f"f{fold}-i{j%200}"
            right = left if j%2 else f"f{fold}-i{(j+1)%200}"
            rows.append(dict(pair_id=f"{fold}-{j}", fold=fold, left_identity=left,
                             right_identity=right, is_genuine=left == right))
    rows.append(dict(pair_id="overlap", fold=2, left_identity="f1-i0", right_identity="other", is_genuine=False))
    cal, test, inventory = pair_partition(pd.DataFrame(rows), 1)
    assert "overlap" not in set(cal.pair_id)
    assert inventory["test_identity_purged_pairs"] == 1
    assert not ((set(cal.left_identity) | set(cal.right_identity)) & set(test.left_identity))


def test_sample_efficiency_never_uses_final_safety():
    from research.calibration.sample_efficiency import quantile_learning_curve
    pop, vectors, q = surv_fixture()
    cal = pop.loc[pop.role.isin(["calibration_fit", "calibration_safety"])].copy()
    cal["is_mated"], cal["score"], cal["fiqa_score"] = False, np.random.default_rng(2).normal(size=len(cal)), q[:len(cal)]
    labels = cal.role.str.removeprefix("calibration_").to_numpy()
    a = quantile_learning_curve(cal, labels, target=.1, score_space="cosine", fractions=[1], folds=2)
    cal.loc[labels == "safety", "score"] = 1e9
    b = quantile_learning_curve(cal, labels, target=.1, score_space="cosine", fractions=[1], folds=2)
    pd.testing.assert_frame_equal(a, b)


def test_producer_consumer_shared_roles_and_tamper_detection(tmp_path):
    producer = Path("F:/FaceX-Zoo/fine-tune/modules/research_protocol.py")
    if not producer.is_file():
        pytest.skip("FaceX producer repository not installed")
    module_spec = importlib.util.spec_from_file_location("research_protocol_fixture", producer)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    raw = tmp_path / "raw"
    raw.mkdir()
    train, test, reference = [], [], []
    for i in range(12):
        for j in range(2):
            path = f"training_set/{i}/{i}_{j}.jpg"
            (raw / path).parent.mkdir(parents=True, exist_ok=True)
            (raw / path).write_bytes(b"fixture")
            train.append(dict(image_path=path, source_identity_id=str(i), protocol_role="training"))
            reference.append(dict(identity_id=f"survface:train:{i}", image_path=path, split="development" if i<8 else "calibration"))
    for role in ("gallery", "registered_probe", "unknown_known_probe"):
        path = f"test/{role}.jpg"
        (raw / path).parent.mkdir(exist_ok=True)
        (raw / path).write_bytes(b"fixture")
        test.append(dict(image_path=path, source_identity_id="100" if role != "unknown_known_probe" else "101", protocol_role=role))
    for name, rows in (("train", train), ("test", test), ("reference", reference)):
        pd.DataFrame(rows).to_csv(tmp_path / (name+".csv"), index=False)
    profile = dict(name="survface", raw_root=str(raw), source_training_manifest=str(tmp_path/"train.csv"),
                   source_evaluation_manifest=str(tmp_path/"test.csv"), split_reference=str(tmp_path/"reference.csv"),
                   gallery_identity_count=3, research_output=str(tmp_path/"out"), allow_test_fixture=True)
    module.prepare_research_protocol(profile)
    record, frame = load_protocol(profile["protocol_record"])
    assert record["num_classes"] == 8
    assert frame.loc[frame.fr_train.eq(1)].identity_id.nunique() == 8
    # A train:/test: prefix must not hide an underlying native identity overlap,
    # even if an invalid producer republishes the CSV with a matching hash.
    import json
    from research.runtime.hashing import sha256_file
    role_path = Path(record["roles_path"])
    original_roles, original_record = role_path.read_bytes(), Path(profile["protocol_record"]).read_bytes()
    forged = pd.read_csv(role_path, keep_default_na=False, dtype=str)
    target = forged.role.eq("test_gallery")
    forged.loc[target, "source_identity_id"] = "0"
    forged.loc[target, "identity_id"] = "survface:test:0"
    forged.to_csv(role_path, index=False)
    record["roles_sha256"] = sha256_file(role_path)
    Path(profile["protocol_record"]).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Native.*overlap"):
        load_protocol(profile["protocol_record"])
    role_path.write_bytes(original_roles)
    Path(profile["protocol_record"]).write_bytes(original_record)
    with Path(record["roles_path"]).open("a", encoding="utf-8") as stream:
        stream.write("changed\n")
    with pytest.raises(ValueError, match="Changed"):
        load_protocol(profile["protocol_record"])


def test_rfw_group_fold_continuous_evaluation_with_named_origin():
    from research.experiments.facex_workflow import evaluate_pairs
    rng = np.random.default_rng(12)
    rows, pairs = [], []
    for fold in (1, 2):
        for i in range(200):
            identity = f"f{fold}:id{i}"
            for j in range(2):
                rows.append(dict(image_id=f"{identity}:{j}", identity_id=identity))
            pairs.append(dict(pair_id=f"g{fold}-{i}", fold=fold, group="African", left_image_id=f"{identity}:0",
                              right_image_id=f"{identity}:1", left_identity=identity, right_identity=identity, is_genuine=True))
            right = f"f{fold}:id{(i+1)%200}"
            pairs.append(dict(pair_id=f"i{fold}-{i}", fold=fold, group="African", left_image_id=f"{identity}:0",
                              right_image_id=f"{right}:1", left_identity=identity, right_identity=right, is_genuine=False))
    pop = pd.DataFrame(rows)
    vectors = normalize(rng.normal(size=(len(pop), 32)))
    codecs = {"origin_32": ("origin", None, {}), "origin_copy": ("origin", None, {})}
    # One actual Origin key; second condition exercises the compressed-vs-Origin lookup.
    codecs["copy"] = codecs.pop("origin_copy")
    metrics, paired, models = evaluate_pairs(pop, vectors, rng.normal(size=len(pop)), pd.DataFrame(pairs), codecs,
        dict(target_rates=[.1], fold_ids=[1], partition_seed=8972, bootstrap_resamples=100))
    assert set(metrics.method) >= {"global_safe", "continuous_fiqa"}
    assert metrics.group.eq("African").all()
    assert not paired.empty
    assert models["partitions"][0]["test_pairs"] == 400


def test_complete_publication_reuse_and_raw_retention(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import yaml
    from research.experiments import facex_workflow as workflow
    from research.experiments.facex_inputs import write_json
    from research.runtime.hashing import canonical_sha256, sha256_file
    pop, vectors, quality = surv_fixture()
    # Add a disjoint PQ-development role even when only Origin is selected.
    extra = pd.DataFrame([dict(image_id="dev", identity_id="dev", role="pq_development", distance="")])
    pop = pd.concat([pop, extra], ignore_index=True)
    vectors = np.pad(np.vstack([vectors, vectors[0]]), ((0, 0), (0, 992)))
    quality = np.r_[quality, 1.]
    source = tmp_path / "source"
    source.mkdir()
    pop.to_csv(source / "population.csv", index=False)
    for name, values in (("embeddings.npy", vectors), ("raw_norm.npy", np.ones(len(pop))), ("quality.npy", quality)):
        np.save(source / name, values)
    write_json(source / "manifest.json", dict(status="completed", model_uid="synthetic", files={
        p.name: sha256_file(p) for p in source.iterdir()}))
    (source / "COMPLETED").write_text("done")
    config = dict(project_root=str(Path(__file__).resolve().parents[2]), embedding_dim=1024,
                  models={}, datasets=["survface"], target_rates=[.1], bootstrap_resamples=100,
                  quality=dict(source="raw_norm"), compression=dict(survface=[dict(family="origin")]))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    models = {"model": (SimpleNamespace(model_uid="synthetic"), "survface")}
    identity = {"fixture": True}
    monkeypatch.setattr(workflow, "_workload", lambda *a: ({}, models, {"survface": (pop, None)}, [], identity))
    root = tmp_path / "outputs"
    index = root / "input_indexes" / (canonical_sha256(identity) + ".json")
    write_json(index, dict(identity=identity, inputs={"model/survface": dict(path=str(source), manifest_sha256=sha256_file(source/"manifest.json"))}))
    first = workflow.run_workflow(path, output_root=root, keep_raw_results=False)
    report = Path(first["report_dir"])
    assert workflow.read_results(report)["manifest"]["completed_jobs"] == 1
    assert not (report / "details").exists()
    assert source.exists()
    before = sha256_file(report / "run_manifest.json")
    assert workflow.run_workflow(path, output_root=root, keep_raw_results=False)["reused"]
    second = workflow.run_workflow(path, output_root=root, keep_raw_results=True)
    assert list((Path(second["report_dir"]) / "details").rglob("*.csv"))
    assert sha256_file(report / "run_manifest.json") == before
    assert not list(root.rglob("RUNNING.lock"))
    summary = json.loads((report / "chatgpt_summary.json").read_text())
    assert summary["summary_parts"]


def test_tar_input_order_preserves_ids_without_repeated_gzip_seeks(tmp_path):
    import io
    import tarfile
    from research.experiments.facex_inputs import archive_order
    with tarfile.open(tmp_path / "images.tar.gz", "w:gz") as handle:
        for name in ("b.jpg", "a.jpg"):
            member = tarfile.TarInfo(name)
            member.size = 1
            handle.addfile(member, io.BytesIO(b"x"))
    pop = pd.DataFrame(dict(image_id=["a", "b"], image_path=["tar://images.tar.gz#a.jpg", "tar://images.tar.gz#b.jpg"]))
    result = archive_order(pop, tmp_path)
    assert list(result.image_id) == ["b", "a"]
    assert set(result.image_id) == set(pop.image_id)
