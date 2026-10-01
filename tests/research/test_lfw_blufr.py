from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from research.datasets.lfw_blufr import (
    load_blufr_lists, bind_image_manifest, REFERENCE_SHA256, BLUFRLists, BLUFRTrial,
)
from research.protocols.blufr import identification_rows, benchmark_curve
from research.runtime.hashing import sha256_file


def fixture_config(tmp_path, mutate=None):
    # Distinct identities across train/test; two gallery identities, plus unknown.
    names = np.array(["A_0001.jpg", "A_0002.jpg", "B_0001.jpg", "B_0002.jpg",
                      "C_0001.jpg", "D_0001.jpg", "D_0002.jpg"], dtype=object)[:, None]
    values = dict(imageList=names, labels=np.array([1, 1, 2, 2, 3, 4, 4])[:, None])
    for key, numbers in dict(trainIndex=[6, 7], testIndex=[1, 2, 3, 4, 5],
                             galIndex=[1, 3], probIndex=[2, 4, 5]).items():
        cells = np.empty((2, 1), dtype=object)
        for i in range(2):
            cells[i, 0] = np.array(numbers)[:, None]
        values[key] = cells
    if mutate:
        mutate(values)
    path = tmp_path / "blufr.mat"
    savemat(path, values)
    return path


def test_matlab_indexes_and_released_order(tmp_path):
    path = fixture_config(tmp_path)
    lists = load_blufr_lists(path, expected_sha256=sha256_file(path), strict_reference=False)
    np.testing.assert_array_equal(lists.trial(1).gallery, [0, 2])
    with pytest.raises(ValueError, match="trial_id"):
        lists.trial(0)
    with pytest.raises(ValueError, match="SHA"):
        load_blufr_lists(path)
    with pytest.raises(ValueError, match="trial inventory"):
        load_blufr_lists(path, expected_sha256=sha256_file(path))


@pytest.mark.parametrize("kind", ["zero", "duplicate", "leak", "wrong_label", "missing_probe"])
def test_corrupt_public_splits_rejected(tmp_path, kind):
    def mutate(values):
        if kind == "zero":
            values["galIndex"][0, 0][0] = 0
        elif kind == "duplicate":
            values["galIndex"][0, 0][1] = 1
        elif kind == "leak":
            values["labels"][6] = 1
        elif kind == "wrong_label":
            values["labels"][1] = 2
        else:
            values["probIndex"][0, 0] = np.array([2, 4])
    path = fixture_config(tmp_path, mutate)
    with pytest.raises(ValueError):
        load_blufr_lists(path, expected_sha256=sha256_file(path), strict_reference=False)


def test_manifest_mapping_uses_names_and_rejects_missing(tmp_path):
    path = fixture_config(tmp_path)
    lists = load_blufr_lists(path, expected_sha256=sha256_file(path), strict_reference=False)
    manifest = pd.DataFrame(dict(image_id=["s" + str(i) for i in range(7)],
        identity_id="lfw:" + lists.images.identity_name, image_path="data/" + lists.images.filename))
    result = bind_image_manifest(lists, manifest.iloc[::-1])
    assert result.image_id.tolist() == manifest.image_id.tolist()
    with pytest.raises(ValueError, match="missing"):
        bind_image_manifest(lists, manifest.iloc[:-1])


def test_curve_requires_genuine_score_not_top1_and_reports_ties():
    # First known probe has a high wrong-person maximum, but true score is low.
    rows = identification_rows(np.array([[.9, .2], [.1, .8], [.7, .1], [.7, .2]]),
                               ["a", "b"], ["q1", "q2", "u1", "u2"], ["b", "b", "x", "y"])
    out = benchmark_curve(rows, (.5,), (1, 2))
    assert out.tpir.tolist() == [.5, .5]
    assert out.toolkit_far.eq(.5).all()
    assert out.realized_fpir.eq(1.).all()
    assert not out.deployment_threshold_selected.any()


def test_matlab_positive_round_and_zero_one_endpoints():
    rows = identification_rows(np.array([[.9, .1], [.4, .2], [.3, .1]]),
                               ["a", "b"], ["g", "u", "v"], ["a", "x", "y"])
    curve = benchmark_curve(rows, (0., .25, 1.), (1,))
    assert curve.toolkit_far.tolist() == [0., .5, 1.]
    assert curve.realized_fpir.tolist() == [0., .5, 1.]
    assert curve.tpir.eq(1.).all()


def test_stable_gallery_order_tie_rank():
    rows = identification_rows(np.array([[.5, .5, .1]]), ["a", "b", "c"], ["q"], ["b"])
    assert rows.true_identity_rank.iloc[0] == 2


def test_public_lists_when_present():
    path = Path("data/external/blufr/blufr_lfw_config.mat")
    if not path.exists():
        pytest.skip("public protocol cache is optional in CI")
    lists = load_blufr_lists(path)
    assert lists.source_sha256 == REFERENCE_SHA256
    assert len(lists.trials) == 10
    assert len(lists.trial(1).train) == 2952
    assert len(lists.trial(1).gallery) == 1000


def test_bounded_cosine_and_adc_match_dense_reference():
    from research.compression import PQCompressor
    from research.experiments.lfw_protocol_experiments import search_rows
    import faiss
    rng = np.random.default_rng(12)
    train = rng.normal(size=(100, 8)).astype("float32")
    gallery = rng.normal(size=(6, 8)).astype("float32")
    query = rng.normal(size=(4, 8)).astype("float32")
    before = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(2)
        codec = PQCompressor(source_dim=8, m=2, nbits=2).fit(train)
        gid, qid, identity = np.arange(6), np.arange(4), np.array([1, 2, 10, 11])
        origin = search_rows(query, gallery, qid, identity, gid, batch_size=2)
        pd.testing.assert_frame_equal(origin, identification_rows(query @ gallery.T, gid, qid, identity))
        actual = search_rows(query, gallery, qid, identity, gid, codec=codec, batch_size=2)
        decoded = codec.decode(codec.encode(gallery))
        dense = -np.square(query[:, None, :] - decoded[None, :, :]).sum(axis=2)
        expected = identification_rows(dense, gid, qid, identity)
        np.testing.assert_allclose(actual.score, expected.score, atol=1e-5)
        np.testing.assert_allclose(actual.true_identity_rank, expected.true_identity_rank, equal_nan=True)
    finally:
        faiss.omp_set_num_threads(before)


def test_missing_embeddings_fail_before_benchmark_write(tmp_path, monkeypatch):
    from research.experiments import lfw_protocol_experiments as module
    monkeypatch.setattr(module, "inspect_blufr", lambda *a, **k: dict(
        ready=False, coverage=pd.DataFrame([dict(model="arcface", missing=38)])))
    with pytest.raises(ValueError, match="every released image"):
        module.run_blufr_benchmark(tmp_path)
    assert not list(tmp_path.iterdir())


def test_matched_extension_rejects_changed_global_test_lists():
    from research.experiments.lfw_protocol_experiments import assert_frozen_test_protocol
    from research.protocols.open_set import OpenSetProtocol
    def frame(samples, ids):
        return pd.DataFrame(dict(image_id=samples, identity_id=ids))
    protocol = OpenSetProtocol(frame(["gallery"], ["a"]), frame(["q"], ["a"]),
                               frame(["u"], ["b"]), frame([], []))
    scores = pd.DataFrame(dict(sample_id=["u", "q"], identity_id=["b", "a"], is_mated=[False, True]))
    assert_frozen_test_protocol(protocol, scores)
    changed = scores.assign(sample_id=["other", "q"])
    with pytest.raises(ValueError, match="cohort"):
        assert_frozen_test_protocol(protocol, changed)
    with pytest.raises(ValueError, match="identity/mated"):
        assert_frozen_test_protocol(protocol, scores.assign(is_mated=True))


def test_benchmark_campaign_resumes_and_does_not_refit_completed(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from research.experiments import lfw_protocol_experiments as module
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=(310, 512)).astype("float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    ids = [f"train{i}" for i in range(300)] + ["a", "b", "a", "b", "u", "v", "u", "v", "u", "v"]
    bound = pd.DataFrame(dict(image_id=[f"q{i}" for i in range(310)], identity_id=ids))
    trial = BLUFRTrial(1, np.arange(300), np.arange(300, 310), np.array([300, 301]), np.arange(302, 310))
    lists = BLUFRLists(pd.DataFrame(), (trial,), "synthetic")
    config = dict(trial_ids=[1], pq_profiles=["pq_512_m32_b8"], codec_seed=12,
                  target_fpirs=[.1], batch_size=3, output_root="output", source_runs={"test": "source"})
    inspection = dict(ready=True, config=config, lists=lists, bound=bound, provenance={"test": {}})
    prepared = SimpleNamespace(sample_ids=bound.image_id.to_numpy(), normalized_embeddings=matrix)
    monkeypatch.setattr(module, "inspect_blufr", lambda *a, **k: inspection)
    monkeypatch.setattr(module, "_source", lambda *a, **k: (None, None, None, prepared, None, {}))
    monkeypatch.setattr(module, "implementation_hashes", lambda: {"version": "fixture"})
    first = module.run_blufr_benchmark(tmp_path, progress=lambda _: None)
    assert first["completed"] and first["new_jobs"] == 1
    monkeypatch.setattr(module, "_source", lambda *a, **k: pytest.fail("completed trial must not reload/refit"))
    second = module.run_blufr_benchmark(tmp_path, progress=lambda _: None)
    assert second["new_jobs"] == 0
    assert first["report_dir"] == second["report_dir"]
    tables, manifest = module.read_lfw_protocol_report(second["report_dir"])
    assert len(tables["benchmark_curves"]) == 4
    assert manifest["spec"]["threshold_source"] == "test_curve_only"
    assert not list((tmp_path / "output").rglob("*.csv"))


def test_matched_calibration_resume_uses_one_database_and_no_old_threshold_models(tmp_path, monkeypatch):
    from test_origin_vs_pq_calibration import inputs, options
    from research.experiments import lfw_protocol_experiments as module
    import research.fiqa
    conditions, fiqa = inputs(tmp_path)
    plan = pd.DataFrame([dict(model="arcface", source_run_dir="source", source_run_id="test-run",
                             dataset_id="lfw", fiqa_dir="quality")])
    monkeypatch.setattr(module, "inspect_origin_pq_experiment", lambda *a, **k: plan)
    monkeypatch.setattr(module, "implementation_hashes", lambda: {"version": "test"})
    monkeypatch.setattr(module, "matched_conditions", lambda *a, **k: (conditions, fiqa, pd.DataFrame([dict(test_unchanged=True)])))
    monkeypatch.setattr(research.fiqa, "load_fiqa_score_artifact", lambda *a, **k: fiqa)
    kwargs = dict(models=("arcface",), partition_seeds=(0, 8972), settings=options(),
                  output_root="out", max_new_jobs=1, execute=True, progress=lambda _: None)
    partial = module.run_matched_lfw_calibration(tmp_path, **kwargs)
    assert not partial["completed"] and partial["completed_jobs"] == 1
    assert "report_dir" not in partial
    complete = module.run_matched_lfw_calibration(tmp_path, **kwargs)
    assert complete["completed"] and complete["new_jobs"] == 1
    monkeypatch.setattr(module, "run_origin_pq_split", lambda *a, **k: pytest.fail("refitted a completed split"))
    reused = module.run_matched_lfw_calibration(tmp_path, **kwargs)
    assert reused["new_jobs"] == 0
    assert reused["report_dir"] == complete["report_dir"]
    assert reused["chat_dir"] == complete["chat_dir"]
    assert len(list((tmp_path / "out").glob("campaign-*.sqlite3"))) == 1
    tables, _ = module.read_lfw_protocol_report(complete["report_dir"])
    assert not tables["models"].fitting_reused.any()
