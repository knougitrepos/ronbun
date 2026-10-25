import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from research.calibration.sample_efficiency import fit_safety_learning_curve
from research.experiments import facex_workflow as flow
from research.experiments import facex_calibration_study as study
from research.experiments.facex_inputs import write_json
from research.runtime.hashing import sha256_file, canonical_sha256
from test_facex_workflow import surv_fixture


def calibration_fixture():
    rng = np.random.default_rng(12)
    n = 600
    q = rng.normal(size=n)
    frame = pd.DataFrame(dict(sample_id=[f"s{i}" for i in range(n)],
        identity_id=[f"id{i//4}" for i in range(n)], is_mated=False,
        score=.1*q+rng.normal(scale=.2, size=n), fiqa_score=q))
    return frame


def test_joint_curve_uses_fit_only_and_counts_both_pool_sizes():
    frame = calibration_fixture()
    roles = np.where(np.arange(len(frame)) < 480, "fit", "safety")
    options = dict(target=.1, score_space="cosine", fit_fractions=[.5, 1.], safety_fractions=[.5, 1.], folds=2)
    first = fit_safety_learning_curve(frame, roles, **options)
    frame.loc[roles == "safety", ["score", "fiqa_score"]] = np.nan
    second = fit_safety_learning_curve(frame, roles, **options)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 2*2*2*2
    assert set(first.method) == {"global_safe", "continuous_fiqa"}
    assert first.status.eq("completed").all()
    assert not first.final_safety_used.any() and not first.final_test_used.any()
    assert first.fit_samples.nunique() == first.safety_samples.nunique() == 2
    # Each fit/safety condition shares the same validation denominator within a fold.
    assert first.groupby("fold").validation_samples.nunique().eq(1).all()


def test_pair_inner_roles_use_both_identities(monkeypatch):
    import research.calibration.sample_efficiency as module
    frame = calibration_fixture()
    frame["left_identity"] = frame.identity_id
    frame["right_identity"] = [f"id{(i//4)^1}" for i in range(len(frame))]
    frame = pd.concat([frame]*4, ignore_index=True)
    frame["sample_id"] = [f"pair{i}" for i in range(len(frame))]
    seen = []
    evaluated = []
    from research.calibration.continuous import ContinuousThresholdModel
    predict = ContinuousThresholdModel.predict
    def observe(self, rows):
        evaluated.append(set(rows.left_identity) | set(rows.right_identity))
        return predict(self, rows)
    monkeypatch.setattr(ContinuousThresholdModel, "predict", observe)
    original = module._fit_continuous_quantile
    def check(fit, **kwargs):
        seen.append(set(fit.left_identity) | set(fit.right_identity))
        return original(fit, **kwargs)
    monkeypatch.setattr(module, "_fit_continuous_quantile", check)
    result = fit_safety_learning_curve(frame, ["fit"]*len(frame), target=.2, score_space="cosine",
        fit_fractions=[1.], safety_fractions=[1.], folds=2)
    assert seen
    for i, fit_ids in enumerate(seen):
        safety_ids, valid_ids = evaluated[2*i:2*i+2]
        assert not (fit_ids & safety_ids or fit_ids & valid_ids or safety_ids & valid_ids)
    assert result.validation_identities.max() < frame.left_identity.nunique()
    assert result.validation_samples.max() < len(frame)//2  # cross-role pairs were dropped


def test_survface_diagnostic_is_invariant_to_final_test_and_safety():
    pop, vectors, q = surv_fixture()
    codecs = {"origin_32": ("origin", None, {})}
    config = dict(target_rates=[.1], sample_efficiency=dict(fit_fractions=[1], safety_fractions=[1], folds=2))
    before = study.diagnostics(pop, vectors, q, codecs, "survface", config)
    excluded = pop.role.str.startswith("test_") | pop.role.eq("calibration_safety")
    vectors[excluded], q[excluded] = np.nan, np.nan
    after = study.diagnostics(pop, vectors, q, codecs, "survface", config)
    pd.testing.assert_frame_equal(before, after)
    assert before.status.eq("insufficient_rows").all()  # do not suppress small-tail conditions


def test_baseline_then_frozen_study_publication_and_lineage(tmp_path, monkeypatch):
    pop, vectors, q = surv_fixture()
    # The baseline needs a codec development row; no model training in this fixture.
    pop = pd.concat([pop, pd.DataFrame([dict(image_id="dev", identity_id="dev", role="pq_development", distance="")])], ignore_index=True)
    vectors = np.pad(np.vstack([vectors, vectors[0]]), ((0, 0), (0, 992)))
    q = np.r_[q, 1.]
    source = tmp_path / "source"; source.mkdir()
    pop.to_csv(source / "population.csv", index=False)
    for name, values in (("embeddings.npy", vectors), ("raw_norm.npy", np.ones(len(pop))), ("quality.npy", q)):
        np.save(source / name, values)
    write_json(source / "manifest.json", dict(status="completed", model_uid="fixture",
        files={p.name: sha256_file(p) for p in source.iterdir()}))
    (source / "COMPLETED").write_text("done")
    config = dict(project_root=str(Path(__file__).resolve().parents[2]), embedding_dim=1024,
        handoff=str(tmp_path / "handoff.json"), models={}, datasets=["survface"], target_rates=[.1],
        quality=dict(source="raw_norm"), bootstrap_resamples=100,
        compression=dict(survface=[dict(family="origin")]),
        sample_efficiency=dict(fit_fractions=[1.], safety_fractions=[1.], folds=2))
    path = tmp_path / "config.yaml"; path.write_text(yaml.safe_dump(config))
    models = {"model": (SimpleNamespace(model_uid="fixture"), "survface")}
    identity = {"fixture": True}
    monkeypatch.setattr(flow, "_workload", lambda *a: ({}, models, {"survface": (pop, None)}, [], identity))
    import research.protocols.facex_handoff as handoff
    monkeypatch.setattr(handoff, "load_handoff", lambda *a: {})
    root = tmp_path / "outputs"
    write_json(root / "input_indexes" / (canonical_sha256(identity)+".json"), dict(identity=identity,
        inputs={"model/survface": dict(path=str(source), manifest_sha256=sha256_file(source/"manifest.json"))}))
    with pytest.raises(ValueError, match="baseline_report"):
        flow.run_workflow(path, output_root=root, study_stage="calibration_size")
    baseline = flow.run_workflow(path, output_root=root, keep_raw_results=False)
    before = sha256_file(Path(baseline["report_dir"]) / "run_manifest.json")
    monkeypatch.setattr(flow, "fit_codecs", lambda *a, **k: pytest.fail("Stage 2 refitted baseline codec"))
    second = flow.run_workflow(path, output_root=root, study_stage="calibration_size",
        baseline_report=baseline["report_dir"], keep_raw_results=False)
    assert second["report_dir"] != baseline["report_dir"]
    result = flow.read_results(second["report_dir"])
    assert result["manifest"]["study_stage"] == "calibration_size"
    assert result["manifest"]["contract"]["baseline_run_uid"] == baseline["run_uid"]
    assert not result["metrics"].final_test_used.any()
    assert sha256_file(Path(baseline["report_dir"]) / "run_manifest.json") == before
    assert flow.run_workflow(path, output_root=root, study_stage="calibration_size",
        baseline_report=baseline["report_dir"], keep_raw_results=False)["reused"]
    changed = copy.deepcopy(config); changed["partition_seed"] = 2
    changed["study"] = dict(stage="calibration_size")
    with pytest.raises(ValueError, match="must keep"):
        study.validate_baseline(changed, baseline["report_dir"], ["survface"])
    with pytest.raises(ValueError, match="rank-only"):
        study.validate_baseline(config, baseline["report_dir"], ["tinyface"])
    assert not list(root.rglob("RUNNING.lock"))
    report = Path(second["report_dir"])
    summary = json.loads((report / "chatgpt_summary.json").read_text())
    assert sum(part["rows"] for part in summary["summary_parts"]) == len(result["metrics"])
    first_part = report / summary["summary_parts"][0]["path"]
    first_part.write_text(first_part.read_text() + "\n")
    with pytest.raises(ValueError, match="Published result changed"):
        flow.read_results(report)


def test_unplanned_fr_ratio_sweep_is_not_silently_enabled():
    with pytest.raises(ValueError, match="separately justified"):
        flow.validate_study(dict(study=dict(stage="fr_ratio_sweep")))


@pytest.mark.parametrize("dataset", ["lfw", "rfw"])
def test_pair_study_with_real_frozen_codecs_excludes_unused_images(tmp_path, dataset):
    import faiss
    from research.protocols.facex_handoff import pair_partition
    from research.evaluation.lfw_verification import pair_scores
    faiss.omp_set_num_threads(2)
    rng = np.random.default_rng(14)
    rows, pairs = [], []
    group = "African" if dataset == "rfw" else "LFW"
    for fold in (1, 2):
        for i in range(300):
            identity = f"f{fold}:id{i}"
            for image in (0, 1):
                rows.append(dict(image_id=f"{identity}:{image}", identity_id=identity))
            for shift in (0, 1, 2, 3):
                right = f"f{fold}:id{(i+shift)%300}"
                pairs.append(dict(pair_id=f"{fold}-{i}-{shift}", fold=fold, group=group,
                    left_image_id=f"{identity}:0", right_image_id=f"{right}:1",
                    left_identity=identity, right_identity=right, is_genuine=shift == 0))
    population, pairs = pd.DataFrame(rows), pd.DataFrame(pairs)
    vectors = flow.normalize(rng.normal(size=(len(population), 32)))
    quality = rng.normal(size=len(population))
    trained = flow.fit_codecs(flow.normalize(rng.normal(size=(256, 32))),
        [dict(family="origin"), dict(family="pca", dimensions=8), dict(family="pq", m=4, nbits=2)],
        tmp_path / "codecs" / "fixture")
    write_json(tmp_path / "fixture_codec_inventory.json", {p: info for p, (_, _, info) in trained.items()})
    codecs = study.frozen_codecs(tmp_path, "fixture")
    # Reloaded PCA/PQ retain identical pair scores; no fit occurs in frozen_codecs.
    subset = pairs.iloc[:12]
    for profile, (family, original, _) in trained.items():
        restored = codecs[profile][1]
        left = flow.normalize(original.transform(vectors)) if family == "pca" else vectors
        right = flow.normalize(restored.transform(vectors)) if family == "pca" else vectors
        np.testing.assert_array_equal(pair_scores(left, subset, population.image_id, codec=original if family == "pq" else None),
                                      pair_scores(right, subset, population.image_id, codec=restored if family == "pq" else None))
    config = dict(target_rates=[.1], fold_ids=[1], partition_seed=8972,
                  sample_efficiency=dict(fit_fractions=[1.], safety_fractions=[1.], folds=2))
    before = study.diagnostics(population, vectors, quality, codecs, dataset, config, pairs)
    calibration, _, _ = pair_partition(pairs, 1)
    candidate = calibration.loc[calibration.calibration_role.eq("fit") & ~calibration.is_genuine]
    used = set(candidate.left_image_id) | set(candidate.right_image_id)
    excluded = ~population.image_id.isin(used)
    vectors[excluded], quality[excluded] = np.nan, np.nan
    after = study.diagnostics(population, vectors, quality, codecs, dataset, config, pairs)
    pd.testing.assert_frame_equal(before, after)
    assert set(before.score_space) == {"cosine", "pca_cosine", "negative_squared_l2_adc"}
    assert before.metric.eq("FMR").all() and before.sample_unit.eq("pair").all()
    assert not before.final_safety_used.any() and not before.final_test_used.any()
    codec_file = next((tmp_path / "codecs" / "fixture").glob("*.joblib"))
    codec_file.write_bytes(codec_file.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Frozen baseline codec changed"):
        study.frozen_codecs(tmp_path, "fixture")


def test_empty_diagnostic_pool_reports_every_insufficient_condition():
    frame = calibration_fixture().iloc[:0]
    result = fit_safety_learning_curve(frame, [], target=.1, score_space="cosine", folds=2,
                                      fit_fractions=[.5, 1.], safety_fractions=[1.])
    assert len(result) == 8 and result.status.eq("insufficient_rows").all()
    assert result.fit_samples.eq(0).all()
