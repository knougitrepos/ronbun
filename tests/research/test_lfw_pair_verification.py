"""Leakage, metrics, actual ADC, retention and notebook routing regressions."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from research.datasets.lfw_pairs import parse_pairs, calibration_partition, ensure_pairs, load_pairs
from research.evaluation.lfw_verification import pair_scores, accuracy_threshold, empirical_fmr_threshold, evaluate_fold
from research.experiments.origin_vs_pq_calibration import OriginPQSettings
from research.experiments.lfw_pair_storage import PairStore, publish, read_report, cleanup_checkpoint

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pairs():
    lines = ["10\t300"]
    for fold in range(10):
        lines.extend(f"P{fold}_{i}\t1\t2" for i in range(300))
        lines.extend(f"P{fold}_{i}\t1\tN{fold}_{i}\t1" for i in range(300))
    return parse_pairs("\n".join(lines))


@pytest.fixture(scope="module")
def inputs(pairs):
    rng = np.random.default_rng(341)
    quality = pd.Series(rng.uniform(0, 1, len(set(pairs.left_image_id))), index=sorted(set(pairs.left_image_id)))
    parts = []
    for profile in ("origin", "pq_512_m128_b8", "pq_512_m64_b8", "pq_512_m32_b8"):
        frame = pairs.copy()
        frame["pair_score"] = rng.normal(0, .15, len(pairs)) + pairs.is_genuine.to_numpy()*.35 + quality.loc[pairs.left_image_id].to_numpy()*.2
        frame["compression_profile"] = profile
        frame["score_space"] = "cosine_similarity" if profile == "origin" else "negative_squared_l2_adc"
        parts.append(frame)
    return pd.concat(parts, ignore_index=True), quality


@pytest.fixture(scope="module")
def evaluated(pairs, inputs):
    settings = OriginPQSettings(target_fpirs=(.1,), diagnostic_fpir_grid=(.001, .1), resamples=100)
    cal, test, inventory = calibration_partition(pairs, 1, 8972)
    tables = evaluate_fold(*inputs, cal, test, seed=8972, settings=settings, keep_decisions=True)
    tables["inventory"] = pd.DataFrame([inventory])
    for frame in tables.values():
        frame["model"] = "arcface"
        frame["fold"] = 1
        frame["partition_seed"] = 8972
    return tables, settings


def test_parser_rejects_cross_fold_identity_overlap():
    lines = ["10\t300"]
    for fold in range(10):
        lines.extend(f"shared_{i}\t1\t2" for i in range(300))
        lines.extend(f"other_{fold}_{i}\t1\tunrelated_{fold}_{i}\t1" for i in range(300))
    with pytest.raises(ValueError, match="identity overlap"):
        parse_pairs("\n".join(lines))


def test_corrupt_protocol_is_not_overwritten(tmp_path):
    p = tmp_path / "pairs.txt"
    p.write_text("wrong")
    with pytest.raises(ValueError, match="hash mismatch"):
        ensure_pairs(p, download=True)
    assert p.read_text() == "wrong"


def test_both_pair_endpoints_are_identity_disjoint(pairs):
    cal, test, inventory = calibration_partition(pairs, 1, 8972)
    def ids(x):
        return set(x.left_identity) | set(x.right_identity)
    fit, safe = cal[cal.calibration_role.eq("fit")], cal[cal.calibration_role.eq("safety")]
    assert not ids(fit) & ids(safe)
    assert not ids(cal) & ids(test)
    assert len(test) == 600
    assert inventory["cross_partition_pairs_excluded"] > 0
    assert len(cal) + inventory["cross_partition_pairs_excluded"] == 5400


def test_development_excludes_every_pair_identity(pairs, tmp_path, monkeypatch):
    from research.datasets import lfw_pairs
    records = {}
    for side in ("left", "right"):
        records.update(zip(pairs[f"{side}_image_id"], pairs[f"{side}_identity"]))
    records.update({f"development_{i}": f"unused_identity_{i}" for i in range(256)})
    # Unused images of a TEST identity must not leak into compression development.
    records["not_in_pairs_but_same_person"] = pairs.left_identity.iloc[0]
    population = pd.DataFrame(dict(image_id=list(records), identity_id=list(records.values())))
    path = tmp_path / "unused.txt"
    path.write_text("fixture")
    monkeypatch.setattr(lfw_pairs, "ensure_pairs", lambda *a, **k: path)
    monkeypatch.setattr(lfw_pairs, "parse_pairs", lambda text: pairs)
    _, dev = load_pairs(path, population)
    assert len(dev) == 256
    assert "not_in_pairs_but_same_person" not in set(dev.image_id)


def test_actual_adc_matches_decoded_l2_and_faiss_search():
    from research.compression import PQCompressor
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(22)
    dev = rng.normal(size=(512, 8)).astype("float32")
    test = rng.normal(size=(8, 8)).astype("float32")
    test /= np.linalg.norm(test, axis=1, keepdims=True)
    rows = pd.DataFrame(dict(left_image_id=["0", "1", "2"], right_image_id=["3", "4", "5"]))
    with threadpool_limits(limits=1):
        codec = PQCompressor(source_dim=8, m=2, nbits=8).fit(dev)
        actual = pair_scores(test, rows, [str(i) for i in range(8)], codec=codec, batch_size=2)
        decoded = codec.decode(codec.encode(test[3:6]))
        np.testing.assert_allclose(actual, -np.square(test[:3]-decoded).sum(axis=1), atol=1e-6)
        for i in range(3):
            d, _ = codec.search_adc(test[i:i+1], codec.encode(test[i+3:i+4]))
            assert actual[i] == pytest.approx(-d[0, 0], abs=1e-6)
    expected = np.einsum("ij,ij->i", test[:3], test[3:6])
    np.testing.assert_allclose(pair_scores(test, rows, [str(i) for i in range(8)]), expected)


def test_accuracy_threshold_is_calibration_optimum_with_ties():
    s = np.array([.1, .2, .2, .3, .6])
    y = np.array([False, False, True, True, True])
    tau = accuracy_threshold(s, y)
    candidates = np.r_[np.nextafter(s.max(), np.inf), np.unique(s)]
    assert ((s >= tau) == y).sum() == max(((s >= t) == y).sum() for t in candidates)


def test_diagnostic_uses_gap_and_never_splits_ties():
    impostors = np.array([.9, .6, .6, .2, .1])
    tau = empirical_fmr_threshold(impostors, .2)
    assert (impostors >= tau).sum() == 1
    assert .7 >= tau  # a genuine in the gap must be accepted at the same FMR
    tau = empirical_fmr_threshold(impostors, .4)
    assert (impostors >= tau).mean() <= .4
    assert (impostors >= empirical_fmr_threshold(impostors, .001)).sum() == 0


def test_counts_and_paired_contrasts(evaluated):
    from research.experiments.lfw_pair_verification import validate_job
    tables, settings = evaluated
    validate_job(tables, settings=settings, profiles=tuple(tables["operating"].compression_profile.unique()), test_count=600)
    for row in tables["operating"].itertuples():
        d = tables["decisions"]
        d = d[d.compression_profile.eq(row.compression_profile) & d.method.eq(row.method)]
        assert row.false_accepts == (d.accepted & ~d.is_genuine).sum()
        assert row.true_accepts == (d.accepted & d.is_genuine).sum()
        assert row.target_met_on_test == (row.realized_fmr <= row.target_fmr)
    assert not any("fpir" in c or "tpir" in c or "rank" in c for c in tables["operating"])
    assert not tables["paired"].empty


def test_test_scores_and_quality_do_not_change_fitted_models(pairs, inputs, evaluated):
    tables, settings = evaluated
    scores, quality = inputs
    scores = scores.copy()
    scores.loc[scores.fold.eq(1), "pair_score"] += 10
    quality = quality.copy()
    quality.loc[pairs.loc[pairs.fold.eq(1), "left_image_id"].unique()] += 20
    cal, test, _ = calibration_partition(pairs, 1, 8972)
    changed = evaluate_fold(scores, quality, cal, test, seed=8972, settings=settings)
    assert changed["models"].model_json.tolist() == tables["models"].model_json.tolist()
    np.testing.assert_array_equal(changed["accuracy_cv"].threshold, tables["accuracy_cv"].threshold)
    assert not np.array_equal(changed["operating"].false_accepts, tables["operating"].false_accepts)


def test_checkpoint_integrity_and_immutability(tmp_path):
    store = PairStore(tmp_path / "checkpoint.sqlite3")
    spec = dict(example="a")
    tables = dict(example=pd.DataFrame({"x": [1, 2]}))
    store.put(spec, tables)
    store.put(spec, tables)
    with pytest.raises(AssertionError):
        store.put(spec, dict(example=pd.DataFrame({"x": [1, 3]})))
    store.connection.execute("UPDATE records SET sha='wrong'")
    store.connection.commit()
    with pytest.raises(ValueError, match="integrity"):
        store.get(spec)
    store.close()


def test_compact_publish_and_safe_retention(tmp_path, evaluated):
    tables, _ = evaluated
    raw = tmp_path / "raw" / "checkpoint.sqlite3"
    store = PairStore(raw)
    store.put(dict(test=1), tables)
    store.close()
    report = publish(tmp_path, tables, spec=dict(keep_raw_results=False), completed_jobs=1, expected_jobs=2,
                     raw_path=raw, guide=ROOT / "notebooks/calibration/LFW_PAIR_VERIFICATION.md")
    compact, manifest = read_report(report)
    assert manifest["status"] == "partial"
    assert "decisions" not in compact and "models" not in compact
    with pytest.raises(ValueError):
        cleanup_checkpoint(raw, tmp_path, report)
    report = publish(tmp_path, tables, spec=dict(keep_raw_results=False), completed_jobs=2, expected_jobs=2,
                     raw_path=raw, guide=ROOT / "notebooks/calibration/LFW_PAIR_VERIFICATION.md")
    cleanup_checkpoint(raw, tmp_path, report)
    assert not raw.exists()
    read_report(report)
    (report / "START_HERE.md").write_text("corrupt")
    with pytest.raises(ValueError, match="integrity"):
        read_report(report)


@pytest.mark.parametrize("keep", [False, True])
def test_campaign_resume_scope_and_retention(tmp_path, pairs, inputs, evaluated, monkeypatch, keep):
    import yaml
    from research.experiments import lfw_pair_verification as module
    from unittest.mock import Mock
    config = yaml.safe_load((ROOT / module.CONFIG_PATH).read_text(encoding="utf8"))
    plan = dict(config=config, settings=evaluated[1], models=("arcface",), folds=(1,), seeds=(0, 8972),
        pairs=pairs, development=pd.DataFrame(dict(image_id=["unused_1", "unused_2"])),
        inventory=pd.DataFrame(), coverage=pd.DataFrame(), quality=inputs[1], sources={"arcface":{"fixture":True}},
        ready=True, expected_jobs=2, development_images=256, development_identities=256,
        protocol_uid=module.PROTOCOL_UID)
    monkeypatch.setattr(module, "inspect_verification", lambda *a, **kw: plan)
    preparation = Mock(return_value=dict(scores=inputs[0], codecs=pd.DataFrame(dict(fixture=[1]))))
    monkeypatch.setattr(module, "prepare_model", preparation)
    kwargs = dict(execute=True, output_root=tmp_path, keep_raw_results=keep, max_new_jobs=1)
    first = module.run_verification(ROOT, **kwargs)
    assert first["completed_jobs"] == 1 and not first["completed"]
    assert first["checkpoint_path"].exists()
    second = module.run_verification(ROOT, **kwargs)
    assert second["completed"] and second["completed_jobs"] == 2 and second["new_jobs"] == 1
    assert preparation.call_count == 1  # scores/codebooks reused, not refitted
    assert second["checkpoint_path"].exists() is keep
    third = module.run_verification(ROOT, **kwargs)
    assert third["new_jobs"] == 0 and third["reused"]
    tables, manifest = read_report(second["report_dir"])
    assert manifest["spec"]["keep_raw_results"] is keep
    assert tables["operating"].seed_count.eq(2).all()
    assert not tables["accuracy_summary"].all_ten_folds_complete.any()
    assert (first["report_dir"] / "manifest.json").exists()  # previous snapshot immutable


@pytest.mark.parametrize("relative", [
    "notebooks/calibration/03_origin_vs_pq_fiqa_calibration.ipynb",
    "notebooks/common/orchestration/00_batch_experiment_runner.ipynb",
    "notebooks/common/orchestration/01_batch_fiqa_saliency_calibration.ipynb",
    "notebooks/common/reports/00_cross_dataset_results.ipynb",
])
def test_default_notebooks_use_pair_path_without_legacy(relative, monkeypatch):
    from research.experiments import lfw_pair_inputs as pair_module
    from research.experiments import pipeline_runner
    from unittest.mock import Mock
    forbidden = Mock(side_effect=AssertionError("legacy GPU path called"))
    monkeypatch.setattr(pipeline_runner, "prepare_common_model_checkpoint", forbidden)
    called = Mock(return_value=dict(coverage=pd.DataFrame(), inventory=pd.DataFrame(), expected_jobs=800))
    monkeypatch.setattr(pair_module, "run_pair_workflow", called)
    monkeypatch.setattr(pair_module, "prepare_pair_inputs", called)
    nb = json.loads((ROOT / relative).read_text(encoding="utf8"))
    ns = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), relative, "exec"), ns)
    forbidden.assert_not_called()
    if "/reports/" not in relative:
        called.assert_called_once()
        assert called.call_args.kwargs["execute"] is False
    assert ns["LFW_PROTOCOL_MODE"].startswith("pair_verification")
