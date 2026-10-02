from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile
import copy
import io

import numpy as np
import pandas as pd
import pytest

from research.datasets.lfw_blufr import (
    BLUFRLists,
    BLUFRTrial,
    load_blufr_lists,
    bind_image_manifest,
)
from research.datasets.lfw_blufr_calibration import build_calibration_split
from research.experiments import lfw_blufr_calibration as module
from test_origin_vs_pq_calibration import inputs, options


def synthetic():
    identities = np.repeat([f"person{i:03}" for i in range(180)], 4)
    images = pd.DataFrame(
        {
            "identity_name": identities,
            "filename": [
                f"{name}_{i % 4 + 1:04}.jpg" for i, name in enumerate(identities)
            ],
        }
    )
    train = np.arange(600)
    gallery = np.array([604, 600, 608, 612, 616])
    test = np.arange(600, 720)
    probe = np.array([i for i in test[::-1] if i not in gallery])
    lists = BLUFRLists(
        images, (BLUFRTrial(1, train, test, gallery, probe),), "synthetic"
    )
    bound = images.assign(
        image_id=[f"s{i}" for i in range(720)],
        identity_id="lfw:" + images.identity_name,
    )
    return lists, bound


def test_nested_ratios_public_lists_and_identity_isolation():
    lists, bound = synthetic()
    previous = set()
    watchlist = None
    for fraction in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        split = build_calibration_split(
            lists,
            bound,
            trial_id=1,
            development_fraction=fraction,
            calibration_gallery_count=5,
            seed=8972,
        )
        frame = split.assignment
        dev = set(frame.loc[frame.role.eq("development"), "identity_id"])
        cal = set(frame.loc[frame.role.str.startswith("calibration"), "identity_id"])
        test = set(frame.loc[frame.role.str.startswith("test"), "identity_id"])
        assert previous <= dev
        assert not (dev & cal or dev & test or cal & test)
        assert len(dev) == int(150 * fraction)
        previous = dev
        current = frame.loc[frame.role.eq("calibration_gallery"), "image_id"].tolist()
        if watchlist is not None:
            assert current == watchlist
        watchlist = current
        np.testing.assert_array_equal(
            frame.loc[frame.role.eq("test_gallery")]
            .sort_values("protocol_order")
            .index,
            lists.trial(1).gallery,
        )
        np.testing.assert_array_equal(
            frame.loc[frame.role.isin(["test_mated", "test_non_mated"])]
            .sort_values("protocol_order")
            .index,
            lists.trial(1).probe,
        )


def test_all_released_trials_and_seven_ratios():
    root = Path(__file__).resolve().parents[2]
    path = root / "data/external/blufr/blufr_lfw_config.mat"
    if not path.exists():
        pytest.skip("public lists not downloaded locally")
    lists = load_blufr_lists(path)
    bound = bind_image_manifest(
        lists, pd.read_csv(root / "data/interim/lfw/face_manifest.csv")
    )
    for trial in lists.trials:
        for fraction in (0.5, 0.6, 0.7, 0.8, 0.4, 0.3, 0.2):
            split = build_calibration_split(
                lists,
                bound,
                trial_id=trial.trial_id,
                development_fraction=fraction,
                calibration_gallery_count=100,
                seed=8972,
            )
            inv = split.inventory
            assert inv["development_identities"] == int(1500 * fraction)
            assert inv["development_identities"] + inv["calibration_identities"] == 1500
            assert inv["development_images"] >= 256
            assert inv["test_gallery_identities"] == 1000
            assert inv["calibration_gallery_identities"] == 100
            assert not inv["gallery_size_matched"]
            frame = split.assignment
            assert set(frame.index[frame.outer_split.eq("train")]) == set(trial.train)
            np.testing.assert_array_equal(
                frame.loc[frame.role.eq("test_gallery")]
                .sort_values("protocol_order")
                .index,
                trial.gallery,
            )
            np.testing.assert_array_equal(
                frame.loc[frame.role.isin(["test_mated", "test_non_mated"])]
                .sort_values("protocol_order")
                .index,
                trial.probe,
            )


@pytest.mark.parametrize("real_pq", [False, True])
def test_prepare_fits_development_only_and_preserves_scores(
    tmp_path, monkeypatch, real_pq
):
    import faiss

    lists, bound = synthetic()
    split = build_calibration_split(
        lists,
        bound,
        trial_id=1,
        development_fraction=0.5,
        calibration_gallery_count=5,
        seed=7,
    )
    rng = np.random.default_rng(10)
    vectors = rng.normal(size=(720, 512)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    prepared = SimpleNamespace(
        sample_ids=bound.image_id.to_numpy(), normalized_embeddings=vectors
    )
    population = bound.assign(aligned_content_sha256="hash")
    seen = []

    class Codec:
        def __init__(self, **kwargs):
            self.index = faiss.IndexFlatL2(512)

        def fit(self, data):
            seen.append(data.copy())
            return self

    real_codec = module.PQCompressor

    class ActualCodec(real_codec):
        def fit(self, data):
            seen.append(data.copy())
            return super().fit(data)

    monkeypatch.setattr(module, "PQCompressor", ActualCodec if real_pq else Codec)
    original = module.search_rows
    # Search semantics are covered by existing ADC tests; spy here isolates fit membership.
    if not real_pq:
        monkeypatch.setattr(
            module,
            "search_rows",
            lambda *a, **kw: original(*a, **{**kw, "codec": None}),
        )
    base, quality = inputs(tmp_path)
    lineage = {
        k: v
        for k, v in base[module.ORIGIN_PROFILE].manifest.items()
        if k
        in (
            "source_run_id",
            "source_run_manifest_sha256",
            "source_freeze_manifest_sha256",
            "selected_manifest_sha256",
            "prepared_population_manifest_sha256",
            "aligned_bundle_manifest_sha256",
        )
    }
    lineage.update(
        dataset_id="lfw",
        model_uid="test",
        extraction_uid="extract",
        origin_embedding_artifact_uid="embed",
    )
    config = dict(
        top_k=5,
        split_seed=7,
        codec_seed=7,
        pq_profiles=["pq_512_m128_b8"],
        query_batch_size=16,
    )
    threads = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(1)
        conditions, packed = module.prepare_conditions(
            prepared,
            population,
            None,
            lineage,
            split,
            config,
            resource_check=lambda: None,
        )
    finally:
        faiss.omp_set_num_threads(threads)
    np.testing.assert_array_equal(
        seen[0], vectors[split.assignment.role.eq("development")]
    )
    assert len(packed["codecs"]) == 1
    origin = conditions[module.ORIGIN_PROFILE]
    assert (
        origin.test.sample_id.tolist()
        == bound.iloc[lists.trial(1).probe].image_id.tolist()
    )
    assert set(origin.calibration.sample_id).isdisjoint(origin.test.sample_id)
    assert (
        conditions["pq_512_m128_b8"].manifest["score_space"]
        == "negative_squared_l2_adc"
    )
    quality = replace(
        quality,
        scores=pd.DataFrame(
            dict(
                sample_id=bound.image_id,
                aligned_content_sha256="hash",
                fiqa_score=np.linspace(0.0, 1.0, len(bound)),
                fiqa_model_uid=quality.manifest["fiqa_model_uid"],
            )
        ),
        manifest={**quality.manifest, "dataset_id": "lfw"},
    )
    result = module.run_origin_pq_split(conditions, quality, settings=options())
    changed = copy.deepcopy(conditions)
    for condition in changed.values():
        condition.test.loc[~condition.test.is_mated, "score"] += 0.5
    shifted = module.run_origin_pq_split(changed, quality, settings=options())
    assert result["models"].model_json.equals(shifted["models"].model_json)


def test_missing_coverage_fails_before_any_write(tmp_path, monkeypatch):
    inspected = dict(ready=False, coverage=pd.DataFrame([dict(missing_embeddings=38)]))
    monkeypatch.setattr(module, "inspect_calibration", lambda *a, **kw: inspected)
    with pytest.raises(ValueError, match="no images may be silently excluded"):
        module.run_calibration(tmp_path, execute=True)
    assert not list(tmp_path.iterdir())


def test_checkpoint_resume_and_ratio_specific_compact(tmp_path, monkeypatch):
    conditions, quality = inputs(tmp_path)
    config = dict(
        output_root="new-results",
        source_runs={"arcface": "source"},
        trial_ids=[1],
        development_fractions=[0.5, 0.6],
    )
    lineage = dict(source_run_id="test-run")
    inspected = dict(
        ready=True,
        config=config,
        models=["arcface"],
        inventory=pd.DataFrame([{}, {}]),
        sources={"arcface": lineage},
        lists=None,
        bound=None,
    )
    monkeypatch.setattr(module, "inspect_calibration", lambda *a, **kw: inspected)
    monkeypatch.setattr(module, "_quality", lambda *a: (quality, None))
    monkeypatch.setattr(
        module, "_source", lambda *a: (None, None, None, None, None, None)
    )
    monkeypatch.setattr(
        module,
        "_split",
        lambda *a: SimpleNamespace(
            inventory={"trial_id": 1, "development_fraction": a[-1]}
        ),
    )
    monkeypatch.setattr(module, "check_resources", lambda **kw: None)
    prepare = Mock(
        side_effect=lambda a, b, c, d, split, f, **kw: (
            conditions,
            module._pack_conditions(conditions, pd.DataFrame([split.inventory])),
        )
    )
    monkeypatch.setattr(module, "prepare_conditions", prepare)
    first = module.run_calibration(
        tmp_path,
        execute=True,
        settings=options(),
        partition_seeds=(1,),
        max_new_jobs=1,
        progress=lambda x: None,
    )
    assert first["completed_jobs"] == 1 and not first["completed"]
    second = module.run_calibration(
        tmp_path,
        execute=True,
        settings=options(),
        partition_seeds=(1,),
        max_new_jobs=None,
        progress=lambda x: None,
    )
    assert second["completed"] and second["new_jobs"] == 1
    third = module.run_calibration(
        tmp_path,
        execute=True,
        settings=options(),
        partition_seeds=(1,),
        max_new_jobs=None,
        progress=lambda x: None,
    )
    assert third["new_jobs"] == 0 and third["report_dir"] == second["report_dir"]
    assert prepare.call_count == 2
    tables, _ = module.read_calibration_report(second["report_dir"])
    assert set(tables["method_summary"].development_fraction) == {0.5, 0.6}
    with ZipFile(Path(second["chat_dir"]) / "analysis.zip") as zipped:
        summary = pd.read_csv(io.BytesIO(zipped.read("performance_by_trial_ratio.csv")))
    assert set(summary.development_fraction) == {0.5, 0.6}
    assert summary.fit_safety_seed_count.eq(1).all()
    assert len(list((tmp_path / "new-results").glob("*.sqlite3"))) == 2
