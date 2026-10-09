"""Baseline identity, provenance and experiment-axis regression checks."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

from research.experiments import lfw_baselines as baselines
from research.experiments.lfw_pair_verification import validate_selection
from research.experiments.lfw_pair_storage import backbone_contrasts
from research.runtime.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def configured(tmp_path):
    config = yaml.safe_load(
        (ROOT / "configs/experiments/lfw_pair_verification.yaml").read_text(
            encoding="utf8"
        )
    )
    config["inputs"]["model_profiles_config"] = str(
        ROOT / "configs/experiments/step2_pytorch_gradcam.yaml"
    )
    first = tmp_path / "original.pth"
    first.write_bytes(b"original fixture, not inference weights")
    second = tmp_path / "tuned.pth"
    second.write_bytes(b"modified fixture, not a trained recognizer")
    config["inputs"]["models"] = {
        "base": dict(
            profile="arcface_ms1mv3_r100",
            checkpoint=str(first),
            expected_sha256=sha256_file(first),
        )
    }
    config["source_runs"] = {"base": "results/fixture/sources/base"}
    config["inputs"]["registry_root"] = "results/fixture/model_registry"
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(config), encoding="utf8")
    kwargs = dict(
        project_root=tmp_path,
        config_path="base.yaml",
        output_config="comparison.yaml",
        alias="tuned",
        profile="arcface_ms1mv3_r100",
        checkpoint=second,
        source_url="https://example.invalid/test-fixture",
        training_dataset="fixture-original-plus-external",
        kind="fine_tuned",
        parent_baseline="base",
        fine_tuning_data="synthetic metadata fixture",
        lfw_identity_overlap="unknown",
        overlap_evidence="unverified fixture",
        evaluation_used_for_training_or_selection=False,
        selection_evidence="unit fixture; no benchmark claim",
    )
    return tmp_path, config, kwargs


def test_finetuned_registration_preserves_parent_and_separates_sources(configured):
    root, original, kwargs = configured
    old = (root / "base.yaml").read_bytes()
    output = baselines.register_baseline(**kwargs)
    config = yaml.safe_load(output.read_text(encoding="utf8"))
    assert (root / "base.yaml").read_bytes() == old
    assert config["inputs"]["models"]["base"] == original["inputs"]["models"]["base"]
    specs = baselines.model_specs(root, config, ("base", "tuned"))
    assert specs["base"].family == specs["tuned"].family == "arcface"
    assert specs["base"].model_uid != specs["tuned"].model_uid
    assert specs["tuned"].training_dataset == kwargs["training_dataset"]
    assert config["source_runs"]["base"] != config["source_runs"]["tuned"]
    assert config["pq_profiles"] == original["pq_profiles"]
    assert (
        config["fold_ids"] == original["fold_ids"]
        and config["partition_seeds"] == original["partition_seeds"]
    )
    assert (
        config["resize_inputs"]["checkpoint_file"]
        != original["resize_inputs"]["checkpoint_file"]
    )
    catalog = baselines.baseline_catalog(config, specs)
    assert not catalog.unseen_identity_claim_supported.any()
    assert catalog[catalog.model.eq("tuned")].parent_baseline.item() == "base"
    with pytest.raises(FileExistsError):
        baselines.register_baseline(**kwargs)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"lfw_identity_overlap": "overlap"}, "evaluation identities"),
        ({"evaluation_used_for_training_or_selection": True}, "training_or_selection"),
        ({"evaluation_used_for_training_or_selection": None}, "training_or_selection"),
        ({"selection_evidence": None}, "selection_evidence"),
        ({"parent_baseline": "absent"}, "parent"),
        ({"alias": "../../escape"}, "aliases"),
    ],
)
def test_invalid_finetuned_provenance_rejected_before_publication(
    configured, change, match
):
    root, _, kwargs = configured
    with pytest.raises(ValueError, match=match):
        baselines.register_baseline(**(kwargs | change))
    assert not (root / "comparison.yaml").exists()


def test_checkpoint_relabeling_and_changed_bytes_rejected(configured):
    root, original, kwargs = configured
    kwargs["checkpoint"] = original["inputs"]["models"]["base"]["checkpoint"]
    with pytest.raises(ValueError, match="duplicate checkpoint"):
        baselines.register_baseline(**kwargs)
    config = deepcopy(original)
    Path(config["inputs"]["models"]["base"]["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        baselines.model_specs(root, config, ("base",))


def test_finetuned_selection_retains_parent_and_rejects_colliding_paths(configured):
    root, _, kwargs = configured
    config = yaml.safe_load(
        baselines.register_baseline(**kwargs).read_text(encoding="utf8")
    )
    with pytest.raises(ValueError, match="parent baseline"):
        validate_selection(config, models=("tuned",))
    config["source_runs"]["tuned"] = config["source_runs"]["base"]
    with pytest.raises(ValueError, match="distinct"):
        validate_selection(config)


def test_invalid_matrix_never_starts_gpu_preparation(configured, monkeypatch):
    root, _, _ = configured
    from research.experiments import lfw_pair_inputs

    prepare = Mock(side_effect=AssertionError("GPU preparation should not start"))
    monkeypatch.setattr(lfw_pair_inputs, "prepare_pair_inputs", prepare)
    for value in ((), (True,), (11,), (1, 1)):
        with pytest.raises(ValueError):
            lfw_pair_inputs.run_pair_workflow(
                root, config_path="base.yaml", execute=True, fold_ids=value
            )
    prepare.assert_not_called()


def test_backbone_effect_and_additional_fiqa_effect_are_separate():
    rows = []
    for model, tar, fmr in (("base", 0.5, 0.09), ("tuned", 0.7, 0.11)):
        for method, gain in (
            ("global_safe", 0),
            ("continuous_fiqa", 0.04 if model == "base" else 0.01),
        ):
            rows.append(
                dict(
                    model=model,
                    fold=1,
                    partition_seed=8972,
                    compression_profile="origin",
                    score_space="cosine_similarity",
                    method=method,
                    target_fmr=0.1,
                    tar=tar + gain,
                    realized_fmr=fmr,
                    test_pairs=600,
                    genuine_pairs=300,
                    impostor_pairs=300,
                )
            )
    frame = pd.DataFrame(rows)
    catalog = pd.DataFrame(
        [
            dict(model="base", parent_baseline="none"),
            dict(model="tuned", parent_baseline="base"),
        ]
    )
    contrasts = backbone_contrasts(frame, catalog).set_index("method")
    assert contrasts.loc["global_safe", "tar_difference"] == pytest.approx(0.2)
    assert contrasts.loc["continuous_fiqa", "fiqa_gain_difference"] == pytest.approx(
        -0.03
    )
    assert contrasts.realized_fmr_difference.tolist() == pytest.approx([0.02, 0.02])
    with pytest.raises(ValueError, match="identical"):
        backbone_contrasts(frame.iloc[:-1], catalog)


def test_declared_pair_roles_match_actual_fitting_roles():
    from research.datasets.lfw_pairs import load_pairs, calibration_partition
    from research.calibration.conditional import deterministic_calibration_partition

    # Real pinned pair list; omit this environment check if the dataset is absent.
    manifest = ROOT / "data/interim/lfw/pair_verification_v1/population.csv"
    if not manifest.is_file():
        pytest.skip("local LFW population not installed")
    pairs, _ = load_pairs(ROOT / "data/external/lfw/pairs.txt", pd.read_csv(manifest))
    for fold in range(1, 11):
        for seed in (*range(19), 8972):
            cal, _, inventory = calibration_partition(pairs, fold, seed)
            actual = deterministic_calibration_partition(
                cal.assign(identity_id=cal.left_identity), seed=seed
            )
            assert actual.tolist() == cal.calibration_role.tolist()
            assert ((actual == "fit") & ~cal.is_genuine).sum() == inventory[
                "fit_impostor"
            ]


def test_survface_explicit_new_uid_keeps_source_and_codec_checks(tmp_path, monkeypatch):
    from research.experiments import calibration_matrix as matrix

    uid = "arcface-fixture-finetuned"
    root = tmp_path / "run"
    workflow = root / "artifacts"
    run = dict(
        run_id="fixture",
        config=dict(
            dataset_id="survface", model_uid=uid, step4={"evaluation": {"top_k": 20}}
        ),
    )
    freeze = dict(
        run_id="fixture",
        dataset_id="survface",
        model_uid=uid,
        extraction_uid="extract",
        scope=dict(data_fraction=1.0, seed=8972),
        fallback_free=True,
        selected_manifest_sha256="sha",
        aligned_bundle_manifest_sha256="aligned",
    )

    def read(path):
        path = Path(path)
        if path.name == "freeze_manifest.json":
            return freeze
        if path.parent.name == "prepared_population":
            return dict(dataset_id="survface", model_uid=uid, extraction_uid="extract")
        return {}

    monkeypatch.setattr(matrix, "_completed_run", lambda p: (root, run, workflow))
    monkeypatch.setattr(matrix, "_read_json", read)
    monkeypatch.setattr(matrix, "sha256_file", lambda p: "sha")
    codec = dict(
        fit_source_run_id="fixture", fit_source_dataset="survface", model_uid=uid
    )
    monkeypatch.setattr(
        matrix,
        "_frozen_pq_codec",
        lambda *a, **kw: (None, {"artifact_sha256": "codec"}, codec),
    )
    monkeypatch.setattr(
        matrix,
        "_ledger_condition",
        lambda *a, **kw: dict(core={"sha256": "core"}, row_count=100),
    )
    monkeypatch.setattr(matrix, "_verified_table", lambda *a: None)
    args = dict(datasets=("survface",), models=("arc_ft",), profiles=("pq_512_m32_b8",))
    with pytest.raises(ValueError, match="models"):
        matrix.inspect_calibration_matrix(
            tmp_path, {"arc_ft": {"survface": root}}, **args
        )
    plan = matrix.inspect_calibration_matrix(
        tmp_path, {"arc_ft": {"survface": root}}, model_uids={"arc_ft": uid}, **args
    )
    assert plan.model_uid.item() == uid and plan.ready.all()
    codec["model_uid"] = "original-checkpoint"
    with pytest.raises(ValueError, match="codec lineage"):
        matrix.inspect_calibration_matrix(
            tmp_path, {"arc_ft": {"survface": root}}, model_uids={"arc_ft": uid}, **args
        )


def test_exported_survface_profile_registers_the_same_checkpoint_uid(configured):
    root, _, kwargs = configured
    config_path = baselines.register_baseline(**kwargs)
    exported = baselines.export_step4_profile(
        root, config_path=config_path, alias="tuned", output_config="step4.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))
    spec = baselines.model_specs(root, config, ("base", "tuned"))["tuned"]
    from research.experiments.pipeline_runner import prepare_common_model_checkpoint

    # Real configuration/registry path. The fixture is NOT loaded for inference.
    registered = prepare_common_model_checkpoint(
        project_root=root,
        model_name="arc",
        model_profile="tuned",
        checkpoint_path=kwargs["checkpoint"],
        step4_config_path=exported,
        run_smoke_validation=False,
    )
    assert registered.model_uid == spec.model_uid
    loaded = yaml.safe_load(exported.read_text(encoding="utf8"))
    assert (
        loaded["models"]["profiles"]["tuned"]["baseline_provenance"]["fine_tuning_data"]
        == kwargs["fine_tuning_data"]
    )


def test_integrated_survface_plan_receives_custom_profile_config(monkeypatch):
    from research.experiments import integrated_pipeline as pipeline
    from types import SimpleNamespace

    builder = Mock(return_value=SimpleNamespace(dataset_id="survface"))
    monkeypatch.setattr(pipeline, "build_common_experiment_plan", builder)
    pipeline.build_integrated_experiment_plans(
        project_root=ROOT,
        dataset_ids=("survface",),
        quick_data_fractions=dict(lfw=0.1, survface=0.02, rfw_custom=0.1, tinyface=0.1),
        seed=8972,
        model_preparation=SimpleNamespace(
            model_profile="tuned", model_uid="fine-uid", checkpoint_path="weights.pth"
        ),
        model_name="arc",
        step4_config_path="registered-step4.yaml",
    )
    assert builder.call_args.kwargs["step4_config_path"] == "registered-step4.yaml"
    assert builder.call_args.kwargs["model_uid"] == "fine-uid"


def test_baseline_rejects_non_512d_dimension(configured):
    root, original, kwargs = configured
    profiles_data = yaml.safe_load((root / original["inputs"]["model_profiles_config"]).read_text(encoding="utf8"))
    profiles_data["models"]["profiles"]["arcface_ms1mv3_r100"]["embedding_dim"] = 256
    custom_profiles = root / "custom_profiles.yaml"
    custom_profiles.write_text(yaml.safe_dump(profiles_data), encoding="utf8")
    cfg = deepcopy(original)
    cfg["inputs"]["model_profiles_config"] = str(custom_profiles)
    with pytest.raises(ValueError, match="512D embedding dimension required"):
        baselines.model_specs(root, cfg, ("base",))


