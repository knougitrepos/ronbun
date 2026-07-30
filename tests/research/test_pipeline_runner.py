from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from research.experiments import pipeline_runner
from research.experiments.pipeline_runner import (
    FULL_DATA_FRACTION,
    QUICK_DATA_FRACTIONS,
    build_common_experiment_plan,
    inspect_common_experiment_plan,
    load_evaluation_contract,
    materialize_effective_step4_config,
    run_common_step4_experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "configs" / "experiments" / "evaluation_contract_v1.yaml"


def _write_plan_inputs(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "step4.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "root": "runs",
                    "dataset_date_dir_template": "{dataset_id}_{date}",
                },
                "execution": {
                    "model_profile": "arcface_registered",
                    "mode": "real",
                    "data_fraction": 1.0,
                    "seed": 42,
                    "execute_stage": True,
                    "write_outputs": True,
                    "overwrite": False,
                    "allow_dirty": False,
                },
                "models": {
                    "profiles": {
                        "arcface_registered": {
                            "family": "arcface",
                        }
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    contract_path = tmp_path / "evaluation_contract_v1.yaml"
    contract_path.write_text(
        CONTRACT_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return config_path, contract_path


@pytest.mark.parametrize(
    ("dataset_id", "run_tier", "expected_fraction", "expected_selected"),
    [
        ("lfw", "quick", 0.10, 10),
        ("survface", "quick", 0.02, 2),
        ("lfw", "full", 1.00, 100),
        ("survface", "full", 1.00, 100),
    ],
)
def test_common_plan_uses_only_confirmed_quick_full_fractions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_id: str,
    run_tier: str,
    expected_fraction: float,
    expected_selected: int,
) -> None:
    config_path, contract_path = _write_plan_inputs(tmp_path)
    source = pd.DataFrame(
        {
            "image_id": [f"image-{index:03d}" for index in range(100)],
            "identity_id": [f"identity-{index:03d}" for index in range(100)],
            "split": ["test"] * 100,
            "protocol_role": ["registered_probe"] * 100,
        }
    )
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_step4_dataset_spec",
        lambda *args, **kwargs: SimpleNamespace(
            dataset_id=dataset_id,
            manifest_paths=(),
        ),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "load_step4_source_manifest",
        lambda spec: source,
    )

    def select(frame, *, dataset_id, scope):
        assert scope.mode == "real"
        assert scope.data_fraction == expected_fraction
        return frame.iloc[:expected_selected].copy()

    monkeypatch.setattr(
        pipeline_runner,
        "select_step4_source_manifest",
        select,
    )
    plan = build_common_experiment_plan(
        project_root=tmp_path,
        dataset_id=dataset_id,
        run_tier=run_tier,
        step4_config_path=config_path,
        evaluation_contract_path=contract_path,
    )

    assert plan.data_fraction == expected_fraction
    assert plan.selected_source_rows == expected_selected
    assert len(plan.selected_image_ids_sha256) == 64
    assert plan.scope_paper_eligible is (run_tier == "full")
    assert plan.comparison_paper_eligible is False
    assert plan.effective_step4_config["execution"]["mode"] == "real"
    assert (
        plan.effective_step4_config["execution"]["data_fraction"] == expected_fraction
    )
    assert plan.effective_step4_config["orchestration"]["run_tier"] == run_tier


def test_contract_constants_match_user_confirmed_values() -> None:
    contract = load_evaluation_contract(CONTRACT_PATH)

    assert dict(QUICK_DATA_FRACTIONS) == {"lfw": 0.10, "survface": 0.02}
    assert FULL_DATA_FRACTION == 1.0
    assert contract["comparison"]["main_profiles"] == [
        "origin_512",
        "pca_256",
        "pca_128",
        "pq_origin_512_m128_nbits8",
    ]
    assert contract["pq_search"]["primary"] == "exhaustive_adc"
    assert contract["calibration"]["enrollment_identity_counts"] == [
        100,
        500,
        1000,
    ]


def test_contract_rejects_unconfirmed_quick_fraction(tmp_path: Path) -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["execution"]["quick_data_fractions"]["survface"] = 0.05
    path = tmp_path / "invalid.yaml"
    path.write_text(
        yaml.safe_dump(contract, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="user-confirmed contract"):
        load_evaluation_contract(path)


def test_materialized_effective_config_is_restart_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, contract_path = _write_plan_inputs(tmp_path)
    source = pd.DataFrame(
        {
            "image_id": ["image-1"],
            "identity_id": ["identity-1"],
            "split": ["test"],
        }
    )
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_step4_dataset_spec",
        lambda *args, **kwargs: SimpleNamespace(
            dataset_id="lfw",
            manifest_paths=(),
        ),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "load_step4_source_manifest",
        lambda spec: source,
    )
    monkeypatch.setattr(
        pipeline_runner,
        "select_step4_source_manifest",
        lambda frame, **kwargs: frame.copy(),
    )
    plan = build_common_experiment_plan(
        project_root=tmp_path,
        dataset_id="lfw",
        run_tier="quick",
        step4_config_path=config_path,
        evaluation_contract_path=contract_path,
    )

    first = materialize_effective_step4_config(plan)
    second = materialize_effective_step4_config(plan)

    assert first == second
    assert yaml.safe_load(first.read_text(encoding="utf-8")) == (
        plan.effective_step4_config
    )


def test_preflight_reports_partial_evaluation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, contract_path = _write_plan_inputs(tmp_path)
    source = pd.DataFrame(
        {
            "image_id": ["image-1"],
            "identity_id": ["identity-1"],
            "split": ["test"],
        }
    )
    monkeypatch.setattr(
        pipeline_runner,
        "resolve_step4_dataset_spec",
        lambda *args, **kwargs: SimpleNamespace(
            dataset_id="lfw",
            manifest_paths=(),
        ),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "load_step4_source_manifest",
        lambda spec: source,
    )
    monkeypatch.setattr(
        pipeline_runner,
        "select_step4_source_manifest",
        lambda frame, **kwargs: frame.copy(),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "inspect_step4_readiness",
        lambda *args, **kwargs: {
            "ready_to_materialize": True,
            "ready_to_run_experiment": False,
        },
    )
    plan = build_common_experiment_plan(
        project_root=tmp_path,
        dataset_id="lfw",
        run_tier="quick",
        step4_config_path=config_path,
        evaluation_contract_path=contract_path,
    )

    report = inspect_common_experiment_plan(plan)

    assert report["ready_to_execute_pipeline"] is True
    assert report["evaluation_contract_coverage"]["pq_exhaustive_adc"] == ("proposed")
    assert plan.comparison_paper_eligible is False


def test_execution_requires_explicit_user_acknowledgement(
    tmp_path: Path,
) -> None:
    plan = SimpleNamespace()

    with pytest.raises(RuntimeError, match="explicit local acknowledgement"):
        run_common_step4_experiment(plan)
