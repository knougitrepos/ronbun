from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from research.experiments import integrated_pipeline
from research.experiments.integrated_pipeline import (
    INTEGRATED_DATASET_IDS,
    OPEN_SET_DATASET_IDS,
    build_integrated_experiment_plans,
    default_integrated_quick_data_fractions,
    is_open_set_dataset,
    validate_completed_run_overrides,
    validate_integrated_dataset_ids,
    validate_integrated_quick_data_fractions,
)


EXPECTED_FRACTIONS = {
    "lfw": 0.10,
    "survface": 0.02,
    "rfw_custom": 0.10,
    "tinyface": 0.10,
}


def test_integrated_contract_contains_all_four_datasets_and_defaults() -> None:
    assert OPEN_SET_DATASET_IDS == ("lfw", "survface", "rfw_custom")
    assert INTEGRATED_DATASET_IDS == (
        "lfw",
        "survface",
        "rfw_custom",
        "tinyface",
    )
    assert default_integrated_quick_data_fractions() == EXPECTED_FRACTIONS
    assert validate_integrated_dataset_ids(INTEGRATED_DATASET_IDS) == (
        "lfw",
        "survface",
        "rfw_custom",
        "tinyface",
    )
    assert validate_integrated_quick_data_fractions(EXPECTED_FRACTIONS) == (
        EXPECTED_FRACTIONS
    )
    assert is_open_set_dataset("lfw") is True
    assert is_open_set_dataset("tinyface") is False


@pytest.mark.parametrize(
    "dataset_ids",
    [(), ("lfw", "lfw"), ("lfw", "unknown")],
)
def test_integrated_dataset_validation_fails_closed(dataset_ids: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        validate_integrated_dataset_ids(dataset_ids)


def test_quick_fraction_and_completed_override_keys_fail_closed() -> None:
    incomplete = dict(EXPECTED_FRACTIONS)
    incomplete.pop("tinyface")
    with pytest.raises(ValueError, match="keys must be exactly"):
        validate_integrated_quick_data_fractions(incomplete)
    with pytest.raises(ValueError, match="not selected"):
        validate_completed_run_overrides(
            {"tinyface": "runs/tinyface/example"},
            selected_dataset_ids=("lfw",),
        )


def test_plan_builder_uses_one_contract_and_full_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_common(**kwargs: object) -> SimpleNamespace:
        calls.append(("common", dict(kwargs)))
        return SimpleNamespace(dataset_id=kwargs["dataset_id"])

    def fake_tinyface(**kwargs: object) -> SimpleNamespace:
        calls.append(("tinyface", dict(kwargs)))
        return SimpleNamespace(dataset_id="tinyface")

    monkeypatch.setattr(
        integrated_pipeline,
        "build_common_experiment_plan",
        fake_common,
    )
    monkeypatch.setattr(
        integrated_pipeline,
        "build_tinyface_experiment_plan",
        fake_tinyface,
    )
    preparation = SimpleNamespace(
        model_profile="edgeface-profile",
        model_uid="edgeface-test",
        model_spec_path=tmp_path / "model_spec.json",
        checkpoint_path=tmp_path / "weights.pt",
    )

    plans = build_integrated_experiment_plans(
        project_root=tmp_path,
        dataset_ids=INTEGRATED_DATASET_IDS,
        quick_data_fractions=EXPECTED_FRACTIONS,
        seed=42,
        model_preparation=preparation,
        model_name="edge",
        include_pq_sdc=False,
    )

    assert tuple(plans) == INTEGRATED_DATASET_IDS
    assert len(calls) == 4
    assert all(kwargs["run_tier"] == "full" for _, kwargs in calls)
    common_calls = [kwargs for kind, kwargs in calls if kind == "common"]
    assert [kwargs["dataset_id"] for kwargs in common_calls] == list(
        OPEN_SET_DATASET_IDS
    )
    assert all(
        kwargs["quick_data_fractions"] == EXPECTED_FRACTIONS
        for kwargs in common_calls
    )
    tinyface_call = [kwargs for kind, kwargs in calls if kind == "tinyface"][0]
    assert tinyface_call["quick_data_fraction"] == 0.10
    assert tinyface_call["include_sdc"] is False


def test_tinyface_execution_uses_integrated_override_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyTinyFacePlan:
        dataset_id = "tinyface"

    plan = DummyTinyFacePlan()
    completed = tmp_path / "runs" / "tinyface" / "complete"
    monkeypatch.setattr(
        integrated_pipeline,
        "TinyFaceExperimentPlan",
        DummyTinyFacePlan,
    )
    monkeypatch.setattr(
        integrated_pipeline,
        "reuse_completed_tinyface_run",
        lambda supplied_plan, run_dir: {
            "dataset_id": supplied_plan.dataset_id,
            "run_dir": str(run_dir),
        },
    )

    result = integrated_pipeline.run_or_reuse_integrated_experiment(
        plan,
        execution_acknowledged=False,
        start_new_run=False,
        completed_run_override=completed,
    )

    assert result == {"dataset_id": "tinyface", "run_dir": str(completed)}
