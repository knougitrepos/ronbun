from __future__ import annotations

import pandas as pd
import pytest

from research.evaluation.report_selection import (
    report_campaign_id,
    select_model_uid_report_cohort,
)


DATASETS = ("lfw", "survface", "rfw_custom")


def _candidate(
    dataset: str,
    run_id: str,
    *,
    model_uid: str = "adaface-model",
    report_ready: bool = True,
    source_commit: str = "commit-a",
    pq_sdc_settings: tuple[tuple[int, int], ...] = (),
) -> dict[str, object]:
    return {
        "campaign_id": report_campaign_id(run_id),
        "dataset": dataset,
        "model_uid": model_uid,
        "report_ready": report_ready,
        "run_id": run_id,
        "source_commit": source_commit,
        "mode": "real",
        "data_fraction": 1.0,
        "is_paper_run": True,
        "pq_sdc_settings": pq_sdc_settings,
    }


def test_selects_newest_complete_model_uid_campaign() -> None:
    rows = [
        _candidate(dataset, f"20260829-R001-{index:08x}")
        for index, dataset in enumerate(DATASETS, start=1)
    ]
    rows.extend(
        _candidate(dataset, f"20260830-R001-{index:08x}")
        for index, dataset in enumerate(DATASETS, start=11)
    )
    rows.append(_candidate("lfw", "20260831-R001-00000099"))

    selected = select_model_uid_report_cohort(
        pd.DataFrame(rows), model_uid="adaface-model"
    )

    assert selected["dataset"].tolist() == list(DATASETS)
    assert set(selected["campaign_id"]) == {"20260830-R001"}
    assert selected.set_index("dataset")["run_id"].to_dict() == {
        dataset: f"20260830-R001-{index:08x}"
        for index, dataset in enumerate(DATASETS, start=11)
    }


def test_does_not_mix_incomplete_independently_latest_runs() -> None:
    rows = [
        _candidate(dataset, f"20260830-R001-{index:08x}")
        for index, dataset in enumerate(DATASETS, start=1)
    ]
    rows.extend(
        [
            _candidate("lfw", "20260831-R001-00000011"),
            _candidate("survface", "20260831-R001-00000012"),
        ]
    )

    selected = select_model_uid_report_cohort(
        pd.DataFrame(rows), model_uid="adaface-model"
    )

    assert set(selected["campaign_id"]) == {"20260830-R001"}


def test_rejects_mixed_contract_inside_campaign() -> None:
    rows = [
        _candidate(dataset, f"20260830-R001-{index:08x}")
        for index, dataset in enumerate(DATASETS, start=1)
    ]
    rows[-1]["source_commit"] = "commit-b"

    with pytest.raises(ValueError, match="mixes source_commit"):
        select_model_uid_report_cohort(
            pd.DataFrame(rows), model_uid="adaface-model"
        )


def test_rejects_missing_complete_campaign() -> None:
    rows = [
        _candidate("lfw", "20260830-R001-00000001"),
        _candidate("survface", "20260830-R001-00000002"),
    ]

    with pytest.raises(ValueError, match="no complete report campaign"):
        select_model_uid_report_cohort(
            pd.DataFrame(rows), model_uid="adaface-model"
        )


@pytest.mark.parametrize("run_id", ["", "20260830", "bad-R001-hash-extra"])
def test_report_campaign_id_rejects_noncanonical_run_ids(run_id: str) -> None:
    with pytest.raises(ValueError, match="unsupported report run_id format"):
        report_campaign_id(run_id)
