from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


OPEN_SET_REPORT_DATASETS = ("lfw", "survface", "rfw_custom")
_REQUIRED_CANDIDATE_COLUMNS = {
    "campaign_id",
    "dataset",
    "model_uid",
    "report_ready",
    "run_id",
    "source_commit",
    "mode",
    "data_fraction",
    "is_paper_run",
    "pq_sdc_settings",
}
_COHORT_CONTRACT_COLUMNS = (
    "model_uid",
    "source_commit",
    "mode",
    "data_fraction",
    "is_paper_run",
    "pq_sdc_settings",
)


def report_campaign_id(run_id: object) -> str:
    """Return the shared YYYYMMDD-RNNN portion of a workflow run id."""

    value = str(run_id)
    parts = value.split("-")
    if len(parts) != 3 or not parts[0].isdigit() or not parts[1].startswith("R"):
        raise ValueError(f"unsupported report run_id format: {value!r}")
    return "-".join(parts[:2])


def select_model_uid_report_cohort(
    candidates: pd.DataFrame,
    *,
    model_uid: str,
    datasets: Sequence[str] = OPEN_SET_REPORT_DATASETS,
) -> pd.DataFrame:
    """Select the newest complete, internally homogeneous run campaign.

    The function never combines independently-latest dataset runs. A campaign is
    eligible only when it contains exactly one report-ready run for every requested
    dataset and all science/provenance contract columns agree.
    """

    requested = tuple(str(dataset) for dataset in datasets)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("report datasets must be a non-empty unique sequence")
    if not isinstance(model_uid, str) or not model_uid.strip():
        raise ValueError("model_uid must be a non-empty string")
    missing_columns = sorted(_REQUIRED_CANDIDATE_COLUMNS - set(candidates.columns))
    if missing_columns:
        raise ValueError(f"report candidates are missing columns: {missing_columns}")

    eligible = candidates.loc[
        candidates["report_ready"].eq(True)
        & candidates["model_uid"].astype(str).eq(model_uid)
        & candidates["dataset"].astype(str).isin(requested)
    ].copy()
    if eligible.empty:
        raise ValueError(f"model_uid {model_uid!r} has no report-ready runs")

    complete: list[pd.DataFrame] = []
    campaign_coverage: dict[str, list[str]] = {}
    for campaign_id, group in eligible.groupby("campaign_id", sort=True):
        campaign_id = str(campaign_id)
        coverage = sorted(set(group["dataset"].astype(str)))
        campaign_coverage[campaign_id] = coverage
        if set(coverage) != set(requested):
            continue
        counts = group.groupby("dataset", sort=False).size()
        duplicates = counts[counts.ne(1)]
        if not duplicates.empty:
            raise ValueError(
                f"campaign {campaign_id!r} has duplicate dataset runs: "
                f"{duplicates.to_dict()}"
            )
        for column in _COHORT_CONTRACT_COLUMNS:
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"campaign {campaign_id!r} mixes {column}: "
                    f"{group[column].astype(str).unique().tolist()}"
                )
        complete.append(group.copy())

    if not complete:
        raise ValueError(
            f"model_uid {model_uid!r} has no complete report campaign for "
            f"datasets={requested}; coverage={campaign_coverage}"
        )

    selected = max(complete, key=lambda frame: str(frame["campaign_id"].iloc[0]))
    order = {dataset: index for index, dataset in enumerate(requested)}
    selected["_dataset_order"] = selected["dataset"].map(order)
    return selected.sort_values("_dataset_order").drop(
        columns="_dataset_order"
    ).reset_index(drop=True)
