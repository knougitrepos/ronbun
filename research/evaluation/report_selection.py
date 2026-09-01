from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


OPEN_SET_REPORT_DATASETS = ("lfw", "survface", "rfw_custom")
CROSS_MODEL_REPORT_FAMILIES = ("arcface", "adaface", "magface", "edgeface")
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


def select_model_uid_report_matrix(
    candidates: pd.DataFrame,
    *,
    model_uids: Mapping[str, str],
    datasets: Sequence[str] = OPEN_SET_REPORT_DATASETS,
) -> tuple[dict[str, dict[str, str]], pd.DataFrame]:
    """Resolve the complete 4-model report matrix from checkpoint UIDs.

    Each checkpoint is resolved independently through
    :func:`select_model_uid_report_cohort`, so dataset-wise latest runs are never
    mixed.  The returned paths are still passed to the strict explicit-run matrix
    loader for artifact and lineage validation.
    """

    if not isinstance(model_uids, Mapping):
        raise TypeError("model_uids must be a model-family to model-UID mapping")
    normalized = {
        str(model_family).strip().lower(): str(model_uid).strip()
        for model_family, model_uid in model_uids.items()
    }
    expected_families = set(CROSS_MODEL_REPORT_FAMILIES)
    if set(normalized) != expected_families:
        raise ValueError(
            "model_uids keys must exactly match the four report model families: "
            f"expected={sorted(expected_families)}, "
            f"actual={sorted(normalized)}"
        )
    if any(not model_uid for model_uid in normalized.values()):
        raise ValueError("model_uids values must be non-empty strings")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("model_uids values must be unique")
    mismatched_prefixes = {
        model_family: model_uid
        for model_family, model_uid in normalized.items()
        if model_uid.split("-", 1)[0].lower() != model_family
    }
    if mismatched_prefixes:
        raise ValueError(
            "model UID prefixes must match their model-family keys: "
            f"{mismatched_prefixes}"
        )
    if "run_dir" not in candidates.columns:
        raise ValueError("report candidates are missing columns: ['run_dir']")

    matrix: dict[str, dict[str, str]] = {}
    selected_frames: list[pd.DataFrame] = []
    for model_family in CROSS_MODEL_REPORT_FAMILIES:
        cohort = select_model_uid_report_cohort(
            candidates,
            model_uid=normalized[model_family],
            datasets=datasets,
        ).copy()
        run_dirs = cohort["run_dir"].astype(str).str.strip()
        if run_dirs.eq("").any():
            raise ValueError(f"{model_family} selected cohort contains an empty run_dir")
        matrix[model_family] = dict(zip(cohort["dataset"], run_dirs))
        cohort.insert(0, "model_family", model_family)
        selected_frames.append(cohort)

    return matrix, pd.concat(selected_frames, ignore_index=True)
