from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Literal, Mapping

import pandas as pd


FAITHFULNESS_ARTIFACT_TYPE = "open_set_gradcam_faithfulness"
FAITHFULNESS_SCHEMA_VERSION = 2
FAITHFULNESS_MAXIMUM_SAMPLES = 10000
_FAITHFULNESS_DIRECTORY_PATTERN = re.compile(
    r"^faithfulness_v2_(?:all|n(?P<maximum_samples>[1-9][0-9]*))$"
)


def normalize_faithfulness_maximum_samples(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("maximum_samples must be None or a positive integer")
    return value


def faithfulness_artifact_directory_name(maximum_samples: object) -> str:
    resolved = normalize_faithfulness_maximum_samples(maximum_samples)
    return "faithfulness_v2_all" if resolved is None else f"faithfulness_v2_n{resolved}"


def resolve_faithfulness_selected_count(
    candidate_count: int,
    maximum_samples: object,
) -> int:
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 0
    ):
        raise ValueError("candidate_count must be a non-negative integer")
    resolved = normalize_faithfulness_maximum_samples(maximum_samples)
    return candidate_count if resolved is None else min(candidate_count, resolved)


def resolve_common_faithfulness_maximum_samples(
    project_root: str | Path,
    *,
    datasets: tuple[str, ...],
    run_ids: Mapping[str, str],
) -> int | None:
    """Select one faithfulness sampling contract available for every run.

    Unlimited ``faithfulness_v2_all`` is preferred when it exists for every
    selected dataset. Otherwise, the largest common finite cap is returned.
    Artifact identity and output hashes are still validated by the loader.
    """

    if not datasets:
        raise ValueError("datasets must not be empty")
    if set(run_ids) != set(datasets):
        raise ValueError("run_ids keys must exactly match datasets")

    root = Path(project_root).resolve()
    available_by_dataset: dict[str, set[int | None]] = {}
    for dataset in datasets:
        run_root = root / "results" / "paper" / dataset / str(run_ids[dataset])
        available: set[int | None] = set()
        if run_root.is_dir():
            for artifact_root in run_root.iterdir():
                if not artifact_root.is_dir():
                    continue
                match = _FAITHFULNESS_DIRECTORY_PATTERN.fullmatch(artifact_root.name)
                if match is None or not (artifact_root / "manifest.json").is_file():
                    continue
                encoded = match.group("maximum_samples")
                available.add(None if encoded is None else int(encoded))
        available_by_dataset[dataset] = available

    common = set.intersection(*available_by_dataset.values())
    if None in common:
        return None
    finite = [value for value in common if value is not None]
    if finite:
        return max(finite)

    rendered = {
        dataset: sorted(
            ("all" if value is None else f"n{value}" for value in values),
        )
        for dataset, values in available_by_dataset.items()
    }
    raise FileNotFoundError(
        "selected runs have no common faithfulness artifact contract: "
        f"{rendered}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SelectedFaithfulnessArtifacts:
    summary: pd.DataFrame
    manifests: dict[str, dict[str, object]]
    roots: dict[str, Path]
    missing_manifests: dict[str, Path]


def load_selected_faithfulness_artifacts(
    project_root: str | Path,
    *,
    datasets: tuple[str, ...],
    model_uids: Mapping[str, str],
    run_ids: Mapping[str, str],
    maximum_samples: int | None = FAITHFULNESS_MAXIMUM_SAMPLES,
    missing_policy: Literal["raise", "omit"] = "raise",
) -> SelectedFaithfulnessArtifacts:
    """Load one SHA-verified, dataset-level faithfulness summary per run.

    ``missing_policy="omit"`` skips only datasets whose manifest is absent.
    Once a manifest exists, all identity, sampling, file, and SHA checks remain
    fail-closed.
    """

    if missing_policy not in {"raise", "omit"}:
        raise ValueError(f"unsupported missing_policy: {missing_policy!r}")
    maximum_samples = normalize_faithfulness_maximum_samples(maximum_samples)

    root = Path(project_root).resolve()
    summaries: list[pd.DataFrame] = []
    manifests: dict[str, dict[str, object]] = {}
    roots: dict[str, Path] = {}
    missing_manifests: dict[str, Path] = {}
    for dataset in datasets:
        artifact_root = (
            root
            / "results"
            / "paper"
            / dataset
            / str(run_ids[dataset])
            / faithfulness_artifact_directory_name(maximum_samples)
        )
        manifest_path = artifact_root / "manifest.json"
        if not manifest_path.is_file():
            if missing_policy == "omit":
                missing_manifests[dataset] = manifest_path
                continue
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "artifact_type": FAITHFULNESS_ARTIFACT_TYPE,
            "schema_version": FAITHFULNESS_SCHEMA_VERSION,
            "dataset_id": dataset,
            "source_run_id": str(run_ids[dataset]),
            "model_uid": str(model_uids[dataset]),
            "threshold_independent": True,
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        sampling = dict(manifest.get("sampling", {}))
        if sampling.get("maximum_samples") != maximum_samples:
            mismatches["maximum_samples"] = (
                sampling.get("maximum_samples"),
                maximum_samples,
            )
        candidate_count = int(sampling.get("candidate_count", -1))
        selected_count = int(sampling.get("selected_count", -1))
        expected_selected_count = resolve_faithfulness_selected_count(
            candidate_count,
            maximum_samples,
        )
        if selected_count != expected_selected_count:
            mismatches["selected_count"] = (
                selected_count,
                expected_selected_count,
            )
        if mismatches:
            raise ValueError(f"{dataset}: faithfulness manifest mismatch: {mismatches}")

        output_entries = {
            str(entry["path"]): dict(entry)
            for entry in manifest.get("outputs", [])
        }
        for required in ("faithfulness_rows.csv", "faithfulness_summary.csv"):
            entry = output_entries.get(required)
            path = artifact_root / required
            if entry is None or not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(entry["bytes"]):
                raise ValueError(f"{dataset}: faithfulness output byte mismatch: {path}")
            if _sha256_file(path) != str(entry["sha256"]):
                raise ValueError(f"{dataset}: faithfulness output SHA-256 mismatch: {path}")

        summary = pd.read_csv(artifact_root / "faithfulness_summary.csv")
        if not summary["dataset"].astype(str).eq(dataset).all():
            raise ValueError(f"{dataset}: summary dataset differs")
        if not summary["source_run_id"].astype(str).eq(str(run_ids[dataset])).all():
            raise ValueError(f"{dataset}: summary run differs")
        if not summary["model_uid"].astype(str).eq(str(model_uids[dataset])).all():
            raise ValueError(f"{dataset}: summary model differs")
        summaries.append(summary)
        manifests[dataset] = manifest
        roots[dataset] = artifact_root

    combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    return SelectedFaithfulnessArtifacts(
        summary=combined,
        manifests=manifests,
        roots=roots,
        missing_manifests=missing_manifests,
    )
