from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd


FAITHFULNESS_ARTIFACT_TYPE = "open_set_gradcam_faithfulness"
FAITHFULNESS_SCHEMA_VERSION = 2
FAITHFULNESS_MAXIMUM_SAMPLES = 10_000


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


def load_selected_faithfulness_artifacts(
    project_root: str | Path,
    *,
    datasets: tuple[str, ...],
    model_uids: Mapping[str, str],
    run_ids: Mapping[str, str],
    maximum_samples: int = FAITHFULNESS_MAXIMUM_SAMPLES,
) -> SelectedFaithfulnessArtifacts:
    """Load one SHA-verified, dataset-level faithfulness summary per run."""

    root = Path(project_root).resolve()
    summaries: list[pd.DataFrame] = []
    manifests: dict[str, dict[str, object]] = {}
    roots: dict[str, Path] = {}
    for dataset in datasets:
        artifact_root = (
            root
            / "results"
            / "paper"
            / dataset
            / str(run_ids[dataset])
            / f"faithfulness_v2_n{int(maximum_samples)}"
        )
        manifest_path = artifact_root / "manifest.json"
        if not manifest_path.is_file():
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
        if sampling.get("maximum_samples") != int(maximum_samples):
            mismatches["maximum_samples"] = (
                sampling.get("maximum_samples"),
                int(maximum_samples),
            )
        candidate_count = int(sampling.get("candidate_count", -1))
        selected_count = int(sampling.get("selected_count", -1))
        if selected_count != min(candidate_count, int(maximum_samples)):
            mismatches["selected_count"] = (
                selected_count,
                min(candidate_count, int(maximum_samples)),
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
    )
