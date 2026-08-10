from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", "--", path],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"git check-ignore failed for {path!r}")
    return result.returncode == 0


@pytest.mark.parametrize("dataset", ["lfw", "survface"])
def test_large_run_payloads_remain_local(dataset: str) -> None:
    root = f"runs/{dataset}_20990101/run-id"
    ignored = (
        "artifacts/step2_workflow/prepared_population/embedding_shards/part-00000.npz",
        "artifacts/step2_workflow/saliency_population/heatmap_shards/part-00000.npz",
        "artifacts/step2_workflow/retrieval_metrics.csv",
        "artifacts/step2_workflow/paired_embedding_metrics.csv",
        "artifacts/step2_workflow/saliency_population/saliency_features.csv",
        "artifacts/step2_workflow/figures/gradcam-case.png",
        "artifacts/04_step2_compression_characterization/pca_384_A001.joblib",
        "logs/events.jsonl",
    )

    assert all(_is_ignored(f"{root}/{relative}") for relative in ignored)


@pytest.mark.parametrize("dataset", ["lfw", "survface"])
def test_compact_analysis_and_lineage_files_remain_visible(dataset: str) -> None:
    root = f"runs/{dataset}_20990101/run-id"
    visible = (
        "COMPLETED",
        "run_manifest.json",
        "phases/04_step2_compression_characterization/attempts/A001/phase_manifest.json",
        "artifacts/step2_workflow/freeze_manifest.json",
        "artifacts/step2_workflow/frozen_codec_manifest.json",
        "artifacts/step2_workflow/origin_calibration_diagnostics.json",
        "artifacts/step2_workflow/step4_summary.json",
        "artifacts/step2_workflow/saliency_validation.json",
        "artifacts/step2_workflow/saliency_geometry_associations.csv",
        "artifacts/step2_workflow/saliency_retrieval_associations.csv",
        "artifacts/step2_workflow/representative_cases.csv",
        "artifacts/step2_workflow/prepared_population/manifest.json",
        "artifacts/step2_workflow/saliency_population/manifest.json",
    )

    assert not any(_is_ignored(f"{root}/{relative}") for relative in visible)


def test_compact_paper_results_remain_visible() -> None:
    assert not _is_ignored(
        "results/paper/lfw/run-id/search_space_v4_multi_fpir/retrieval_summary.csv"
    )
