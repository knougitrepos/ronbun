from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import refresh_step4_search_spaces as module


def _compact_retrieval(*, family: str) -> pd.DataFrame:
    modes_and_policies = (
        (
            (
                "pca_direct_cosine",
                "frozen_origin",
            ),
            (
                "pca_direct_cosine",
                "recalibrated_compressed",
            ),
            (
                "pca_reconstruction_cosine",
                "frozen_origin",
            ),
            (
                "pca_reconstruction_cosine",
                "recalibrated_compressed",
            ),
        )
        if family == "pca"
        else (
            ("pq_reconstruction_cosine", "frozen_origin"),
            ("pq_reconstruction_cosine", "recalibrated_compressed"),
            ("pq_adc_exhaustive", "recalibrated_compressed"),
        )
    )
    rows = []
    for target in (0.10, 0.01):
        for search_mode, threshold_policy in modes_and_policies:
            is_adc = search_mode == "pq_adc_exhaustive"
            rows.append(
                {
                    "compression_family": family,
                    "compression_profile": f"{family}_test",
                    "search_mode": search_mode,
                    "threshold_policy": threshold_policy,
                    "target_fpir": target,
                    "origin_dir_rank1_count": 8,
                    "origin_dir_rank1_denominator": 10,
                    "origin_dir_rank1": 0.8,
                    "origin_dir_rank1_wilson95_low": 0.5,
                    "origin_dir_rank1_wilson95_high": 0.95,
                    "compressed_dir_rank1_count": 7,
                    "compressed_dir_rank1_denominator": 10,
                    "compressed_dir_rank1": 0.7,
                    "compressed_dir_rank1_wilson95_low": 0.4,
                    "compressed_dir_rank1_wilson95_high": 0.9,
                    "compressed_minus_origin_dir_rank1": -0.1,
                    "compressed_minus_origin_dir_rank1_paired_bootstrap95_low": -0.5,
                    "compressed_minus_origin_dir_rank1_paired_bootstrap95_high": 0.2,
                    "origin_false_accept_count": 1,
                    "origin_fpir_denominator": 10,
                    "origin_fpir": 0.1,
                    "origin_realized_fpir": 0.1,
                    "origin_fpir_wilson95_low": 0.0,
                    "origin_fpir_wilson95_high": 0.4,
                    "compressed_false_accept_count": 2,
                    "compressed_fpir_denominator": 10,
                    "compressed_fpir": 0.2,
                    "compressed_realized_fpir": 0.2,
                    "compressed_fpir_wilson95_low": 0.0,
                    "compressed_fpir_wilson95_high": 0.5,
                    "compressed_minus_origin_fpir": 0.1,
                    "compressed_minus_origin_fpir_paired_bootstrap95_low": -0.2,
                    "compressed_minus_origin_fpir_paired_bootstrap95_high": 0.5,
                    "confidence_interval_unit": "probe",
                    "rate_confidence_interval_method": "wilson_score",
                    "difference_confidence_interval_method": (
                        "paired_nonparametric_bootstrap_percentile"
                    ),
                    "difference_confidence_interval_resamples": 2_000,
                    "difference_confidence_interval_random_seed": 42,
                    "threshold_crossing_count": 0,
                    "accept_to_reject_count": 0,
                    "reject_to_accept_count": 0,
                    "score_spaces_comparable": not is_adc,
                }
            )
    return pd.DataFrame.from_records(rows)


@pytest.mark.parametrize("family", ["pca", "pq"])
def test_v4_compact_validation_requires_both_fpir_targets(family: str) -> None:
    compression = pd.DataFrame(
        {
            "compression_family": [family],
            "compression_profile": [f"{family}_test"],
        }
    )
    retrieval = _compact_retrieval(family=family)

    module._validate_compact_frames(
        compression,
        retrieval,
        family=family,
        expected_profiles=1,
        target_fpirs=(0.10, 0.01),
    )

    with pytest.raises(ValueError, match="missing confidence fields"):
        module._validate_compact_frames(
            compression,
            retrieval.drop(columns="compressed_fpir_wilson95_high"),
            family=family,
            expected_profiles=1,
            target_fpirs=(0.10, 0.01),
        )

    with pytest.raises(ValueError, match="row mismatch|coverage mismatch"):
        module._validate_compact_frames(
            compression,
            retrieval.loc[retrieval["target_fpir"].eq(0.10)],
            family=family,
            expected_profiles=1,
            target_fpirs=(0.10, 0.01),
        )


def test_v4_compact_validation_allows_csv_roundtrip_but_rejects_real_delta_drift():
    compression = pd.DataFrame(
        {
            "compression_family": ["pca"],
            "compression_profile": ["pca_test"],
        }
    )
    retrieval = _compact_retrieval(family="pca")
    csv_roundtrip = retrieval.copy()
    csv_roundtrip.loc[0, "compressed_minus_origin_dir_rank1"] += 1.5e-12

    module._validate_compact_frames(
        compression,
        csv_roundtrip,
        family="pca",
        expected_profiles=1,
        target_fpirs=(0.10, 0.01),
    )

    drifted = csv_roundtrip.copy()
    drifted.loc[0, "compressed_minus_origin_dir_rank1"] += 1e-6
    with pytest.raises(ValueError, match="dir_rank1 compressed-minus-origin"):
        module._validate_compact_frames(
            compression,
            drifted,
            family="pca",
            expected_profiles=1,
            target_fpirs=(0.10, 0.01),
        )


def test_refresh_passes_explicit_targets_to_each_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, tuple[float, ...]]] = []
    monkeypatch.setattr(
        module,
        "_source_context",
        lambda run_dir: {
            "dataset_id": "lfw",
            "run_id": "R001",
        },
    )
    monkeypatch.setattr(
        module,
        "inspect_git_provenance",
        lambda *args, **kwargs: {"dirty": False},
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    def fake_run_family(context, *, family, output_root, target_fpirs):
        observed.append((family, target_fpirs))
        return {"family": family, "status": "completed"}

    monkeypatch.setattr(module, "_run_family", fake_run_family)

    result = module.refresh(
        tmp_path / "source-run",
        output_dir=tmp_path / "derived",
        families=("pca",),
        target_fpirs=(0.10, 0.01),
    )

    assert observed == [("pca", (0.10, 0.01))]
    assert result["output_dir"] == str((tmp_path / "derived").resolve())
