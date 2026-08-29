from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import refresh_step4_search_spaces as module


def test_rfw_custom_refresh_rejects_legacy_80_gallery_source_run(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "legacy-rfw-run"
    run_dir.mkdir()
    (run_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "RFW-LEGACY",
                "config": {
                    "dataset_id": "rfw_custom",
                    "step4": {
                        "evaluation": {
                            "rfw_custom_calibration_gallery_identities": 80,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="predates gallery-size-matched"):
        module._source_context(run_dir)


def _compact_retrieval(*, family: str) -> pd.DataFrame:
    modes = module.PCA_SEARCH_MODES if family == "pca" else module.PQ_SEARCH_MODES
    modes_and_policies = tuple(
        (mode, policy)
        for mode in modes
        for policy in module.threshold_policies_for_search_mode(mode)
    )
    rows = []
    for target in module.DEFAULT_TARGET_FPIRS:
        for search_mode, threshold_policy in modes_and_policies:
            condition = module.search_condition(search_mode)
            rows.append(
                {
                    "compression_family": family,
                    "compression_profile": (
                        module.REQUIRED_PQ_SDC_PROFILE
                        if family == "pq"
                        else f"{family}_test"
                    ),
                    "search_mode": search_mode,
                    "query_representation": condition.query_representation,
                    "gallery_representation": condition.gallery_representation,
                    "distance_function": condition.distance_function,
                    "compressed_score_space": condition.compressed_score_space,
                    "frozen_origin_threshold_applicable": (
                        condition.frozen_origin_threshold_applicable
                    ),
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
                    "top_k": 20,
                    "tpir_rank": 20,
                    "origin_tpir_at_rank_k_count": 6,
                    "origin_tpir_at_rank_k_denominator": 10,
                    "origin_tpir_at_rank_k": 0.6,
                    "origin_tpir_at_rank_k_wilson95_low": 0.3,
                    "origin_tpir_at_rank_k_wilson95_high": 0.85,
                    "compressed_tpir_at_rank_k_count": 5,
                    "compressed_tpir_at_rank_k_denominator": 10,
                    "compressed_tpir_at_rank_k": 0.5,
                    "compressed_tpir_at_rank_k_wilson95_low": 0.2,
                    "compressed_tpir_at_rank_k_wilson95_high": 0.8,
                    "compressed_minus_origin_tpir_at_rank_k": -0.1,
                    "compressed_minus_origin_tpir_at_rank_k_paired_bootstrap95_low": -0.4,
                    "compressed_minus_origin_tpir_at_rank_k_paired_bootstrap95_high": 0.2,
                    "compressed_tpir_at_rank_k_retention": 5 / 6,
                    "origin_tpir20_count": 6,
                    "origin_tpir20_denominator": 10,
                    "origin_tpir20": 0.6,
                    "compressed_tpir20_count": 5,
                    "compressed_tpir20_denominator": 10,
                    "compressed_tpir20": 0.5,
                    "compressed_tpir20_retention": 5 / 6,
                    "origin_closed_set_rank20_recall": 0.9,
                    "compressed_closed_set_rank20_recall": 0.8,
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
                    "score_spaces_comparable": condition.score_spaces_comparable,
                }
            )
    return pd.DataFrame.from_records(rows)


@pytest.mark.parametrize("family", ["pca", "pq"])
def test_v6_compact_validation_requires_tpir20_fpir_grid(family: str) -> None:
    compression = pd.DataFrame(
        {
            "compression_family": [family],
            "compression_profile": [
                module.REQUIRED_PQ_SDC_PROFILE
                if family == "pq"
                else f"{family}_test"
            ],
        }
    )
    retrieval = _compact_retrieval(family=family)

    module._validate_compact_frames(
        compression,
        retrieval,
        family=family,
        expected_profiles=1,
        target_fpirs=module.DEFAULT_TARGET_FPIRS,
    )

    with pytest.raises(ValueError, match="missing confidence fields"):
        module._validate_compact_frames(
            compression,
            retrieval.drop(columns="compressed_fpir_wilson95_high"),
            family=family,
            expected_profiles=1,
            target_fpirs=module.DEFAULT_TARGET_FPIRS,
        )

    with pytest.raises(ValueError, match="row mismatch|coverage mismatch"):
        module._validate_compact_frames(
            compression,
            retrieval.loc[retrieval["target_fpir"].eq(0.10)],
            family=family,
            expected_profiles=1,
            target_fpirs=module.DEFAULT_TARGET_FPIRS,
        )


def test_v6_compact_validation_rejects_sdc_outside_m128() -> None:
    compression = pd.DataFrame(
        {
            "compression_family": ["pq"],
            "compression_profile": [module.REQUIRED_PQ_SDC_PROFILE],
        }
    )
    retrieval = _compact_retrieval(family="pq")
    retrieval.loc[
        retrieval["search_mode"].eq("pq_sdc_exhaustive"),
        "compression_profile",
    ] = "pq_origin_512_m64_b8"

    with pytest.raises(ValueError, match="profile/search-mode coverage mismatch"):
        module._validate_compact_frames(
            compression,
            retrieval,
            family="pq",
            expected_profiles=1,
            target_fpirs=module.DEFAULT_TARGET_FPIRS,
        )


def test_v6_compact_validation_accepts_exactly_zero_sdc_when_disabled() -> None:
    compression = pd.DataFrame(
        {
            "compression_family": ["pq"],
            "compression_profile": [module.REQUIRED_PQ_SDC_PROFILE],
        }
    )
    retrieval = _compact_retrieval(family="pq")
    without_sdc = retrieval.loc[
        retrieval["search_mode"].ne("pq_sdc_exhaustive")
    ].copy()

    module._validate_compact_frames(
        compression,
        without_sdc,
        family="pq",
        expected_profiles=1,
        target_fpirs=module.DEFAULT_TARGET_FPIRS,
        expected_pq_sdc_profiles=(),
    )

    with pytest.raises(ValueError, match="row mismatch|coverage mismatch"):
        module._validate_compact_frames(
            compression,
            retrieval,
            family="pq",
            expected_profiles=1,
            target_fpirs=module.DEFAULT_TARGET_FPIRS,
            expected_pq_sdc_profiles=(),
        )


def test_v6_compact_validation_allows_csv_roundtrip_but_rejects_real_delta_drift():
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
        target_fpirs=module.DEFAULT_TARGET_FPIRS,
    )

    drifted = csv_roundtrip.copy()
    drifted.loc[0, "compressed_minus_origin_dir_rank1"] += 1e-6
    with pytest.raises(ValueError, match="dir_rank1 compressed-minus-origin"):
        module._validate_compact_frames(
            compression,
            drifted,
            family="pca",
            expected_profiles=1,
            target_fpirs=module.DEFAULT_TARGET_FPIRS,
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
        target_fpirs=module.DEFAULT_TARGET_FPIRS,
    )

    assert observed == [("pca", module.DEFAULT_TARGET_FPIRS)]
    assert result["output_dir"] == str((tmp_path / "derived").resolve())


def test_phase04_compact_cache_reuses_normalized_ledger_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_root = tmp_path / "artifacts" / "step2_workflow"
    ledger_path = workflow_root / "retrieval_ledger" / "manifest.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text("{}\n", encoding="utf-8")
    paired_path = workflow_root / "paired.csv"
    paired_path.write_text("placeholder\n", encoding="utf-8")
    diagnostics_path = workflow_root / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "diagnostics_by_target": {
                    f"{target:.12g}": {"target_fpir": target}
                    for target in module.DEFAULT_TARGET_FPIRS
                }
            }
        ),
        encoding="utf-8",
    )
    selected_path = workflow_root / "selected.csv"
    pd.DataFrame({"sample_id": ["s1", "s2"]}).to_csv(
        selected_path,
        index=False,
    )
    compression = pd.DataFrame(
        {
            "compression_family": ["pca", "pq"],
            "sample_count": [2, 2],
        }
    )
    retrieval = pd.DataFrame(
        {
            "compression_family": ["pca", "pq"],
            "query_count": [4, 6],
        }
    )
    calls = {"compression": 0, "retrieval": 0}

    def fake_compression(*args: object, **kwargs: object):
        calls["compression"] += 1
        return compression, 4

    def fake_retrieval(*args: object, **kwargs: object):
        calls["retrieval"] += 1
        return retrieval, 10

    monkeypatch.setattr(module, "summarize_compression", fake_compression)
    monkeypatch.setattr(module, "summarize_retrieval", fake_retrieval)
    context = {
        "step4": {
            "workflow": {
                "retrieval_ledger_manifest_path": (
                    "retrieval_ledger/manifest.json"
                ),
                "paired_metrics_path": "paired.csv",
                "calibration_diagnostics_path": "diagnostics.json",
            }
        },
        "workflow_root": workflow_root,
        "selected_path": selected_path,
        "prepared_manifest": {
            "origin_embedding_artifact_uid": "origin-a"
        },
    }

    first = module._phase04_compact_cache(
        context,
        target_fpirs=module.DEFAULT_TARGET_FPIRS,
    )
    second = module._phase04_compact_cache(
        context,
        target_fpirs=module.DEFAULT_TARGET_FPIRS,
    )

    assert first is second
    assert first is not None
    assert first["selected_count"] == 2
    assert first["origin_embedding_artifact_uid"] == "origin-a"
    assert calls == {"compression": 1, "retrieval": 1}
