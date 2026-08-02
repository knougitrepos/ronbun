from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research.experiments import step2_compression as module
from research.explainability.gradcam.extraction import PreparedPopulationInputs
from research.explainability.gradcam.templates import (
    build_leave_one_out_identity_templates,
)


class _FakePQ:
    def __init__(
        self,
        source_dim: int,
        m: int,
        nbits: int,
        *,
        source_profile: str,
    ):
        self.m = m
        self.nbits = nbits

    def fit(self, development_vectors: np.ndarray):
        return self

    def transform_profile(self, vectors: np.ndarray):
        restored = np.asarray(vectors, dtype=np.float32).copy()
        return SimpleNamespace(
            vectors=restored,
            reconstructed_vectors=restored,
            metadata={
                "code_bytes": self.m * self.nbits,
                "codebook_bytes": 128,
                "codebook_bytes_source": "test",
            },
        )

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        return np.asarray(vectors, dtype=np.float32).copy()

    def search_adc(
        self,
        queries: np.ndarray,
        gallery_codes: np.ndarray,
        *,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(queries, dtype=np.float32)
        gallery = np.asarray(gallery_codes, dtype=np.float32)
        distances = np.sum(
            (query[:, None, :] - gallery[None, :, :]) ** 2,
            axis=2,
        )
        indices = np.argsort(distances, axis=1, kind="stable")[:, :top_k]
        selected = np.take_along_axis(distances, indices, axis=1)
        return selected.astype(np.float32), indices.astype(np.int64)

    def search_adc_with_metrics(
        self,
        queries: np.ndarray,
        gallery_codes: np.ndarray,
        *,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        distances, indices = self.search_adc(
            queries,
            gallery_codes,
            top_k=top_k,
        )
        return distances, indices, {
            "latency_measurement_repeats": 1,
            "latency_timer": "test_clock",
            "compressed_index_build_latency_ms": 1.0,
            "compressed_gallery_add_latency_ms": 2.0,
            "compressed_search_latency_ms_total": 3.0,
            "compressed_search_latency_ms_per_query": 1.5,
            "compressed_search_queries_per_second": 666.0,
        }


class _FakePCA:
    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return np.asarray(vectors, dtype=np.float32)[:, :2]

    def transform_profile(self, vectors: np.ndarray):
        source = np.asarray(vectors, dtype=np.float32)
        return SimpleNamespace(
            vectors=source[:, :2],
            reconstructed_vectors=source.copy(),
            metadata={"storage_bytes_per_vector": 8},
        )


def _prepared() -> tuple[PreparedPopulationInputs, pd.DataFrame]:
    rows = [
        ("d1", "dev-1", "development", 0.0),
        ("d2", "dev-2", "development", 0.5),
        ("d3", "dev-3", "development", 1.0),
        ("d4", "dev-4", "development", 1.5),
        ("ca1", "cal-a", "calibration", 0.2),
        ("ca2", "cal-a", "calibration", 0.2),
        ("cb1", "cal-b", "calibration", 2.2),
        ("tg1", "test-gallery", "test", 0.4),
        ("tg2", "test-gallery", "test", 0.4),
        ("tk1", "test-known-unknown", "test", 1.8),
        ("tu1", "test-unknown-unknown", "test", 2.6),
    ]
    vectors = np.zeros((len(rows), 512), dtype=np.float32)
    for index, (_, _, _, angle) in enumerate(rows):
        vectors[index, 0] = np.cos(angle)
        vectors[index, 1] = np.sin(angle)
    sample_ids = np.asarray([row[0] for row in rows])
    identity_ids = np.asarray([row[1] for row in rows])
    splits = np.asarray([row[2] for row in rows])
    loo = build_leave_one_out_identity_templates(
        sample_ids,
        identity_ids,
        vectors,
        model_uid="arcface-test",
        scope_ids=splits,
    )
    raw = vectors * 2.0
    prepared = PreparedPopulationInputs(
        extraction_uid="extract-test",
        dataset_id="lfw",
        sample_ids=sample_ids,
        identity_ids=identity_ids,
        scope_ids=splits,
        raw_embeddings=raw,
        raw_norms=np.linalg.norm(raw, axis=1),
        normalized_embeddings=vectors,
        loo_templates=loo,
        model_uid="arcface-test",
        checkpoint_sha256="a" * 64,
        preprocess_hash="b" * 64,
        origin_embedding_artifact_uid="origin-test",
    )
    selected = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "identity_id": identity_ids,
            "split": splits,
            "image_path": [f"{sample}.jpg" for sample in sample_ids],
        }
    )
    return prepared, selected


def test_step2_runner_fits_calibrates_and_evaluates_one_lineage(
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "PQCompressor", _FakePQ)
    monkeypatch.setattr(
        module,
        "fit_pca_family",
        lambda development_vectors, dimensions, random_state: {
            "pca_2": _FakePCA()
        },
    )
    prepared, selected = _prepared()

    result = module.characterize_step2_compression(
        prepared,
        selected,
        gallery_identities=["test-gallery"],
        unknown_unknown_identities=["test-unknown-unknown"],
        pca_dimensions=[2],
        pq_settings=[(8, 1)],
        seed=42,
        target_fpir=1.0,
        enrollment_count=1,
        calibration_gallery_identities=1,
        top_k=1,
    )

    assert set(result.paired_metrics["compression_family"]) == {"pca", "pq"}
    assert set(result.retrieval_metrics["threshold_policy"]) == {
        "frozen_origin",
        "recalibrated_compressed",
    }
    assert not result.paired_metrics["origin_fallback_used"].any()
    assert not result.retrieval_metrics["origin_fallback_used"].any()
    assert set(result.retrieval_metrics["evaluation_split"]) == {"test"}
    assert set(result.retrieval_metrics["threshold_source_split"]) == {
        "calibration"
    }
    assert set(result.origin_score_audit["evaluation_split"]) == {
        "calibration",
        "test",
    }
    assert result.origin_score_audit["query_id"].notna().all()
    assert not result.origin_score_audit.duplicated(
        ["evaluation_split", "query_id"]
    ).any()
    assert result.origin_score_audit["threshold_comparator"].eq(">=").all()
    calibration_rows = result.origin_score_audit.loc[
        result.origin_score_audit["evaluation_split"].eq("calibration")
    ]
    expected_false_accepts = int(
        (
            calibration_rows["origin_accepted"].astype(bool)
            & ~calibration_rows["is_mated"].astype(bool)
        ).sum()
    )
    calibration_summary = result.calibration_diagnostics["splits"][
        "calibration"
    ]
    assert calibration_summary["origin_false_accept_count"] == expected_false_accepts
    assert calibration_summary["origin_fpir"] <= 1.0
    assert (
        calibration_summary["origin_fpir_wilson95_low"]
        <= calibration_summary["origin_fpir"]
        <= calibration_summary["origin_fpir_wilson95_high"]
    )
    assert result.calibration_diagnostics["threshold_comparator"] == ">="
    assert result.calibration_diagnostics[
        "independent_threshold_verification"
    ]["matches"] is True
    assert result.calibration_diagnostics[
        "calibration_transfer_assessment"
    ]["test_used_for_threshold_selection"] is False
    assert set(result.retrieval_metrics["search_mode"]) == {
        "pca_direct_cosine",
        "pca_reconstruction_cosine",
        "pq_reconstruction_cosine",
        "pq_adc_exhaustive",
    }
    adc = result.retrieval_metrics.loc[
        result.retrieval_metrics["search_mode"].eq("pq_adc_exhaustive")
    ]
    assert set(adc["threshold_policy"]) == {"recalibrated_compressed"}
    assert not adc["frozen_origin_threshold_applicable"].any()
    assert adc["compressed_score_space"].eq(
        "negative_squared_l2_adc"
    ).all()
    assert adc["top1_score_drift"].isna().all()
    assert adc["compressed_search_latency_ms_total"].eq(3.0).all()
    assert adc["compressed_gallery_encode_latency_ms"].notna().all()
    assert len(result.summary) == 7
    assert result.summary["gallery_template_count"].eq(1).all()
    assert result.summary["origin_gallery_storage_bytes"].eq(2048).all()
    pq_summary = result.summary.loc[
        result.summary["compression_family"].eq("pq")
    ]
    assert pq_summary["compressed_gallery_storage_bytes"].eq(136).all()
    assert pq_summary[
        "amortized_storage_bytes_per_gallery_template"
    ].eq(136.0).all()


def test_step2_runner_rejects_manifest_order_mismatch() -> None:
    prepared, selected = _prepared()
    selected = selected.iloc[::-1].reset_index(drop=True)

    try:
        module.characterize_step2_compression(
            prepared,
            selected,
            gallery_identities=["test-gallery"],
            unknown_unknown_identities=["test-unknown-unknown"],
            pca_dimensions=[2],
            pq_settings=[(8, 1)],
        )
    except ValueError as exc:
        assert "row order" in str(exc)
    else:
        raise AssertionError("manifest order mismatch was not rejected")


@pytest.mark.parametrize(
    ("pca_dimensions", "pq_settings", "expected_family", "expected_modes"),
    [
        (
            [2],
            [],
            "pca",
            {"pca_direct_cosine", "pca_reconstruction_cosine"},
        ),
        (
            [],
            [(8, 1)],
            "pq",
            {"pq_reconstruction_cosine", "pq_adc_exhaustive"},
        ),
    ],
)
def test_step2_runner_supports_one_compression_family_for_bounded_refresh(
    monkeypatch,
    pca_dimensions,
    pq_settings,
    expected_family,
    expected_modes,
) -> None:
    monkeypatch.setattr(module, "PQCompressor", _FakePQ)
    monkeypatch.setattr(
        module,
        "fit_pca_family",
        lambda development_vectors, dimensions, random_state: {
            "pca_2": _FakePCA()
        },
    )
    prepared, selected = _prepared()

    result = module.characterize_step2_compression(
        prepared,
        selected,
        gallery_identities=["test-gallery"],
        unknown_unknown_identities=["test-unknown-unknown"],
        pca_dimensions=pca_dimensions,
        pq_settings=pq_settings,
        target_fpir=1.0,
        enrollment_count=1,
        calibration_gallery_identities=1,
        top_k=1,
    )

    assert set(result.paired_metrics["compression_family"]) == {
        expected_family
    }
    assert set(result.retrieval_metrics["search_mode"]) == expected_modes
    assert len(result.summary) == (4 if expected_family == "pca" else 3)


def test_step2_runner_rejects_empty_profile_selection() -> None:
    prepared, selected = _prepared()

    with pytest.raises(ValueError, match="at least one PCA or PQ"):
        module.characterize_step2_compression(
            prepared,
            selected,
            gallery_identities=["test-gallery"],
            unknown_unknown_identities=["test-unknown-unknown"],
            pca_dimensions=[],
            pq_settings=[],
            enrollment_count=1,
        )


def test_independent_threshold_audit_preserves_greater_equal_ties() -> None:
    comparison = pd.DataFrame(
        {
            "origin_top1_score": [0.90, 0.80, 0.80, 0.70],
            "is_mated": [True, False, False, True],
            "origin_rank1_correct": [True, False, False, True],
        }
    )

    selected = module._threshold(
        comparison,
        score_column="origin_top1_score",
        correct_column="origin_rank1_correct",
        target_fpir=0.50,
    )
    audited = module._independent_threshold(
        comparison,
        target_fpir=0.50,
    )

    assert selected == pytest.approx(0.90)
    assert audited == pytest.approx(selected)


def test_survface_runner_preserves_official_protocol_and_training_boundary(
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "PQCompressor", _FakePQ)
    monkeypatch.setattr(
        module,
        "fit_pca_family",
        lambda development_vectors, dimensions, random_state: {
            "pca_2": _FakePCA()
        },
    )
    rows = [
        ("d1", "dev-a", "development", 0.0, "training", None),
        ("d2", "dev-a", "development", 0.5, "training", None),
        ("d3", "dev-b", "development", 1.0, "training", None),
        ("d4", "dev-b", "development", 1.5, "training", None),
        ("ca1", "cal-a", "calibration", 0.2, "training", None),
        ("ca2", "cal-a", "calibration", 0.2, "training", None),
        ("cb1", "cal-b", "calibration", 2.2, "training", None),
        ("cb2", "cal-b", "calibration", 2.2, "training", None),
        ("g1", "official-a", "test", 0.4, "gallery", 5),
        ("g2", "official-a", "test", 0.4, "gallery", 9),
        ("p1", "official-a", "test", 0.4, "registered_probe", 7),
        (
            "u1",
            "official-unknown",
            "test",
            2.6,
            "unknown_unknown_probe",
            11,
        ),
    ]
    vectors = np.zeros((len(rows), 512), dtype=np.float32)
    for index, row in enumerate(rows):
        vectors[index, 0] = np.cos(row[3])
        vectors[index, 1] = np.sin(row[3])
    sample_ids = np.asarray([row[0] for row in rows])
    identity_ids = np.asarray([row[1] for row in rows])
    splits = np.asarray([row[2] for row in rows])
    loo = build_leave_one_out_identity_templates(
        sample_ids,
        identity_ids,
        vectors,
        model_uid="arcface-test",
        scope_ids=splits,
    )
    prepared = PreparedPopulationInputs(
        extraction_uid="extract-survface-test",
        dataset_id="survface",
        sample_ids=sample_ids,
        identity_ids=identity_ids,
        scope_ids=splits,
        raw_embeddings=vectors * 2.0,
        raw_norms=np.linalg.norm(vectors * 2.0, axis=1),
        normalized_embeddings=vectors,
        loo_templates=loo,
        model_uid="arcface-test",
        checkpoint_sha256="a" * 64,
        preprocess_hash="b" * 64,
        origin_embedding_artifact_uid="origin-survface-test",
    )
    selected = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "identity_id": identity_ids,
            "split": splits,
            "image_path": [f"{value}.jpg" for value in sample_ids],
            "protocol_role": [row[4] for row in rows],
            "probe_type": [
                (
                    "registered"
                    if row[4] == "registered_probe"
                    else "unknown_unknown"
                    if row[4] == "unknown_unknown_probe"
                    else "not_applicable"
                    if row[4] == "gallery"
                    else None
                )
                for row in rows
            ],
            "protocol_index": [row[5] for row in rows],
        }
    )

    result = module.characterize_step2_survface_compression(
        prepared,
        selected,
        pca_dimensions=[2],
        pq_settings=[(8, 1)],
        target_fpir=1.0,
        calibration_gallery_identities=1,
        top_k=1,
    )

    assert set(result.retrieval_metrics["protocol_uid"]) == {
        "qmul-survface-v1-training-derived-1-watchlist-calibration-v2"
    }
    assert set(result.retrieval_metrics["query_id"]) == {"p1", "u1"}
    assert set(result.retrieval_metrics["threshold_policy"]) == {
        "frozen_origin",
        "recalibrated_compressed",
    }
    diagnostics = result.calibration_diagnostics["splits"]
    assert diagnostics["calibration"]["template_count"] == 1
    assert diagnostics["calibration"]["source_image_count"] == 1
    assert result.calibration_diagnostics["threshold_selection"].startswith(
        "non-mated maximum-score"
    )
    calibration_contract = result.calibration_diagnostics[
        "calibration_contract"
    ]
    assert calibration_contract == {
        "name": "training_1_half_gallery_v2",
        "seed": 42,
        "source_identity_count": 4,
        "gallery_identity_count": 1,
        "gallery_source_image_count": 1,
        "gallery_enrollment_policy": "half_floor",
        "registered_probe_count": 1,
        "non_mated_identity_count": 3,
        "non_mated_probe_count": 6,
        "compressor_fit_source": "watchlist_enrollment_images_only",
        "compressor_fit_image_count": 1,
        "official_test_used_for_threshold": False,
    }
    assert diagnostics["test"]["template_count"] == 1
    assert diagnostics["test"]["source_image_count"] == 2
    audit = result.origin_score_audit
    assert set(audit.loc[audit["evaluation_split"].eq("calibration"), "probe_type"]) == {
        "registered",
        "known_unknown",
    }
    assert set(audit.loc[audit["evaluation_split"].eq("test"), "probe_type"]) == {
        "registered",
        "unknown_unknown",
    }

    sweep = module.diagnose_step2_survface_origin_calibration(
        prepared,
        selected,
        calibration_conditions=[(1, 1)],
        target_fpir=1.0,
        top_k=1,
    )
    assert sweep.condition_summary["calibration_condition_uid"].tolist() == [
        "gallery-1_enrollment-1"
    ]
    assert sweep.condition_summary["test_gallery_template_count"].tolist() == [1]
    assert sweep.condition_summary[
        "test_gallery_source_image_count"
    ].tolist() == [2]
    assert set(sweep.score_audit["evaluation_split"]) == {
        "calibration",
        "test",
        "test_matched",
    }
    assert sweep.condition_summary["matched_test_fpir"].notna().all()
    assert sweep.diagnostics["test_threshold_recalibration"] is False
