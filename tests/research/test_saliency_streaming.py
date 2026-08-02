from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.evaluation.saliency_compression import (
    DEFAULT_GEOMETRY_METRICS,
    DEFAULT_RETRIEVAL_METRICS,
    DEFAULT_SALIENCY_FEATURES,
    join_population_saliency_with_compression,
    join_population_saliency_with_retrieval,
)
from research.evaluation.saliency_streaming import (
    stream_join_population_saliency_with_compression,
    stream_join_population_saliency_with_retrieval,
)


def _saliency() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "extraction_uid": ["extract-1"] * 3,
            "dataset_id": ["survface"] * 3,
            "sample_id": ["sample-1", "sample-2", "sample-3"],
            "model_uid": ["model-a"] * 3,
            "identity_id": ["identity-1", "identity-1", "identity-2"],
            "origin_embedding_artifact_uid": ["origin-a"] * 3,
            "saliency_spec_uid": ["saliency-v1"] * 3,
            "saliency_target_eligible": [True] * 3,
            "heatmap_available": [True] * 3,
        }
    )
    for index, column in enumerate(DEFAULT_SALIENCY_FEATURES, start=1):
        frame[column] = [index / 100.0, index / 50.0, index / 25.0]
    return frame


def _geometry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for profile_index, profile in enumerate(("pca_128", "pca_64"), start=1):
        for sample_index, sample_id in enumerate(
            ("sample-1", "sample-2", "sample-3"),
            start=1,
        ):
            rows.append(
                {
                    "extraction_uid": "extract-1",
                    "dataset_id": "survface",
                    "sample_id": sample_id,
                    "model_uid": "model-a",
                    "compression_family": "pca",
                    "compression_profile": profile,
                    "origin_embedding_artifact_uid": "origin-a",
                    "origin_fallback_used": False,
                    "angular_error_rad": profile_index * sample_index / 100.0,
                    "reconstruction_mse": profile_index * sample_index / 1000.0,
                    "cosine_to_origin": 1.0
                    - profile_index * sample_index / 100.0,
                }
            )
    return pd.DataFrame.from_records(rows)


def _retrieval() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy in ("frozen_origin", "recalibrated_compressed"):
        for sample_index, (sample_id, is_mated) in enumerate(
            (
                ("sample-1", True),
                ("sample-2", True),
                ("sample-3", False),
            ),
            start=1,
        ):
            rows.append(
                {
                    "extraction_uid": "extract-1",
                    "dataset_id": "survface",
                    "query_id": sample_id,
                    "model_uid": "model-a",
                    "compression_family": "pca",
                    "compression_profile": "pca_128",
                    "origin_embedding_artifact_uid": "origin-a",
                    "origin_fallback_used": False,
                    "protocol_uid": "survface-official-v1",
                    "threshold_source_split": "calibration",
                    "evaluation_split": "official_test",
                    "threshold_policy": policy,
                    "is_mated": is_mated,
                    "top1_score_drift": sample_index / 100.0,
                    "agreement_with_origin": sample_index != 3,
                    "threshold_crossing": sample_index == 2,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_streaming_geometry_join_matches_in_memory_join(tmp_path: Path) -> None:
    source = tmp_path / "geometry.csv"
    joined_path = tmp_path / "geometry_join.csv"
    projection_path = tmp_path / "geometry_projection.parquet"
    geometry = _geometry()
    geometry.to_csv(source, index=False)

    result = stream_join_population_saliency_with_compression(
        _saliency(),
        source,
        joined_output_path=joined_path,
        association_projection_path=projection_path,
        chunksize=2,
        expected_rows=len(geometry),
    )

    expected = join_population_saliency_with_compression(
        _saliency(),
        geometry,
    ).sort_values(["sample_id", "compression_profile"], ignore_index=True)
    actual = pd.read_csv(joined_path).sort_values(
        ["sample_id", "compression_profile"],
        ignore_index=True,
    )
    assert result.row_count == len(expected)
    assert result.chunk_count == 3
    assert list(actual.columns) == list(expected.columns)
    pd.testing.assert_series_equal(
        actual["saliency_entropy"],
        expected["saliency_entropy"],
        check_names=False,
    )
    projection = pd.read_parquet(projection_path)
    assert len(projection) == len(expected)
    assert set(DEFAULT_GEOMETRY_METRICS).issubset(projection)


def test_streaming_retrieval_join_matches_in_memory_join(tmp_path: Path) -> None:
    source = tmp_path / "retrieval.csv"
    joined_path = tmp_path / "retrieval_join.csv"
    projection_path = tmp_path / "retrieval_projection.parquet"
    retrieval = _retrieval()
    retrieval.to_csv(source, index=False)

    result = stream_join_population_saliency_with_retrieval(
        _saliency(),
        source,
        joined_output_path=joined_path,
        association_projection_path=projection_path,
        chunksize=2,
        expected_rows=len(retrieval),
    )

    expected = join_population_saliency_with_retrieval(
        _saliency(),
        retrieval,
    ).sort_values(["sample_id", "threshold_policy"], ignore_index=True)
    actual = pd.read_csv(joined_path).sort_values(
        ["sample_id", "threshold_policy"],
        ignore_index=True,
    )
    assert result.row_count == len(expected)
    assert result.chunk_count == 3
    assert list(actual.columns) == list(expected.columns)
    pd.testing.assert_series_equal(
        actual["saliency_entropy"],
        expected["saliency_entropy"],
        check_names=False,
    )
    projection = pd.read_parquet(projection_path)
    assert len(projection) == len(expected)
    assert set(DEFAULT_RETRIEVAL_METRICS).issubset(projection)
    assert set(projection["is_mated"]) == {False, True}


def test_streaming_retrieval_join_keeps_search_modes_separate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "retrieval_search_modes.csv"
    joined_path = tmp_path / "retrieval_join.csv"
    projection_path = tmp_path / "retrieval_projection.parquet"
    retrieval = pd.concat(
        [
            _retrieval().assign(search_mode="pca_direct_cosine"),
            _retrieval().assign(search_mode="pca_reconstruction_cosine"),
        ],
        ignore_index=True,
    )
    retrieval.to_csv(source, index=False)

    result = stream_join_population_saliency_with_retrieval(
        _saliency(),
        source,
        joined_output_path=joined_path,
        association_projection_path=projection_path,
        chunksize=4,
        expected_rows=len(retrieval),
    )

    joined = pd.read_csv(joined_path)
    projection = pd.read_parquet(projection_path)
    expected_modes = {
        "pca_direct_cosine",
        "pca_reconstruction_cosine",
    }
    assert result.row_count == len(retrieval)
    assert set(joined["search_mode"]) == expected_modes
    assert set(projection["search_mode"]) == expected_modes


def test_streaming_join_detects_duplicate_keys_across_chunks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "geometry_duplicate.csv"
    joined_path = tmp_path / "geometry_join.csv"
    projection_path = tmp_path / "geometry_projection.parquet"
    geometry = _geometry()
    duplicated = pd.concat(
        [geometry.iloc[[0]], geometry.iloc[[1]], geometry.iloc[[0]]],
        ignore_index=True,
    )
    duplicated.to_csv(source, index=False)

    with pytest.raises(ValueError, match="rows are not unique"):
        stream_join_population_saliency_with_compression(
            _saliency(),
            source,
            joined_output_path=joined_path,
            association_projection_path=projection_path,
            chunksize=1,
        )
