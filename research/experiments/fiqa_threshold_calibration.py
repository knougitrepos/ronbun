"""Derived FIQA threshold experiment over immutable completed Step-4 runs.

Only the selected compressed calibration search is replayed. Existing face
embeddings, fitted codecs, official test scores, and completed runs remain
unchanged. The replay is audited against the threshold persisted by Step 4.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from research.fiqa import FIQAScoreArtifact
from research.calibration import (
    apply_threshold_model,
    fit_conditional_threshold,
    fit_global_threshold,
    paired_method_comparison,
)
from research.calibration.rejection import choose_non_mated_fpir_threshold
from research.compression import ORIGIN_512, PQCompressor
from research.explainability.gradcam.artifacts import (
    read_prepared_population_artifact,
)
from research.experiments.step2_compression import (
    open_set_protocol_arrays,
    prepared_population_frame,
)
from research.protocols import build_survface_matched_calibration_protocol
from research.runtime.hashing import canonical_sha256, sha256_file


SURVFACE_ADC_SEARCH_MODE = "pq_adc_exhaustive"
ADC_SCORE_SPACE = "negative_squared_l2_adc"


@dataclass(frozen=True)
class ConditionScoreTables:
    calibration: pd.DataFrame
    test: pd.DataFrame
    manifest: dict[str, Any]

    @property
    def condition_uid(self) -> str:
        return str(self.manifest["condition_uid"])


@dataclass(frozen=True)
class CalibrationComparison:
    method_summary: pd.DataFrame
    thresholds: pd.DataFrame
    paired_comparisons: pd.DataFrame
    manifest: dict[str, Any]

    @property
    def comparison_uid(self) -> str:
        return str(self.manifest["comparison_uid"])


@dataclass(frozen=True)
class SaliencyIncrementalReadiness:
    status: str
    primary_analysis_supported: bool
    secondary_calibration_supported: bool
    calibration_coverage: float
    test_coverage: float
    reasons: tuple[str, ...]
    saliency_target_name: str | None
    requested_features: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "primary_analysis_supported": self.primary_analysis_supported,
            "secondary_calibration_supported": self.secondary_calibration_supported,
            "calibration_coverage": self.calibration_coverage,
            "test_coverage": self.test_coverage,
            "reasons": list(self.reasons),
            "saliency_target_name": self.saliency_target_name,
            "requested_features": list(self.requested_features),
        }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _completed_run(run_dir: str | Path) -> tuple[Path, dict[str, Any], Path]:
    root = Path(run_dir).resolve()
    marker = root / "COMPLETED"
    manifest_path = root / "run_manifest.json"
    if not marker.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"completed Step-4 run is incomplete: {root}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError("run_manifest status is not completed")
    config = dict(manifest.get("config", {}))
    if config.get("dataset_id") != "survface":
        raise ValueError("the initial FIQA replay supports SurvFace completed runs")
    workflow = root / "artifacts" / "step2_workflow"
    if not workflow.is_dir():
        raise FileNotFoundError(f"Step-4 workflow artifact is missing: {workflow}")
    return root, manifest, workflow


def _ledger_condition(
    ledger_manifest: dict[str, Any],
    *,
    compression_profile: str,
    search_mode: str,
) -> dict[str, Any]:
    matches = [
        dict(item)
        for item in ledger_manifest.get("conditions", [])
        if str(item.get("condition", {}).get("compression_profile"))
        == compression_profile
        and str(item.get("condition", {}).get("search_mode")) == search_mode
        and str(item.get("condition", {}).get("evaluation_split")) == "test"
    ]
    if len(matches) != 1:
        raise ValueError(
            "retrieval ledger must contain exactly one matching test core: "
            f"profile={compression_profile}, mode={search_mode}, count={len(matches)}"
        )
    return matches[0]


def _verified_table(root: Path, entry: dict[str, Any]) -> Path:
    path = (root / str(entry["path"])).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("artifact path escapes its verified root")
    if not path.is_file() or sha256_file(path) != str(entry["sha256"]):
        raise ValueError(f"artifact is missing or has a hash mismatch: {path}")
    return path


def _standard_scores(
    frame: pd.DataFrame,
    *,
    split: str,
    expected_top_k: int,
) -> pd.DataFrame:
    required = {
        "query_id",
        "query_identity_id",
        "is_mated",
        "compressed_top1_score",
        "compressed_rank1_correct",
        "compressed_top_k_correct",
        "compression_profile",
        "top_k",
        "search_mode",
        "compressed_score_space",
        "protocol_uid",
        "model_uid",
        "dataset_id",
        "origin_embedding_artifact_uid",
        "extraction_uid",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"compressed score frame is missing columns: {missing}")
    output = frame[
        [
            "query_id",
            "query_identity_id",
            "is_mated",
            "compressed_top1_score",
            "compressed_rank1_correct",
            "compressed_top_k_correct",
            "compression_profile",
            "top_k",
            "search_mode",
            "compressed_score_space",
            "protocol_uid",
            "model_uid",
            "dataset_id",
            "origin_embedding_artifact_uid",
            "extraction_uid",
        ]
    ].copy()
    output = output.rename(
        columns={
            "query_id": "sample_id",
            "query_identity_id": "identity_id",
            "compressed_top1_score": "score",
            "compressed_rank1_correct": "rank1_correct",
            "compressed_top_k_correct": "top_k_correct",
            "compressed_score_space": "score_space",
        }
    )
    output.insert(2, "evaluation_split", split)
    top_k_values = pd.to_numeric(output["top_k"], errors="coerce")
    if top_k_values.isna().any() or not top_k_values.eq(
        int(expected_top_k)
    ).all():
        observed = sorted(output["top_k"].dropna().astype(str).unique())
        raise ValueError(
            f"{split} compressed score rows have unexpected top_k: "
            f"expected={int(expected_top_k)}, observed={observed}"
        )
    output["top_k"] = top_k_values.astype(np.int64)
    score_spaces = output["score_space"].dropna().astype(str).unique()
    if len(score_spaces) != 1 or score_spaces[0] != ADC_SCORE_SPACE:
        raise ValueError(
            f"{split} compressed score rows are not in the required ADC "
            f"score space: expected={ADC_SCORE_SPACE!r}, "
            f"observed={sorted(score_spaces.tolist())}"
        )
    output["threshold_comparator"] = ">="
    if output["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"{split} compressed score rows have duplicate sample IDs")
    if not np.isfinite(output["score"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{split} compressed score rows contain non-finite values")
    return output.reset_index(drop=True)


def _adc_condition_score_frame(
    adc_distances: np.ndarray,
    adc_indices: np.ndarray,
    *,
    query_ids: np.ndarray,
    gallery_identity_ids: np.ndarray,
    query_identity_ids: np.ndarray,
    compression_profile: str,
    search_mode: str,
) -> pd.DataFrame:
    """Build the compressed score fields needed by FIQA calibration.

    The FIQA replay consumes only ADC scores and rank correctness.  It does not
    require an origin-cosine comparison, so this adapter avoids repeating and
    materializing origin top-k retrieval solely to satisfy the broader Step-4
    comparison schema.
    """

    distances = np.asarray(adc_distances, dtype=np.float64)
    indices = np.asarray(adc_indices, dtype=np.int64)
    queries = np.asarray(query_ids, dtype=object)
    query_identities = np.asarray(query_identity_ids, dtype=object)
    gallery_identities = np.asarray(gallery_identity_ids, dtype=object)
    if distances.ndim != 2 or indices.shape != distances.shape:
        raise ValueError("ADC distances and indices must be equal-shape 2D arrays")
    if distances.shape[0] != len(queries) or len(query_identities) != len(queries):
        raise ValueError("ADC result shape must match the query identifiers")
    if distances.shape[1] == 0:
        raise ValueError("ADC result must contain at least one rank")
    if not np.isfinite(distances).all() or (distances < -1e-5).any():
        raise ValueError("ADC squared-L2 distances must be finite and non-negative")
    if (indices < 0).any() or (indices >= len(gallery_identities)).any():
        raise ValueError("ADC indices are outside the gallery")

    ranked_identities = gallery_identities[indices]
    gallery_identity_set = set(gallery_identities.tolist())
    return pd.DataFrame(
        {
            "query_id": queries,
            "query_identity_id": query_identities,
            "is_mated": [
                identity in gallery_identity_set for identity in query_identities
            ],
            "compressed_top1_score": -distances[:, 0],
            "compressed_rank1_correct": (
                ranked_identities[:, 0] == query_identities
            ),
            "compressed_top_k_correct": np.any(
                ranked_identities == query_identities[:, np.newaxis],
                axis=1,
            ),
            "compression_family": "pq",
            "compression_profile": compression_profile,
            "top_k": int(distances.shape[1]),
            "search_mode": search_mode,
            "compressed_score_space": ADC_SCORE_SPACE,
        }
    )


def _attach_alignment_hashes(
    scores: pd.DataFrame,
    selected_manifest: pd.DataFrame,
) -> pd.DataFrame:
    required = {"sample_id", "aligned_content_sha256"}
    missing = sorted(required - set(selected_manifest.columns))
    if missing:
        raise ValueError(f"selected manifest is missing columns: {missing}")
    alignment = selected_manifest[
        ["sample_id", "aligned_content_sha256"]
    ].copy()
    alignment["sample_id"] = alignment["sample_id"].astype(str)
    if alignment["sample_id"].duplicated().any():
        raise ValueError("selected manifest sample IDs are not unique")
    if alignment["aligned_content_sha256"].isna().any():
        raise ValueError("selected manifest has missing aligned content hashes")
    output = scores.copy()
    output["sample_id"] = output["sample_id"].astype(str)
    output = output.merge(
        alignment,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if output["aligned_content_sha256"].isna().any():
        raise ValueError("one or more condition scores lack an alignment hash")
    return output


def _frozen_pq_codec(
    run_root: Path,
    workflow: Path,
    *,
    compression_profile: str,
) -> tuple[PQCompressor, dict[str, Any], dict[str, Any]]:
    manifest_path = workflow / "frozen_codec_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_type") != "frozen_compression_codec_bundle":
        raise ValueError("unexpected frozen codec artifact_type")
    matches = [
        dict(item)
        for item in manifest.get("codecs", [])
        if item.get("family") == "pq"
        and item.get("profile_name") == compression_profile
    ]
    if len(matches) != 1:
        raise ValueError("frozen codec manifest has no unique requested PQ codec")
    entry = matches[0]
    codec_path = (run_root / str(entry["artifact"])).resolve()
    if run_root not in codec_path.parents:
        raise ValueError("frozen codec path escapes the completed run")
    if not codec_path.is_file() or sha256_file(codec_path) != str(
        entry["artifact_sha256"]
    ):
        raise ValueError("frozen PQ codec is missing or has a hash mismatch")
    codec = PQCompressor.load(codec_path)
    codec.source_profile = ORIGIN_512
    codec.fit_count = int(entry["fit_count"])
    return codec, entry, manifest


def replay_survface_adc_condition_scores(
    run_dir: str | Path,
    *,
    compression_profile: str = "pq_512_m128_b8",
    search_mode: str = SURVFACE_ADC_SEARCH_MODE,
) -> ConditionScoreTables:
    """Replay only SurvFace calibration ADC and reuse the persisted test core."""

    if search_mode != SURVFACE_ADC_SEARCH_MODE:
        raise ValueError("the initial replay contract supports PQ ADC only")
    run_root, run_manifest, workflow = _completed_run(run_dir)
    step4 = dict(run_manifest["config"]["step4"])
    evaluation = dict(step4["evaluation"])
    selected_path = workflow / "selected_manifest.csv"
    prepared_dir = workflow / "prepared_population"
    freeze_path = workflow / "freeze_manifest.json"
    selected = pd.read_csv(selected_path)
    prepared = read_prepared_population_artifact(prepared_dir)
    freeze = _read_json(freeze_path)
    expected_freeze = {
        "run_id": str(run_manifest["run_id"]),
        "dataset_id": "survface",
        "model_uid": str(run_manifest["config"]["model_uid"]),
        "extraction_uid": str(prepared.extraction_uid),
    }
    for key, expected in expected_freeze.items():
        if str(freeze.get(key)) != expected:
            raise ValueError(
                f"freeze manifest {key} differs from the completed run: "
                f"expected={expected}, actual={freeze.get(key)}"
            )
    selected_sha256 = sha256_file(selected_path)
    if selected_sha256 != str(freeze.get("selected_manifest_sha256")):
        raise ValueError("selected manifest SHA-256 differs from freeze manifest")
    if len(selected) != int(freeze.get("selected_sample_count", -1)):
        raise ValueError("selected manifest row count differs from freeze manifest")
    population = prepared_population_frame(prepared, selected)

    codec, codec_entry, codec_manifest = _frozen_pq_codec(
        run_root,
        workflow,
        compression_profile=compression_profile,
    )
    seed = int(codec_manifest["fit_seed"])
    calibration_protocol = build_survface_matched_calibration_protocol(
        population,
        gallery_identity_count=int(
            evaluation["survface_calibration_gallery_identities"]
        ),
        seed=seed,
    )
    arrays = open_set_protocol_arrays(calibration_protocol, population)
    top_k = min(int(evaluation["top_k"]), len(arrays["gallery"]))
    gallery_codes = codec.encode(arrays["gallery"])
    distances, indices, _ = codec.search_adc_with_metrics(
        arrays["queries"],
        gallery_codes,
        top_k=top_k,
    )
    calibration_raw = _adc_condition_score_frame(
        distances,
        indices,
        query_ids=arrays["query_ids"],
        query_identity_ids=arrays["query_identity_ids"],
        gallery_identity_ids=arrays["gallery_identity_ids"],
        compression_profile=compression_profile,
        search_mode=search_mode,
    )
    protocol_uid = (
        "qmul-survface-v1-training-derived-"
        f"{int(evaluation['survface_calibration_gallery_identities'])}"
        "-watchlist-calibration-v2"
    )
    calibration_raw.insert(0, "dataset", "survface")
    calibration_raw.insert(1, "model_uid", prepared.model_uid)
    calibration_raw["protocol_uid"] = protocol_uid
    calibration_raw["threshold_source_split"] = "calibration"
    calibration_raw["evaluation_split"] = "calibration"
    calibration_raw["dataset_id"] = "survface"
    calibration_raw["extraction_uid"] = prepared.extraction_uid
    calibration_raw["origin_embedding_artifact_uid"] = (
        prepared.origin_embedding_artifact_uid
    )
    calibration = _attach_alignment_hashes(
        _standard_scores(
            calibration_raw,
            split="calibration",
            expected_top_k=top_k,
        ),
        selected,
    )

    ledger_root = workflow / "retrieval_ledger"
    ledger_manifest_path = ledger_root / "manifest.json"
    ledger_manifest = _read_json(ledger_manifest_path)
    condition = _ledger_condition(
        ledger_manifest,
        compression_profile=compression_profile,
        search_mode=search_mode,
    )
    core_path = _verified_table(ledger_root, dict(condition["core"]))
    test_raw = pd.read_parquet(core_path)
    test = _attach_alignment_hashes(
        _standard_scores(
            test_raw,
            split="test",
            expected_top_k=top_k,
        ),
        selected,
    )
    if set(calibration["sample_id"]).intersection(test["sample_id"]):
        raise ValueError("calibration and test compressed score IDs overlap")
    if set(calibration["identity_id"]).intersection(test["identity_id"]):
        raise ValueError("calibration and test compressed score identities overlap")

    threshold_audit_rows: list[dict[str, Any]] = []
    for decision in condition.get("decisions", []):
        item = dict(decision)
        decision_path = _verified_table(ledger_root, dict(item["artifact"]))
        persisted = pd.read_parquet(
            decision_path,
            columns=["compressed_decision_threshold"],
        )["compressed_decision_threshold"].to_numpy(dtype=np.float64)
        if len(persisted) != int(condition["row_count"]):
            raise ValueError("persisted decision row count differs from test core")
        unique = np.unique(persisted)
        if len(unique) != 1:
            raise ValueError("persisted decision artifact mixes thresholds")
        target = float(item["target_fpir"])
        reproduced = float(
            choose_non_mated_fpir_threshold(
                calibration["score"].to_numpy(dtype=np.float64),
                calibration["is_mated"].to_numpy(dtype=bool),
                target_fpir=target,
            )
        )
        threshold_audit_rows.append(
            {
                "target_fpir": target,
                "persisted_threshold": float(unique[0]),
                "reproduced_threshold": reproduced,
                "absolute_difference": abs(reproduced - float(unique[0])),
                "exact_match": bool(reproduced == float(unique[0])),
            }
        )
    if not threshold_audit_rows or not all(
        row["absolute_difference"] <= 1e-12 for row in threshold_audit_rows
    ):
        raise RuntimeError(
            "replayed calibration scores do not reproduce persisted Step-4 thresholds"
        )

    manifest_base: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "compressed_calibration_test_score_tables",
        "status": "completed_in_memory",
        "source_run_id": str(run_manifest["run_id"]),
        "source_run_manifest_sha256": sha256_file(run_root / "run_manifest.json"),
        "source_freeze_manifest_sha256": sha256_file(freeze_path),
        "aligned_bundle_manifest_sha256": str(
            freeze["aligned_bundle_manifest_sha256"]
        ),
        "dataset_id": "survface",
        "model_uid": prepared.model_uid,
        "extraction_uid": prepared.extraction_uid,
        "origin_embedding_artifact_uid": prepared.origin_embedding_artifact_uid,
        "compression_family": "pq",
        "compression_profile": compression_profile,
        "search_mode": search_mode,
        "score_space": ADC_SCORE_SPACE,
        "threshold_comparator": ">=",
        "protocol_uid": protocol_uid,
        "calibration_protocol": "training_3000_half_gallery_v2",
        "calibration_seed": seed,
        "top_k": top_k,
        "prepared_population_manifest_sha256": sha256_file(
            prepared_dir / "manifest.json"
        ),
        "selected_manifest_sha256": selected_sha256,
        "frozen_codec": {
            "profile_name": codec_entry["profile_name"],
            "sha256": codec_entry["artifact_sha256"],
            "fit_count": codec_entry["fit_count"],
            "fit_seed": codec_entry["fit_seed"],
        },
        "persisted_test_core_sha256": condition["core"]["sha256"],
        "calibration_rows": int(len(calibration)),
        "test_rows": int(len(test)),
        "global_threshold_reproduction": threshold_audit_rows,
    }
    manifest_base["condition_uid"] = (
        "compressed-scores-" + canonical_sha256(manifest_base)[:24]
    )
    return ConditionScoreTables(
        calibration=calibration,
        test=test,
        manifest=manifest_base,
    )


def join_fiqa_scores(
    condition_scores: pd.DataFrame,
    fiqa_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Fail closed unless every condition query has one matching FIQA score."""

    required = {"sample_id", "fiqa_score", "fiqa_model_uid"}
    missing = sorted(required - set(fiqa_scores.columns))
    if missing:
        raise ValueError(f"FIQA score table is missing columns: {missing}")
    quality_columns = ["sample_id", "fiqa_score", "fiqa_model_uid"]
    if "aligned_content_sha256" in fiqa_scores:
        quality_columns.append("aligned_content_sha256")
    quality = fiqa_scores[quality_columns].copy()
    quality["sample_id"] = quality["sample_id"].astype(str)
    if quality["sample_id"].duplicated().any():
        raise ValueError("FIQA score table contains duplicate sample IDs")
    joined = condition_scores.copy()
    joined["sample_id"] = joined["sample_id"].astype(str)
    if "aligned_content_sha256" in quality:
        quality = quality.rename(
            columns={"aligned_content_sha256": "fiqa_aligned_content_sha256"}
        )
    joined = joined.merge(
        quality,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if joined["fiqa_score"].isna().any():
        missing_count = int(joined["fiqa_score"].isna().sum())
        raise ValueError(f"condition queries are missing {missing_count} FIQA scores")
    if joined["fiqa_model_uid"].astype(str).nunique() != 1:
        raise ValueError("joined condition rows mix FIQA model UIDs")
    if "aligned_content_sha256" in joined:
        if "fiqa_aligned_content_sha256" not in joined:
            raise ValueError("FIQA scores lack aligned content hashes")
        condition_hash = joined["aligned_content_sha256"].astype(str)
        fiqa_hash = joined["fiqa_aligned_content_sha256"].astype(str)
        if not condition_hash.eq(fiqa_hash).all():
            mismatch_count = int((~condition_hash.eq(fiqa_hash)).sum())
            raise ValueError(
                f"condition/FIQA alignment hashes differ for {mismatch_count} rows"
            )
        joined = joined.drop(columns=["fiqa_aligned_content_sha256"])
    return joined


def join_fiqa_score_artifacts(
    condition_tables: ConditionScoreTables,
    fiqa_artifact: FIQAScoreArtifact,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join two verified artifacts only when dataset/alignment lineage matches."""

    condition_manifest = condition_tables.manifest
    fiqa_manifest = fiqa_artifact.manifest
    if condition_manifest.get("artifact_type") != (
        "compressed_calibration_test_score_tables"
    ):
        raise ValueError("unexpected condition score artifact_type")
    if fiqa_manifest.get("artifact_type") != "fiqa_score_table":
        raise ValueError("unexpected FIQA score artifact_type")
    if condition_manifest.get("status") != "completed":
        raise ValueError("condition score artifact must be completed")
    if fiqa_manifest.get("status") != "completed":
        raise ValueError("FIQA score artifact must be completed")
    if str(condition_manifest.get("dataset_id")) != str(
        fiqa_manifest.get("dataset_id")
    ):
        raise ValueError("condition and FIQA artifacts use different datasets")
    condition_alignment = str(
        condition_manifest.get("aligned_bundle_manifest_sha256", "")
    )
    fiqa_alignment = str(
        fiqa_manifest.get("aligned_bundle_manifest_sha256", "")
    )
    if not condition_alignment or condition_alignment != fiqa_alignment:
        raise ValueError("condition and FIQA aligned-bundle lineage differs")
    if "aligned_content_sha256" not in condition_tables.calibration:
        raise ValueError("calibration condition scores lack alignment hashes")
    if "aligned_content_sha256" not in condition_tables.test:
        raise ValueError("test condition scores lack alignment hashes")
    if "aligned_content_sha256" not in fiqa_artifact.scores:
        raise ValueError("FIQA score artifact lacks alignment hashes")
    return (
        join_fiqa_scores(condition_tables.calibration, fiqa_artifact.scores),
        join_fiqa_scores(condition_tables.test, fiqa_artifact.scores),
    )


def _single_value(frame: pd.DataFrame, column: str, *, split: str) -> str:
    if column not in frame:
        raise ValueError(f"{split} rows are missing {column}")
    values = frame[column].dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError(f"{split} rows do not have one {column}")
    return str(values[0])


def _validate_comparison_contract(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    *,
    condition_manifest: dict[str, Any],
    fiqa_manifest: dict[str, Any],
) -> tuple[str, int, str, str]:
    if condition_manifest.get("artifact_type") != (
        "compressed_calibration_test_score_tables"
    ) or condition_manifest.get("status") != "completed":
        raise ValueError("comparison requires a completed condition artifact")
    if (
        fiqa_manifest.get("artifact_type") != "fiqa_score_table"
        or fiqa_manifest.get("status") != "completed"
    ):
        raise ValueError("comparison requires a completed FIQA artifact")
    condition_uid = str(condition_manifest.get("condition_uid", ""))
    fiqa_uid = str(fiqa_manifest.get("fiqa_uid", ""))
    if not condition_uid or not fiqa_uid:
        raise ValueError("comparison source artifacts lack immutable UIDs")
    dataset_id = str(condition_manifest.get("dataset_id", ""))
    if not dataset_id or dataset_id != str(fiqa_manifest.get("dataset_id", "")):
        raise ValueError("comparison source artifacts use different datasets")
    condition_alignment = str(
        condition_manifest.get("aligned_bundle_manifest_sha256", "")
    )
    if not condition_alignment or condition_alignment != str(
        fiqa_manifest.get("aligned_bundle_manifest_sha256", "")
    ):
        raise ValueError("comparison source artifacts use different alignments")
    score_space = str(condition_manifest.get("score_space", ""))
    if not score_space:
        raise ValueError("condition artifact lacks score_space")
    rank_k = int(condition_manifest.get("top_k", 0))
    if rank_k <= 0:
        raise ValueError("condition artifact lacks a positive top_k")

    required = {
        "sample_id",
        "identity_id",
        "evaluation_split",
        "is_mated",
        "score",
        "fiqa_score",
        "fiqa_model_uid",
        "top_k_correct",
        "top_k",
        "aligned_content_sha256",
    }
    for split, frame in (("calibration", calibration), ("test", test)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{split} comparison rows are missing: {missing}")
        if set(frame["evaluation_split"].astype(str)) != {split}:
            raise ValueError(f"{split} rows have an invalid evaluation_split")
        if frame["identity_id"].isna().any() or frame["identity_id"].astype(
            str
        ).eq("").any():
            raise ValueError(f"{split} rows have missing identity IDs")
        row_top_k = pd.to_numeric(frame["top_k"], errors="coerce")
        if row_top_k.isna().any() or not row_top_k.eq(rank_k).all():
            observed = sorted(frame["top_k"].dropna().astype(str).unique())
            raise ValueError(
                f"{split} top_k differs from its source manifest: "
                f"expected={rank_k}, observed={observed}"
            )
        expected_values = {
            "dataset_id": dataset_id,
            "model_uid": str(condition_manifest["model_uid"]),
            "compression_profile": str(
                condition_manifest["compression_profile"]
            ),
            "search_mode": str(condition_manifest["search_mode"]),
            "score_space": score_space,
            "protocol_uid": str(condition_manifest["protocol_uid"]),
            "origin_embedding_artifact_uid": str(
                condition_manifest["origin_embedding_artifact_uid"]
            ),
            "extraction_uid": str(condition_manifest["extraction_uid"]),
            "fiqa_model_uid": str(fiqa_manifest["fiqa_model_uid"]),
        }
        for column, expected in expected_values.items():
            if _single_value(frame, column, split=split) != expected:
                raise ValueError(
                    f"{split} {column} differs from its source manifest"
                )
    calibration_ids = set(calibration["sample_id"].astype(str))
    test_ids = set(test["sample_id"].astype(str))
    overlap = calibration_ids.intersection(test_ids)
    if overlap:
        raise ValueError(
            f"calibration and test comparison IDs overlap: {len(overlap)}"
        )
    calibration_identities = set(calibration["identity_id"].astype(str))
    test_identities = set(test["identity_id"].astype(str))
    identity_overlap = calibration_identities.intersection(test_identities)
    if identity_overlap:
        raise ValueError(
            "calibration and test comparison identities overlap: "
            f"{len(identity_overlap)}"
        )
    return score_space, rank_k, condition_uid, fiqa_uid


def run_global_vs_fiqa_calibration(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_fpirs: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.30),
    bin_count: int = 2,
    shrinkage_strength: float = 200.0,
    minimum_group_non_mated: int = 100,
    safety_fraction: float = 0.30,
    partition_seed: int = 42,
    condition_manifest: dict[str, Any],
    fiqa_manifest: dict[str, Any],
) -> CalibrationComparison:
    """Compare legacy global, safety-global, and FIQA-conditional thresholds."""

    targets = tuple(float(value) for value in target_fpirs)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("target_fpirs must be non-empty and unique")
    if tuple(sorted(targets)) != targets:
        raise ValueError("target_fpirs must be in increasing order")
    if not 0.0 < float(safety_fraction) < 1.0:
        raise ValueError(
            "global-vs-FIQA comparison requires safety_fraction inside (0, 1)"
        )
    score_space, rank_k, condition_uid, fiqa_uid = _validate_comparison_contract(
        calibration,
        test,
        condition_manifest=condition_manifest,
        fiqa_manifest=fiqa_manifest,
    )
    summary_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    model_uids: list[str] = []

    for target in targets:
        models = (
            fit_global_threshold(
                calibration,
                target_fpir=target,
                safety_fraction=0.0,
                partition_seed=partition_seed,
                partition_column="identity_id",
                score_space=score_space,
            ),
            fit_global_threshold(
                calibration,
                target_fpir=target,
                safety_fraction=safety_fraction,
                partition_seed=partition_seed,
                partition_column="identity_id",
                score_space=score_space,
            ),
            fit_conditional_threshold(
                calibration,
                target_fpir=target,
                bin_count=bin_count,
                shrinkage_strength=shrinkage_strength,
                minimum_group_non_mated=minimum_group_non_mated,
                safety_fraction=safety_fraction,
                partition_seed=partition_seed,
                partition_column="identity_id",
                score_space=score_space,
            ),
        )
        evaluations = {
            model.method: apply_threshold_model(test, model) for model in models
        }
        if len(evaluations) != len(models):
            raise RuntimeError("calibration methods have duplicate names")
        for model in models:
            model_uids.append(model.model_uid)
            summary_rows.append(
                {
                    **evaluations[model.method].summary,
                    "rank_k": rank_k,
                    "condition_uid": condition_uid,
                    "fiqa_uid": fiqa_uid,
                }
            )
            for group in model.groups:
                threshold_rows.append(
                    {
                        "method": model.method,
                        "model_uid": model.model_uid,
                        "target_fpir": target,
                        "score_space": score_space,
                        "rank_k": rank_k,
                        "quality_group": group.name,
                        "quality_cutpoints": json.dumps(
                            list(model.quality_cutpoints),
                            separators=(",", ":"),
                        ),
                        **group.as_dict(),
                    }
                )
        candidate = evaluations[models[2].method]
        for reference_model in models[:2]:
            paired = paired_method_comparison(
                evaluations[reference_model.method],
                candidate,
            )
            for metric in ("fpir", "tpir_at_rank_k"):
                evidence = dict(paired[metric])
                paired_rows.append(
                    {
                        "reference_method": paired["reference_method"],
                        "candidate_method": paired["candidate_method"],
                        "target_fpir": target,
                        "score_space": score_space,
                        "rank_k": rank_k,
                        "metric": metric,
                        **evidence,
                        "confidence_interval_method": paired[
                            "confidence_interval_method"
                        ],
                        "resamples": paired["resamples"],
                        "random_seed": paired["random_seed"],
                    }
                )

    manifest_base: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "global_vs_fiqa_threshold_calibration",
        "status": "completed_in_memory",
        "condition_uid": str(condition_uid),
        "fiqa_uid": str(fiqa_uid),
        "condition_manifest_sha256": canonical_sha256(condition_manifest),
        "fiqa_manifest_sha256": canonical_sha256(fiqa_manifest),
        "target_fpirs": list(targets),
        "score_space": str(score_space),
        "threshold_comparator": ">=",
        "rank_k": rank_k,
        "calibration_rows": int(len(calibration)),
        "test_rows": int(len(test)),
        "quality_condition": {
            "column": "fiqa_score",
            "bin_count": int(bin_count),
            "cutpoint_source": "calibration_fit_only",
            "shrinkage_strength": float(shrinkage_strength),
            "minimum_group_non_mated": int(minimum_group_non_mated),
        },
        "safety_calibration": {
            "fraction": float(safety_fraction),
            "partition_seed": int(partition_seed),
            "partition_key": "sha256(identity_id)",
            "partition_unit": "identity_cluster",
        },
        "methods": [
            "global_empirical",
            "global_safe",
            f"fiqa_{int(bin_count)}bin_conservative_shrunk_safe",
        ],
        "threshold_fit_on_test": False,
        "saliency_contract": {
            "primary_purpose": (
                "analyze associations between FR spatial evidence and "
                "compression-induced embedding/retrieval behavior"
            ),
            "secondary_purpose": (
                "test incremental value beyond FIQA only after calibration "
                "saliency coverage passes a leakage-safe readiness gate"
            ),
            "included_in_this_comparison": False,
        },
        "model_uids": sorted(set(model_uids)),
    }
    manifest_base["comparison_uid"] = (
        "global-fiqa-" + canonical_sha256(manifest_base)[:24]
    )
    return CalibrationComparison(
        method_summary=pd.DataFrame.from_records(summary_rows),
        thresholds=pd.DataFrame.from_records(threshold_rows),
        paired_comparisons=pd.DataFrame.from_records(paired_rows),
        manifest=manifest_base,
    )


def assess_saliency_incremental_readiness(
    calibration_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    saliency_features: pd.DataFrame,
    *,
    requested_features: tuple[str, ...] = (
        "outside_face_attention",
        "saliency_entropy",
    ),
    minimum_coverage: float = 0.95,
) -> SaliencyIncrementalReadiness:
    """Gate FIQA+Saliency so test-only Grad-CAM can never train thresholds."""

    coverage_requirement = float(minimum_coverage)
    if (
        not np.isfinite(coverage_requirement)
        or not 0.0 < coverage_requirement <= 1.0
    ):
        raise ValueError("minimum_coverage must be finite and inside (0, 1]")
    reasons: list[str] = []
    required = {
        "sample_id",
        "saliency_target_name",
        "heatmap_available",
        "gradcam_valid_heatmap",
        *requested_features,
    }
    missing = sorted(required - set(saliency_features.columns))
    if missing:
        reasons.append(f"saliency feature columns are missing: {missing}")
        return SaliencyIncrementalReadiness(
            status="blocked",
            primary_analysis_supported=False,
            secondary_calibration_supported=False,
            calibration_coverage=0.0,
            test_coverage=0.0,
            reasons=tuple(reasons),
            saliency_target_name=None,
            requested_features=requested_features,
        )
    saliency = saliency_features.copy()
    saliency["sample_id"] = saliency["sample_id"].astype(str)
    sample_ids_unique = not saliency["sample_id"].duplicated().any()
    if not sample_ids_unique:
        reasons.append("saliency sample IDs are not unique")
    targets = saliency["saliency_target_name"].dropna().astype(str).unique()
    target_name = str(targets[0]) if len(targets) == 1 else None
    if target_name != "origin_top1_gallery_cosine":
        reasons.append(
            "saliency target is not the prespecified label-free top-1 gallery target"
        )
    available = _strict_boolean_series(
        saliency["heatmap_available"],
        column="heatmap_available",
    ) & _strict_boolean_series(
        saliency["gradcam_valid_heatmap"],
        column="gradcam_valid_heatmap",
    )
    finite = np.ones(len(saliency), dtype=bool)
    for feature in requested_features:
        finite &= np.isfinite(saliency[feature].to_numpy(dtype=np.float64))
    valid_ids = set(saliency.loc[available & finite, "sample_id"].astype(str))
    calibration_ids = set(calibration_scores["sample_id"].astype(str))
    test_ids = set(test_scores["sample_id"].astype(str))
    calibration_coverage = (
        len(calibration_ids.intersection(valid_ids)) / len(calibration_ids)
        if calibration_ids
        else 0.0
    )
    test_coverage = (
        len(test_ids.intersection(valid_ids)) / len(test_ids) if test_ids else 0.0
    )
    if calibration_coverage < coverage_requirement:
        reasons.append(
            "calibration saliency coverage is below the prespecified minimum; "
            "fitting FIQA+Saliency would leak test information"
        )
    if test_coverage < coverage_requirement:
        reasons.append("test saliency coverage is below the prespecified minimum")
    primary_supported = bool(
        target_name == "origin_top1_gallery_cosine"
        and sample_ids_unique
        and test_coverage > 0.0
    )
    secondary_supported = bool(not reasons)
    return SaliencyIncrementalReadiness(
        status="ready" if secondary_supported else "blocked",
        primary_analysis_supported=primary_supported,
        secondary_calibration_supported=secondary_supported,
        calibration_coverage=float(calibration_coverage),
        test_coverage=float(test_coverage),
        reasons=tuple(reasons),
        saliency_target_name=target_name,
        requested_features=requested_features,
    )


def _strict_boolean_series(series: pd.Series, *, column: str) -> pd.Series:
    """Parse persisted bool/string-bool columns without truthy-string errors."""

    normalized = series.astype("string").str.strip().str.lower()
    true_values = {"true", "1", "1.0", "yes"}
    false_values = {"false", "0", "0.0", "no"}
    recognized = normalized.isna() | normalized.isin(true_values | false_values)
    if not bool(recognized.all()):
        unknown = sorted(normalized.loc[~recognized].dropna().unique().tolist())
        raise ValueError(f"{column} contains invalid boolean values: {unknown}")
    return normalized.isin(true_values).astype(bool)


def load_saliency_primary_diagnostics(
    run_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load compact first-purpose saliency evidence without reinterpreting it."""

    _, _, workflow = _completed_run(run_dir)
    names = {
        "geometry": "saliency_geometry_associations.csv",
        "retrieval": "saliency_retrieval_associations.csv",
        "threshold_instability": "saliency_threshold_instability_associations.csv",
        "threshold_policy": "saliency_threshold_policy_comparisons.csv",
        "threshold_policy_rho": "saliency_threshold_policy_rho_comparisons.csv",
    }
    result: dict[str, pd.DataFrame] = {}
    for key, name in names.items():
        path = workflow / name
        if not path.is_file():
            raise FileNotFoundError(f"saliency diagnostic is missing: {path}")
        result[key] = pd.read_csv(path)
    return result


def write_condition_score_artifact(
    output_dir: str | Path,
    tables: ConditionScoreTables,
    *,
    overwrite: bool = False,
) -> ConditionScoreTables:
    destination = Path(output_dir).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"condition score artifact exists: {destination}")
    staging = destination.with_name(f".{destination.name}.tmp-{uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        calibration_path = staging / "calibration_scores.parquet"
        test_path = staging / "test_scores.parquet"
        tables.calibration.to_parquet(calibration_path, index=False)
        tables.test.to_parquet(test_path, index=False)
        manifest = dict(tables.manifest)
        manifest["status"] = "completed"
        manifest["files"] = {
            "calibration_scores.parquet": {
                "sha256": sha256_file(calibration_path),
                "bytes": calibration_path.stat().st_size,
                "row_count": int(len(tables.calibration)),
            },
            "test_scores.parquet": {
                "sha256": sha256_file(test_path),
                "bytes": test_path.stat().st_size,
                "row_count": int(len(tables.test)),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            backup = destination.with_name(f".{destination.name}.old-{uuid4().hex}")
            os.replace(destination, backup)
            try:
                os.replace(staging, destination)
            except BaseException:
                os.replace(backup, destination)
                raise
            shutil.rmtree(backup)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_condition_score_artifact(destination)


def load_condition_score_artifact(directory: str | Path) -> ConditionScoreTables:
    root = Path(directory).resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("artifact_type") != "compressed_calibration_test_score_tables":
        raise ValueError("unexpected condition score artifact_type")
    if manifest.get("status") != "completed":
        raise ValueError("condition score artifact is not completed")
    files = dict(manifest.get("files", {}))
    frames: dict[str, pd.DataFrame] = {}
    for split in ("calibration", "test"):
        name = f"{split}_scores.parquet"
        entry = dict(files.get(name, {}))
        path = root / name
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"condition score artifact hash mismatch: {name}")
        frame = pd.read_parquet(path)
        if len(frame) != int(entry.get("row_count", -1)):
            raise ValueError(f"condition score artifact row count mismatch: {name}")
        frames[split] = frame
    return ConditionScoreTables(
        calibration=frames["calibration"],
        test=frames["test"],
        manifest=manifest,
    )


def write_calibration_comparison_artifact(
    output_dir: str | Path,
    comparison: CalibrationComparison,
    *,
    overwrite: bool = False,
) -> CalibrationComparison:
    destination = Path(output_dir).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"calibration comparison exists: {destination}")
    staging = destination.with_name(f".{destination.name}.tmp-{uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    frames = {
        "method_summary.csv": comparison.method_summary,
        "thresholds.csv": comparison.thresholds,
        "paired_comparisons.csv": comparison.paired_comparisons,
    }
    try:
        files: dict[str, Any] = {}
        for name, frame in frames.items():
            path = staging / name
            frame.to_csv(path, index=False, encoding="utf-8")
            files[name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "row_count": int(len(frame)),
                "columns": list(frame.columns),
            }
        manifest = dict(comparison.manifest)
        manifest["status"] = "completed"
        manifest["files"] = files
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            backup = destination.with_name(f".{destination.name}.old-{uuid4().hex}")
            os.replace(destination, backup)
            try:
                os.replace(staging, destination)
            except BaseException:
                os.replace(backup, destination)
                raise
            shutil.rmtree(backup)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_calibration_comparison_artifact(destination)


def load_calibration_comparison_artifact(
    directory: str | Path,
) -> CalibrationComparison:
    root = Path(directory).resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("artifact_type") != "global_vs_fiqa_threshold_calibration":
        raise ValueError("unexpected calibration comparison artifact_type")
    if manifest.get("status") != "completed":
        raise ValueError("calibration comparison artifact is not completed")
    frames: dict[str, pd.DataFrame] = {}
    for key, name in (
        ("method_summary", "method_summary.csv"),
        ("thresholds", "thresholds.csv"),
        ("paired_comparisons", "paired_comparisons.csv"),
    ):
        entry = dict(manifest.get("files", {}).get(name, {}))
        path = root / name
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"calibration comparison hash mismatch: {name}")
        frame = pd.read_csv(path)
        if len(frame) != int(entry.get("row_count", -1)):
            raise ValueError(f"calibration comparison row count mismatch: {name}")
        frames[key] = frame
    return CalibrationComparison(
        method_summary=frames["method_summary"],
        thresholds=frames["thresholds"],
        paired_comparisons=frames["paired_comparisons"],
        manifest=manifest,
    )
