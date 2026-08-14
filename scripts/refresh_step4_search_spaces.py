from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.evaluation import (  # noqa: E402
    annotate_compression_lineage,
    rate_ratio_matches_counts_or_compact_csv,
)
from research.experiments import (  # noqa: E402
    characterize_step2_compression,
    characterize_step2_rfw_custom_compression,
    characterize_step2_survface_compression,
)
from research.explainability.gradcam import (  # noqa: E402
    read_prepared_population_artifact,
)
from research.runtime import inspect_git_provenance  # noqa: E402
from scripts.generate_step4_compact_summaries import (  # noqa: E402
    summarize_compression,
    summarize_retrieval,
)


SCHEMA_VERSION = 5
ARTIFACT_TYPE = "step4_search_space_tpir20_multi_fpir_v5"
FAMILY_ARTIFACT_TYPE = "step4_search_space_tpir20_multi_fpir_family_v5"
SUPPORTED_FAMILIES = ("pca", "pq")
DEFAULT_TARGET_FPIRS = (0.01, 0.05, 0.10, 0.20, 0.30)
# Compact CSVs use ``%.12g``. Origin, compressed, and their independently
# rounded delta can differ from recomputation by at most about 1.5e-12 for
# rates in [0, 1]; keep the check strict enough to reject scientific drift.
COMPACT_RATE_DELTA_ROUNDTRIP_ATOL = 2e-12
REQUIRED_OPEN_SET_CONFIDENCE_COLUMNS = (
    "origin_dir_rank1_count",
    "origin_dir_rank1_denominator",
    "origin_dir_rank1",
    "origin_dir_rank1_wilson95_low",
    "origin_dir_rank1_wilson95_high",
    "compressed_dir_rank1_count",
    "compressed_dir_rank1_denominator",
    "compressed_dir_rank1",
    "compressed_dir_rank1_wilson95_low",
    "compressed_dir_rank1_wilson95_high",
    "compressed_minus_origin_dir_rank1",
    "compressed_minus_origin_dir_rank1_paired_bootstrap95_low",
    "compressed_minus_origin_dir_rank1_paired_bootstrap95_high",
    "tpir_rank",
    "origin_tpir_at_rank_k_count",
    "origin_tpir_at_rank_k_denominator",
    "origin_tpir_at_rank_k",
    "origin_tpir_at_rank_k_wilson95_low",
    "origin_tpir_at_rank_k_wilson95_high",
    "compressed_tpir_at_rank_k_count",
    "compressed_tpir_at_rank_k_denominator",
    "compressed_tpir_at_rank_k",
    "compressed_tpir_at_rank_k_wilson95_low",
    "compressed_tpir_at_rank_k_wilson95_high",
    "compressed_minus_origin_tpir_at_rank_k",
    "compressed_minus_origin_tpir_at_rank_k_paired_bootstrap95_low",
    "compressed_minus_origin_tpir_at_rank_k_paired_bootstrap95_high",
    "compressed_tpir_at_rank_k_retention",
    "origin_tpir20_count",
    "origin_tpir20_denominator",
    "origin_tpir20",
    "compressed_tpir20_count",
    "compressed_tpir20_denominator",
    "compressed_tpir20",
    "compressed_tpir20_retention",
    "origin_closed_set_rank20_recall",
    "compressed_closed_set_rank20_recall",
    "origin_false_accept_count",
    "origin_fpir_denominator",
    "origin_fpir",
    "origin_realized_fpir",
    "origin_fpir_wilson95_low",
    "origin_fpir_wilson95_high",
    "compressed_false_accept_count",
    "compressed_fpir_denominator",
    "compressed_fpir",
    "compressed_realized_fpir",
    "compressed_fpir_wilson95_low",
    "compressed_fpir_wilson95_high",
    "compressed_minus_origin_fpir",
    "compressed_minus_origin_fpir_paired_bootstrap95_low",
    "compressed_minus_origin_fpir_paired_bootstrap95_high",
    "confidence_interval_unit",
    "rate_confidence_interval_method",
    "difference_confidence_interval_method",
    "difference_confidence_interval_resamples",
    "difference_confidence_interval_random_seed",
)
_PROGRESS_STATE: dict[str, tuple[int, int]] = {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_entry(path: Path, *, root: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        portable = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        portable = str(resolved)
    return {
        "path": portable,
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _named_file_entry(path: Path, *, name: str) -> dict[str, object]:
    return {
        "path": name,
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.12g",
    )
    os.replace(temporary, path)


def _source_context(run_dir: Path) -> dict[str, Any]:
    source = run_dir.resolve()
    run_manifest_path = source / "run_manifest.json"
    completed_path = source / "COMPLETED"
    if not run_manifest_path.is_file() or not completed_path.is_file():
        raise FileNotFoundError("source run must contain run_manifest.json and COMPLETED")
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("status") != "completed":
        raise ValueError("source run status must be completed")
    config = run_manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("source run manifest is missing the frozen config")
    step4 = config.get("step4")
    if not isinstance(step4, dict):
        raise ValueError("source run manifest is missing the frozen Step 4 config")
    dataset_id = str(config.get("dataset_id", "")).strip()
    if dataset_id not in {"lfw", "survface", "rfw_custom"}:
        raise ValueError(f"unsupported source dataset: {dataset_id!r}")
    if dataset_id == "rfw_custom":
        evaluation = step4.get("evaluation")
        if not isinstance(evaluation, dict) or evaluation.get(
            "rfw_custom_calibration_gallery_policy"
        ) != "evaluation_group_matched":
            raise ValueError(
                "RFW-Custom source run predates gallery-size-matched "
                "calibration; start a new run"
            )
    workflow = step4["workflow"]
    workflow_root = source / str(workflow["artifact_subdir"])
    selected_path = workflow_root / str(workflow["selected_manifest_path"])
    prepared_dir = workflow_root / str(workflow["prepared_population_dir"])
    prepared_manifest_path = prepared_dir / "manifest.json"
    freeze_manifest_path = workflow_root / str(workflow["freeze_manifest_path"])
    for path in (
        selected_path,
        prepared_manifest_path,
        freeze_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    freeze_manifest = _read_json(freeze_manifest_path)
    if freeze_manifest.get("fallback_free") is not True:
        raise ValueError("source freeze manifest must declare fallback_free=true")
    prepared_manifest = _read_json(prepared_manifest_path)
    run_id = str(run_manifest["run_id"])
    model_uid = str(config["model_uid"])
    for label, observed in (
        ("freeze dataset", freeze_manifest.get("dataset_id")),
        ("prepared dataset", prepared_manifest.get("dataset_id")),
    ):
        if str(observed) != dataset_id:
            raise ValueError(f"{label} does not match source run")
    for label, observed in (
        ("freeze model", freeze_manifest.get("model_uid")),
        ("prepared model", prepared_manifest.get("model_uid")),
    ):
        if str(observed) != model_uid:
            raise ValueError(f"{label} does not match source run")
    return {
        "run_dir": source,
        "run_manifest_path": run_manifest_path,
        "run_manifest": run_manifest,
        "config": config,
        "step4": step4,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "model_uid": model_uid,
        "workflow_root": workflow_root,
        "selected_path": selected_path,
        "prepared_dir": prepared_dir,
        "prepared_manifest_path": prepared_manifest_path,
        "prepared_manifest": prepared_manifest,
        "freeze_manifest_path": freeze_manifest_path,
        "freeze_manifest": freeze_manifest,
    }


def _progress(message: str, details: dict[str, object]) -> None:
    processed = int(details.get("processed", 0))
    total = int(details.get("total", 0))
    if total > 0:
        previous_processed, previous_bucket = _PROGRESS_STATE.get(
            message, (-1, -1)
        )
        if processed < previous_processed:
            previous_bucket = -1
        bucket = min(10, (processed * 10) // total)
        _PROGRESS_STATE[message] = (processed, max(previous_bucket, bucket))
        if bucket <= previous_bucket and processed != total:
            return
    print(
        json.dumps(
            {"event": message, **details},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _read_lfw_ids(step4: dict[str, Any], key: str) -> tuple[str, ...]:
    path = PROJECT_ROOT / str(step4["datasets"]["lfw"][key])
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _diagnostic_signature(payload: dict[str, object]) -> dict[str, object]:
    splits = dict(payload["splits"])
    calibration = dict(splits["calibration"])
    test = dict(splits["test"])
    assessment = dict(payload["calibration_transfer_assessment"])
    return {
        "protocol_uid": str(payload["protocol_uid"]),
        "threshold_selection": str(payload["threshold_selection"]),
        "target_fpir": float(payload["target_fpir"]),
        "decision_threshold": float(payload["origin_decision_threshold"]),
        "calibration_non_mated_count": int(calibration["non_mated_count"]),
        "calibration_false_accept_count": int(
            calibration["origin_false_accept_count"]
        ),
        "calibration_fpir": float(calibration["origin_fpir"]),
        "test_non_mated_count": int(test["non_mated_count"]),
        "test_false_accept_count": int(test["origin_false_accept_count"]),
        "test_fpir": float(test["origin_fpir"]),
        "status": str(assessment["status"]),
    }


def _validate_compact_frames(
    compression: pd.DataFrame,
    retrieval: pd.DataFrame,
    *,
    family: str,
    expected_profiles: int,
    target_fpirs: tuple[float, ...],
) -> None:
    if len(compression) != expected_profiles:
        raise ValueError(
            f"{family} compression summary row mismatch: "
            f"{len(compression)} != {expected_profiles}"
        )
    expected_retrieval_rows = (
        expected_profiles
        * (4 if family == "pca" else 3)
        * len(target_fpirs)
    )
    if len(retrieval) != expected_retrieval_rows:
        raise ValueError(
            f"{family} retrieval summary row mismatch: "
            f"{len(retrieval)} != {expected_retrieval_rows}"
        )
    if not compression["compression_family"].eq(family).all():
        raise ValueError("compression family escaped the requested family")
    if not retrieval["compression_family"].eq(family).all():
        raise ValueError("retrieval family escaped the requested family")
    observed_targets = set(
        pd.to_numeric(retrieval["target_fpir"], errors="raise").astype(float)
    )
    if observed_targets != set(target_fpirs):
        raise ValueError(
            f"{family} target FPIR coverage mismatch: "
            f"{sorted(observed_targets)} != {sorted(target_fpirs)}"
        )
    missing_confidence = sorted(
        set(REQUIRED_OPEN_SET_CONFIDENCE_COLUMNS).difference(retrieval.columns)
    )
    if missing_confidence:
        raise ValueError(
            f"{family} retrieval summary is missing confidence fields: "
            f"{missing_confidence}"
        )
    if retrieval["confidence_interval_unit"].ne("probe").any():
        raise ValueError("open-set confidence intervals must use probe units")
    if retrieval["rate_confidence_interval_method"].ne(
        "wilson_score"
    ).any():
        raise ValueError("open-set rate intervals must use Wilson score")
    if retrieval["difference_confidence_interval_method"].ne(
        "paired_nonparametric_bootstrap_percentile"
    ).any():
        raise ValueError("origin deltas must use the paired bootstrap contract")
    if (
        pd.to_numeric(
            retrieval["difference_confidence_interval_resamples"],
            errors="raise",
        )
        <= 0
    ).any():
        raise ValueError("paired-bootstrap resamples must be positive")
    for source in ("origin", "compressed"):
        realized = pd.to_numeric(
            retrieval[f"{source}_realized_fpir"], errors="raise"
        )
        fpir = pd.to_numeric(retrieval[f"{source}_fpir"], errors="raise")
        if not np.allclose(realized, fpir, rtol=0.0, atol=1e-12):
            raise ValueError(f"{source} realized FPIR alias drifted")
    for metric in ("dir_rank1", "fpir"):
        origin = pd.to_numeric(retrieval[f"origin_{metric}"], errors="raise")
        compressed = pd.to_numeric(
            retrieval[f"compressed_{metric}"], errors="raise"
        )
        delta = pd.to_numeric(
            retrieval[f"compressed_minus_origin_{metric}"], errors="raise"
        )
        if not np.allclose(
            delta,
            compressed - origin,
            rtol=0.0,
            atol=COMPACT_RATE_DELTA_ROUNDTRIP_ATOL,
        ):
            raise ValueError(f"{metric} compressed-minus-origin delta drifted")
        for source, rate in (("origin", origin), ("compressed", compressed)):
            low = pd.to_numeric(
                retrieval[f"{source}_{metric}_wilson95_low"],
                errors="raise",
            )
            high = pd.to_numeric(
                retrieval[f"{source}_{metric}_wilson95_high"],
                errors="raise",
            )
            if ((low > rate) | (rate > high)).any():
                raise ValueError(f"{source} {metric} escaped its Wilson interval")
        delta_low = pd.to_numeric(
            retrieval[
                f"compressed_minus_origin_{metric}_paired_bootstrap95_low"
            ],
            errors="raise",
        )
        delta_high = pd.to_numeric(
            retrieval[
                f"compressed_minus_origin_{metric}_paired_bootstrap95_high"
            ],
            errors="raise",
        )
        if (delta_low > delta_high).any():
            raise ValueError(f"{metric} paired-bootstrap interval is inverted")
    if not retrieval["top_k"].astype(int).eq(20).all():
        raise ValueError("TPIR20 compact artifacts require top_k=20")
    if not retrieval["tpir_rank"].astype(int).eq(20).all():
        raise ValueError("TPIR rank metadata must equal 20")
    origin_tpir = pd.to_numeric(
        retrieval["origin_tpir_at_rank_k"], errors="raise"
    )
    compressed_tpir = pd.to_numeric(
        retrieval["compressed_tpir_at_rank_k"], errors="raise"
    )
    tpir_delta = pd.to_numeric(
        retrieval["compressed_minus_origin_tpir_at_rank_k"], errors="raise"
    )
    if not np.allclose(
        tpir_delta,
        compressed_tpir - origin_tpir,
        rtol=0.0,
        atol=COMPACT_RATE_DELTA_ROUNDTRIP_ATOL,
    ):
        raise ValueError("TPIR20 compressed-minus-origin delta drifted")
    for source, rate in (
        ("origin", origin_tpir),
        ("compressed", compressed_tpir),
    ):
        low = pd.to_numeric(
            retrieval[f"{source}_tpir_at_rank_k_wilson95_low"],
            errors="raise",
        )
        high = pd.to_numeric(
            retrieval[f"{source}_tpir_at_rank_k_wilson95_high"],
            errors="raise",
        )
        if ((low > rate) | (rate > high)).any():
            raise ValueError(f"{source} TPIR20 escaped its Wilson interval")
    for alias, canonical in (
        ("origin_tpir20", origin_tpir),
        ("compressed_tpir20", compressed_tpir),
    ):
        observed = pd.to_numeric(retrieval[alias], errors="raise")
        if not np.allclose(observed, canonical, rtol=0.0, atol=1e-12):
            raise ValueError(f"{alias} alias drifted")
    if not rate_ratio_matches_counts_or_compact_csv(
        retrieval["compressed_tpir20_retention"],
        reference_successes=retrieval["origin_tpir20_count"],
        reference_totals=retrieval["origin_tpir20_denominator"],
        candidate_successes=retrieval["compressed_tpir20_count"],
        candidate_totals=retrieval["compressed_tpir20_denominator"],
    ):
        raise ValueError("TPIR20 retention drifted")
    crossing = retrieval["threshold_crossing_count"].astype(int)
    directional = (
        retrieval["accept_to_reject_count"].astype(int)
        + retrieval["reject_to_accept_count"].astype(int)
    )
    if not crossing.equals(directional):
        raise ValueError("directional threshold crossing counts do not reconcile")
    if family == "pca":
        expected_modes = {"pca_direct_cosine", "pca_reconstruction_cosine"}
        if set(retrieval["search_mode"].astype(str)) != expected_modes:
            raise ValueError("PCA search modes are incomplete")
    else:
        expected_modes = {"pq_reconstruction_cosine", "pq_adc_exhaustive"}
        if set(retrieval["search_mode"].astype(str)) != expected_modes:
            raise ValueError("PQ search modes are incomplete")
        adc = retrieval["search_mode"].eq("pq_adc_exhaustive")
        if retrieval.loc[adc, "threshold_policy"].ne(
            "recalibrated_compressed"
        ).any():
            raise ValueError("PQ ADC must use recalibrated-compressed only")
        if retrieval.loc[adc, "score_spaces_comparable"].astype(bool).any():
            raise ValueError("PQ ADC cannot be cosine-score comparable")


def _run_family(
    context: dict[str, Any],
    *,
    family: str,
    output_root: Path,
    target_fpirs: tuple[float, ...],
) -> dict[str, object]:
    family_dir = output_root / family
    if family_dir.exists():
        manifest_path = family_dir / "family_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(
                f"incomplete family output exists; inspect manually: {family_dir}"
            )
        existing = _read_json(manifest_path)
        if existing.get("artifact_type") != FAMILY_ARTIFACT_TYPE:
            raise ValueError(f"unexpected family artifact: {manifest_path}")
        if tuple(float(value) for value in existing.get("target_fpirs", [])) != (
            target_fpirs
        ):
            raise ValueError("existing family artifact uses different FPIR targets")
        return {"family": family, "status": "already_completed"}

    step4 = context["step4"]
    compression_config = step4["compression"]["families"]
    pca_dimensions = tuple(
        int(value) for value in compression_config["pca"]["dimensions"]
    )
    pq_settings = tuple(
        (int(item["m"]), int(item["nbits"]))
        for item in compression_config["pq"]["settings"]
    )
    requested_pca = pca_dimensions if family == "pca" else ()
    requested_pq = pq_settings if family == "pq" else ()
    expected_profiles = len(requested_pca or requested_pq)
    prepared = read_prepared_population_artifact(context["prepared_dir"])
    selected = pd.read_csv(context["selected_path"])
    evaluation = step4["evaluation"]
    execution = step4["execution"]
    common = {
        "pca_dimensions": requested_pca,
        "pq_settings": requested_pq,
        "seed": int(execution["seed"]),
        "top_k": int(evaluation["top_k"]),
        "progress": _progress,
        "target_fpirs": target_fpirs,
    }
    if context["dataset_id"] == "survface":
        result = characterize_step2_survface_compression(
            prepared,
            selected,
            target_fpir=float(evaluation["survface_target_fpir"]),
            calibration_gallery_identities=3000,
            **common,
        )
    elif context["dataset_id"] == "rfw_custom":
        result = characterize_step2_rfw_custom_compression(
            prepared,
            selected,
            target_fpir=float(evaluation["rfw_custom_target_fpir"]),
            **common,
        )
    else:
        result = characterize_step2_compression(
            prepared,
            selected,
            gallery_identities=_read_lfw_ids(
                step4, "gallery_identities_path"
            ),
            unknown_unknown_identities=_read_lfw_ids(
                step4, "unknown_unknown_identities_path"
            ),
            target_fpir=float(evaluation["target_fpir"]),
            enrollment_count=int(evaluation["lfw_enrollment_count"]),
            calibration_gallery_identities=int(
                evaluation["calibration_gallery_identities"]
            ),
            **common,
        )
    lineage = {
        "extraction_uid": prepared.extraction_uid,
        "dataset_id": prepared.dataset_id,
        "model_uid": prepared.model_uid,
        "origin_embedding_artifact_uid": prepared.origin_embedding_artifact_uid,
    }
    paired = annotate_compression_lineage(result.paired_metrics, **lineage)
    retrieval_rows = annotate_compression_lineage(result.retrieval_metrics, **lineage)
    compression_summary, paired_count = summarize_compression(
        None,
        chunksize=max(1, len(paired)),
        source_frame=paired,
    )
    retrieval_summary, retrieval_count = summarize_retrieval(
        None,
        chunksize=max(1, len(retrieval_rows)),
        source_frame=retrieval_rows,
    )
    for frame in (compression_summary, retrieval_summary):
        frame.loc[:, "model_uid"] = context["model_uid"]
        frame.loc[:, "run_id"] = context["run_id"]
    _validate_compact_frames(
        compression_summary,
        retrieval_summary,
        family=family,
        expected_profiles=expected_profiles,
        target_fpirs=target_fpirs,
    )

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{family}.tmp-",
            dir=str(output_root),
        )
    )
    compression_path = temporary / "compression_summary.csv"
    retrieval_path = temporary / "retrieval_summary.csv"
    diagnostics_path = temporary / "origin_calibration_diagnostics.json"
    _write_csv(compression_path, compression_summary)
    _write_csv(retrieval_path, retrieval_summary)
    _write_json(
        diagnostics_path,
        {
            "schema_version": 1,
            "artifact_type": "origin_open_set_multi_fpir_diagnostics",
            "target_fpirs": list(target_fpirs),
            "diagnostics_by_target": result.calibration_diagnostics_by_target,
        },
    )
    family_manifest_path = temporary / "family_manifest.json"
    family_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": FAMILY_ARTIFACT_TYPE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "dataset_id": context["dataset_id"],
        "model_uid": context["model_uid"],
        "source_run_id": context["run_id"],
        "target_fpirs": list(target_fpirs),
        "source_origin_embedding_artifact_uid": prepared.origin_embedding_artifact_uid,
        "profile_count": expected_profiles,
        "source_row_counts": {
            "selected_samples": int(len(selected)),
            "paired_rows": int(paired_count),
            "retrieval_rows": int(retrieval_count),
        },
        "summary_row_counts": {
            "compression": int(len(compression_summary)),
            "retrieval": int(len(retrieval_summary)),
        },
        "origin_calibration_signatures": {
            target: _diagnostic_signature(dict(payload))
            for target, payload in result.calibration_diagnostics_by_target.items()
        },
        "outputs": {
            "compression_summary.csv": _named_file_entry(
                compression_path, name="compression_summary.csv"
            ),
            "retrieval_summary.csv": _named_file_entry(
                retrieval_path, name="retrieval_summary.csv"
            ),
            "origin_calibration_diagnostics.json": _named_file_entry(
                diagnostics_path, name="origin_calibration_diagnostics.json"
            ),
        },
    }
    _write_json(family_manifest_path, family_manifest)
    os.replace(temporary, family_dir)
    del result, paired, retrieval_rows, compression_summary, retrieval_summary
    del prepared, selected
    gc.collect()
    return {"family": family, "status": "completed", "path": str(family_dir)}


def _verified_family(output_root: Path, family: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    family_dir = output_root / family
    manifest_path = family_dir / "family_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_type") != FAMILY_ARTIFACT_TYPE:
        raise ValueError(f"unexpected family artifact: {manifest_path}")
    for name, entry in dict(manifest["outputs"]).items():
        path = family_dir / name
        if not path.is_file() or _sha256_file(path) != str(dict(entry)["sha256"]):
            raise ValueError(f"family output hash mismatch: {path}")
    return (
        pd.read_csv(family_dir / "compression_summary.csv"),
        pd.read_csv(family_dir / "retrieval_summary.csv"),
        manifest,
    )


def _legacy_consistency(
    context: dict[str, Any],
    compression: pd.DataFrame,
    retrieval: pd.DataFrame,
) -> dict[str, object]:
    legacy_dir = (
        PROJECT_ROOT
        / "results"
        / "paper"
        / context["dataset_id"]
        / context["run_id"]
        / "summaries"
    )
    legacy_compression_path = legacy_dir / "compression_summary.csv"
    legacy_retrieval_path = legacy_dir / "retrieval_summary.csv"
    if not legacy_compression_path.is_file() or not legacy_retrieval_path.is_file():
        return {"status": "not_available"}
    legacy_compression = pd.read_csv(legacy_compression_path)
    legacy_retrieval = pd.read_csv(legacy_retrieval_path)
    if "target_fpir" in retrieval:
        source_evaluation = context["step4"]["evaluation"]
        primary_target = float(
            source_evaluation[
                "survface_target_fpir"
                if context["dataset_id"] == "survface"
                else "rfw_custom_target_fpir"
                if context["dataset_id"] == "rfw_custom"
                else "target_fpir"
            ]
        )
        retrieval = retrieval.loc[
            np.isclose(
                pd.to_numeric(retrieval["target_fpir"], errors="raise"),
                primary_target,
                rtol=0.0,
                atol=1e-12,
            )
        ].copy()
    if "search_mode" not in legacy_retrieval:
        legacy_retrieval["search_mode"] = legacy_retrieval[
            "compression_family"
        ].map(
            {
                "pca": "pca_direct_cosine",
                "pq": "pq_reconstruction_cosine",
            }
        )
    compression_key = ["compression_family", "compression_profile"]
    retrieval_key = [
        "compression_family",
        "compression_profile",
        "search_mode",
        "threshold_policy",
    ]
    shared_modes = {"pca_direct_cosine", "pq_reconstruction_cosine"}
    current_shared = retrieval.loc[
        retrieval["search_mode"].astype(str).isin(shared_modes)
    ]
    compression_join = legacy_compression.merge(
        compression,
        on=compression_key,
        suffixes=("_legacy", "_refresh"),
        validate="one_to_one",
    )
    retrieval_join = legacy_retrieval.merge(
        current_shared,
        on=retrieval_key,
        suffixes=("_legacy", "_refresh"),
        validate="one_to_one",
    )
    if len(compression_join) != len(legacy_compression):
        raise ValueError("legacy compression profiles are not fully reproduced")
    if len(retrieval_join) != len(legacy_retrieval):
        raise ValueError("legacy retrieval strata are not fully reproduced")
    compression_metrics = (
        "mean_reconstruction_mse",
        "p95_reconstruction_mse",
        "mean_angular_error_rad",
        "p95_angular_error_rad",
        "mean_cosine_to_origin",
        "p05_cosine_to_origin",
    )
    retrieval_metrics = (
        "origin_decision_threshold",
        "compressed_decision_threshold",
        "origin_rank1_rate",
        "compressed_rank1_rate",
        "origin_top_k_rate",
        "compressed_top_k_rate",
        "origin_dir_rank1",
        "compressed_dir_rank1",
        "origin_fpir",
        "compressed_fpir",
        "agreement_with_origin_rate",
        "threshold_crossing_rate",
    )
    maximum_absolute_difference = 0.0
    for joined, metrics in (
        (compression_join, compression_metrics),
        (retrieval_join, retrieval_metrics),
    ):
        for metric in metrics:
            left = pd.to_numeric(
                joined[f"{metric}_legacy"], errors="coerce"
            ).to_numpy(dtype=float)
            right = pd.to_numeric(
                joined[f"{metric}_refresh"], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.array_equal(pd.isna(left), pd.isna(right)):
                raise ValueError(f"legacy missing-value pattern changed: {metric}")
            finite = ~pd.isna(left)
            difference = (
                float(abs(left[finite] - right[finite]).max())
                if finite.any()
                else 0.0
            )
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                difference,
            )
            if difference > 1e-10:
                raise ValueError(
                    f"legacy shared-mode metric changed: {metric} diff={difference}"
                )
    return {
        "status": "matched",
        "compression_rows": int(len(compression_join)),
        "retrieval_rows": int(len(retrieval_join)),
        "maximum_absolute_difference": maximum_absolute_difference,
        "legacy_files": {
            "compression_summary.csv": _file_entry(
                legacy_compression_path, root=PROJECT_ROOT
            ),
            "retrieval_summary.csv": _file_entry(
                legacy_retrieval_path, root=PROJECT_ROOT
            ),
        },
    }


def _merge(context: dict[str, Any], output_root: Path) -> dict[str, object]:
    family_data = {
        family: _verified_family(output_root, family)
        for family in SUPPORTED_FAMILIES
    }
    compression = pd.concat(
        [family_data[family][0] for family in SUPPORTED_FAMILIES],
        ignore_index=True,
    ).sort_values(["compression_family", "compression_profile"])
    retrieval = pd.concat(
        [family_data[family][1] for family in SUPPORTED_FAMILIES],
        ignore_index=True,
    ).sort_values(
        [
            "compression_family",
            "compression_profile",
            "search_mode",
            "threshold_policy",
            "target_fpir",
        ]
    )
    if compression.duplicated(
        ["compression_family", "compression_profile"]
    ).any():
        raise ValueError("merged compression summary has duplicate keys")
    if retrieval.duplicated(
        [
            "compression_family",
            "compression_profile",
            "search_mode",
            "threshold_policy",
            "target_fpir",
        ]
    ).any():
        raise ValueError("merged retrieval summary has duplicate keys")
    signatures_by_target = [
        dict(family_data[family][2]["origin_calibration_signatures"])
        for family in SUPPORTED_FAMILIES
    ]
    if signatures_by_target[0] != signatures_by_target[1]:
        raise ValueError("PCA and PQ origin calibration signatures differ")
    target_fpirs = tuple(
        float(value) for value in family_data["pca"][2]["target_fpirs"]
    )
    _validate_compact_frames(
        compression.loc[compression["compression_family"].eq("pca")],
        retrieval.loc[retrieval["compression_family"].eq("pca")],
        family="pca",
        expected_profiles=int(
            family_data["pca"][2]["profile_count"]
        ),
        target_fpirs=target_fpirs,
    )
    legacy_consistency = (
        {
            "status": "not_applicable_protocol_changed",
            "reason": (
                "SurvFace 3,000-ID half-gallery calibration and watch-list "
                "compressor fit intentionally replace the legacy 200-ID regime"
            ),
        }
        if (
            context["dataset_id"] == "survface"
            and output_root.name in {
                "search_space_v3_matched_calibration",
                "search_space_v4_multi_fpir",
                "search_space_v5_tpir20_multi_fpir",
            }
        )
        else _legacy_consistency(context, compression, retrieval)
    )
    _validate_compact_frames(
        compression.loc[compression["compression_family"].eq("pq")],
        retrieval.loc[retrieval["compression_family"].eq("pq")],
        family="pq",
        expected_profiles=int(
            family_data["pq"][2]["profile_count"]
        ),
        target_fpirs=target_fpirs,
    )

    compression_path = output_root / "compression_summary.csv"
    retrieval_path = output_root / "retrieval_summary.csv"
    _write_csv(compression_path, compression.reset_index(drop=True))
    _write_csv(retrieval_path, retrieval.reset_index(drop=True))
    git = dict(context["evaluator_git"])
    source_scope = dict(context["config"].get("scope", {}))
    source_is_full_paper_run = bool(
        source_scope.get("is_full_dataset")
        and source_scope.get("is_paper_run")
        and not bool(context["run_manifest"].get("git", {}).get("dirty", True))
    )
    evaluator_git_clean = not bool(git.get("dirty", True))
    failed_transfer_targets = [
        target
        for target, signature in signatures_by_target[0].items()
        if str(signature["status"]) == "failed_target_fpir"
    ]
    manifest_path = output_root / "summary_manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": context["dataset_id"],
        "model_uid": context["model_uid"],
        "run_id": context["run_id"],
        "source_run_id": context["run_id"],
        "source_run_preserved_immutable": True,
        "compact_only": True,
        "row_level_search_space_ledgers_retained": False,
        "source_run_full_paper_eligible": source_is_full_paper_run,
        "evaluator_git_clean": evaluator_git_clean,
        "derived_evaluation_paper_eligible": bool(
            source_is_full_paper_run and evaluator_git_clean
        ),
        "claim_status": (
            "validation_required"
            if evaluator_git_clean
            else "exploratory_dirty_evaluator"
        ),
        "producer_script": "scripts/refresh_step4_search_spaces.py",
        "target_fpirs": list(target_fpirs),
        "origin_calibration_signatures": signatures_by_target[0],
        "legacy_shared_mode_consistency": legacy_consistency,
        "evaluator_git": git,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "source_files": {
            "run_manifest.json": _file_entry(
                context["run_manifest_path"], root=PROJECT_ROOT
            ),
            "freeze_manifest.json": _file_entry(
                context["freeze_manifest_path"], root=PROJECT_ROOT
            ),
            "prepared_population/manifest.json": _file_entry(
                context["prepared_manifest_path"], root=PROJECT_ROOT
            ),
            "selected_manifest.csv": _file_entry(
                context["selected_path"], root=PROJECT_ROOT
            ),
        },
        "family_manifests": {
            family: _file_entry(
                output_root / family / "family_manifest.json",
                root=PROJECT_ROOT,
            )
            for family in SUPPORTED_FAMILIES
        },
        "validated_counts": {
            "compression_summary_rows": int(len(compression)),
            "retrieval_summary_rows": int(len(retrieval)),
        },
        "summary_contract": {
            "compression_grain": "family x profile",
            "retrieval_grain": (
                "family x profile x search_mode x threshold_policy x target_fpir"
            ),
            "tpir20_definition": (
                "mated probe true identity is within top 20 and its score is "
                "greater than or equal to the calibration threshold"
            ),
            "tpir20_threshold_score": "true_identity_score_within_top20",
            "closed_set_rank20_definition": (
                "mated probe true identity is within top 20 without thresholding"
            ),
            "search_modes": sorted(retrieval["search_mode"].astype(str).unique()),
            "pq_adc_frozen_origin": "not_applicable",
            "threshold_crossing_directions": [
                "accept_to_reject",
                "reject_to_accept",
            ],
            "rate_confidence_intervals": (
                "probe-level two-sided 95% Wilson score intervals"
            ),
            "origin_difference_confidence_intervals": (
                "compressed minus origin probe-level paired nonparametric "
                "bootstrap percentile intervals"
            ),
            "latency": "single observation; repeated benchmark still required",
        },
        "limitations": [
            "No row-level derived search-space ledger is retained by this compact refresh.",
            "Confidence intervals resample probes and do not represent identity-cluster or checkpoint-training uncertainty.",
            *(
                [
                    "SurvFace threshold-dependent claims remain blocked for "
                    "targets whose held-out realized FPIR exceeds the calibrated "
                    f"target: {failed_transfer_targets}."
                ]
                if context["dataset_id"] == "survface"
                and failed_transfer_targets
                else []
            ),
            *(
                [
                    "RFW-Custom uses a non-official 1:N identity split and must not be reported as RFW Official.",
                    "Checkpoint training identity overlap with RFW is UNKNOWN; this is not strict unseen-identity evidence.",
                ]
                if context["dataset_id"] == "rfw_custom"
                else []
            ),
            "Latency is a one-shot wall-clock observation and is not a stable systems benchmark.",
        ],
        "output_files": {
            "compression_summary.csv": _file_entry(
                compression_path, root=PROJECT_ROOT
            ),
            "retrieval_summary.csv": _file_entry(
                retrieval_path, root=PROJECT_ROOT
            ),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def _verified_merged_manifest(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "summary_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(f"unexpected merged artifact: {manifest_path}")
    for name, entry in dict(manifest["output_files"]).items():
        path = output_root / name
        metadata = dict(entry)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(metadata["bytes"]):
            raise ValueError(f"merged output byte-size mismatch: {path}")
        if _sha256_file(path) != str(metadata["sha256"]):
            raise ValueError(f"merged output hash mismatch: {path}")
    for family, entry in dict(manifest["family_manifests"]).items():
        path = output_root / str(family) / "family_manifest.json"
        metadata = dict(entry)
        if not path.is_file() or _sha256_file(path) != str(metadata["sha256"]):
            raise ValueError(f"merged family manifest hash mismatch: {path}")
    return manifest


def refresh(
    run_dir: Path,
    *,
    output_dir: Path | None,
    families: tuple[str, ...],
    target_fpirs: tuple[float, ...] = DEFAULT_TARGET_FPIRS,
) -> dict[str, object]:
    normalized_targets = tuple(dict.fromkeys(float(value) for value in target_fpirs))
    if not normalized_targets or any(
        not 0.0 <= value <= 1.0 for value in normalized_targets
    ):
        raise ValueError("target_fpirs must contain values between 0 and 1")
    context = _source_context(run_dir)
    context["evaluator_git"] = inspect_git_provenance(
        PROJECT_ROOT,
        run_root=PROJECT_ROOT / "runs",
    )
    output_root = (
        output_dir.resolve()
        if output_dir is not None
        else PROJECT_ROOT
        / "results"
        / "paper"
        / context["dataset_id"]
        / context["run_id"]
        / "search_space_v5_tpir20_multi_fpir"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    merged_manifest_path = output_root / "summary_manifest.json"
    if merged_manifest_path.is_file():
        existing = _verified_merged_manifest(output_root)
        if tuple(float(value) for value in existing.get("target_fpirs", [])) != (
            normalized_targets
        ):
            raise ValueError("existing merged artifact uses different FPIR targets")
        return {
            "output_dir": str(output_root),
            "families": [
                {"family": family, "status": "already_completed"}
                for family in families
            ],
            "merged": existing,
            "status": "already_completed",
        }
    family_results = [
        _run_family(
            context,
            family=family,
            output_root=output_root,
            target_fpirs=normalized_targets,
        )
        for family in families
    ]
    merged: dict[str, object] | None = None
    if all((output_root / family / "family_manifest.json").is_file() for family in SUPPORTED_FAMILIES):
        merged = _merge(context, output_root)
    return {
        "output_dir": str(output_root),
        "families": family_results,
        "merged": merged,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate PCA direct/reconstruction and PQ reconstruction/ADC "
            "from an immutable completed Step 4 population."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=SUPPORTED_FAMILIES,
        default=list(SUPPORTED_FAMILIES),
    )
    parser.add_argument(
        "--target-fpir",
        action="append",
        type=float,
        dest="target_fpirs",
        help=(
            "Repeat for each calibrated operating point. Defaults to "
            "0.01, 0.05, 0.10, 0.20, and 0.30."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    families = tuple(dict.fromkeys(str(value) for value in args.families))
    result = refresh(
        args.run_dir,
        output_dir=args.output_dir,
        families=families,
        target_fpirs=(
            tuple(args.target_fpirs)
            if args.target_fpirs
            else DEFAULT_TARGET_FPIRS
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
