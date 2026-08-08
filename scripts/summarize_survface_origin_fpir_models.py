from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY_COLUMNS = (
    "dataset_id",
    "model_uid",
    "protocol_uid",
    "calibration_condition_uid",
    "seed",
    "target_fpir",
    "calibration_gallery_identity_count",
    "calibration_enrollment_count",
    "origin_decision_threshold",
    "calibration_non_mated_count",
    "calibration_false_accept_count",
    "calibration_fpir",
    "calibration_fpir_wilson95_low",
    "calibration_fpir_wilson95_high",
    "matched_test_non_mated_count",
    "matched_test_false_accept_count",
    "matched_test_fpir",
    "matched_test_fpir_wilson95_low",
    "matched_test_fpir_wilson95_high",
    "test_gallery_template_count",
    "test_gallery_source_image_count",
    "test_non_mated_count",
    "test_false_accept_count",
    "test_fpir",
    "test_fpir_wilson95_low",
    "test_fpir_wilson95_high",
    "test_to_target_fpir_ratio",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_family(model_uid: str) -> str:
    family = str(model_uid).split("-", 1)[0].strip().lower()
    if family not in {"arcface", "adaface", "magface", "edgeface"}:
        raise ValueError(f"unsupported model family in UID: {model_uid!r}")
    return family


def summarize(
    diagnostic_dirs: tuple[Path, ...],
    output_dir: Path,
) -> dict[str, Any]:
    if len(diagnostic_dirs) < 2:
        raise ValueError("at least two diagnostic directories are required")
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")

    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    reference_parameters: dict[str, Any] | None = None
    seen_models: set[str] = set()
    for directory in diagnostic_dirs:
        source_dir = directory.resolve()
        manifest_path = source_dir / "diagnostic_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError(f"diagnostic bundle is not completed: {source_dir}")
        parameters = manifest["parameters"]
        comparable_parameters = {
            "conditions": parameters["conditions"],
            "target_fpir": parameters["target_fpir"],
            "seed": parameters["seed"],
            "test_threshold_recalibration": parameters[
                "test_threshold_recalibration"
            ],
        }
        if reference_parameters is None:
            reference_parameters = comparable_parameters
        elif comparable_parameters != reference_parameters:
            raise ValueError("diagnostic bundles use different audit parameters")

        summary_path = source_dir / manifest["outputs"]["condition_summary"]
        expected_hash = manifest["output_sha256"][summary_path.name]
        actual_hash = _sha256_file(summary_path)
        if actual_hash != expected_hash:
            raise ValueError(f"source summary hash mismatch: {summary_path}")
        frame = pd.read_csv(summary_path, usecols=list(SUMMARY_COLUMNS))
        model_uid = str(manifest["source"]["model_uid"])
        if model_uid in seen_models:
            raise ValueError(f"duplicate model UID: {model_uid}")
        seen_models.add(model_uid)
        if set(frame["model_uid"].astype(str)) != {model_uid}:
            raise ValueError("summary model UID differs from diagnostic manifest")
        frame.insert(1, "model_family", _model_family(model_uid))
        frame.insert(2, "source_run_id", str(manifest["source"]["run_id"]))
        frames.append(frame)
        sources.append(
            {
                "diagnostic_dir": source_dir.as_posix(),
                "diagnostic_manifest_sha256": _sha256_file(manifest_path),
                "condition_summary_sha256": actual_hash,
                "source_run_id": str(manifest["source"]["run_id"]),
                "model_uid": model_uid,
            }
        )

    result = pd.concat(frames, ignore_index=True).sort_values(
        [
            "calibration_gallery_identity_count",
            "calibration_enrollment_count",
            "model_family",
        ],
        kind="stable",
        ignore_index=True,
    )
    expected_rows = len(reference_parameters["conditions"]) * len(seen_models)
    if len(result) != expected_rows:
        raise ValueError(
            f"cross-model row count mismatch: {len(result)} != {expected_rows}"
        )

    destination.mkdir(parents=True, exist_ok=False)
    output_path = destination / "cross_model_origin_fpir_summary.csv"
    result.to_csv(output_path, index=False, float_format="%.12g")
    output_manifest = {
        "schema_version": 1,
        "artifact_type": "survface_cross_model_origin_fpir_summary",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": reference_parameters,
        "model_count": len(seen_models),
        "row_count": len(result),
        "sources": sources,
        "outputs": {
            output_path.name: {
                "rows": len(result),
                "sha256": _sha256_file(output_path),
            }
        },
    }
    manifest_output_path = destination / "cross_model_summary_manifest.json"
    manifest_output_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine comparable SurvFace origin FPIR diagnostic bundles."
    )
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        action="append",
        required=True,
        dest="diagnostic_dirs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(tuple(args.diagnostic_dirs), args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
