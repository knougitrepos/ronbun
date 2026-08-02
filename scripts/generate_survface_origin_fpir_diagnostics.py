from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.experiments import (  # noqa: E402
    diagnose_step2_survface_origin_calibration,
)
from research.explainability.gradcam import (  # noqa: E402
    read_prepared_population_artifact,
)


DEFAULT_CONDITIONS = (
    (100, 1),
    (200, 1),
    (500, 1),
    (1000, 1),
    (200, 5),
    (200, 20),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_condition(value: str) -> tuple[int, int]:
    pieces = str(value).split(":")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError(
            "condition must be GALLERY_IDENTITIES:ENROLLMENT_COUNT"
        )
    try:
        gallery_count, enrollment_count = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("condition counts must be integers") from exc
    if gallery_count <= 0 or enrollment_count <= 0:
        raise argparse.ArgumentTypeError("condition counts must be positive")
    return gallery_count, enrollment_count


def _read_run_manifest(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or not (run_dir / "COMPLETED").is_file():
        raise ValueError("source run must be immutable and completed")
    config = payload.get("config", {})
    if config.get("dataset_id") != "survface":
        raise ValueError("source run must be a SurvFace run")
    return manifest_path, payload


class _ProgressPrinter:
    def __init__(self) -> None:
        self._last_bucket: dict[str, int] = {}

    def __call__(self, message: str, details: dict[str, object]) -> None:
        processed = int(details.get("processed", 0))
        total = int(details.get("total", 0))
        if total <= 0:
            return
        bucket = min(10, int(processed * 10 / total))
        condition = str(details.get("calibration_condition_uid", "test"))
        key = f"{message}:{condition}"
        if self._last_bucket.get(key) == bucket:
            return
        self._last_bucket[key] = bucket
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] {message} "
            f"condition={condition} processed={processed}/{total}"
        )


def generate_diagnostics(
    *,
    run_dir: Path,
    output_dir: Path,
    conditions: tuple[tuple[int, int], ...],
    target_fpir: float,
    seed: int,
) -> dict[str, Any]:
    source_run_dir = run_dir.resolve()
    destination = output_dir.resolve()
    manifest_path, run_manifest = _read_run_manifest(source_run_dir)
    if destination.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new path: {destination}"
        )

    workflow_config = run_manifest["config"]["step4"]["workflow"]
    workflow_root = source_run_dir / workflow_config["artifact_subdir"]
    selected_path = workflow_root / workflow_config["selected_manifest_path"]
    prepared_dir = workflow_root / workflow_config["prepared_population_dir"]
    if not selected_path.is_file() or not prepared_dir.is_dir():
        raise FileNotFoundError("source Step 4 population artifacts are incomplete")

    selected = pd.read_csv(selected_path, low_memory=False)
    prepared = read_prepared_population_artifact(prepared_dir)
    result = diagnose_step2_survface_origin_calibration(
        prepared,
        selected,
        calibration_conditions=conditions,
        seed=seed,
        target_fpir=target_fpir,
        top_k=1,
        progress=_ProgressPrinter(),
    )

    destination.mkdir(parents=True, exist_ok=False)
    summary_path = destination / "origin_calibration_sweep_summary.csv"
    audit_path = destination / "origin_calibration_sweep_scores.csv.gz"
    diagnostics_path = destination / "origin_calibration_sweep_diagnostics.json"
    manifest_output_path = destination / "diagnostic_manifest.json"
    result.condition_summary.to_csv(summary_path, index=False)
    result.score_audit.to_csv(
        audit_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 1, "mtime": 0},
    )
    diagnostics_path.write_text(
        json.dumps(result.diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_hashes = {
        path.name: _sha256_file(path)
        for path in (summary_path, audit_path, diagnostics_path)
    }
    output_manifest: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": "survface_origin_fpir_diagnostic_bundle",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "run_id": str(run_manifest["run_id"]),
            "run_dir": source_run_dir.as_posix(),
            "run_manifest_sha256": _sha256_file(manifest_path),
            "selected_manifest_sha256": _sha256_file(selected_path),
            "prepared_population_manifest_sha256": _sha256_file(
                prepared_dir / "manifest.json"
            ),
            "dataset_id": str(prepared.dataset_id),
            "model_uid": str(prepared.model_uid),
            "extraction_uid": str(prepared.extraction_uid),
            "origin_embedding_artifact_uid": str(
                prepared.origin_embedding_artifact_uid
            ),
        },
        "parameters": {
            "conditions": [
                {
                    "gallery_identity_count": gallery_count,
                    "enrollment_count": enrollment_count,
                }
                for gallery_count, enrollment_count in conditions
            ],
            "target_fpir": float(target_fpir),
            "seed": int(seed),
            "test_threshold_recalibration": False,
        },
        "outputs": {
            "condition_summary": summary_path.name,
            "row_score_audit": audit_path.name,
            "diagnostics": diagnostics_path.name,
        },
        "output_sha256": output_hashes,
        "counts": {
            "condition_count": int(len(result.condition_summary)),
            "score_audit_rows": int(len(result.score_audit)),
        },
    }
    manifest_output_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an origin-only SurvFace calibration/test FPIR diagnostic "
            "bundle from one completed Step 4 run."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--condition",
        type=_parse_condition,
        action="append",
        dest="conditions",
        help="GALLERY_IDENTITIES:ENROLLMENT_COUNT; may be repeated",
    )
    parser.add_argument("--target-fpir", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    conditions = tuple(args.conditions or DEFAULT_CONDITIONS)
    manifest = generate_diagnostics(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        conditions=conditions,
        target_fpir=args.target_fpir,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
