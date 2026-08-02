from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.experiments import (  # noqa: E402
    validate_survface_sidak_calibration_holdout,
)
from research.explainability.gradcam import (  # noqa: E402
    read_prepared_population_artifact,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Sidak gallery calibration on SurvFace calibration folds."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-fpir", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "completed" or not (run_dir / "COMPLETED").is_file():
        raise ValueError("source run must be completed")
    if run_manifest["config"].get("dataset_id") != "survface":
        raise ValueError("source run must be SurvFace")
    workflow = run_manifest["config"]["step4"]["workflow"]
    workflow_root = run_dir / workflow["artifact_subdir"]
    selected_path = workflow_root / workflow["selected_manifest_path"]
    prepared_dir = workflow_root / workflow["prepared_population_dir"]
    selected = pd.read_csv(selected_path, low_memory=False)
    prepared = read_prepared_population_artifact(prepared_dir)

    result = validate_survface_sidak_calibration_holdout(
        prepared,
        selected,
        target_fpir=args.target_fpir,
        seed=args.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    fold_path = output_dir / "sidak_holdout_fold_results.csv"
    summary_path = output_dir / "sidak_holdout_summary.csv"
    result.fold_results.to_csv(fold_path, index=False)
    result.summary.to_csv(summary_path, index=False)
    manifest = {
        "schema_version": 2,
        "artifact_type": "survface_sidak_calibration_holdout",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_id": str(run_manifest["run_id"]),
        "source_run_manifest_sha256": _sha256_file(run_manifest_path),
        "selected_manifest_sha256": _sha256_file(selected_path),
        "prepared_population_manifest_sha256": _sha256_file(
            prepared_dir / "manifest.json"
        ),
        "dataset_id": str(prepared.dataset_id),
        "model_uid": str(prepared.model_uid),
        "target_fpir": float(args.target_fpir),
        "seed": int(args.seed),
        "conditions": [
            {
                "fit_gallery_identity_count": fit_count,
                "validation_gallery_identity_count": validation_count,
                "enrollment_count": enrollment_count,
            }
            for fit_count, validation_count, enrollment_count in (
                (50, 100, 1),
                (100, 200, 1),
                (50, 100, 5),
                (100, 200, 5),
                (50, 100, 20),
            )
        ],
        "test_data_used": False,
        "methods": [
            "empirical_sidak_pair_tail_v1",
            "empirical_effective_gallery_pair_tail_v1",
        ],
        "outputs": {
            fold_path.name: _sha256_file(fold_path),
            summary_path.name: _sha256_file(summary_path),
        },
    }
    manifest_path = output_dir / "validation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(result.summary.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
