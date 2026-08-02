from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd


ARTIFACT_TYPE = "survface_gradcam_faithfulness_cross_model"
SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def summarize_models(
    artifact_dirs: list[str | Path],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    sources: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    contracts: list[dict[str, object]] = []
    for raw_directory in artifact_dirs:
        directory = Path(raw_directory).resolve()
        manifest_path = directory / "manifest.json"
        manifest = _read_json(manifest_path)
        if manifest.get("artifact_type") != "survface_gradcam_faithfulness":
            raise ValueError(f"unexpected source artifact: {directory}")
        if manifest.get("dataset_id") != "survface":
            raise ValueError("cross-model faithfulness sources must be SurvFace")
        output_entries = {str(entry["path"]): entry for entry in manifest["outputs"]}
        summary_path = directory / "faithfulness_summary.csv"
        summary_entry = output_entries.get("faithfulness_summary.csv")
        if summary_entry is None or _sha256_file(summary_path) != str(
            summary_entry["sha256"]
        ):
            raise ValueError(f"source summary hash mismatch: {summary_path}")
        model_uid = str(manifest["model_uid"])
        model_family = model_uid.split("-", maxsplit=1)[0]
        frame = pd.read_csv(summary_path)
        frame.insert(0, "source_run_id", str(manifest["source_run_id"]))
        frame.insert(0, "model_uid", model_uid)
        frame.insert(0, "model_family", model_family)
        frames.append(frame)
        contracts.append(
            {
                "selected_count": int(manifest["sampling"]["selected_count"]),
                "maximum_samples": int(manifest["sampling"]["maximum_samples"]),
                "strata": list(manifest["sampling"]["strata"]),
                "occlusion_fraction": float(manifest["occlusion"]["fraction"]),
                "strategies": list(manifest["occlusion"]["strategies"]),
                "random_repeats": int(manifest["occlusion"]["random_repeats"]),
                "bootstrap_repeats": int(
                    manifest["statistics"]["bootstrap_repeats"]
                ),
                "target_name": str(manifest["saliency_target_name"]),
            }
        )
        sources.append(
            {
                "model_family": model_family,
                "model_uid": model_uid,
                "source_run_id": str(manifest["source_run_id"]),
                "manifest": {
                    "path": str(manifest_path),
                    "bytes": int(manifest_path.stat().st_size),
                    "sha256": _sha256_file(manifest_path),
                },
                "summary": {
                    "path": str(summary_path),
                    "bytes": int(summary_path.stat().st_size),
                    "sha256": _sha256_file(summary_path),
                },
            }
        )
    if len(frames) != 3 or {frame["model_family"].iat[0] for frame in frames} != {
        "arcface",
        "adaface",
        "magface",
    }:
        raise ValueError("exactly one ArcFace, AdaFace, and MagFace artifact is required")
    canonical_contract = contracts[0]
    if any(contract != canonical_contract for contract in contracts[1:]):
        raise ValueError("faithfulness artifacts do not share one evaluation contract")

    combined = pd.concat(frames, ignore_index=True)
    primary = combined.loc[
        combined["group"].eq("all")
        & combined["metric"].isin(
            {
                "high_saliency_occlusion_score_drop",
                "low_saliency_occlusion_score_drop",
                "random_occlusion_score_drop",
                "faithfulness_gain_over_low_saliency",
                "faithfulness_gain_over_random",
            }
        )
    ].copy()
    primary = primary.pivot(
        index=["model_family", "model_uid", "source_run_id"],
        columns="metric",
        values=["mean", "mean_ci_lower", "mean_ci_upper", "positive_fraction"],
    )
    primary.columns = [f"{metric}__{stat}" for stat, metric in primary.columns]
    primary = primary.reset_index()
    primary["passes_high_over_low"] = (
        primary["faithfulness_gain_over_low_saliency__mean_ci_lower"] > 0.0
    )
    primary["passes_high_over_random"] = (
        primary["faithfulness_gain_over_random__mean_ci_lower"] > 0.0
    )
    primary["passes_strong_faithfulness"] = (
        primary["passes_high_over_low"] & primary["passes_high_over_random"]
    )

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    combined_path = temporary / "faithfulness_cross_model_long.csv"
    primary_path = temporary / "faithfulness_cross_model_primary.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8", lineterminator="\n")
    primary.to_csv(primary_path, index=False, encoding="utf-8", lineterminator="\n")
    manifest = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "survface",
        "model_families": ["arcface", "adaface", "magface"],
        "evaluation_contract": canonical_contract,
        "claim_status": "exploratory_dirty_evaluator",
        "paper_eligible": False,
        "threshold_independent": True,
        "strong_faithfulness_pass_count": int(
            primary["passes_strong_faithfulness"].sum()
        ),
        "sources": sources,
        "outputs": [
            _entry(combined_path, root=temporary),
            _entry(primary_path, root=temporary),
        ],
        "interpretation_boundary": (
            "High-saliency occlusion must exceed both low-saliency and random "
            "controls with a positive identity-cluster bootstrap CI to pass the "
            "strong faithfulness criterion."
        ),
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = summarize_models(args.artifact_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
