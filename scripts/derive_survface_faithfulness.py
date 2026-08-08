from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.embeddings import (  # noqa: E402
    create_pytorch_adapter_from_spec,
    select_model_spec_by_profile,
)
from research.evaluation import (  # noqa: E402
    select_stratified_faithfulness_sample,
    summarize_faithfulness,
)
from research.explainability.gradcam import (  # noqa: E402
    measure_population_faithfulness,
    read_population_heatmap_subset,
    read_prepared_population_template_subset,
)
from research.runtime import inspect_git_provenance  # noqa: E402


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "survface_gradcam_faithfulness"


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


def _source_context(run_dir: Path) -> dict[str, Any]:
    source = run_dir.resolve()
    run_manifest_path = source / "run_manifest.json"
    completed_path = source / "COMPLETED"
    if not run_manifest_path.is_file() or not completed_path.is_file():
        raise FileNotFoundError("source run must contain run_manifest.json and COMPLETED")
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("status") != "completed":
        raise ValueError("source run must be completed")
    config = run_manifest.get("config")
    if not isinstance(config, dict) or str(config.get("dataset_id")) != "survface":
        raise ValueError("source run must be a frozen SurvFace Step 4 run")
    step4 = config.get("step4")
    if not isinstance(step4, dict):
        raise ValueError("source run manifest is missing frozen Step 4 config")
    workflow = step4["workflow"]
    workflow_root = source / str(workflow["artifact_subdir"])
    paths = {
        "selected": workflow_root / str(workflow["selected_manifest_path"]),
        "prepared": workflow_root / str(workflow["prepared_population_dir"]),
        "saliency": workflow_root / str(workflow["saliency_population_dir"]),
        "freeze": workflow_root / str(workflow["freeze_manifest_path"]),
    }
    for path in (
        paths["selected"],
        paths["prepared"] / "manifest.json",
        paths["saliency"] / "manifest.json",
        paths["freeze"],
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    freeze = _read_json(paths["freeze"])
    if freeze.get("fallback_free") is not True:
        raise ValueError("source run must declare fallback_free=true")
    if _sha256_file(paths["selected"]) != str(freeze["selected_manifest_sha256"]):
        raise ValueError("selected manifest differs from frozen SHA-256")
    prepared_manifest = _read_json(paths["prepared"] / "manifest.json")
    saliency_manifest = _read_json(paths["saliency"] / "manifest.json")
    model_uid = str(config["model_uid"])
    for name, payload in (
        ("prepared", prepared_manifest),
        ("saliency", saliency_manifest),
    ):
        if str(payload.get("dataset_id")) != "survface":
            raise ValueError(f"{name} artifact dataset differs from SurvFace")
        if str(payload.get("model_uid")) != model_uid:
            raise ValueError(f"{name} artifact model differs from source run")
    if (
        str(prepared_manifest.get("origin_embedding_artifact_uid"))
        != str(saliency_manifest.get("origin_embedding_artifact_uid"))
    ):
        raise ValueError("prepared and saliency origin lineages differ")
    return {
        "run_dir": source,
        "run_manifest_path": run_manifest_path,
        "run_manifest": run_manifest,
        "config": config,
        "step4": step4,
        "run_id": str(run_manifest["run_id"]),
        "model_uid": model_uid,
        "workflow_root": workflow_root,
        "selected_path": paths["selected"],
        "prepared_dir": paths["prepared"],
        "saliency_dir": paths["saliency"],
        "freeze_path": paths["freeze"],
        "freeze": freeze,
        "prepared_manifest": prepared_manifest,
        "saliency_manifest": saliency_manifest,
    }


def _read_candidates(context: dict[str, Any]) -> pd.DataFrame:
    selected = pd.read_csv(
        context["selected_path"],
        usecols=[
            "sample_id",
            "identity_id",
            "protocol_role",
            "aligned_face_index",
        ],
        low_memory=False,
    )
    saliency_manifest = context["saliency_manifest"]
    feature_entry = dict(saliency_manifest["saliency_features"])
    feature_path = context["saliency_dir"] / str(feature_entry["path"])
    if _sha256_file(feature_path) != str(feature_entry["sha256"]):
        raise ValueError("saliency feature CSV differs from artifact SHA-256")
    features = pd.read_csv(
        feature_path,
        usecols=[
            "sample_id",
            "raw_embedding_norm",
            "gradcam_target_score",
            "gradcam_valid_heatmap",
            "heatmap_available",
            "heatmap_index",
        ],
        low_memory=False,
    )
    if len(selected) != int(context["prepared_manifest"]["row_count"]):
        raise ValueError("selected and prepared row counts differ")
    if not np.array_equal(
        selected["sample_id"].astype(str).to_numpy(),
        features["sample_id"].astype(str).to_numpy(),
    ):
        raise ValueError("selected and saliency feature sample order differs")
    selected["prepared_row_index"] = np.arange(len(selected), dtype=np.int64)
    frame = selected.merge(
        features,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    valid = (
        frame["heatmap_available"].astype(bool)
        & frame["gradcam_valid_heatmap"].fillna(False).astype(bool)
        & frame["protocol_role"].astype(str).isin(
            {"registered_probe", "unknown_unknown_probe"}
        )
    )
    candidates = frame.loc[valid].copy()
    if candidates.empty:
        raise ValueError("no valid SurvFace probe heatmaps are available")
    return candidates


def _verify_existing_output(output: Path) -> dict[str, Any] | None:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("existing output has an unexpected artifact type")
    for entry in manifest.get("outputs", []):
        path = output / str(entry["path"])
        if not path.is_file() or _sha256_file(path) != str(entry["sha256"]):
            raise ValueError(f"existing faithfulness output failed verification: {path}")
    return manifest


def derive_survface_faithfulness(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    maximum_samples: int = 2048,
    seed: int = 42,
    occlusion_fraction: float = 0.10,
    random_repeats: int = 5,
    batch_size: int = 32,
    chunk_size: int = 256,
    bootstrap_repeats: int = 1000,
) -> dict[str, Any]:
    context = _source_context(Path(run_dir))
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (
            PROJECT_ROOT
            / "results"
            / "paper"
            / "survface"
            / context["run_id"]
            / "faithfulness_v1"
        ).resolve()
    )
    existing = _verify_existing_output(output)
    if existing is not None:
        return existing
    provenance = inspect_git_provenance(
        PROJECT_ROOT,
        run_root=context["run_dir"],
    )

    candidates = _read_candidates(context)
    selected = select_stratified_faithfulness_sample(
        candidates,
        maximum_samples=int(maximum_samples),
        seed=int(seed),
    )
    sample_ids = selected["sample_id"].astype(str).to_numpy()
    heatmap_ids, heatmaps = read_population_heatmap_subset(
        context["saliency_dir"],
        selected["heatmap_index"].to_numpy(dtype=np.int64),
        expected_sample_ids=sample_ids,
    )
    prepared_ids, templates, prepared_scores = (
        read_prepared_population_template_subset(
            context["prepared_dir"],
            selected["prepared_row_index"].to_numpy(dtype=np.int64),
            expected_sample_ids=sample_ids,
        )
    )
    if not np.array_equal(heatmap_ids, prepared_ids):
        raise ValueError("heatmap and prepared subset sample IDs differ")
    feature_scores = selected["gradcam_target_score"].to_numpy(np.float64)
    maximum_score_difference = float(
        np.max(np.abs(feature_scores - prepared_scores.astype(np.float64)))
    )
    if maximum_score_difference > 1e-5:
        raise ValueError(
            "stored Grad-CAM and prepared target scores differ; "
            f"maximum absolute difference={maximum_score_difference:.6g}"
        )

    step4 = context["step4"]
    execution = step4["execution"]
    profile_id = str(execution["model_profile"])
    profile_config = step4["models"]["profiles"][profile_id]
    _, model_spec = select_model_spec_by_profile(
        PROJECT_ROOT / str(step4["models"]["registry_root"]),
        profile_id=profile_id,
        profile_config=profile_config,
        verify_checkpoint=True,
    )
    if model_spec.model_uid != context["model_uid"]:
        raise ValueError("verified model differs from source run")
    if model_spec.checkpoint.sha256 != str(context["freeze"]["checkpoint_sha256"]):
        raise ValueError("verified checkpoint differs from source freeze manifest")
    adapter = create_pytorch_adapter_from_spec(
        model_spec,
        device=str(execution["device"]),
    )
    aligned_path = (
        PROJECT_ROOT
        / str(step4["datasets"]["survface"]["aligned_bundle_dir"])
        / "aligned_faces.npy"
    )
    source_faces = np.load(aligned_path, mmap_mode="r", allow_pickle=False)
    aligned_indices = selected["aligned_face_index"].to_numpy(dtype=np.int64)

    parts: dict[str, list[np.ndarray]] = {}
    step = int(chunk_size)
    if step <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(selected), step):
        stop = min(start + step, len(selected))
        images = np.asarray(source_faces[aligned_indices[start:stop]])
        measured = measure_population_faithfulness(
            adapter,
            images,
            heatmaps[start:stop],
            templates[start:stop],
            sample_ids[start:stop],
            prepared_scores[start:stop].astype(np.float64),
            fraction=float(occlusion_fraction),
            random_repeats=int(random_repeats),
            seed=int(seed),
            batch_size=int(batch_size),
        )
        for name, values in measured.items():
            parts.setdefault(name, []).append(np.asarray(values))
        print(
            f"faithfulness forward passes: {stop}/{len(selected)}",
            flush=True,
        )
    for name, values in parts.items():
        selected[name] = np.concatenate(values, axis=0)
    summary = summarize_faithfulness(
        selected,
        bootstrap_repeats=int(bootstrap_repeats),
        seed=int(seed),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        rows_path = temporary / "faithfulness_rows.csv"
        summary_path = temporary / "faithfulness_summary.csv"
        selected.to_csv(
            rows_path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
            float_format="%.12g",
        )
        summary.to_csv(
            summary_path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
            float_format="%.12g",
        )
        manifest: dict[str, Any] = {
            "artifact_type": ARTIFACT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_id": "survface",
            "source_run_id": context["run_id"],
            "model_uid": context["model_uid"],
            "checkpoint_sha256": model_spec.checkpoint.sha256,
            "origin_embedding_artifact_uid": context["prepared_manifest"][
                "origin_embedding_artifact_uid"
            ],
            "saliency_spec_uid": context["saliency_manifest"]["saliency_spec_uid"],
            "saliency_target_name": context["saliency_manifest"]["target_name"],
            "claim_status": "exploratory_dirty_evaluator",
            "paper_eligible": False,
            "threshold_independent": True,
            "source_full_paper_run": bool(
                context["freeze"]["scope"]["is_paper_run"]
            ),
            "sampling": {
                "candidate_count": int(len(candidates)),
                "selected_count": int(len(selected)),
                "maximum_samples": int(maximum_samples),
                "seed": int(seed),
                "unit": "sample_id",
                "strata": [
                    "protocol_role",
                    "raw_embedding_norm_within_role_quartile",
                    "gradcam_target_score_within_role_quartile",
                ],
                "selected_sample_sha256": hashlib.sha256(
                    "\n".join(sample_ids).encode("utf-8")
                ).hexdigest(),
                "stratum_count": int(selected["faithfulness_stratum"].nunique()),
            },
            "occlusion": {
                "fraction": float(occlusion_fraction),
                "strategies": ["high_saliency", "low_saliency", "random"],
                "random_repeats": int(random_repeats),
                "random_seed_unit": "sample_id",
                "fill_value": "model_preprocessing_channel_mean",
            },
            "statistics": {
                "bootstrap_method": "identity_cluster",
                "bootstrap_repeats": int(bootstrap_repeats),
                "confidence_level": 0.95,
                "grouping": ["all", "protocol_role"],
            },
            "validation": {
                "maximum_stored_target_score_abs_difference": (
                    maximum_score_difference
                ),
                "source_completed": True,
                "source_fallback_free": True,
                "source_artifacts_immutable": True,
            },
            "evaluator_git": provenance,
            "sources": [
                _file_entry(context["run_manifest_path"], root=PROJECT_ROOT),
                _file_entry(context["freeze_path"], root=PROJECT_ROOT),
                _file_entry(context["selected_path"], root=PROJECT_ROOT),
                _file_entry(
                    context["prepared_dir"] / "manifest.json",
                    root=PROJECT_ROOT,
                ),
                _file_entry(
                    context["saliency_dir"] / "manifest.json",
                    root=PROJECT_ROOT,
                ),
            ],
            "outputs": [
                _file_entry(rows_path, root=temporary),
                _file_entry(summary_path, root=temporary),
            ],
            "limitations": [
                "This is a deterministic stratified subset, not all 182159 valid probe heatmaps.",
                "The source run is clean and full, but the derived evaluator is running from a dirty Step 6 worktree.",
                "Each artifact validates one model; matched-condition cross-model interpretation requires the ArcFace, AdaFace, and MagFace baseline trio and may add EdgeFace.",
                "No FPIR threshold or threshold-crossing metric is used because SurvFace calibration transfer failed.",
            ],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(output)
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive threshold-independent SurvFace Grad-CAM faithfulness.",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--maximum-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--occlusion-fraction", type=float, default=0.10)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = derive_survface_faithfulness(
        args.run_dir,
        output_dir=args.output_dir,
        maximum_samples=args.maximum_samples,
        seed=args.seed,
        occlusion_fraction=args.occlusion_fraction,
        random_repeats=args.random_repeats,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
