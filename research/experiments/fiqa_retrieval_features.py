"""Restartable top-2 ADC and frozen gallery reconstruction-error features."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from research.experiments.fiqa_threshold_calibration import (
    _completed_run, _frozen_pq_codec, _read_json,
)
from research.experiments.step2_compression import prepared_population_frame, open_set_protocol_arrays
from research.explainability.gradcam.artifacts import read_prepared_population_artifact
from research.protocols.open_set import (
    build_survface_matched_calibration_protocol, build_survface_official_protocol,
)
from research.runtime.hashing import canonical_sha256, sha256_file

FEATURE_COLUMNS = ("adc_s1", "adc_s2", "adc_margin", "top1_gallery_pq_distortion")


def adc_feature_frame(distances, indices, query_ids, gallery_ids, distortion):
    d, ix = np.asarray(distances), np.asarray(indices)
    if (d.shape != ix.shape or d.ndim != 2 or d.shape != (len(query_ids), 2)
            or not np.isfinite(d).all() or (d < -1e-5).any()
            or (np.diff(d, axis=1) < 0).any()
            or not np.issubdtype(ix.dtype, np.integer)
            or (ix < 0).any() or (ix >= len(gallery_ids)).any()
            or (ix[:, 0] == ix[:, 1]).any()):
        raise ValueError("invalid sorted top-2 ADC results")
    distortion = np.asarray(distortion, dtype=float)
    if (distortion.shape != (len(gallery_ids),) or not np.isfinite(distortion).all()
            or (distortion < 0).any()):
        raise ValueError("invalid gallery PQ distortion")
    return pd.DataFrame({
        "sample_id": np.asarray(query_ids, dtype=str),
        "adc_s1": -d[:, 0].astype(float), "adc_s2": -d[:, 1].astype(float),
        "adc_margin": d[:, 1].astype(float) - d[:, 0].astype(float),
        "top1_gallery_id": np.asarray(gallery_ids, dtype=str)[ix[:, 0]],
        "top1_gallery_pq_distortion": distortion[ix[:, 0]],
    })


def join_retrieval_features(rows, features, *, score_tolerance=1e-6):
    if (features.sample_id.duplicated().any() or rows.sample_id.duplicated().any()
            or set(features.sample_id) != set(rows.sample_id)):
        raise ValueError("retrieval features require exact one-to-one sample coverage")
    overlap = set(rows.columns) & (set(features.columns) - {"sample_id"})
    if overlap:
        raise ValueError(f"already attached feature columns: {sorted(overlap)}")
    out = rows.merge(features, on="sample_id", how="left", sort=False, validate="one_to_one")
    if not np.isfinite(out[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("retrieval features must be finite")
    if (not np.allclose(out.score, out.adc_s1, atol=score_tolerance, rtol=0)
            or (out.adc_s1 < out.adc_s2).any() or (out.adc_margin < 0).any()
            or (out.top1_gallery_pq_distortion < 0).any()
            or not np.allclose(out.adc_margin, out.adc_s1-out.adc_s2, atol=1e-12, rtol=0)):
        raise ValueError("ADC feature scores disagree with the frozen condition")
    if not np.array_equal(rows.sample_id.to_numpy(), out.sample_id.to_numpy()):
        raise ValueError("retrieval join changed query order")
    return out


def _atomic_json(path, value):
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    os.replace(partial, path)


def load_retrieval_features(directory, condition):
    root = Path(directory)
    manifest = _read_json(root / "manifest.json")
    spec = manifest["spec"]
    if (manifest.get("status") != "completed"
            or spec.get("artifact_type") != "fiqa_retrieval_features"
            or spec.get("condition_manifest_sha256") != canonical_sha256(condition.manifest)
            or manifest.get("feature_uid") != "fiqa-retrieval-" + canonical_sha256(spec)[:24]):
        raise ValueError("feature artifact condition lineage/status mismatch")
    frames = {}
    for split in ("calibration", "test"):
        parts = []
        for entry in manifest["shards"][split]:
            filename = entry["file"]
            if Path(filename).name != filename or sha256_file(root / filename) != entry["sha256"]:
                raise ValueError("feature shard hash/path mismatch")
            part = pd.read_parquet(root / filename)
            if len(part) != entry["rows"]:
                raise ValueError("feature shard row count mismatch")
            parts.append(part)
        frames[split] = pd.concat(parts, ignore_index=True)
        join_retrieval_features(getattr(condition, split), frames[split],
                                score_tolerance=spec["score_tolerance"])
    return {**frames, "manifest": manifest, "directory": root}


def build_retrieval_features(run_dir, condition, output_root, *, batch_size=8192,
                             score_tolerance=1e-6, progress=None):
    """Replay both galleries' top-2 ADC once; verify s1 against every saved query.

    Completed per-batch shards resume without another search. Gallery templates
    are decoded only to precompute ||g - decode(PQ(g))||_2 in codec input space;
    all retrieval remains ADC over PQ codes. No FIQA/FR inference or SDC occurs.
    """
    if (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
            or not np.isfinite(score_tolerance) or not 0 <= score_tolerance <= 1e-5):
        raise ValueError("invalid batch size or score tolerance")
    root, run, workflow = _completed_run(run_dir)
    cm = condition.manifest
    if (cm.get("status") != "completed" or cm.get("search_mode") != "pq_adc_exhaustive"
            or cm.get("source_run_id") != run["run_id"]
            or cm.get("source_run_manifest_sha256") != sha256_file(root / "run_manifest.json")):
        raise ValueError("feature replay requires the exact completed source condition")
    import faiss
    source_paths = [Path(__file__), Path(__file__).with_name("step2_compression.py"),
                    Path(__file__).with_name("fiqa_threshold_calibration.py"),
                    Path(__file__).parents[1] / "protocols/open_set.py",
                    Path(__file__).parents[1] / "compression/profiles.py"]
    spec = {"artifact_type": "fiqa_retrieval_features", "schema_version": 1,
            "condition_manifest_sha256": canonical_sha256(cm), "batch_size": batch_size,
            "score_tolerance": score_tolerance,
            "implementation_sha256": {p.name: sha256_file(p) for p in source_paths},
            "faiss_version": faiss.__version__,
            "retrieval_backend": "faiss_cpu_pq_adc", "top_k": 2,
            "distortion_definition": "l2(gallery_template - decoded_pq_template); no renormalization",
            "margin_definition": "s1 - s2; distinct gallery identity templates"}
    uid = "fiqa-retrieval-" + canonical_sha256(spec)[:24]
    destination = Path(output_root) / uid
    if (destination / "manifest.json").exists():
        return load_retrieval_features(destination, condition)
    destination.mkdir(parents=True, exist_ok=True)
    progress_path = destination / "progress.json"
    journal = _read_json(progress_path) if progress_path.exists() else {
        "spec": spec, "feature_uid": uid, "shards": {"calibration": [], "test": []}}
    if journal["spec"] != spec:
        raise ValueError("feature replay resume settings mismatch")
    expected_files = {
        workflow / "freeze_manifest.json": "source_freeze_manifest_sha256",
        workflow / "selected_manifest.csv": "selected_manifest_sha256",
        workflow / "prepared_population/manifest.json": "prepared_population_manifest_sha256",
    }
    for path, key in expected_files.items():
        if sha256_file(path) != cm[key]:
            raise ValueError(f"frozen source hash mismatch: {key}")
    selected = pd.read_csv(workflow / "selected_manifest.csv")
    prepared = read_prepared_population_artifact(workflow / "prepared_population")
    population = prepared_population_frame(prepared, selected)
    codec, entry, bundle = _frozen_pq_codec(root, workflow, compression_profile=cm["compression_profile"])
    if entry["artifact_sha256"] != cm["frozen_codec"]["sha256"] or bundle["fit_seed"] != cm["calibration_seed"]:
        raise ValueError("frozen codec or protocol seed mismatch")
    evaluation = run["config"]["step4"]["evaluation"]
    official_roles = {"gallery", "registered_probe", "unknown_unknown_probe"}
    for split in ("calibration", "test"):
        protocol = (build_survface_matched_calibration_protocol(
            population, gallery_identity_count=int(evaluation["survface_calibration_gallery_identities"]),
            seed=int(cm["calibration_seed"])) if split == "calibration" else
            build_survface_official_protocol(population.loc[population.protocol_role.isin(official_roles)].copy()))
        arrays = open_set_protocol_arrays(protocol, population)
        rows = getattr(condition, split)
        ids = pd.Index(arrays["query_ids"].astype(str))
        if ids.has_duplicates or set(ids) != set(rows.sample_id):
            raise ValueError("reconstructed protocol differs from saved query cohort")
        order = ids.get_indexer(rows.sample_id)
        if not np.array_equal(arrays["query_identity_ids"][order].astype(str), rows.identity_id.astype(str)):
            raise ValueError("reconstructed query identities disagree")
        codes = codec.encode(arrays["gallery"])
        distortion = np.linalg.norm(arrays["gallery"].astype(float) - codec.decode(codes).astype(float), axis=1)
        for shard, start in enumerate(range(0, len(rows), batch_size)):
            selected_rows = rows.iloc[start:start+batch_size]
            filename = f"{split}-{shard:06d}.parquet"
            if shard < len(journal["shards"][split]):
                receipt = journal["shards"][split][shard]
                if receipt["file"] != filename or sha256_file(destination / filename) != receipt["sha256"]:
                    raise ValueError("resume shard hash mismatch")
                frame = pd.read_parquet(destination / filename)
            else:
                distances, indices, _ = codec.search_adc_with_metrics(
                    arrays["queries"][order[start:start+batch_size]], codes, top_k=2)
                frame = adc_feature_frame(distances, indices, selected_rows.sample_id,
                                          arrays["gallery_ids"], distortion)
                join_retrieval_features(selected_rows, frame, score_tolerance=score_tolerance)
                partial = destination / (filename + ".tmp")
                frame.to_parquet(partial, index=False)
                os.replace(partial, destination / filename)
                journal["shards"][split].append({"file": filename, "rows": len(frame),
                                                   "sha256": sha256_file(destination / filename)})
                _atomic_json(progress_path, journal)
            join_retrieval_features(selected_rows, frame, score_tolerance=score_tolerance)
            if progress:
                progress({"split": split, "queries_done": min(start+batch_size, len(rows)), "total": len(rows)})
    _atomic_json(destination / "manifest.json", {**journal, "status": "completed"})
    return load_retrieval_features(destination, condition)
