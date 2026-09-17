"""Restartable split-matched Grad-CAM and calibration-only masking evidence for 02."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.embeddings import create_pytorch_adapter_from_spec, select_model_spec_by_profile
from research.evaluation.saliency_faithfulness import summarize_faithfulness, FAITHFULNESS_METRICS
from research.experiments.fiqa_threshold_calibration import _completed_run, _read_json, _strict_boolean_series
from research.experiments.fiqa_split_stability import _frame_hash
from research.experiments.step2_compression import prepared_population_frame, open_set_protocol_arrays
from research.explainability.gradcam.artifacts import read_prepared_population_artifact, _replace_with_retry
from research.explainability.gradcam.extraction import measure_population_faithfulness
from research.explainability.gradcam.features import summarize_saliency_features
from research.explainability.gradcam.landmark_regions import read_landmark_region_bundle
from research.explainability.gradcam.pair import PairCosineGradCAM
from research.protocols.open_set import build_survface_matched_calibration_protocol, build_survface_official_protocol
from research.runtime.hashing import canonical_sha256, sha256_file

TARGET = "origin_top1_gallery_cosine"
FEATURES = ("outside_face_attention", "saliency_entropy")
# Audited c6fda99 producer: migration is limited to this I/O-only repair.
_LEGACY_PRODUCER_SHA256 = "4a1fdf7ba3f527efaa8708dc78ecb798e4fcc36db865b3d467dc90870cdbf0a8"
_LEGACY_GENERATION_SHA256 = "de36b2e2e01ff69554a670bebb684b738244d930c322a56a6bc6433f016350cd"


def _atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    _replace_with_retry(tmp, path)


def _generation_contract_hash():
    """Keep I/O repair compatibility separate from the scientific implementation."""
    names = {"TARGET", "FEATURES", "_source_context", "select_calibration_faithfulness",
             "_origin_targets", "_generate_chunk", "_reuse_test_features"}
    tree = ast.parse(Path(__file__).read_text(encoding="utf8"))
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name in names)
             or (isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]
    return hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[]),
                                   include_attributes=False).encode()).hexdigest()


def _resume_journal(source, spec):
    """Read an explicit recovery source without modifying its files or provenance."""
    source = Path(source).resolve()
    path = source / "progress.json"
    journal = _read_json(path)
    old = journal["spec"]
    if old != spec:
        normalized = json.loads(json.dumps(spec))
        if (old.get("implementation_sha256", {}).get("saliency_calibration_inputs.py") != _LEGACY_PRODUCER_SHA256
                or normalized.pop("generation_contract_sha256", None) != _LEGACY_GENERATION_SHA256):
            raise ValueError("resume source implementation is not compatible with this I/O-only repair")
        normalized["implementation_sha256"]["saliency_calibration_inputs.py"] = _LEGACY_PRODUCER_SHA256
        if normalized != old:
            raise ValueError("resume source settings/data/runtime/implementation differ")
    if source.name != "saliency-inputs-" + canonical_sha256(old)[:24]:
        raise ValueError("resume source UID differs from its specification")
    pending_path = source / "progress.tmp"
    if pending_path.exists():
        try:
            pending = _read_json(pending_path)
        except json.JSONDecodeError:
            pending = None  # Interrupted write: retain the committed journal.
        if pending is not None:
            if pending.get("spec") != old or any(
                pending.get("shards", {}).get(s, [])[:len(journal["shards"][s])] != journal["shards"][s]
                for s in ("calibration", "test")
            ):
                raise ValueError("pending resume journal is not an exact extension")
            journal, path = pending, pending_path
    for split in ("calibration", "test"):
        for index, receipt in enumerate(journal["shards"][split]):
            if receipt["path"] != f"{split}-{index:06d}.parquet":
                raise ValueError("resume source shard order differs")
            _verified(source, receipt)
    return journal, dict(directory=str(source), journal=path.name, journal_sha256=sha256_file(path),
                         spec_sha256=canonical_sha256(old))


def _receipt(path):
    return dict(path=path.name, bytes=path.stat().st_size, sha256=sha256_file(path))


def _verified(root, receipt):
    path = (root / receipt["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("input shard path escapes artifact")
    if path.stat().st_size != receipt["bytes"] or sha256_file(path) != receipt["sha256"]:
        raise ValueError(f"input shard hash mismatch: {path}")
    return path


def _source_context(run_dir, condition):
    root, run, workflow = _completed_run(run_dir)
    cm = condition.manifest
    if (cm.get("dataset_id") != "survface" or cm.get("status") != "completed"
            or cm.get("source_run_id") != run["run_id"]
            or cm.get("model_uid") != run["config"]["model_uid"]
            or cm.get("source_run_manifest_sha256") != sha256_file(root / "run_manifest.json")):
        raise ValueError("02 input generation requires the exact completed SurvFace source")
    for path, key in {
        workflow / "freeze_manifest.json": "source_freeze_manifest_sha256",
        workflow / "selected_manifest.csv": "selected_manifest_sha256",
        workflow / "prepared_population/manifest.json": "prepared_population_manifest_sha256",
    }.items():
        if sha256_file(path) != cm[key]:
            raise ValueError(f"frozen source mismatch: {key}")
    freeze = _read_json(workflow / "freeze_manifest.json")
    selected = pd.read_csv(workflow / "selected_manifest.csv")
    prepared = read_prepared_population_artifact(workflow / "prepared_population")
    for key in ("extraction_uid", "model_uid", "origin_embedding_artifact_uid"):
        if getattr(prepared, key) != cm[key]:
            raise ValueError(f"prepared population mismatch: {key}")
    population = prepared_population_frame(prepared, selected)
    config = run["config"]["step4"]
    project = Path(__file__).resolve().parents[2]
    dataset = config["datasets"]["survface"]
    aligned_root = project / dataset["aligned_bundle_dir"]
    mask_root = project / dataset["landmark_region_bundle_dir"]
    for path, expected in (
        (aligned_root / "bundle_manifest.json", cm["aligned_bundle_manifest_sha256"]),
        (mask_root / "bundle_manifest.json", freeze["landmark_region_manifest_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"alignment/landmark manifest mismatch: {path}")
    profile = config["execution"]["model_profile"]
    _, model_spec = select_model_spec_by_profile(
        project / config["models"]["registry_root"], profile_id=profile,
        profile_config=config["models"]["profiles"][profile], verify_checkpoint=True)
    if (model_spec.model_uid != cm["model_uid"] or model_spec.checkpoint.sha256 != freeze["checkpoint_sha256"]
            or model_spec.preprocessing.preprocess_hash != freeze["preprocess_hash"]
            or model_spec.target_layer != freeze["target_layer"]):
        raise ValueError("Grad-CAM model/checkpoint/preprocessing/target layer differs from source")
    masks = read_landmark_region_bundle(mask_root)
    faces = np.load(aligned_root / "aligned_faces.npy", mmap_mode="r", allow_pickle=False)
    if selected.sample_id.duplicated().any():
        raise ValueError("selected population IDs must be unique")
    index = selected.set_index("sample_id")
    splits = {}
    for split in ("calibration", "test"):
        protocol = (build_survface_matched_calibration_protocol(
            population, gallery_identity_count=int(config["evaluation"]["survface_calibration_gallery_identities"]),
            seed=int(cm["calibration_seed"])) if split == "calibration" else
            build_survface_official_protocol(population.loc[population.protocol_role.isin(
                {"gallery", "registered_probe", "unknown_unknown_probe"})].copy()))
        arrays = open_set_protocol_arrays(protocol, population)
        rows = getattr(condition, split)
        ids = pd.Index(arrays["query_ids"].astype(str))
        if ids.has_duplicates or set(ids) != set(rows.sample_id):
            raise ValueError(f"{split} reconstructed query cohort differs")
        order = ids.get_indexer(rows.sample_id)
        if not np.array_equal(arrays["query_identity_ids"][order].astype(str), rows.identity_id.astype(str)):
            raise ValueError("query identity contract differs")
        aligned = index.loc[rows.sample_id]
        if not np.array_equal(aligned.aligned_content_sha256, rows.aligned_content_sha256):
            raise ValueError("query alignment hash differs")
        face_indices = aligned.aligned_face_index.to_numpy(np.int64)
        masks.subset(face_indices, expected_sample_ids=rows.sample_id)
        splits[split] = dict(queries=arrays["queries"][order], gallery=arrays["gallery"],
                             gallery_ids=arrays["gallery_ids"], face_indices=face_indices)
    return dict(root=root, workflow=workflow, freeze=freeze, spec=model_spec, masks=masks, faces=faces,
                splits=splits, expected_grid=tuple(config["gradcam"]["extraction"]["expected_heatmap_size"]))


def select_calibration_faithfulness(condition, *, maximum_samples=10000, seed=8972, limit=None):
    """Predeclare a hash-ranked calibration sample; exclude test content/identities."""
    rows = condition.calibration.iloc[:limit].copy() if limit else condition.calibration.copy()
    test = condition.test
    eligible = rows.loc[~rows.sample_id.isin(test.sample_id)
                        & ~rows.identity_id.isin(test.identity_id)
                        & ~rows.aligned_content_sha256.isin(test.aligned_content_sha256)].copy()
    eligible["_order"] = eligible.sample_id.map(lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    eligible = eligible.sort_values(["_order", "sample_id"], kind="stable")
    selected = eligible if maximum_samples is None else eligible.iloc[:maximum_samples]
    if selected.empty:
        raise ValueError("no test-disjoint calibration images available for faithfulness")
    return selected.drop(columns="_order").reset_index(drop=True)


def _origin_targets(queries, gallery):
    with threadpool_limits(limits=1, user_api="blas"):
        scores = queries @ gallery.T
    winners = np.argmax(scores, axis=1)  # canonical template order breaks ties
    return winners, scores[np.arange(len(queries)), winners]


def _generate_chunk(context, adapter, rows, split, start, settings, faith_ids):
    import torch
    arrays = context["splits"][split]
    ix = slice(start, start + len(rows))
    queries, face_indices = arrays["queries"][ix], arrays["face_indices"][ix]
    images = np.asarray(context["faces"][face_indices])
    observed = [hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest() for im in images]
    if not np.array_equal(observed, rows.aligned_content_sha256):
        raise ValueError("aligned image bytes differ from frozen query hashes")
    winners, scores = _origin_targets(queries, arrays["gallery"])
    templates = arrays["gallery"][winners]
    analyzer = PairCosineGradCAM(adapter.model, adapter.target_layer,
                                embedding_extractor=adapter.select_embedding_tensor)
    heatmaps, valid, measured_scores, repeat_cosine = [], [], [], []
    batch = settings["gradcam_batch_size"]
    for offset in range(0, len(rows), batch):
        sub = slice(offset, offset + batch)
        generated = analyzer.generate(adapter.preprocess(images[sub]),
            torch.from_numpy(templates[sub]).to(adapter.device),
            batch_mode="single" if len(images[sub]) == 1 else "independent", target_name=TARGET)
        if tuple(generated.heatmaps.shape[1:]) != context["expected_grid"]:
            raise ValueError("Grad-CAM spatial grid differs from frozen source")
        repeat = np.sum(queries[sub] * generated.normalized_embeddings, axis=1)
        if np.any(repeat < .99999) or not np.allclose(generated.target_scores, scores[sub], atol=1e-5, rtol=0):
            raise ValueError("Grad-CAM forward differs from frozen origin embeddings/targets")
        heatmaps.append(generated.heatmaps)
        valid.append(generated.valid_heatmap)
        measured_scores.append(generated.target_scores)
        repeat_cosine.append(repeat)
    heatmaps = np.concatenate(heatmaps)
    masks = context["masks"].build_region_masks(face_indices, image_size=context["expected_grid"])
    spatial = summarize_saliency_features(heatmaps, region_masks=masks)
    out = rows[["sample_id", "identity_id", "aligned_content_sha256"]].reset_index(drop=True).copy()
    out["split"] = split
    out["saliency_target_name"] = TARGET
    out["heatmap_available"] = True
    out["gradcam_valid_heatmap"] = np.concatenate(valid)
    out["gradcam_target_score"] = np.concatenate(measured_scores)
    out["saliency_reference_identity_id"] = arrays["gallery_ids"][winners]
    out["pass_a_pass_b_embedding_cosine"] = np.concatenate(repeat_cosine)
    for feature in FEATURES:
        out[feature] = spatial[feature].to_numpy()
    chosen = np.flatnonzero(rows.sample_id.isin(faith_ids).to_numpy()) if split == "calibration" else np.array([], int)
    if len(chosen):
        # Use the fresh unoccluded score of this same forward, not an ADC score.
        measured = measure_population_faithfulness(adapter, images[chosen], heatmaps[chosen], templates[chosen],
            rows.sample_id.to_numpy()[chosen], out.gradcam_target_score.to_numpy()[chosen].astype(float),
            fraction=settings["occlusion_fraction"], random_repeats=settings["random_repeats"],
            seed=settings["seed"], batch_size=settings["faithfulness_batch_size"])
        for key, values in measured.items():
            out[key] = np.nan
            out.loc[chosen, key] = values
    return out


def _reuse_test_features(context, condition):
    """Validate old test targets against the newly reconstructed official gallery."""
    from research.experiments.saliency_incremental_calibration import _verified_file
    root = context["workflow"] / "saliency_population"
    manifest = _read_json(root / "manifest.json")
    for key in ("dataset_id", "model_uid", "extraction_uid", "origin_embedding_artifact_uid"):
        if manifest.get(key) != condition.manifest[key]:
            raise ValueError(f"test saliency lineage mismatch: {key}")
    if manifest.get("target_name") != TARGET or manifest.get("target_layer") != context["freeze"]["target_layer"]:
        raise ValueError("test saliency target mismatch; disable reuse to regenerate")
    columns = {"sample_id", "saliency_target_name", "heatmap_available", "gradcam_valid_heatmap", *FEATURES,
               "region_mask_uid", "checkpoint_sha256", "preprocess_hash", "gradcam_target_score",
               "saliency_reference_identity_id"}
    frame = pd.read_csv(_verified_file(root, manifest["saliency_features"]),
                        usecols=lambda c: c in columns, low_memory=False,
                        dtype={"sample_id": str, "saliency_reference_identity_id": str})
    if frame.sample_id.duplicated().any():
        raise ValueError("duplicate test saliency IDs")
    frame = frame.set_index("sample_id").loc[condition.test.sample_id].reset_index()
    for key, expected in {"saliency_target_name": TARGET, "region_mask_uid": context["masks"].region_mask_uid,
                          "checkpoint_sha256": context["freeze"]["checkpoint_sha256"],
                          "preprocess_hash": context["freeze"]["preprocess_hash"]}.items():
        if not frame[key].eq(expected).all():
            raise ValueError(f"test saliency {key} differs; disable reuse to regenerate")
    arrays = context["splits"]["test"]
    for start in range(0, len(frame), 1024):
        part = frame.iloc[start:start+1024]
        winners, scores = _origin_targets(arrays["queries"][start:start+len(part)], arrays["gallery"])
        if (not np.array_equal(part.saliency_reference_identity_id.astype(str), arrays["gallery_ids"][winners].astype(str))
                or not np.allclose(part.gradcam_target_score, scores, atol=1e-5, rtol=0)):
            raise ValueError("test saliency gallery targets differ; disable reuse to regenerate")
    for col in ("heatmap_available", "gradcam_valid_heatmap"):
        frame[col] = _strict_boolean_series(frame[col], column=col)
    for col in ("identity_id", "aligned_content_sha256"):
        frame[col] = condition.test[col].to_numpy()
    frame["split"] = "test"
    return frame, canonical_sha256(manifest)


def build_saliency_calibration_inputs(
    run_dir, condition, output_root, *, device="cuda", reuse_test_saliency=True,
    gradcam_batch_size=4, chunk_size=128, faithfulness_batch_size=32,
    faithfulness_maximum_samples=10000, occlusion_fraction=.1, random_repeats=5,
    seed=8972, bootstrap_repeats=2000, max_queries_per_split=None, progress=None, resume_from=None,
):
    """Build both inputs once; checkpoint each chunk. Limited builds cannot authorize 02."""
    import torch
    for name, value in dict(gradcam_batch_size=gradcam_batch_size, chunk_size=chunk_size,
                            faithfulness_batch_size=faithfulness_batch_size, random_repeats=random_repeats).items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for value in (faithfulness_maximum_samples, max_queries_per_split):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("sample limits must be None or positive integers")
    if type(seed) is not int or seed < 0 or type(bootstrap_repeats) is not int or bootstrap_repeats < 100:
        raise ValueError("invalid seed/bootstrap settings")
    if not np.isfinite(occlusion_fraction) or not 0 < occlusion_fraction <= 1:
        raise ValueError("invalid occlusion fraction")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("02 production input generation requires CUDA; CPU fallback is disabled")
    runtime = dict(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                   device=str(device), device_name=torch.cuda.get_device_name(torch.device(device)))
    if progress:
        progress(dict(stage="preflight", **runtime))
    context = _source_context(run_dir, condition)
    root = Path(output_root).resolve()
    if root.is_relative_to(context["root"].resolve()):
        raise ValueError("derived inputs must be outside the completed source run")
    selected = select_calibration_faithfulness(condition, maximum_samples=faithfulness_maximum_samples,
                                              seed=seed, limit=max_queries_per_split)
    if progress:
        progress(dict(stage="source_verified", calibration_queries=len(condition.calibration),
                      test_queries=len(condition.test), faithfulness_samples=len(selected),
                      reuse_test_saliency=reuse_test_saliency))
    reused, reuse_hash = _reuse_test_features(context, condition) if reuse_test_saliency else (None, None)
    settings = dict(gradcam_batch_size=gradcam_batch_size, chunk_size=chunk_size,
                    faithfulness_batch_size=faithfulness_batch_size,
                    faithfulness_maximum_samples=faithfulness_maximum_samples,
                    occlusion_fraction=occlusion_fraction, random_repeats=random_repeats,
                    seed=seed, bootstrap_repeats=bootstrap_repeats, max_queries_per_split=max_queries_per_split,
                    reuse_test_saliency=reuse_test_saliency)
    source_modules = [Path(__file__), Path(__file__).with_name("step2_compression.py"),
                      Path(__file__).parents[1]/"protocols/open_set.py",
                      Path(__file__).parents[1]/"explainability/gradcam/pair.py",
                      Path(__file__).parents[1]/"explainability/gradcam/features.py",
                      Path(__file__).parents[1]/"explainability/gradcam/extraction.py",
                      Path(__file__).parents[1]/"explainability/gradcam/landmark_regions.py",
                      Path(__file__).parents[1]/"explainability/gradcam/metrics.py",
                      Path(__file__).parents[1]/"embeddings/pytorch/adapter.py",
                      Path(__file__).parents[1]/"embeddings/pytorch/official_loaders.py",
                      Path(__file__).parents[1]/"evaluation/cluster_bootstrap.py",
                      Path(__file__).parents[1]/"evaluation/saliency_faithfulness.py"]
    spec = dict(artifact_type="saliency_calibration_input_bundle", schema_version=1,
                condition_manifest_sha256=canonical_sha256(condition.manifest),
                condition_frame_sha256={s:_frame_hash(getattr(condition,s)) for s in ("calibration","test")},
                settings=settings, runtime=runtime, source_test_saliency_manifest_sha256=reuse_hash,
                generation_contract_sha256=_generation_contract_hash(),
                selection_sha256=_frame_hash(selected[["sample_id","identity_id","aligned_content_sha256"]]),
                implementation_sha256={p.name:sha256_file(p) for p in source_modules})
    uid = "saliency-inputs-" + canonical_sha256(spec)[:24]
    destination = root / uid
    destination.mkdir(parents=True, exist_ok=True)
    journal_path = destination / "progress.json"
    journal = _read_json(journal_path) if journal_path.exists() else dict(spec=spec, shards={"calibration":[],"test":[]})
    if journal["spec"] != spec:
        raise ValueError("resume settings differ")
    if (destination / "manifest.json").exists():
        complete = _read_json(destination / "manifest.json")
        if complete["spec"] != spec or complete.get("status") != "completed":
            raise ValueError("completed input bundle differs")
        for entry in complete["outputs"]:
            _verified(destination, entry)
        if progress:
            progress(dict(stage="reuse_completed", directory=str(destination)))
        return dict(directory=destination, saliency_directory=destination/"saliency",
                    faithfulness_directory=destination/"faithfulness", manifest=complete)
    recovery = None
    if resume_from is not None and Path(resume_from).resolve() != destination:
        recovery, provenance = _resume_journal(resume_from, spec)
        if any(len(recovery["shards"][s]) > (len(getattr(condition, s).iloc[:max_queries_per_split]) + chunk_size - 1) // chunk_size
               for s in ("calibration", "test")):
            raise ValueError("resume source has more shards than the configured cohort")
        if "resume_from" in journal and journal["resume_from"] != provenance:
            raise ValueError("explicit resume source changed since recovery started")
        journal["resume_from"] = provenance
        _atomic_json(journal_path, journal)
        if progress:
            progress(dict(stage="recover_verified_shards", **provenance,
                          shard_counts={s:len(v) for s,v in recovery["shards"].items()}))
    adapter = None
    frames = []
    faith_ids = set(selected.sample_id)
    for split in ("calibration", "test"):
        rows = getattr(condition, split).iloc[:max_queries_per_split]
        for shard, start in enumerate(range(0, len(rows), chunk_size)):
            part = rows.iloc[start:start+chunk_size]
            name = f"{split}-{shard:06d}.parquet"
            if shard < len(journal["shards"][split]):
                entry = journal["shards"][split][shard]
                if entry["path"] != name:
                    raise ValueError("resume shard order differs")
                frame = pd.read_parquet(_verified(destination, entry))
            else:
                source_shard = None
                if recovery is not None and shard < len(recovery["shards"][split]):
                    source_shard = _verified(Path(resume_from), recovery["shards"][split][shard])
                    frame = pd.read_parquet(source_shard)
                elif split == "test" and reused is not None:
                    frame = reused.iloc[start:start+len(part)].copy()
                else:
                    if adapter is None:
                        adapter = create_pytorch_adapter_from_spec(context["spec"], device=device)
                        if adapter.device.type != "cuda":
                            raise RuntimeError("adapter did not use CUDA")
                    frame = _generate_chunk(context, adapter, part, split, start, settings, faith_ids)
                path = destination / name
                tmp = path.with_suffix(".tmp")
                if (not np.array_equal(frame.sample_id, part.sample_id)
                        or not np.array_equal(frame.aligned_content_sha256, part.aligned_content_sha256)
                        or not frame.split.eq(split).all()):
                    raise ValueError("generated/recovered shard cohort differs")
                if source_shard is None:
                    frame.to_parquet(tmp, index=False)
                else:
                    shutil.copyfile(source_shard, tmp)
                    _verified(destination, {**recovery["shards"][split][shard], "path":tmp.name})
                _replace_with_retry(tmp, path)
                journal["shards"][split].append(_receipt(path))
                _atomic_json(journal_path, journal)
            if (not np.array_equal(frame.sample_id, part.sample_id)
                    or not np.array_equal(frame.aligned_content_sha256, part.aligned_content_sha256)
                    or not frame.split.eq(split).all()):
                raise ValueError("resume shard cohort differs")
            frames.append(frame)
            if progress:
                progress(dict(stage="generate_inputs", split=split, queries_done=start+len(part), total=len(rows)))
    combined = pd.concat(frames, ignore_index=True)
    sal_dir, faith_dir = destination / "saliency", destination / "faithfulness"
    sal_dir.mkdir(exist_ok=True)
    faith_dir.mkdir(exist_ok=True)
    cols = ["sample_id","identity_id","split","aligned_content_sha256","saliency_target_name",
            "heatmap_available","gradcam_valid_heatmap",*FEATURES]
    combined[cols].to_csv(sal_dir / "saliency_features.csv", index=False)
    faith = combined.set_index("sample_id").loc[selected.sample_id].reset_index()
    faith = faith[["sample_id","identity_id","aligned_content_sha256","split",*FAITHFULNESS_METRICS]]
    if not np.isfinite(faith[list(FAITHFULNESS_METRICS)].to_numpy(float)).all():
        raise ValueError("incomplete faithfulness rows")
    faith.to_csv(faith_dir / "faithfulness_rows.csv", index=False)
    # Re-read persisted floating-point values before computing gate statistics.
    faith = pd.read_csv(faith_dir / "faithfulness_rows.csv")
    summary = summarize_faithfulness(faith, group_columns=(), bootstrap_repeats=bootstrap_repeats, seed=seed)
    summary.to_csv(faith_dir / "faithfulness_summary.csv", index=False)
    cm = condition.manifest
    lineage = {k:cm[k] for k in ("dataset_id","model_uid","origin_embedding_artifact_uid")}
    sal_uid = "saliency-calibration-" + canonical_sha256(spec)[:24]
    sm = dict(artifact_type="saliency_calibration_features", schema_version=1, status="completed", **lineage,
              extraction_uid=cm["extraction_uid"], condition_manifest_sha256=canonical_sha256(cm),
              gallery_contract="split_matched_origin_top1", target_name=TARGET, saliency_spec_uid=sal_uid,
              smoke_only=max_queries_per_split is not None, producer_spec=spec,
              saliency_features=_receipt(sal_dir/"saliency_features.csv"))
    fm = dict(artifact_type="open_set_gradcam_faithfulness", schema_version=2, status="completed", **lineage,
              source_run_id=cm["source_run_id"], saliency_spec_uid=sal_uid, saliency_target_name=TARGET,
              evaluation_split="calibration", condition_manifest_sha256=canonical_sha256(cm),
              sampling=dict(method="sha256_sample_id", seed=seed, maximum_samples=faithfulness_maximum_samples,
                            selected_count=len(faith), selected_identity_count=int(faith.identity_id.nunique()),
                            selected_sample_sha256=spec["selection_sha256"], excluded_test_content=True),
              statistics=dict(bootstrap_method="identity_cluster", confidence_level=.95,
                              bootstrap_repeats=bootstrap_repeats, seed=seed),
              occlusion=dict(fraction=occlusion_fraction, random_repeats=random_repeats, seed=seed),
              outputs=[_receipt(faith_dir/n) for n in ("faithfulness_rows.csv","faithfulness_summary.csv")])
    if "resume_from" in journal:
        sm["resume_from"] = fm["resume_from"] = journal["resume_from"]
    _atomic_json(sal_dir/"manifest.json", sm)
    _atomic_json(faith_dir/"manifest.json", fm)
    outputs = []
    for directory in (sal_dir,faith_dir):
        for path in directory.iterdir():
            if path.is_file():
                outputs.append({**_receipt(path), "path":path.relative_to(destination).as_posix()})
    complete = dict(spec=spec, status="completed", input_uid=uid, outputs=outputs)
    if "resume_from" in journal:
        complete["resume_from"] = journal["resume_from"]
    _atomic_json(destination/"manifest.json", complete)
    return dict(directory=destination, saliency_directory=sal_dir, faithfulness_directory=faith_dir, manifest=complete)
