"""LFW public benchmark and a separate matched-gallery calibration extension.

These are derived experiments over explicitly selected immutable source runs.
No GPU inference, source-run mutation, or implicit latest-run selection occurs.
"""
from __future__ import annotations

from copy import deepcopy
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.compression import PQCompressor
from research.datasets.lfw_blufr import bind_image_manifest, load_blufr_lists, trial_inventory
from research.experiments.calibration_matrix import DEFAULT_RUN_MATRIX, MODEL_UIDS, PQ_PROFILES
from research.experiments.calibration_protocols import calibration_protocol
from research.experiments.fiqa_threshold_calibration import (
    ConditionScoreTables, _completed_run, _frozen_pq_codec,
)
from research.experiments.origin_pq_inputs import (
    ORIGIN_PROFILE, inspect_origin_pq_experiment, load_run_inputs,
)
from research.experiments.origin_pq_resources import check_resources, ResourceBudgetExceeded
from research.experiments.origin_pq_storage import SplitStore, write_report, read_report
from research.experiments.origin_vs_pq_calibration import (
    OriginPQSettings, TABLES, run_origin_pq_split, science_hashes, _validate_split_tables,
)
from research.experiments.step2_compression import prepared_population_frame, open_set_protocol_arrays
from research.explainability.gradcam.artifacts import read_prepared_population_artifact
from research.protocols.blufr import identification_rows, benchmark_curve
from research.protocols.open_set import build_calibration_protocol
from research.runtime.hashing import sha256_file, canonical_sha256

MATCHED_PROTOCOL = "lfw-custom-matched-gallery-calibration-v2"


def implementation_hashes():
    root = Path(__file__).parents[1]
    return {**science_hashes(), **{name: sha256_file(root / name) for name in (
        "datasets/lfw_blufr.py", "protocols/blufr.py", "experiments/lfw_protocol_experiments.py",
        "compression/profiles.py", "explainability/gradcam/artifacts.py")}}


def runtime_versions():
    import faiss
    import scipy
    import sys
    return dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                scipy=scipy.__version__, faiss=faiss.__version__)


def _source(project_root, source_run):
    root, run, workflow = _completed_run(Path(project_root) / source_run)
    if run["config"]["dataset_id"] != "lfw":
        raise ValueError("an explicit LFW source run is required")
    freeze_path = workflow / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf8"))
    selected_path = workflow / "selected_manifest.csv"
    if sha256_file(selected_path) != freeze["selected_manifest_sha256"]:
        raise ValueError("source selected manifest hash mismatch")
    prepared = read_prepared_population_artifact(workflow / "prepared_population")
    for key in ("model_uid", "checkpoint_sha256", "preprocess_hash", "extraction_uid"):
        if str(freeze.get(key)) != str(getattr(prepared, key)):
            raise ValueError(f"source freeze/prepared mismatch: {key}")
    if (freeze["run_id"] != run["run_id"] or freeze["model_uid"] != run["config"]["model_uid"]
            or freeze.get("dataset_id") != "lfw"):
        raise ValueError("source run/freeze mismatch")
    selected = pd.read_csv(selected_path)
    population = prepared_population_frame(prepared, selected)
    lineage = dict(source_run_id=run["run_id"], model_uid=prepared.model_uid,
        source_run_manifest_sha256=sha256_file(root / "run_manifest.json"),
        source_freeze_manifest_sha256=sha256_file(freeze_path),
        selected_manifest_sha256=sha256_file(selected_path),
        prepared_population_manifest_sha256=sha256_file(workflow / "prepared_population/manifest.json"),
        aligned_bundle_manifest_sha256=freeze["aligned_bundle_manifest_sha256"],
        checkpoint_sha256=prepared.checkpoint_sha256, preprocess_hash=prepared.preprocess_hash,
        embedding_reused=True, checkpoint_training_overlap_verified=False)
    return root, run, workflow, prepared, population, lineage


def inspect_blufr(project_root, *, config_path="configs/experiments/lfw_blufr.yaml", models=None):
    """Read-only public-list and embedding-coverage preflight (no source creation)."""
    import yaml
    project = Path(project_root).resolve()
    config = yaml.safe_load((project / config_path).read_text(encoding="utf8"))
    profiles = tuple(config["pq_profiles"])
    if not profiles or len(set(profiles)) != len(profiles) or not set(profiles).issubset(PQ_PROFILES):
        raise ValueError("BLUFR requires unique supported PQ profiles")
    lists = load_blufr_lists(project / config["protocol_file"], expected_sha256=config["protocol_sha256"])
    bound = bind_image_manifest(lists, pd.read_csv(project / config["image_manifest"]))
    selected_models = tuple(models or config["source_runs"])
    if not selected_models or len(set(selected_models)) != len(selected_models):
        raise ValueError("unique nonempty models required")
    coverage, missing, provenance = [], [], {}
    for model in selected_models:
        source = config["source_runs"][model]
        _, _, _, prepared, population, lineage = _source(project, source)
        if model not in MODEL_UIDS or prepared.model_uid != MODEL_UIDS[model]:
            raise ValueError("BLUFR model label/checkpoint UID mismatch")
        vectors = prepared.normalized_embeddings
        if (vectors.shape != (len(prepared.sample_ids), 512) or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-4)):
            raise ValueError("BLUFR requires finite normalized 512D source embeddings")
        lookup = pd.Series(prepared.identity_ids, index=prepared.sample_ids)
        if lookup.index.has_duplicates:
            raise ValueError("duplicate source embedding sample IDs")
        available = bound.image_id.isin(lookup.index)
        if not np.array_equal(lookup.loc[bound.loc[available, "image_id"]].astype(str),
                              bound.loc[available, "identity_id"].astype(str)):
            raise ValueError("embedding identity differs from public image identity")
        missing.append(bound.loc[~available, ["filename", "image_id", "identity_id"]].assign(model=model))
        coverage.append(dict(model=model, required=len(bound), available=int(available.sum()),
                             missing=int((~available).sum()), ready=bool(available.all())))
        provenance[model] = lineage
        del prepared, population, vectors
    return dict(config=config, lists=lists, bound=bound, coverage=pd.DataFrame(coverage),
                missing=pd.concat(missing, ignore_index=True), inventory=trial_inventory(lists),
                provenance=provenance, ready=all(x["ready"] for x in coverage))


def search_rows(queries, gallery, query_ids, identity_ids, gallery_ids, *, codec=None, batch_size=256,
                resource_check=check_resources):
    """Bounded exhaustive search; retain full ranks for the public benchmark."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("positive integer batch size required")
    codes = codec.encode(gallery) if codec is not None else None
    parts = []
    for start in range(0, len(queries), batch_size):
        resource_check()
        stop = min(start + batch_size, len(queries))
        q = np.ascontiguousarray(queries[start:stop], dtype=np.float32)
        if codec is None:
            scores = q @ np.asarray(gallery, dtype=np.float32).T
        else:
            distance, indices = codec.search_adc(q, codes, top_k=len(gallery))
            scores = np.empty_like(distance)
            np.put_along_axis(scores, indices, -distance, axis=1)
        parts.append(identification_rows(scores, gallery_ids, query_ids[start:stop], identity_ids[start:stop]))
    if not parts:
        raise ValueError("empty query cohort")
    return pd.concat(parts, ignore_index=True)


def _summary(curves):
    grouped = curves.groupby(["model", "compression_profile", "rank", "target_fpir"], sort=False)
    result = grouped.agg(trial_count=("trial_id", "nunique"), tpir_mean=("tpir", "mean"),
        tpir_std=("tpir", "std"), fpir_min=("realized_fpir", "min"), fpir_max=("realized_fpir", "max")).reset_index()
    result["tpir_mean_minus_std"] = result.tpir_mean - result.tpir_std
    result["aggregation"] = "trial_descriptive_sample_std_not_independent_ci"
    return result


def run_blufr_benchmark(project_root, *, config_path="configs/experiments/lfw_blufr.yaml",
                        models=None, max_new_jobs=1, progress=print):
    """Refit codecs on each released training set; fail before writing on missing images."""
    project = Path(project_root).resolve()
    inspection = inspect_blufr(project, config_path=config_path, models=models)
    if not inspection["ready"]:
        raise ValueError("BLUFR requires every released image; missing embeddings per model: "
                         + inspection["coverage"].to_json(orient="records"))
    config, lists, bound = (inspection[k] for k in ("config", "lists", "bound"))
    trials = tuple(config["trial_ids"])
    if not trials or len(set(trials)) != len(trials):
        raise ValueError("unique trial IDs required")
    for trial in trials:
        lists.trial(trial)
    if max_new_jobs is not None and (type(max_new_jobs) is not int or max_new_jobs < 1):
        raise ValueError("max_new_jobs must be positive or None")
    spec = dict(kind="blufr_public_open_set_benchmark", config=config,
                models=list(inspection["provenance"]), provenance=inspection["provenance"],
                implementation=implementation_hashes(), versions=runtime_versions(),
                threshold_source="test_curve_only")
    root = project / config["output_root"]
    checkpoint = root / ("benchmark-" + canonical_sha256(spec)[:24] + ".sqlite3")
    frames, costs_accumulated, receipts, new_jobs = [], [], [], 0
    expected = len(trials) * len(spec["models"])
    stop_reason = None
    import faiss
    previous_threads = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(2)
        with SplitStore(checkpoint) as store, threadpool_limits(limits=2):
            for model in spec["models"]:
                prepared = matrix = None
                for trial_id in trials:
                    job = dict(campaign=spec, model=model, trial_id=trial_id)
                    saved = store.get(job)
                    if saved is None:
                        if max_new_jobs is not None and new_jobs >= max_new_jobs:
                            stop_reason = "new_job_budget_reached"
                            continue
                        check_resources()
                        if prepared is None:
                            _, _, _, prepared, _, _ = _source(project, config["source_runs"][model])
                            order = pd.Index(prepared.sample_ids).get_indexer(bound.image_id)
                            if (order < 0).any():
                                raise ValueError("embedding coverage changed after preflight")
                            matrix = prepared.normalized_embeddings[order]
                        trial = lists.trial(trial_id)
                        ids, samples = bound.identity_id.to_numpy(), bound.image_id.to_numpy()
                        tables, curves, costs = {}, [], []
                        for profile in (ORIGIN_PROFILE, *config["pq_profiles"]):
                            progress(f"BLUFR {model} trial={trial_id} {profile}")
                            codec = None
                            if profile != ORIGIN_PROFILE:
                                m = int(profile.split("_m")[1].split("_")[0])
                                codec = PQCompressor(source_dim=512, m=m, nbits=8,
                                    random_state=int(config["codec_seed"])).fit(matrix[trial.train])
                                tables[profile + "_codec"] = pd.DataFrame({
                                    "serialized_faiss": [bytes(faiss.serialize_index(codec.index))]})
                            rows = search_rows(matrix[trial.probe], matrix[trial.gallery], samples[trial.probe],
                                ids[trial.probe], ids[trial.gallery], codec=codec, batch_size=config["batch_size"])
                            tables[profile + "_scores"] = rows
                            curves.append(benchmark_curve(rows, tuple(config["target_fpirs"]), (1, 20)).assign(
                                model=model, trial_id=trial_id, compression_profile=profile,
                                score_space="cosine_similarity" if codec is None else "negative_squared_l2_adc"))
                            costs.append(dict(compression_profile=profile, model=model, trial_id=trial_id,
                                training_image_count=len(trial.train), gallery_count=len(trial.gallery),
                                vector_payload_bytes=2048 if codec is None else codec.index.sa_code_size(),
                                codebook_bytes=0 if codec is None else int(faiss.vector_to_array(codec.index.pq.centroids).nbytes),
                                query_compression="origin_float32", gallery_compression=profile))
                        tables["benchmark_curves"] = pd.concat(curves, ignore_index=True)
                        tables["storage"] = pd.DataFrame(costs)
                        manifest = store.put(job, tables)
                        new_jobs += 1
                        del codec, tables
                        saved = store.get(job)
                    tables, manifest = saved
                    frames.append(tables["benchmark_curves"])
                    costs_accumulated.append(tables["storage"])
                    receipts.append(manifest["result_uid"])
                del prepared, matrix
                gc.collect()
    except (ResourceBudgetExceeded, KeyboardInterrupt) as error:
        stop_reason = str(error) or "user_interrupted"
    finally:
        faiss.omp_set_num_threads(previous_threads)
    result = dict(completed=len(receipts) == expected, completed_jobs=len(receipts), expected_jobs=expected,
                  new_jobs=new_jobs, stop_reason=stop_reason, checkpoint_path=str(checkpoint))
    if frames:
        curves = pd.concat(frames, ignore_index=True)
        summary = _summary(curves)
        report_spec = dict(**spec, completed=result["completed"], completed_jobs=len(receipts),
                           expected_jobs=expected, receipts=receipts)
        report_tables = {"benchmark_curves": curves, "benchmark_summary": summary,
                         "storage": pd.concat(costs_accumulated, ignore_index=True)}
        # Partial detail stays in SQLite. A compact snapshot is useful after each batch.
        result["chat_dir"] = str(write_benchmark_chat(root / "chat", report_tables, report_spec))
        if result["completed"]:
            result["report_dir"] = str(write_benchmark_report(root / "reports", report_tables, report_spec))
        result["benchmark_summary"] = summary
    return result


def assert_frozen_test_protocol(protocol, scores):
    """Reject drift in global LFW list files before deriving matched counts."""
    probes = pd.concat([protocol.registered_probes, protocol.known_unknown_probes,
                        protocol.unknown_unknown_probes], ignore_index=True)
    if (probes.image_id.duplicated().any() or scores.sample_id.duplicated().any()
            or set(probes.image_id) != set(scores.sample_id)):
        raise ValueError("reconstructed test cohort differs from frozen score cohort")
    ordered = probes.set_index("image_id").loc[scores.sample_id]
    if (not np.array_equal(ordered.identity_id.astype(str), scores.identity_id.astype(str))
            or not np.array_equal(ordered.identity_id.isin(protocol.gallery.identity_id), scores.is_mated)):
        raise ValueError("reconstructed test identity/mated labels differ from frozen scores")


def matched_conditions(project_root, group, output_root, *, resource_check=check_resources):
    """Freeze old test scores/codecs; regenerate only a size/enrollment-matched calibration search."""
    project = Path(project_root).resolve()
    conditions, fiqa = load_run_inputs(group, Path(output_root) / "legacy_origin_cache", progress=lambda _: None)
    root, run, workflow, prepared, population, lineage = _source(project, group.iloc[0].source_run_dir)
    seed = int(conditions[ORIGIN_PROFILE].manifest["calibration_seed"])
    test = calibration_protocol(run, population, "test", seed)
    assert_frozen_test_protocol(test, conditions[ORIGIN_PROFILE].test)
    enrollment = int(run["config"]["step4"]["evaluation"]["lfw_enrollment_count"])
    gallery_count = int(test.gallery.identity_id.nunique())
    if not test.gallery.groupby("identity_id").size().eq(enrollment).all():
        raise ValueError("test gallery enrollment is not uniform")
    cal = build_calibration_protocol(population, split_name="calibration",
        gallery_identity_count=gallery_count, enrollment_count=enrollment, seed=seed)
    arrays = open_set_protocol_arrays(cal, population)
    development_ids = set(population.loc[population.split.eq("development"), "identity_id"])
    if development_ids & set(cal.gallery.identity_id) or set(cal.gallery.identity_id) & set(test.gallery.identity_id):
        raise ValueError("development/calibration/test identity leakage")
    hashes = fiqa.scores.set_index("sample_id").aligned_content_sha256
    result = {}
    for profile, old in conditions.items():
        codec = None
        if profile != ORIGIN_PROFILE:
            codec, entry, bundle = _frozen_pq_codec(root, workflow, compression_profile=profile)
            if bundle["fit_seed"] != seed or int(entry["fit_count"]) != int(population.split.eq("development").sum()):
                raise ValueError("frozen PQ development fit contract differs")
        rows = search_rows(arrays["queries"], arrays["gallery"], arrays["query_ids"],
            arrays["query_identity_ids"], arrays["gallery_identity_ids"], codec=codec, resource_check=resource_check)
        cm = deepcopy(old.manifest)
        # Parent file hashes and old-threshold reproduction apply to the parent only.
        for key in ("files", "global_threshold_reproduction", "spec"):
            cm.pop(key, None)
        cm.update(protocol_uid=MATCHED_PROTOCOL, calibration_gallery_identities=gallery_count,
                  calibration_enrollment_count=enrollment, parent_condition_uid=old.condition_uid,
                  parent_condition_manifest_sha256=canonical_sha256(old.manifest),
                  calibration_protocol=dict(policy=MATCHED_PROTOCOL, gallery_identities=gallery_count,
                                            enrollment_count=enrollment, template="mean_then_l2"),
                  calibration_rows=len(rows), test_rows=len(old.test),
                  calibration_assignment_sha256=canonical_sha256(cal.gallery.image_id.tolist()),
                  test_scores_reused=True, frozen_codec_reused=codec is not None)
        cm["condition_uid"] = "lfw-matched-" + canonical_sha256(cm)[:24]
        rows["top_k"] = int(cm["top_k"])
        rows["top_k_correct"] = rows.is_mated & rows.true_identity_rank.le(cm["top_k"])
        rows.loc[~rows.top_k_correct, ["true_identity_rank", "true_identity_score"]] = np.nan
        rows["evaluation_split"] = "calibration"
        rows["aligned_content_sha256"] = hashes.loc[rows.sample_id].to_numpy()
        for name in ("dataset_id", "model_uid", "compression_profile", "search_mode", "score_space",
                     "protocol_uid", "extraction_uid", "origin_embedding_artifact_uid"):
            rows[name] = cm[name]
        test_rows = old.test.copy()
        test_rows["protocol_uid"] = MATCHED_PROTOCOL
        result[profile] = ConditionScoreTables(rows, test_rows, cm)
    inventory = pd.DataFrame([dict(protocol_uid=MATCHED_PROTOCOL, gallery_identities=gallery_count,
        enrollment_count=enrollment, calibration_queries=len(arrays["query_ids"]),
        calibration_non_mated=int((~result[ORIGIN_PROFILE].calibration.is_mated).sum()),
        development_identities=len(development_ids), test_unchanged=True,
        calibration_gallery_assignment_sha256=canonical_sha256(cal.gallery.image_id.tolist()))])
    return result, fiqa, inventory


def _pack_conditions(conditions, inventory):
    tables = {"inventory": inventory}
    manifests = []
    for profile, condition in conditions.items():
        tables[profile + "_cal"] = condition.calibration
        tables[profile + "_test"] = condition.test
        manifests.append(dict(profile=profile, manifest=json.dumps(condition.manifest)))
    tables["condition_manifests"] = pd.DataFrame(manifests)
    return tables


def _unpack_conditions(tables):
    return {row.profile: ConditionScoreTables(tables[row.profile + "_cal"],
            tables[row.profile + "_test"], json.loads(row.manifest))
            for row in tables["condition_manifests"].itertuples()}


def run_matched_lfw_calibration(project_root, *, run_matrix=None, models=None, partition_seeds=(*range(19), 8972),
        settings=None, output_root="results/calibration/lfw_matched", prepare_only=False,
        max_new_jobs=20, execute=False, progress=print, blas_threads=2,
        minimum_available_gb=8., maximum_process_gb=16., profiles=PQ_PROFILES, fiqa_variant="L"):
    """Use the existing calibrated-score core with separate, immutable matched inputs.

    ``prepare_only`` is the optional batch-01 entry point; notebook 03 can prepare
    missing inputs itself. Old PQ threshold models are NEVER reused here.
    """
    from research.fiqa import load_fiqa_score_artifact
    project = Path(project_root).resolve()
    matrix = run_matrix or DEFAULT_RUN_MATRIX
    selected = tuple(models or matrix)
    settings = settings or OriginPQSettings()
    settings.validate()
    seeds = tuple(partition_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("unique nonnegative partition seeds required")
    if max_new_jobs is not None and (type(max_new_jobs) is not int or max_new_jobs < 1):
        raise ValueError("max_new_jobs must be a positive integer or None")
    if type(blas_threads) is not int or blas_threads < 1:
        raise ValueError("blas_threads must be a positive integer")
    def resource_check():
        return check_resources(minimum_available_gb=minimum_available_gb, maximum_process_gb=maximum_process_gb)
    plan = inspect_origin_pq_experiment(project, run_matrix=matrix, datasets=("lfw",), models=selected,
                                       profiles=profiles, variant=fiqa_variant)
    if not execute:
        return dict(plan=plan, protocol_uid=MATCHED_PROTOCOL, stage="preflight", execute=False)
    root = project / output_root
    implementation = implementation_hashes()
    spec = dict(protocol_uid=MATCHED_PROTOCOL, plan=plan.to_dict("records"),
                partition_seeds=list(seeds), settings=settings.as_dict(), implementation=implementation,
                versions=runtime_versions(), blas_threads=blas_threads)
    checkpoint = root / ("campaign-" + canonical_sha256(spec)[:24] + ".sqlite3")
    accumulated = {name: [] for name in TABLES}
    inventories, receipts, new_jobs, stop_reason = [], [], 0, None
    import faiss
    previous_threads = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(blas_threads)
        with SplitStore(root / "matched_inputs.sqlite3") as inputs, SplitStore(checkpoint) as store, threadpool_limits(limits=blas_threads):
            for run_dir, group in plan.groupby("source_run_dir", sort=False):
                if not prepare_only and max_new_jobs is not None and new_jobs >= max_new_jobs:
                    stop_reason = "new_job_budget_reached"
                    break
                resource_check()
                input_spec = dict(protocol_uid=MATCHED_PROTOCOL, plan=group.to_dict("records"),
                                  implementation=implementation, versions=runtime_versions(), blas_threads=blas_threads)
                saved = inputs.get(input_spec)
                if saved is None:
                    progress(f"LFW matched calibration inputs: {group.iloc[0].model}")
                    conditions, fiqa, inventory = matched_conditions(project, group, root, resource_check=resource_check)
                    inputs.put(input_spec, _pack_conditions(conditions, inventory))
                else:
                    packed, _ = saved
                    conditions, inventory = _unpack_conditions(packed), packed["inventory"]
                    fiqa = load_fiqa_score_artifact(group.iloc[0].fiqa_dir)
                context = dict(dataset_id="lfw", model=group.iloc[0].model,
                               source_run_id=group.iloc[0].source_run_id)
                inventories.append(inventory.assign(**context))
                if prepare_only:
                    continue
                for seed in seeds:
                    job = dict(campaign=spec, context=context, partition_seed=seed)
                    saved = store.get(job)
                    if saved is None:
                        if max_new_jobs is not None and new_jobs >= max_new_jobs:
                            stop_reason = "new_job_budget_reached"
                            continue
                        resource_check()
                        progress(f"LFW matched {context['model']} partition_seed={seed}")
                        tables = run_origin_pq_split(conditions, fiqa, partition_seed=seed, settings=settings)
                        _validate_split_tables(tables, conditions, seed, settings)
                        tables = {name: frame.assign(**context) for name, frame in tables.items()}
                        manifest = store.put(job, tables)
                        new_jobs += 1
                    else:
                        tables, manifest = saved
                        _validate_split_tables(tables, conditions, seed, settings)
                    for name in TABLES:
                        accumulated[name].append(tables[name])
                    receipts.append(dict(**context, partition_seed=seed, result_uid=manifest["result_uid"]))
                del conditions, fiqa
                gc.collect()
    except (ResourceBudgetExceeded, KeyboardInterrupt) as error:
        stop_reason = str(error) or "user_interrupted"
    finally:
        faiss.omp_set_num_threads(previous_threads)
    expected = len(selected) * len(seeds)
    result = dict(protocol_uid=MATCHED_PROTOCOL, completed=len(receipts) == expected,
        completed_jobs=len(receipts), expected_jobs=expected, new_jobs=new_jobs,
        stop_reason=stop_reason, checkpoint_path=str(checkpoint), prepare_only=prepare_only)
    if prepare_only:
        result.update(completed=len(inventories) == len(selected), completed_jobs=len(inventories),
                      expected_jobs=len(selected), stage="matched_input_preparation")
    if inventories:
        result["inventory"] = pd.concat(inventories, ignore_index=True)
    if receipts:
        from research.experiments.origin_pq_compact import write_chat_bundle
        frames = {name: pd.concat(parts, ignore_index=True) for name, parts in accumulated.items()}
        report_spec = dict(**spec, receipts=receipts, completed=result["completed"],
                           protocol_inventory=result["inventory"].to_dict("records"))
        # Keep partial numerical details in SQLite; only publish a full report when complete.
        if result["completed"]:
            result["report_dir"] = str(write_report(root / "reports",
                {**frames, "protocol_inventory": result["inventory"]}, report_spec))
        expected_jobs = [f"{group.iloc[0].source_run_id}:{seed}"
                         for _, group in plan.groupby("source_run_dir") for seed in seeds]
        result["chat_dir"] = str(write_chat_bundle(root / "chat", frames, report_spec,
            expected_jobs=expected_jobs,
            completed_jobs=[f"{r['source_run_id']}:{r['partition_seed']}" for r in receipts]))
        result["method_summary"] = frames["method_summary"]
    return result


def read_lfw_protocol_report(directory):
    """Explicit report selection, with protocol and curve/operating interpretation."""
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    if metadata.get("artifact_type") == "blufr_open_set_benchmark":
        from zipfile import ZipFile
        from research.experiments.origin_pq_storage import _decode_frame
        if (metadata.get("status") != "completed" or metadata["spec"].get("completed") is not True
                or directory.name != "blufr-report-" + canonical_sha256(metadata["spec"])[:24]
                or sha256_file(directory / "results.zip") != metadata["archive_sha256"]):
            raise ValueError("invalid completed BLUFR report")
        with ZipFile(directory / "results.zip") as archive:
            expected = {name + ".parquet" for name in metadata["tables"]}
            if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
                raise ValueError("BLUFR archive inventory mismatch")
            tables = {name: _decode_frame(archive.read(name + ".parquet"), entry)
                      for name, entry in metadata["tables"].items()}
        return tables, metadata
    tables, manifest = read_report(directory)
    spec = manifest["spec"]
    if spec.get("kind") != "blufr_public_open_set_benchmark" and spec.get("protocol_uid") != MATCHED_PROTOCOL:
        raise ValueError("not a BLUFR benchmark or matched LFW calibration report")
    return tables, manifest


def write_benchmark_report(root, tables, spec):
    """Keep public test-curve semantics distinct from calibration report metadata."""
    from zipfile import ZipFile, ZIP_STORED
    from uuid import uuid4
    from research.experiments.origin_pq_storage import _encode_frame, _entry
    uid = "blufr-report-" + canonical_sha256(spec)[:24]
    destination = Path(root) / uid
    if destination.exists():
        existing, _ = read_lfw_protocol_report(destination)
        if set(existing) != set(tables):
            raise ValueError("cannot replace completed BLUFR report")
        for name, table in tables.items():
            pd.testing.assert_frame_equal(existing[name], table, check_exact=True)
        return destination
    if spec.get("completed") is not True:
        raise ValueError("partial benchmark details stay in the checkpoint")
    staging = Path(root) / (".staging-" + uuid4().hex)
    staging.mkdir(parents=True)
    entries = {}
    with ZipFile(staging / "results.zip", "w", compression=ZIP_STORED) as archive:
        for name, table in tables.items():
            blob = _encode_frame(table)
            archive.writestr(name + ".parquet", blob)
            entries[name] = _entry(table, blob)
    manifest = dict(artifact_type="blufr_open_set_benchmark", schema_version=1, status="completed",
        spec=spec, tables=entries, archive_sha256=sha256_file(staging / "results.zip"),
        threshold_source="test_curve_only", deployment_threshold_selected=False,
        uncertainty="descriptive_mean_and_sample_std_over_overlapping_trials_no_ci",
        formal_fpir_guarantee=False, checkpoint_training_overlap_verified=False)
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    staging.rename(destination)
    return destination


def write_benchmark_chat(root, tables, spec):
    """Lossless, paged CSV ZIP plus provenance and a concise reading guide."""
    from zipfile import ZipFile, ZIP_DEFLATED
    from uuid import uuid4
    uid = "blufr-chat-" + canonical_sha256(spec)[:24]
    destination = Path(root) / uid
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf8"))
        if manifest["spec"] != json.loads(json.dumps(spec)):
            raise ValueError("BLUFR chat provenance mismatch")
        for name, digest in manifest["files"].items():
            if sha256_file(destination / name) != digest:
                raise ValueError("BLUFR chat file hash mismatch")
        return destination
    staging = Path(root) / (".staging-" + uuid4().hex)
    staging.mkdir(parents=True)
    with ZipFile(staging / "analysis.zip", "w", compression=ZIP_DEFLATED) as zipped:
        for name, table in tables.items():
            for i, start in enumerate(range(0, len(table), 125)):
                zipped.writestr(f"{name}-{i:03d}.csv", table.iloc[start:start + 125].to_csv(index=False))
    text = ("# BLUFR public-list open-set benchmark\n\n"
            f"Completed: {spec['completed']}; jobs {spec['completed_jobs']}/{spec['expected_jobs']}\n\n"
            "Thresholds describe test score curves, not deployment calibration.\n"
            "toolkit_far and realized_fpir are both retained (ties can separate them).\n"
            "tpir_mean_minus_std uses sample std across trials; it is not a CI.\n"
            "No confidence/independence claim across overlapping trials.\n"
            "Lists came from the pinned toolkit mirror recorded in manifest.json; author authentication is unverified.\n"
            "Checkpoint training overlap and face-crop target correctness remain unaudited.\n"
            "Calibration extension is a separate LFW-Custom experiment.\n")
    (staging / "START_HERE.md").write_text(text, encoding="utf8")
    manifest = dict(spec=spec, files={name: sha256_file(staging / name)
                                    for name in ("analysis.zip", "START_HERE.md")})
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    staging.rename(destination)
    return destination
