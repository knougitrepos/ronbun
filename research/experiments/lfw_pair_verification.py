"""UCFace-style LFW pair evaluation with additional held-out compression/FIQA controls.

This is an adaptation using standard pairs, not a reproduction of UCFace weights,
preprocessing, training or its unreported threshold-selection implementation.
"""

from pathlib import Path
from functools import partial
import argparse
import json
import shutil

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from research.compression import PQCompressor
from research.datasets.lfw_pairs import load_pairs, calibration_partition, PROTOCOL_UID, PAIRS_SHA256, PAIRS_URL
from research.evaluation.lfw_verification import pair_scores, evaluate_fold
from research.experiments.calibration_matrix import MODEL_UIDS
from research.experiments.lfw_blufr_calibration import _quality
from research.experiments.lfw_resize_inputs import load_resize_source
from research.experiments.lfw_protocol_experiments import runtime_versions
from research.experiments.origin_vs_pq_calibration import OriginPQSettings, science_hashes
from research.experiments.origin_pq_resources import check_resources, ResourceBudgetExceeded
from research.experiments.lfw_pair_storage import PairStore, publish, read_report, cleanup_checkpoint
from research.runtime.hashing import canonical_sha256, sha256_file

CONFIG_PATH = "configs/experiments/lfw_pair_verification.yaml"
GUIDE = "notebooks/calibration/LFW_PAIR_VERIFICATION.md"


def settings_from_config(config):
    values = dict(config["fit_settings"])
    values["target_fpirs"] = tuple(values.pop("target_fmrs"))
    values["diagnostic_fpir_grid"] = tuple(values.pop("diagnostic_fmr_grid"))
    values["knot_quantiles"] = tuple(values["knot_quantiles"])
    settings = OriginPQSettings(**values)
    settings.validate()
    return settings


def inspect_verification(project_root, *, config_path=CONFIG_PATH, models=None, fold_ids=None,
                         partition_seeds=None, download_pairs=False):
    project = Path(project_root).resolve()
    config = yaml.safe_load((project / config_path).read_text(encoding="utf8"))
    selected = tuple(models if models is not None else config["source_runs"])
    folds = tuple(fold_ids if fold_ids is not None else config["fold_ids"])
    seeds = tuple(partition_seeds if partition_seeds is not None else config["partition_seeds"])
    if (not selected or len(set(selected)) != len(selected) or not set(selected) <= set(config["source_runs"])
            or not set(selected) <= set(MODEL_UIDS) or not folds or len(set(folds)) != len(folds)
            or not set(folds) <= set(range(1, 11)) or not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)):
        raise ValueError("invalid explicit model/fold/seed matrix")
    if config["pq_profiles"] != ["pq_512_m128_b8", "pq_512_m64_b8", "pq_512_m32_b8"]:
        raise ValueError("preserve the predeclared three-PQ matrix")
    settings = settings_from_config(config)
    population = pd.read_csv(project / config["image_manifest"])
    pairs, development = load_pairs(project / config["pairs_file"], population, download=download_pairs)
    required_ids = set(pairs.left_image_id) | set(pairs.right_image_id) | set(development.image_id)
    inventory = pd.DataFrame([calibration_partition(pairs, f, s, settings.safety_fraction)[2]
                              for f in folds for s in seeds])
    quality, _ = _quality(project, config)
    if quality.scores.sample_id.duplicated().any():
        raise ValueError("duplicate FIQA image IDs")
    quality_rows = quality.scores.set_index("sample_id")
    missing_quality = len(required_ids - set(quality_rows.index))
    sources, coverage = {}, []
    for model in selected:
        _, _, _, prepared, frame, lineage = load_resize_source(project, config["source_runs"][model])
        if prepared.model_uid != MODEL_UIDS[model]:
            raise ValueError("model/source label mismatch")
        if quality.manifest["aligned_bundle_manifest_sha256"] != lineage["aligned_bundle_manifest_sha256"]:
            raise ValueError("FIQA and recognition preprocessing differ")
        available = frame.image_id.isin(quality_rows.index)
        if not np.array_equal(frame.loc[available, "aligned_content_sha256"].astype(str),
                              quality_rows.loc[frame.loc[available, "image_id"], "aligned_content_sha256"].astype(str)):
            raise ValueError("FIQA/embedding per-image hash mismatch")
        lookup = frame.set_index("image_id").identity_id.astype(str)
        shared = population.loc[population.image_id.isin(lookup.index)]
        if not np.array_equal(lookup.loc[shared.image_id].to_numpy(), shared.identity_id.astype(str)):
            raise ValueError("source/population identity mismatch")
        missing = len(required_ids - set(frame.image_id))
        coverage.append(dict(model=model, required_images=len(required_ids), missing_embeddings=missing,
                             missing_fiqa=missing_quality, ready=not missing and not missing_quality))
        sources[model] = dict(**lineage, fiqa_manifest_sha256=canonical_sha256(quality.manifest))
    return dict(config=config, settings=settings, models=selected, folds=folds, seeds=seeds,
        pairs=pairs, development=development, inventory=inventory, coverage=pd.DataFrame(coverage),
        quality=quality_rows.fiqa_score, sources=sources, ready=all(x["ready"] for x in coverage),
        expected_jobs=len(selected)*len(folds)*len(seeds), development_images=len(development),
        development_identities=development.identity_id.nunique(), protocol_uid=PROTOCOL_UID)


def _implementation(project):
    paths = ("datasets/lfw_pairs.py", "evaluation/lfw_verification.py",
             "experiments/lfw_pair_verification.py", "experiments/lfw_pair_storage.py",
             "experiments/lfw_resize_inputs.py", "compression/profiles.py")
    return {**science_hashes(), **{p: sha256_file(project / "research" / p) for p in paths},
            "interpretation_guide": sha256_file(project / GUIDE)}


def prepare_model(project, plan, model, resource_check):
    import faiss
    _, _, _, prepared, _, _ = load_resize_source(project, plan["config"]["source_runs"][model])
    ids = pd.Index(prepared.sample_ids)
    vectors = prepared.normalized_embeddings
    development = vectors[ids.get_indexer(plan["development"].image_id)]
    scores, codecs = [], []
    for profile in ("origin", *plan["config"]["pq_profiles"]):
        resource_check()
        codec = None
        if profile != "origin":
            codec = PQCompressor(source_dim=512, m=int(profile.split("_m")[1].split("_")[0]),
                                 nbits=8, random_state=plan["config"]["codec_seed"]).fit(development)
            codecs.append(dict(compression_profile=profile, serialized_faiss=bytes(faiss.serialize_index(codec.index)),
                development_images=len(development), development_assignment_sha256=canonical_sha256(plan["development"].image_id.tolist()),
                codebook_bytes=512*256*4, code_payload_bytes=codec.m,
                faiss_recommended_training_images=39*256, below_recommended_training_count=len(development)<39*256))
        frame = plan["pairs"].copy()
        frame["pair_score"] = pair_scores(vectors, frame, ids, codec=codec, batch_size=128)
        frame["compression_profile"] = profile
        frame["score_space"] = "cosine_similarity" if codec is None else "negative_squared_l2_adc"
        scores.append(frame)
    return dict(scores=pd.concat(scores, ignore_index=True), codecs=pd.DataFrame(codecs))


def validate_job(tables, *, settings, profiles, test_count):
    rows = tables["operating"]
    if len(rows) != len(profiles)*3*len(settings.target_fpirs):
        raise ValueError("incomplete pair method/profile/target matrix")
    if rows.duplicated(["compression_profile", "method", "target_fmr"]).any():
        raise ValueError("duplicate operating point")
    expected = {(p, m, t) for p in profiles for m in ("global_safe", "fiqa_5bin", "continuous_fiqa") for t in settings.target_fpirs}
    if set(rows[["compression_profile", "method", "target_fmr"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("incomplete operating-point matrix")
    if (not rows.test_pairs.eq(test_count).all()
            or not (rows.true_accepts + rows.false_rejects).eq(rows.genuine_pairs).all()
            or not (rows.false_accepts + rows.true_rejects).eq(rows.impostor_pairs).all()
            or not np.array_equal(rows.target_met_on_test, rows.realized_fmr <= rows.target_fmr)
            or not np.allclose(rows.realized_fmr, rows.false_accepts/rows.impostor_pairs, rtol=0, atol=0)
            or not np.allclose(rows.tar, rows.true_accepts/rows.genuine_pairs, rtol=0, atol=0)):
        raise ValueError("invalid pair counts/rates/target flags")
    if (tables["diagnostic"].achieved_fmr > tables["diagnostic"].requested_fmr).any():
        raise ValueError("diagnostic FMR ceiling violated")


def run_verification(project_root, *, config_path=CONFIG_PATH, models=None, fold_ids=None,
                     partition_seeds=None, execute=False, download_pairs=False,
                     output_root=None, keep_raw_results=None, max_new_jobs=None, prepare_only=False,
                     blas_threads=2, minimum_available_gb=8., maximum_process_gb=16., progress=None):
    """Inspect by default; explicit execution resumes immutable model/fold/seed jobs."""
    project = Path(project_root).resolve()
    plan = inspect_verification(project, config_path=config_path, models=models, fold_ids=fold_ids,
                               partition_seeds=partition_seeds, download_pairs=download_pairs)
    result = {k: plan[k] for k in ("ready", "coverage", "inventory", "expected_jobs", "development_images", "development_identities", "protocol_uid")}
    if not execute:
        return result
    if not plan["ready"]:
        raise ValueError("complete LFW pair embeddings/FIQA required; no silent pair exclusion")
    if max_new_jobs is not None and (isinstance(max_new_jobs, bool) or not isinstance(max_new_jobs, int) or max_new_jobs < 1):
        raise ValueError("max_new_jobs must be a positive integer or None")
    keep = plan["config"]["keep_raw_results"] if keep_raw_results is None else keep_raw_results
    if not isinstance(keep, bool) or not isinstance(blas_threads, int) or blas_threads < 1:
        raise ValueError("invalid retention/thread setting")
    spec = dict(protocol_uid=PROTOCOL_UID, protocol_kind="pair_verification_1to1",
        config=plan["config"], config_sha256=sha256_file(project / config_path),
        models=plan["models"], folds=plan["folds"], seeds=plan["seeds"],
        sources=plan["sources"], implementation=_implementation(project), runtime=runtime_versions(),
        keep_raw_results=keep, blas_threads=blas_threads, pairs_sha256=PAIRS_SHA256, pairs_url=PAIRS_URL,
        pair_orientation="left_original_query_right_compressed_reference", quality_policy="left_query_only",
        development_assignment_sha256=canonical_sha256(plan["development"].image_id.tolist()),
        development_images=plan["development_images"], development_identities=int(plan["development_identities"]),
        image_manifest_sha256=sha256_file(project / plan["config"]["image_manifest"]))
    root = (project / (output_root or plan["config"]["output_root"])).resolve()
    campaign = root / ("campaign-" + canonical_sha256(spec)[:24])
    raw = campaign / "raw" / "checkpoint.sqlite3"
    expected = plan["expected_jobs"]
    final_uid = "lfw-pairs-report-" + canonical_sha256(dict(campaign=spec, completed_jobs=expected, expected_jobs=expected))[:24]
    final = campaign / "reports" / final_uid
    if final.exists():
        tables, meta = read_report(final)
        if not keep:
            cleanup_checkpoint(raw, campaign, final)
        return dict(result, completed=True, completed_jobs=expected, new_jobs=0,
                    report_dir=final, checkpoint_path=raw, reused=True, tables=tables)
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    # Conservative planning bound; actual filesystem size is recorded after each run.
    estimate = expected * (3_000_000 if keep else 300_000) + len(plan["models"])*30_000_000
    available = shutil.disk_usage(ancestor).free
    if available < estimate + 512*2**20:
        raise ResourceBudgetExceeded(f"insufficient disk: free={available}, planned={estimate} bytes; choose another output_root")
    result.update(estimated_peak_bytes=estimate, available_disk_bytes=available)
    resource_check = partial(check_resources, minimum_available_gb=minimum_available_gb, maximum_process_gb=maximum_process_gb)
    resource_check()
    campaign.mkdir(parents=True, exist_ok=True)
    ignore = campaign / ".gitignore"
    if not ignore.exists():
        ignore.write_text("raw/\nreports/.staging-*/\n", encoding="utf8")
    aggregate = {k: [] for k in ("operating", "diagnostic", "accuracy_cv", "paired", "inventory")}
    completed = new_jobs = 0
    stop_reason = None
    store = PairStore(raw)
    try:
        with threadpool_limits(limits=blas_threads):
            for model in plan["models"]:
                input_spec = dict(campaign=spec, kind="pair_scores_and_codecs", model=model)
                inputs = None
                for fold in plan["folds"]:
                    for seed in plan["seeds"]:
                        job_spec = dict(campaign=spec, kind="calibration", model=model, fold=fold, seed=seed)
                        saved = store.get(job_spec)
                        if saved is None:
                            if max_new_jobs is not None and new_jobs >= max_new_jobs and not prepare_only:
                                stop_reason = "max_new_jobs"
                                continue
                            resource_check()
                            if inputs is None:
                                inputs = store.get(input_spec)
                                if inputs is None:
                                    if progress:
                                        progress(dict(stage="PQ_fit_and_pair_scores", model=model))
                                    inputs = prepare_model(project, plan, model, resource_check)
                                    store.put(input_spec, inputs)
                            if prepare_only:
                                break
                            cal, test, inventory = calibration_partition(plan["pairs"], fold, seed, plan["settings"].safety_fraction)
                            if progress:
                                progress(dict(stage="calibration", model=model, fold=fold, seed=seed,
                                              completed=completed, expected=expected))
                            saved = evaluate_fold(inputs["scores"], plan["quality"], cal, test, seed=seed,
                                                  settings=plan["settings"], keep_decisions=keep)
                            saved["inventory"] = pd.DataFrame([inventory])
                            for name, frame in saved.items():
                                for key, value in dict(model=model, fold=fold, partition_seed=seed).items():
                                    frame[key] = value
                            validate_job(saved, settings=plan["settings"], profiles=("origin", *plan["config"]["pq_profiles"]), test_count=len(test))
                            store.put(job_spec, saved)
                            new_jobs += 1
                        if prepare_only:
                            break
                        validate_job(saved, settings=plan["settings"], profiles=("origin", *plan["config"]["pq_profiles"]), test_count=600)
                        completed += 1
                        for name in aggregate:
                            aggregate[name].append(saved[name])
                    if prepare_only:
                        break
    except ResourceBudgetExceeded as error:
        stop_reason = str(error)
    finally:
        store.close()
    result.update(completed=completed == expected, completed_jobs=completed, new_jobs=new_jobs,
                  checkpoint_path=raw, report_dir=None, stop_reason=stop_reason,
                  actual_checkpoint_bytes=raw.stat().st_size, keep_raw_results=keep)
    if completed:
        tables = {k: pd.concat(v, ignore_index=True) for k, v in aggregate.items()}
        report = publish(campaign, tables, spec=spec, completed_jobs=completed, expected_jobs=expected,
                         raw_path=raw, guide=project / GUIDE)
        compact, _ = read_report(report)
        result.update(report_dir=report, tables=compact)
        if completed == expected and not keep:
            cleanup_checkpoint(raw, campaign, report)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--download-pairs", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--max-new-jobs", type=int)
    parser.add_argument("--keep-raw-results", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    result = run_verification(args.project_root, config_path=args.config, execute=args.execute,
        download_pairs=args.download_pairs, output_root=args.output_root,
        max_new_jobs=args.max_new_jobs, keep_raw_results=args.keep_raw_results, progress=print)
    print(json.dumps({k: v for k, v in result.items() if k not in ("tables", "inventory", "coverage")}, default=str))
    print(result["coverage"].to_string(index=False))


if __name__ == "__main__":
    main()
