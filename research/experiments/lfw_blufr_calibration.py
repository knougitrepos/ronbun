"""Public BLUFR splits with independent compression/FIQA calibration.

The archived BLUFR benchmark evaluator is never called. Only its existing
read-only source and bounded-search helpers are reused, without modifying it.
"""

from pathlib import Path
import gc
import json
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from research.compression import PQCompressor
from research.datasets.lfw_blufr import load_blufr_lists, bind_image_manifest
from research.datasets.lfw_blufr_calibration import build_calibration_split
from research.experiments.calibration_matrix import MODEL_UIDS, PQ_PROFILES
from research.experiments.fiqa_threshold_calibration import ConditionScoreTables
from research.experiments.lfw_protocol_experiments import (
    _source,
    search_rows,
    _pack_conditions,
    _unpack_conditions,
    runtime_versions,
)
from research.experiments.origin_pq_inputs import ORIGIN_PROFILE, ORIGIN_MODE
from research.experiments.origin_pq_resources import (
    check_resources,
    ResourceBudgetExceeded,
)
from research.experiments.origin_pq_storage import SplitStore, write_report, read_report
from research.experiments.origin_vs_pq_calibration import (
    OriginPQSettings,
    run_origin_pq_split,
    _validate_split_tables,
    science_hashes,
)
from research.fiqa import CRFIQA_VARIANTS, load_fiqa_score_artifact
from research.runtime.hashing import canonical_sha256, sha256_file

PROTOCOL_UID = "lfw-blufr-outer-split-independent-calibration-v1"
CONFIG_PATH = "configs/experiments/lfw_blufr_calibration.yaml"


def implementation_hashes():
    root = Path(__file__).resolve().parents[1]
    names = (
        "datasets/lfw_blufr.py",
        "datasets/lfw_blufr_calibration.py",
        "experiments/lfw_blufr_calibration.py",
        "experiments/lfw_protocol_experiments.py",
        "protocols/blufr.py",
        "compression/profiles.py",
    )
    return {**science_hashes(), **{n: sha256_file(root / n) for n in names}}


def _configuration(project, config_path):
    config = yaml.safe_load((project / config_path).read_text(encoding="utf8"))
    for name in ("trial_ids", "development_fractions", "pq_profiles"):
        values = config[name]
        if (
            not isinstance(values, list)
            or not values
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"unique nonempty {name} required")
    if not set(config["pq_profiles"]).issubset(PQ_PROFILES):
        raise ValueError("unsupported PQ profile")
    for name in ("calibration_gallery_identities", "top_k", "query_batch_size"):
        value = config[name]
        if type(value) is not int or value < 1:
            raise ValueError(f"positive integer {name} required")
    for name in ("split_seed", "codec_seed"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"nonnegative integer {name} required")
    if config["top_k"] > min(config["calibration_gallery_identities"], 1000):
        raise ValueError("top_k exceeds gallery size")
    if config["fiqa_variant"] not in CRFIQA_VARIANTS:
        raise ValueError("unsupported FIQA variant")
    return config


def _split(lists, bound, config, trial_id, fraction):
    return build_calibration_split(
        lists,
        bound,
        trial_id=trial_id,
        development_fraction=fraction,
        calibration_gallery_count=config["calibration_gallery_identities"],
        seed=config["split_seed"],
    )


def _quality(project, config):
    variant = CRFIQA_VARIANTS[config["fiqa_variant"]]
    path = project / config["fiqa_root"] / "lfw" / variant.model_uid
    quality = load_fiqa_score_artifact(path)
    if (
        quality.manifest.get("checkpoint_sha256") != variant.expected_sha256
        or quality.manifest.get("dataset_id") != "lfw"
    ):
        raise ValueError("FIQA checkpoint/dataset mismatch")
    return quality, path


def inspect_calibration(project_root, *, config_path=CONFIG_PATH, models=None):
    """Read-only inventory; missing embeddings/quality block execution, not inspection."""
    project = Path(project_root).resolve()
    config = _configuration(project, config_path)
    selected = tuple(models if models is not None else config["source_runs"])
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(MODEL_UIDS)
    ):
        raise ValueError("unique supported models required")
    lists = load_blufr_lists(
        project / config["protocol_file"], expected_sha256=config["protocol_sha256"]
    )
    bound = bind_image_manifest(lists, pd.read_csv(project / config["image_manifest"]))
    inventory = pd.DataFrame(
        [
            _split(lists, bound, config, trial, fraction).inventory
            for trial in config["trial_ids"]
            for fraction in config["development_fractions"]
        ]
    )
    if (inventory.development_images < 256).any():
        raise ValueError("8-bit PQ needs at least 256 training images in every split")
    quality = None
    quality_error = None
    try:
        quality, _ = _quality(project, config)
    except FileNotFoundError as error:
        quality_error = str(error)
    quality_ids = (
        set() if quality is None else set(quality.scores.sample_id.astype(str))
    )
    quality_missing = int((~bound.image_id.isin(quality_ids)).sum())
    coverage, missing, sources = [], [], {}
    for model in selected:
        _, run, _, prepared, population, lineage = _source(
            project, config["source_runs"][model]
        )
        if prepared.model_uid != MODEL_UIDS[model]:
            raise ValueError("source checkpoint/model label mismatch")
        vectors = prepared.normalized_embeddings
        if (
            vectors.shape != (len(prepared.sample_ids), 512)
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
        ):
            raise ValueError("finite normalized 512D embeddings required")
        lookup = pd.Series(prepared.identity_ids, index=prepared.sample_ids)
        available = bound.image_id.isin(lookup.index)
        if lookup.index.has_duplicates or not np.array_equal(
            lookup.loc[bound.loc[available, "image_id"]].astype(str),
            bound.loc[available, "identity_id"].astype(str),
        ):
            raise ValueError("embedding identity mapping differs from public list")
        if quality is not None:
            if (
                quality.manifest["aligned_bundle_manifest_sha256"]
                != lineage["aligned_bundle_manifest_sha256"]
            ):
                raise ValueError("FIQA and source alignment bundles differ")
            hashes = quality.scores.set_index("sample_id").aligned_content_sha256
            shared = population.loc[population.image_id.isin(hashes.index)]
            if not np.array_equal(
                shared.aligned_content_sha256.astype(str),
                hashes.loc[shared.image_id].astype(str),
            ):
                raise ValueError("FIQA/source per-image alignment hashes differ")
        missing.append(
            bound.loc[~available, ["filename", "image_id", "identity_id"]].assign(
                model=model
            )
        )
        coverage.append(
            dict(
                model=model,
                required=len(bound),
                available=int(available.sum()),
                missing_embeddings=int((~available).sum()),
                missing_fiqa=quality_missing,
                ready=bool(available.all() and quality_missing == 0),
                fiqa_error=quality_error,
            )
        )
        sources[model] = dict(
            **lineage,
            dataset_id="lfw",
            extraction_uid=prepared.extraction_uid,
            origin_embedding_artifact_uid=prepared.origin_embedding_artifact_uid,
            fiqa_manifest_sha256=None
            if quality is None
            else canonical_sha256(quality.manifest),
        )
        del prepared, population, vectors
    return dict(
        config=config,
        lists=lists,
        bound=bound,
        models=selected,
        sources=sources,
        coverage=pd.DataFrame(coverage),
        inventory=inventory,
        missing=pd.concat(missing, ignore_index=True),
        ready=all(row["ready"] for row in coverage),
        protocol_uid=PROTOCOL_UID,
        gallery_size_mismatch=True,
        benchmark_evaluator_used=False,
    )


def prepare_conditions(
    prepared,
    population,
    quality,
    lineage,
    split,
    config,
    *,
    resource_check=check_resources,
):
    """Fit codecs only on development; recompute both score spaces on fixed cohorts."""
    import faiss

    frame = split.assignment
    indices = pd.Index(prepared.sample_ids).get_indexer(frame.image_id)
    if (indices < 0).any():
        raise ValueError("complete public image embeddings required")
    vectors = prepared.normalized_embeddings[indices]
    source_hashes = population.set_index("image_id").aligned_content_sha256
    common = dict(
        **lineage,
        protocol_uid=PROTOCOL_UID,
        top_k=config["top_k"],
        calibration_seed=config["split_seed"],
        status="completed",
        schema_version=2,
        metric_contract="genuine-score-topk-v2",
        split_inventory=split.inventory,
        threshold_fit_on_test=False,
        gallery_size_mismatch=not split.inventory["gallery_size_matched"],
    )
    conditions, codecs = {}, []
    development = vectors[frame.role.eq("development")]
    for profile in (ORIGIN_PROFILE, *config["pq_profiles"]):
        resource_check()
        codec = None
        if profile != ORIGIN_PROFILE:
            m = int(profile.split("_m")[1].split("_")[0])
            codec = PQCompressor(
                source_dim=512, m=m, nbits=8, random_state=config["codec_seed"]
            ).fit(development)
            payload = bytes(faiss.serialize_index(codec.index))
            codecs.append(
                dict(
                    profile=profile,
                    serialized_faiss=payload,
                    fit_image_count=len(development),
                    fit_assignment_sha256=canonical_sha256(
                        frame.loc[frame.role.eq("development"), "image_id"].tolist()
                    ),
                )
            )
        score_space = (
            "cosine_similarity" if codec is None else "negative_squared_l2_adc"
        )
        mode = ORIGIN_MODE if codec is None else "pq_adc_exhaustive"
        manifest = dict(
            **common,
            compression_profile=profile,
            score_space=score_space,
            search_mode=mode,
            artifact_type="origin_calibration_test_score_tables"
            if codec is None
            else "compressed_calibration_test_score_tables",
        )
        manifest["condition_uid"] = "lfw-blufr-cal-" + canonical_sha256(manifest)[:24]
        tables = {}
        for phase in ("calibration", "test"):
            gallery = frame.loc[frame.role.eq(phase + "_gallery")].sort_values(
                "protocol_order"
            )
            queries = frame.loc[
                frame.role.isin([phase + "_mated", phase + "_non_mated"])
            ].sort_values("protocol_order")
            rows = search_rows(
                vectors[queries.index],
                vectors[gallery.index],
                queries.image_id.to_numpy(),
                queries.identity_id.to_numpy(),
                gallery.identity_id.to_numpy(),
                codec=codec,
                batch_size=config["query_batch_size"],
                resource_check=resource_check,
            )
            rows["top_k"] = config["top_k"]
            rows["rank1_correct"] = rows.is_mated & rows.true_identity_rank.eq(1)
            rows["top_k_correct"] = rows.is_mated & rows.true_identity_rank.le(
                config["top_k"]
            )
            rows.loc[
                ~rows.top_k_correct, ["true_identity_score", "true_identity_rank"]
            ] = np.nan
            rows["evaluation_split"] = phase
            rows["aligned_content_sha256"] = source_hashes.loc[
                rows.sample_id
            ].to_numpy()
            for key in (
                "dataset_id",
                "model_uid",
                "compression_profile",
                "search_mode",
                "score_space",
                "protocol_uid",
                "extraction_uid",
                "origin_embedding_artifact_uid",
            ):
                rows[key] = manifest[key]
            tables[phase] = rows
        conditions[profile] = ConditionScoreTables(
            tables["calibration"], tables["test"], manifest
        )
    packed = _pack_conditions(conditions, pd.DataFrame([split.inventory]))
    packed["assignment"] = frame
    packed["codecs"] = pd.DataFrame(codecs)
    return conditions, packed


def _compact(root, summary, inventory, spec):
    """CSV ZIP preserves trial and ratio; overlapping repeats are descriptive only."""
    keys = [
        "model",
        "trial_id",
        "development_fraction",
        "compression_profile",
        "method",
        "target_fpir",
    ]
    compact = (
        summary.groupby(keys, sort=False)
        .agg(
            fit_safety_seed_count=("partition_seed", "nunique"),
            target_met_count=("target_met_on_test", "sum"),
            fpir_min=("realized_fpir", "min"),
            fpir_median=("realized_fpir", "median"),
            fpir_max=("realized_fpir", "max"),
            tpir_min=("tpir_at_rank_k", "min"),
            tpir_median=("tpir_at_rank_k", "median"),
            tpir_max=("tpir_at_rank_k", "max"),
            test_mated_count=("test_mated_count", "first"),
            test_non_mated_count=("test_non_mated_count", "first"),
        )
        .reset_index()
    )
    uid = "calibration-chat-" + canonical_sha256(spec)[:24]
    directory = root / uid
    if directory.exists():
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
        if (
            manifest["spec"] != json.loads(json.dumps(spec))
            or sha256_file(directory / "analysis.zip") != manifest["sha256"]
        ):
            raise ValueError("compact report provenance/hash mismatch")
        return directory
    staging = root / (".staging-" + uuid4().hex)
    staging.mkdir(parents=True)
    with ZipFile(staging / "analysis.zip", "w", compression=ZIP_DEFLATED) as zipped:
        zipped.writestr("performance_by_trial_ratio.csv", compact.to_csv(index=False))
        zipped.writestr("split_inventory.csv", inventory.to_csv(index=False))
        zipped.writestr(
            "README.txt",
            "Derived BLUFR-split calibration, not the BLUFR benchmark.\n"
            "Calibration gallery=100 by default; released test gallery=1000. Size mismatch is intentional.\n"
            "Trials/ratios/fit-safety seeds overlap: ranges are descriptive, not independent CIs.\n"
            "Detailed fitted models, paired CIs, scores, codecs and assignments remain in SQLite.\n"
            f"Completed jobs: {spec['completed_jobs']} / {spec['expected_jobs']}\n",
        )
    (staging / "manifest.json").write_text(
        json.dumps(
            dict(spec=spec, sha256=sha256_file(staging / "analysis.zip")),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf8",
    )
    staging.rename(directory)
    return directory


def run_calibration(
    project_root,
    *,
    config_path=CONFIG_PATH,
    models=None,
    settings=None,
    partition_seeds=(*range(19), 8972),
    execute=False,
    prepare_only=False,
    max_new_jobs=1,
    blas_threads=2,
    minimum_available_gb=8.0,
    maximum_process_gb=16.0,
    progress=print,
):
    """Resume model/trial/ratio/fit-safety jobs in two SQLite files, never test-fit thresholds."""
    project = Path(project_root).resolve()
    settings = settings or OriginPQSettings()
    settings.validate()
    seeds = tuple(partition_seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("unique nonnegative partition seeds required")
    if max_new_jobs is not None and (type(max_new_jobs) is not int or max_new_jobs < 1):
        raise ValueError("positive max_new_jobs or None required")
    if type(blas_threads) is not int or blas_threads < 1:
        raise ValueError("positive thread count required")
    inspected = inspect_calibration(project, config_path=config_path, models=models)
    if not execute:
        return {
            key: inspected[key]
            for key in ("coverage", "inventory", "ready", "protocol_uid", "missing")
        }
    if not inspected["ready"]:
        raise ValueError(
            "Complete embeddings and FIQA required; no images may be silently excluded: "
            + inspected["coverage"].to_json(orient="records")
        )
    config = inspected["config"]
    root = (project / config["output_root"]).resolve()
    for source in config["source_runs"].values():
        source = (project / source).resolve()
        if root == source or source in root.parents or root in source.parents:
            raise ValueError("output must be separate from completed source runs")
    input_spec = dict(
        protocol_uid=PROTOCOL_UID,
        config=config,
        sources=inspected["sources"],
        implementation=implementation_hashes(),
        runtime=runtime_versions(),
    )
    spec = dict(**input_spec, settings=settings.as_dict(), partition_seeds=list(seeds))
    checkpoint = root / ("campaign-" + canonical_sha256(spec)[:24] + ".sqlite3")
    expected_inputs = len(inspected["models"]) * len(inspected["inventory"])
    expected = expected_inputs if prepare_only else expected_inputs * len(seeds)

    def resource_check():
        return check_resources(
            minimum_available_gb=minimum_available_gb,
            maximum_process_gb=maximum_process_gb,
        )

    import faiss

    old_threads = faiss.omp_get_max_threads()
    receipts, summaries, inventories = [], [], []
    new_jobs, stop_reason = 0, None
    try:
        faiss.omp_set_num_threads(blas_threads)
        with (
            SplitStore(root / "inputs.sqlite3") as inputs,
            SplitStore(checkpoint) as store,
            threadpool_limits(limits=blas_threads),
        ):
            quality, _ = _quality(project, config)
            for model in inspected["models"]:
                prepared = population = None
                for trial in config["trial_ids"]:
                    for fraction in config["development_fractions"]:
                        context = dict(
                            dataset_id="lfw",
                            model=model,
                            trial_id=trial,
                            development_fraction=fraction,
                            source_run_id=inspected["sources"][model]["source_run_id"],
                        )
                        key = dict(input_spec=input_spec, context=context)
                        saved = inputs.get(key)
                        if saved is None:
                            if max_new_jobs is not None and new_jobs >= max_new_jobs:
                                stop_reason = "new_job_budget_reached"
                                continue
                            resource_check()
                            if prepared is None:
                                _, _, _, prepared, population, _ = _source(
                                    project, config["source_runs"][model]
                                )
                            split = _split(
                                inspected["lists"],
                                inspected["bound"],
                                config,
                                trial,
                                fraction,
                            )
                            progress(
                                f"LFW inputs {model} trial={trial} compression_fraction={fraction}"
                            )
                            conditions, packed = prepare_conditions(
                                prepared,
                                population,
                                quality,
                                inspected["sources"][model],
                                split,
                                config,
                                resource_check=resource_check,
                            )
                            inputs.put(key, packed)
                            if prepare_only:
                                new_jobs += 1
                        else:
                            packed, _ = saved
                            conditions = _unpack_conditions(packed)
                        inventories.append(packed["inventory"].assign(model=model))
                        if prepare_only:
                            receipts.append(context)
                            continue
                        for seed in seeds:
                            job = dict(
                                campaign=spec, context=context, partition_seed=seed
                            )
                            saved_result = store.get(job)
                            if saved_result is None:
                                if (
                                    max_new_jobs is not None
                                    and new_jobs >= max_new_jobs
                                ):
                                    stop_reason = "new_job_budget_reached"
                                    continue
                                resource_check()
                                progress(
                                    f"LFW calibration {model} trial={trial} fraction={fraction} seed={seed}"
                                )
                                tables = run_origin_pq_split(
                                    conditions,
                                    quality,
                                    partition_seed=seed,
                                    settings=settings,
                                )
                                _validate_split_tables(
                                    tables, conditions, seed, settings
                                )
                                tables = {
                                    name: frame.assign(**context)
                                    for name, frame in tables.items()
                                }
                                result_manifest = store.put(job, tables)
                                new_jobs += 1
                            else:
                                tables, result_manifest = saved_result
                                _validate_split_tables(
                                    tables, conditions, seed, settings
                                )
                            summaries.append(tables["method_summary"])
                            receipts.append(
                                dict(
                                    **context,
                                    partition_seed=seed,
                                    result_uid=result_manifest["result_uid"],
                                )
                            )
                        del conditions, packed
                del prepared, population
                gc.collect()
    except (ResourceBudgetExceeded, KeyboardInterrupt) as error:
        stop_reason = str(error) or "user_interrupted"
    finally:
        faiss.omp_set_num_threads(old_threads)
    result = dict(
        protocol_uid=PROTOCOL_UID,
        completed=len(receipts) == expected,
        completed_jobs=len(receipts),
        expected_jobs=expected,
        new_jobs=new_jobs,
        stop_reason=stop_reason,
        checkpoint_path=str(checkpoint),
        prepare_only=prepare_only,
        gallery_size_mismatch=True,
    )
    if inventories:
        result["inventory"] = pd.concat(inventories, ignore_index=True)
    if summaries:
        summary = pd.concat(summaries, ignore_index=True)
        report_spec = dict(
            **spec,
            receipts=receipts,
            completed=result["completed"],
            completed_jobs=len(receipts),
            expected_jobs=expected,
            checkpoint_path=str(checkpoint),
            detailed_evidence="SQLite per-job tables; report contains operating metrics and split inventory",
        )
        result["chat_dir"] = str(
            _compact(root / "chat", summary, result["inventory"], report_spec)
        )
        if result["completed"]:
            result["report_dir"] = str(
                write_report(
                    root / "reports",
                    {
                        "method_summary": summary,
                        "protocol_inventory": result["inventory"],
                    },
                    report_spec,
                )
            )
        result["method_summary"] = summary
    return result


def read_calibration_report(directory):
    tables, manifest = read_report(directory)
    if (
        manifest["spec"].get("protocol_uid") != PROTOCOL_UID
        or manifest["spec"].get("completed") is not True
    ):
        raise ValueError("expected a derived BLUFR-split calibration report")
    return tables, manifest
