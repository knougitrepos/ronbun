"""Restartable 3 datasets x 4 checkpoints x 3 PQ budgets calibration campaign.

The notebook only configures this runner. Frozen source runs are read-only;
each fitted split is checkpointed, and reports keep all requested conditions.
"""

import gc
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import numpy as np
import scipy

from research.calibration.conditional import IDENTIFICATION_METRIC_CONTRACT
from research.experiments.calibration_protocols import OPEN_SET_DATASETS
from research.experiments.fiqa_threshold_calibration import (
    _completed_run, _frozen_pq_codec, _ledger_condition, _read_json, _verified_table,
    load_condition_score_artifact, replay_open_set_adc_condition_scores,
    write_condition_score_artifact,
)
from research.experiments.fiqa_continuous_calibration import run_continuous_calibration, write_continuous_calibration
from research.experiments.saliency_calibration_inputs import build_saliency_calibration_inputs, _atomic_json
from research.experiments.saliency_condition_binding import bind_saliency_condition
from research.experiments.saliency_incremental_calibration import (
    load_saliency_incremental_inputs, assess_incremental_gate,
    run_saliency_incremental_calibration, write_saliency_incremental_result,
)
from research.fiqa import (
    CRFIQA_VARIANTS, load_cr_fiqa, load_fiqa_score_artifact,
    materialize_aligned_bundle_score_artifact,
)
from research.runtime.hashing import canonical_sha256, sha256_file
from research.explainability.gradcam.artifacts import _publish_atomic_directory

MODEL_UIDS = {
    "arcface": "arcface-7972a704552df378345f", "adaface": "adaface-4df25b75e065b0b9ed43",
    "magface": "magface-6931178ad2025e1b3799", "edgeface": "edgeface-a348c305af33c223b337",
}
# Explicit audited campaign; no latest-run selection.
DEFAULT_RUN_MATRIX = {
    "arcface": {
        "lfw": "runs/lfw_20260902/20260902-R001-6c8b08b1_step4_lfw_arcface-7972a704552df378345f",
        "rfw_custom": "runs/rfw_custom_20260903/20260903-R001-af875d0f_step4_rfw_custom_arcface-7972a704552df378345f",
        "survface": "runs/survface_20260902/20260902-R001-61915edf_step4_survface_arcface-7972a704552df378345f",
    },
    "adaface": {
        "lfw": "runs/lfw_20260830/20260830-R001-2c810919_step4_lfw_adaface-4df25b75e065b0b9ed43",
        "rfw_custom": "runs/rfw_custom_20260830/20260830-R001-9f3f1375_step4_rfw_custom_adaface-4df25b75e065b0b9ed43",
        "survface": "runs/survface_20260830/20260830-R001-ec6e5d4a_step4_survface_adaface-4df25b75e065b0b9ed43",
    },
    "magface": {
        "lfw": "runs/lfw_20260831/20260831-R001-937290ca_step4_lfw_magface-6931178ad2025e1b3799",
        "rfw_custom": "runs/rfw_custom_20260831/20260831-R001-7c74c5f7_step4_rfw_custom_magface-6931178ad2025e1b3799",
        "survface": "runs/survface_20260831/20260831-R001-6695386d_step4_survface_magface-6931178ad2025e1b3799",
    },
    "edgeface": {
        "lfw": "runs/lfw_20260901/20260901-R001-03c98fe5_step4_lfw_edgeface-a348c305af33c223b337",
        "rfw_custom": "runs/rfw_custom_20260901/20260901-R001-0c811e8a_step4_rfw_custom_edgeface-a348c305af33c223b337",
        "survface": "runs/survface_20260901/20260901-R001-56c2f3ed_step4_survface_edgeface-a348c305af33c223b337",
    },
}
DEFAULT_SALIENCY_INPUTS = {
    "20260902-R001-61915edf": "saliency-inputs-a9b8252b75742667b08f67d8",
    "20260830-R001-ec6e5d4a": "saliency-inputs-7c807016541f24e49ac2a2ef",
    "20260831-R001-6695386d": "saliency-inputs-3e174994b9baae468caf21fd",
    "20260901-R001-56c2f3ed": "saliency-inputs-45dffd1416a3a8c075660f2e",
}
PQ_PROFILES = ("pq_512_m128_b8", "pq_512_m64_b8", "pq_512_m32_b8")


def inspect_calibration_matrix(project_root, run_matrix, *, datasets=OPEN_SET_DATASETS,
                               models=tuple(MODEL_UIDS), profiles=PQ_PROFILES):
    """Read-only preflight; fail before GPU work if any source condition is absent."""
    project = Path(project_root).resolve()
    for name, values, allowed in (("datasets", datasets, OPEN_SET_DATASETS),
                                   ("models", models, MODEL_UIDS), ("profiles", profiles, PQ_PROFILES)):
        if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
            raise ValueError(f"invalid unique {name}")
    rows = []
    dataset_contracts = {}
    for model in models:
        for dataset in datasets:
            root, run, workflow = _completed_run(project / run_matrix[model][dataset])
            if run["config"]["dataset_id"] != dataset or run["config"]["model_uid"] != MODEL_UIDS[model]:
                raise ValueError(f"source dataset/checkpoint differs: {model}/{dataset}")
            freeze = _read_json(workflow / "freeze_manifest.json")
            for key, expected in dict(run_id=run["run_id"], dataset_id=dataset, model_uid=MODEL_UIDS[model]).items():
                if freeze.get(key) != expected:
                    raise ValueError(f"source/freeze lineage mismatch: {key}")
            if freeze["scope"].get("data_fraction") != 1. or not freeze.get("fallback_free"):
                raise ValueError("matrix requires full, fallback-free completed source runs")
            if sha256_file(workflow / "selected_manifest.csv") != freeze["selected_manifest_sha256"]:
                raise ValueError("source selected manifest hash mismatch")
            contract = dict(selected_sha256=freeze["selected_manifest_sha256"],
                            aligned_sha256=freeze["aligned_bundle_manifest_sha256"],
                            seed=freeze["scope"]["seed"], evaluation=run["config"]["step4"]["evaluation"])
            if dataset in dataset_contracts and dataset_contracts[dataset] != contract:
                raise ValueError(f"mixed cohort/protocol across FR models: {dataset}")
            dataset_contracts[dataset] = contract
            prepared = _read_json(workflow / "prepared_population/manifest.json")
            for key in ("dataset_id", "model_uid", "extraction_uid"):
                if prepared[key] != freeze[key]:
                    raise ValueError(f"prepared/freeze lineage mismatch: {key}")
            ledger_root = workflow / "retrieval_ledger"
            ledger = _read_json(ledger_root / "manifest.json")
            source_saliency = _read_json(workflow / "saliency_population/manifest.json")
            for profile in profiles:
                _, codec, codec_manifest = _frozen_pq_codec(root, workflow, compression_profile=profile)
                for key, expected in dict(fit_source_run_id=run["run_id"], fit_source_dataset=dataset,
                                          model_uid=MODEL_UIDS[model]).items():
                    if codec_manifest.get(key) != expected:
                        raise ValueError(f"source/codec lineage mismatch: {key}")
                item = _ledger_condition(ledger, compression_profile=profile, search_mode="pq_adc_exhaustive")
                _verified_table(ledger_root, item["core"])
                rows.append(dict(dataset_id=dataset, model=model, model_uid=MODEL_UIDS[model],
                                 compression_profile=profile, source_run_id=run["run_id"],
                                 source_run_dir=str(root), source_manifest_sha256=sha256_file(root / "run_manifest.json"),
                                 codec_sha256=codec["artifact_sha256"], test_core_sha256=item["core"]["sha256"],
                                 test_rows=item["row_count"], ready=True,
                                 source_saliency_target=source_saliency.get("target_name"),
                                 needs_origin_top1_saliency=source_saliency.get("target_name") != "origin_top1_gallery_cosine"))
    return pd.DataFrame(rows)


def _fiqa_artifact(project, run, result_root, variant, batch_size, progress):
    dataset = run["config"]["dataset_id"]
    aligned = project / run["config"]["step4"]["datasets"][dataset]["aligned_bundle_dir"]
    spec = CRFIQA_VARIANTS[variant]
    destination = result_root / "fiqa_scores" / dataset / spec.model_uid
    if destination.exists():
        artifact = load_fiqa_score_artifact(destination)
    else:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("FIQA inference requires CUDA; CPU fallback disabled")
        progress(dict(stage="fiqa_inference", dataset=dataset, device=torch.cuda.get_device_name(0)))
        model, loaded = load_cr_fiqa(project / "models/fiqa" / f"CR-FIQA({variant}).pth", variant=variant, device="cuda")
        artifact = materialize_aligned_bundle_score_artifact(
            aligned, destination, model=model, model_uid=loaded.model_uid,
            checkpoint_sha256=loaded.expected_sha256, variant=variant,
            batch_size=batch_size, device="cuda", overwrite=False)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    expected = dict(dataset_id=dataset, fiqa_model_uid=spec.model_uid,
                    checkpoint_sha256=spec.expected_sha256,
                    aligned_bundle_manifest_sha256=sha256_file(aligned / "bundle_manifest.json"))
    for key, value in expected.items():
        if artifact.manifest.get(key) != value:
            raise ValueError(f"existing FIQA artifact mismatch: {key}")
    return destination


def prepare_calibration_matrix(project_root, plan, *, variant="L", result_root=None,
                               saliency_overrides=None, input_settings=None,
                               fiqa_batch_size=64, progress=print):
    """Prepare all conditions; expensive origin saliency is built once per run."""
    project = Path(project_root).resolve()
    result_root = Path(result_root or project / "results/calibration")
    options = dict(device="cuda", reuse_test_saliency=True, gradcam_batch_size=4, chunk_size=128,
                   faithfulness_batch_size=32, faithfulness_maximum_samples=10000,
                   occlusion_fraction=.1, random_repeats=5, seed=8972, bootstrap_repeats=2000)
    options.update(input_settings or {})
    if options.get("max_queries_per_split") is not None:
        raise ValueError("matrix input preparation requires full cohorts")
    overrides = dict(saliency_overrides or {})
    inputs = []
    for run_dir, group in plan.groupby("source_run_dir", sort=False):
        root, run, workflow = _completed_run(run_dir)
        if sha256_file(root / "run_manifest.json") != group.iloc[0].source_manifest_sha256:
            raise ValueError("source manifest changed since preflight")
        run_id = run["run_id"]
        fiqa_dir = _fiqa_artifact(project, run, result_root, variant, fiqa_batch_size, progress)
        anchor = bundle = None
        for row in group.itertuples(index=False):
            progress(dict(stage="condition", dataset=row.dataset_id, model=row.model, profile=row.compression_profile))
            directory = result_root / "condition_scores" / run_id / f"{row.compression_profile}__pq_adc_exhaustive" / IDENTIFICATION_METRIC_CONTRACT
            if directory.exists():
                condition = load_condition_score_artifact(directory)
            else:
                condition = write_condition_score_artifact(directory, replay_open_set_adc_condition_scores(
                    root, compression_profile=row.compression_profile), overwrite=False)
            for key, expected in dict(source_run_id=run_id, dataset_id=row.dataset_id, model_uid=row.model_uid,
                                      compression_profile=row.compression_profile, search_mode="pq_adc_exhaustive",
                                      source_run_manifest_sha256=row.source_manifest_sha256).items():
                if condition.manifest.get(key) != expected:
                    raise ValueError(f"condition lineage mismatch: {key}")
            if anchor is None:
                anchor = condition
                if run_id in overrides:
                    if row.compression_profile != "pq_512_m128_b8":
                        raise ValueError("pinned historical input override requires m128 first")
                    d = Path(overrides[run_id])
                    complete = _read_json(d / "manifest.json")
                    if complete.get("status") != "completed":
                        raise ValueError("saliency override is not completed")
                    # Overrides are explicit historical evidence, with its real producer retained.
                    for key in ("seed", "occlusion_fraction", "random_repeats", "faithfulness_maximum_samples", "bootstrap_repeats"):
                        if complete["spec"]["settings"][key] != options[key]:
                            raise ValueError(f"saliency override settings mismatch: {key}")
                    bundle = dict(saliency_directory=d / "saliency", faithfulness_directory=d / "faithfulness")
                else:
                    run_options = dict(options)
                    # LFW/RFW source saliency can use an identity prototype target.
                    # Regenerate those valid but incompatible features with origin-top1.
                    source_saliency = workflow / "saliency_population/manifest.json"
                    if run_options["reuse_test_saliency"] and (
                            not source_saliency.exists() or
                            _read_json(source_saliency).get("target_name") != "origin_top1_gallery_cosine"):
                        run_options["reuse_test_saliency"] = False
                        progress(dict(stage="saliency_regenerate", dataset=row.dataset_id, model=row.model,
                                      reason="existing target is not split-matched origin-top1"))
                    bundle = build_saliency_calibration_inputs(root, condition, result_root / "saliency_inputs" / run_id,
                                                               progress=progress, **run_options)
                bound = bundle
            else:
                bound = bind_saliency_condition(anchor, condition, bundle["saliency_directory"],
                                                  bundle["faithfulness_directory"], result_root / "saliency_bindings" / run_id)
            evidence = load_saliency_incremental_inputs(condition, bound["saliency_directory"], bound["faithfulness_directory"])
            gate = assess_incremental_gate(condition, evidence)
            if not gate["comparison_enabled"]:
                raise ValueError(f"saliency integrity gate blocked: {gate['reasons']}")
            inputs.append(dict(dataset_id=row.dataset_id, model=row.model, model_uid=row.model_uid,
                               compression_profile=row.compression_profile, source_run_id=run_id,
                               condition_dir=str(directory), condition_sha256=canonical_sha256(condition.manifest),
                               fiqa_dir=str(fiqa_dir), saliency_dir=str(bound["saliency_directory"]),
                               faithfulness_dir=str(bound["faithfulness_directory"]),
                               faithfulness_status=gate["faithfulness_status"]))
        gc.collect()
    return pd.DataFrame(inputs)


def _load_result(directory, *, allowed_status=("completed",)):
    directory = Path(directory)
    manifest = _read_json(directory / "manifest.json")
    if manifest.get("status") not in allowed_status:
        raise ValueError("unfinished matrix result")
    tables = {}
    for name, digest in manifest["files"].items():
        if Path(name).name != name or sha256_file(directory / name) != digest:
            raise ValueError("matrix result file hash/path mismatch")
        tables[Path(name).stem] = pd.read_csv(directory / name)
    return {**tables, "manifest": manifest}


def run_calibration_matrix(inputs, output_root, *, partition_seeds=(8972,),
                           target_fpirs=(.01, .05, .1, .2, .3), resamples=2000,
                           bootstrap_seed=8972, safety_fraction=.3, knot_quantiles=(1/3, 2/3),
                           smoothing=.01, ridge=.001, max_iterations=2000,
                           minimum_group_non_mated=100, shrinkage_strength=200., progress=print):
    """Checkpoint every method family/seed; reruns load checked results before fitting."""
    if not partition_seeds or len(set(partition_seeds)) != len(partition_seeds):
        raise ValueError("unique partition seeds required")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    common = dict(target_fpirs=target_fpirs, resamples=resamples, bootstrap_seed=bootstrap_seed,
                  safety_fraction=safety_fraction, knot_quantiles=knot_quantiles, smoothing=smoothing,
                  ridge=ridge, max_iterations=max_iterations)
    # Include transitive scientific modules so cached fits cannot hide code changes.
    research = Path(__file__).parents[1]
    modules = sorted(set(research.joinpath("calibration").glob("*.py")) |
                     set(research.joinpath("experiments").glob("*fiqa*.py")) |
                     set(research.joinpath("experiments").glob("*saliency*.py")) |
                     {Path(__file__), research / "evaluation/cluster_bootstrap.py", research / "evaluation/metrics.py"})
    implementation = {str(p.relative_to(research)): sha256_file(p) for p in modules}
    receipts = []
    for row in inputs.itertuples(index=False):
        condition = load_condition_score_artifact(row.condition_dir)
        if canonical_sha256(condition.manifest) != row.condition_sha256:
            raise ValueError("condition changed after matrix input preparation")
        fiqa = load_fiqa_score_artifact(row.fiqa_dir)
        saliency = load_saliency_incremental_inputs(condition, row.saliency_dir, row.faithfulness_dir)
        input_hashes = dict(condition=row.condition_sha256, fiqa=canonical_sha256(fiqa.manifest),
                            saliency=canonical_sha256(saliency["saliency_manifest"]),
                            faithfulness=canonical_sha256(saliency["faithfulness_manifest"]))
        for seed in partition_seeds:
            pair = {}
            for family in ("fiqa", "saliency"):
                spec = json.loads(json.dumps(dict(input_hashes=input_hashes, seed=seed, family=family,
                    common=common, minimum_group_non_mated=minimum_group_non_mated,
                    shrinkage_strength=shrinkage_strength, implementation=implementation,
                    versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__))))
                job = root / ("job-" + canonical_sha256(spec)[:24])
                receipt_path = job / "receipt.json"
                progress(dict(stage=family, dataset=row.dataset_id, model=row.model,
                              profile=row.compression_profile, seed=seed, reuse=receipt_path.exists()))
                if receipt_path.exists():
                    receipt = _read_json(receipt_path)
                    if receipt["spec"] != spec:
                        raise ValueError("matrix checkpoint settings differ")
                    destination = Path(receipt["result_dir"])
                    result = _load_result(destination)
                    if canonical_sha256(result["manifest"]) != receipt["result_manifest_sha256"]:
                        raise ValueError("matrix checkpoint manifest hash mismatch")
                else:
                    job.mkdir(parents=True, exist_ok=True)
                    if family == "fiqa":
                        result = run_continuous_calibration(condition, fiqa, partition_seeds=(seed,),
                            minimum_group_non_mated=minimum_group_non_mated, shrinkage_strength=shrinkage_strength, **common)
                        destination = write_continuous_calibration(job, result)
                    else:
                        result = run_saliency_incremental_calibration(condition, fiqa, saliency,
                            baseline_method="continuous_fiqa", partition_seeds=(seed,), **common)
                        destination = write_saliency_incremental_result(job, result)
                    result = _load_result(destination)
                    _atomic_json(receipt_path, dict(spec=spec, result_dir=str(destination.resolve()),
                        result_manifest_sha256=canonical_sha256(result["manifest"])))
                pair[family] = result["method_summary"]
                receipts.append(dict(dataset_id=row.dataset_id, model=row.model, model_uid=row.model_uid,
                                     compression_profile=row.compression_profile, source_run_id=row.source_run_id,
                                     condition_sha256=row.condition_sha256, partition_seed=seed, family=family,
                                     result_dir=str(destination.resolve()),
                                     result_manifest_sha256=canonical_sha256(result["manifest"])))
            # Both notebooks refit the same baseline. A discrepancy invalidates aggregation.
            cols = ["target_fpir", "realized_fpir", "tpir_at_rank_k"]
            left = pair["fiqa"].loc[pair["fiqa"].method.eq("continuous_fiqa"), cols].reset_index(drop=True)
            right = pair["saliency"].loc[pair["saliency"].method.eq("baseline"), cols].reset_index(drop=True)
            pd.testing.assert_frame_equal(left, right, check_exact=True)
    return pd.DataFrame(receipts)


def summarize_calibration_matrix(receipts, plan, *, expected_seeds, output_root=None):
    """Keep all condition/seed rows and explicit completeness; never pool datasets."""
    expected = {(r.dataset_id, r.model, r.compression_profile, seed, family)
                for r in plan.itertuples(index=False) for seed in expected_seeds for family in ("fiqa", "saliency")}
    observed = set()
    metrics, pairs = [], []
    for row in receipts.itertuples(index=False):
        key = (row.dataset_id, row.model, row.compression_profile, row.partition_seed, row.family)
        if key in observed:
            raise ValueError("duplicate matrix result receipt")
        observed.add(key)
        result = _load_result(row.result_dir)
        if canonical_sha256(result["manifest"]) != row.result_manifest_sha256:
            raise ValueError("report source manifest hash mismatch")
        manifest = result["manifest"]
        expected_lineage = dict(dataset_id=row.dataset_id, compression_profile=row.compression_profile,
                                source_run_id=row.source_run_id, model_uid=row.model_uid,
                                condition_manifest_sha256=row.condition_sha256,
                                artifact_type={"fiqa": "fiqa_continuous_calibration",
                                               "saliency": "saliency_incremental_calibration"}[row.family])
        for name, value in expected_lineage.items():
            if manifest.get(name) != value:
                raise ValueError(f"report receipt/source lineage mismatch: {name}")
        if (manifest["settings"]["partition_seeds"] != [row.partition_seed]
                or set(result["method_summary"].partition_seed) != {row.partition_seed}):
            raise ValueError("report receipt/source partition mismatch")
        labels = {k: getattr(row, k) for k in ("dataset_id", "model", "model_uid", "compression_profile", "source_run_id", "family")}
        for name, out in (("method_summary", metrics), ("paired_comparisons", pairs)):
            frame = result[name].copy()
            if name == "method_summary" and row.family == "saliency":
                frame = frame.loc[~frame.method.eq("baseline")].copy()
            out.append(frame.assign(**labels, result_dir=row.result_dir))
    if observed - expected:
        raise ValueError("unrequested matrix conditions in report")
    all_metrics = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame(columns=["method", "target_fpir"])
    all_pairs = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=["metric", "target_fpir"])
    summary = pd.DataFrame(columns=["method", "target_fpir"])
    if not all_metrics.empty:
        summary = all_metrics.groupby(["dataset_id", "model", "compression_profile", "method", "target_fpir"]).agg(
            split_count=("partition_seed", "nunique"), target_met_split_count=("target_met_on_test", "sum"),
            fpir_min=("realized_fpir", "min"), fpir_median=("realized_fpir", "median"), fpir_max=("realized_fpir", "max"),
            tpir_min=("tpir_at_rank_k", "min"), tpir_median=("tpir_at_rank_k", "median"), tpir_max=("tpir_at_rank_k", "max")).reset_index()
    manifest = dict(artifact_type="calibration_matrix_report", status="completed" if observed == expected else "partial",
                    expected_jobs=len(expected), observed_jobs=len(observed), missing_jobs=sorted(expected - observed),
                    sources=receipts.to_dict("records"), metric_contract=IDENTIFICATION_METRIC_CONTRACT,
                    interpretation="same test/gallery; split ranges descriptive, not independent replications",
                    formal_fpir_guarantee=False, multiple_comparison_adjustment="none",
                    unseen_identity_claim_for_rfw_edgeface=False)
    directory = None
    if output_root is not None:
        directory = Path(output_root) / ("matrix-report-" + canonical_sha256(manifest)[:24])
        if directory.exists():
            saved = _load_result(directory, allowed_status=("completed", "partial"))["manifest"]
            if {k: v for k, v in saved.items() if k != "files"} != json.loads(json.dumps(manifest)):
                raise ValueError("matrix report manifest differs from requested sources")
        else:
            directory.parent.mkdir(parents=True, exist_ok=True)
            staging = directory.parent / (".staging-" + uuid4().hex)
            staging.mkdir()
            files = {}
            for name, frame in dict(method_summary=all_metrics, paired_comparisons=all_pairs, split_summary=summary).items():
                path = staging / f"{name}.csv"
                frame.to_csv(path, index=False)
                files[path.name] = sha256_file(path)
            _atomic_json(staging / "manifest.json", {**manifest, "files": files})
            _publish_atomic_directory(staging, directory, overwrite=False)
    return dict(method_summary=all_metrics, paired_comparisons=all_pairs, split_summary=summary,
                manifest=manifest, directory=directory)
