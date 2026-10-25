"""Stage 2: frozen-model/codec fit+safety diagnostics after a completed baseline.

Separate artifact, no FR training, no final-test scoring, no choice of a winner.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.calibration.sample_efficiency import fit_safety_learning_curve
from research.experiments.facex_inputs import read_inputs, write_json
from research.runtime.hashing import canonical_sha256, sha256_file


def validate_baseline(config, baseline, selected):
    from research.experiments.facex_workflow import read_results, _workload
    if not baseline:
        raise ValueError("Stage 2 requires an explicit completed baseline_report")
    path = Path(baseline).resolve()
    manifest = read_results(path)["manifest"]
    previous = manifest["contract"]
    if previous["config"].get("study", {}).get("stage", "baseline") != "baseline":
        raise ValueError("The prerequisite must be a baseline, not another diagnostic")
    if not set(selected) <= set(previous["datasets"]) or not set(selected) <= {"survface", "lfw", "rfw"}:
        raise ValueError("Stage 2 selects baseline threshold datasets only; TinyFace/SCFace have rank-only protocols")
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select unique stage-2 datasets")
    # Stage controls and retention can differ; model/quality/split/codec settings cannot.
    exclude = {"study", "sample_efficiency", "keep_raw_results"}
    if {k: v for k, v in config.items() if k not in exclude} != {k: v for k, v in previous["config"].items() if k not in exclude}:
        raise ValueError("Stage 2 must keep the baseline model, protocol, quality and codec configuration")
    _, models, _, _, identity = _workload(config, previous["datasets"])
    if identity != previous["inputs"]["identity"]:
        raise ValueError("Baseline input/model/code lineage changed; create a new baseline")
    return path, manifest, models


def frozen_codecs(baseline, alias):
    from research.compression import PCACompressor, PQCompressor
    inventory = json.loads((baseline / f"{alias}_codec_inventory.json").read_text(encoding="utf-8"))
    codecs = {}
    for profile, info in inventory.items():
        family = info["family"]
        codec = None
        if family != "origin":
            path = baseline / "codecs" / alias / info["codec_filename"]
            if sha256_file(path) != info["codec_sha256"]:
                raise ValueError("Frozen baseline codec changed")
            codec = (PCACompressor if family == "pca" else PQCompressor).load(path)
        codecs[profile] = (family, codec, info)
    return codecs


def write_summary_parts(root, metrics):
    """Small, complete condition tables; repeated context is kept in the index."""
    dimensions = ["dataset", "model_alias", "model_uid", "group", "outer_fold",
                  "compression_profile", "metric", "target_rate", "score_space", "sample_unit"]
    constant = ["scope", "interval_scope", "final_test_used", "final_safety_used", "formal_fpir_guarantee"]
    parts, restored = [], []
    for values, table in metrics.groupby(dimensions, dropna=False, sort=True):
        condition = dict(zip(dimensions, values))
        filename = f"summary_parts/{canonical_sha256(condition)[:24]}.csv"
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        compact = table.drop(columns=[*dimensions, *[c for c in constant if c in table]])
        compact.to_csv(path, index=False)
        loaded = pd.read_csv(path)
        pd.testing.assert_frame_equal(loaded, compact.reset_index(drop=True), check_dtype=False,
                                      check_exact=False, rtol=1e-12, atol=1e-12)
        restored.extend(loaded.status.tolist())
        parts.append(dict(path=filename, condition=condition, rows=len(table), bytes=path.stat().st_size,
                          estimated_text_tokens=path.stat().st_size//3))
    if sorted(restored) != sorted(metrics.status):
        raise ValueError("Diagnostic summary coverage mismatch")
    return parts


def diagnostics(population, vectors, quality, codecs, dataset, config, pairs=None):
    from research.experiments.facex_workflow import rank_scores, _gallery, normalize
    from research.protocols.facex_handoff import pair_partition
    from research.evaluation.lfw_verification import pair_scores, _frame
    setting = config.get("sample_efficiency", {})
    args = dict(fit_fractions=setting.get("fit_fractions", [.25, .5, .75, 1.]),
                safety_fractions=setting.get("safety_fractions", [.25, .5, .75, 1.]),
                folds=setting.get("folds", 5), seed=config.get("partition_seed", 8972),
                inner_safety_fraction=setting.get("inner_safety_fraction", .3))
    tables = []
    if dataset == "survface":
        # Exclude final safety, registered probes and all test rows before scoring.
        mask = population.role.eq("calibration_fit").to_numpy()
        rows = population.loc[mask].reset_index(drop=True)
        gallery, ids = _gallery(population, vectors, "calibration_gallery", templates=True)
        for profile, (family, codec, _) in codecs.items():
            frame, space = rank_scores(vectors[mask], gallery, rows.identity_id, ids, family=family, codec=codec)
            frame["sample_id"], frame["identity_id"], frame["fiqa_score"] = rows.image_id, rows.identity_id, quality[mask]
            for target in config["target_rates"]:
                table = fit_safety_learning_curve(frame, ["fit"]*len(frame), target=target, score_space=space, **args)
                tables.append(table.assign(compression_profile=profile, metric="FPIR", group="", outer_fold=0))
    elif dataset in ("lfw", "rfw"):
        q = pd.Series(quality, index=population.image_id)
        for group, group_pairs in pairs.groupby("group", sort=True):
            for fold in config["fold_ids"]:
                calibration, _, _ = pair_partition(group_pairs, fold, seed=config.get("partition_seed", 8972))
                candidate = calibration.loc[calibration.calibration_role.eq("fit") & ~calibration.is_genuine].copy()
                endpoints = set(candidate.left_image_id) | set(candidate.right_image_id)
                mask = population.image_id.isin(endpoints).to_numpy()
                candidate_vectors = vectors[mask]
                candidate_images = population.loc[mask, "image_id"]
                for profile, (family, codec, _) in codecs.items():
                    values = normalize(codec.transform(candidate_vectors)) if family == "pca" else candidate_vectors
                    scores = pair_scores(values, candidate, candidate_images, codec=codec if family == "pq" else None)
                    frame = _frame(candidate, pd.Series(scores, index=candidate.pair_id), q)
                    space = "negative_squared_l2_adc" if family == "pq" else "pca_cosine" if family == "pca" else "cosine"
                    for target in config["target_rates"]:
                        table = fit_safety_learning_curve(frame, ["fit"]*len(frame), target=target, score_space=space, **args)
                        tables.append(table.assign(compression_profile=profile, metric="FMR", group=group, outer_fold=fold))
    else:
        raise ValueError("No fit/safety threshold study for rank-only datasets")
    return pd.concat(tables, ignore_index=True)


def run_study(config, selected, root, *, keep_raw_results):
    from research.experiments.facex_workflow import _population, _science_hash, read_results
    from research.protocols.facex_handoff import load_handoff
    baseline, prior, models = validate_baseline(config, config["study"].get("baseline_report"), selected)
    contract = dict(config=config, datasets=selected, science=_science_hash(),
        baseline_report=str(baseline), baseline_run_uid=prior["run_uid"],
        baseline_manifest_sha256=sha256_file(baseline / "run_manifest.json"),
        inputs=prior["contract"]["inputs"], keep_raw_results=keep_raw_results)
    uid = "facex-fit-safety-" + canonical_sha256(contract)[:24]
    destination = root / "reports" / uid
    if destination.exists():
        read_results(destination)
        return dict(report_dir=str(destination), run_uid=uid, reused=True)
    work = destination.with_name(uid + ".partial")
    work.mkdir(parents=True, exist_ok=True)
    lock = work / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), run_uid=uid)))
    try:
        write_json(work / "run_manifest.json", dict(status="running", run_uid=uid, contract=contract))
        all_tables = []
        handoff = load_handoff(config["handoff"])
        with threadpool_limits(limits=2):
            for alias, (spec, trained_on) in models.items():
                if not any(d in (trained_on, "lfw", "rfw") for d in selected):
                    continue
                codecs = frozen_codecs(baseline, alias)
                for dataset in selected:
                    if dataset not in (trained_on, "lfw", "rfw"):
                        continue
                    item = contract["inputs"]["inputs"][f"{alias}/{dataset}"]
                    if sha256_file(Path(item["path"]) / "manifest.json") != item["manifest_sha256"]:
                        raise ValueError("Baseline input manifest changed")
                    _, population, vectors, _, quality = read_inputs(item["path"], expected_uid=spec.model_uid)
                    pairs = _population(config, handoff, dataset)[1] if dataset in ("lfw", "rfw") else None
                    table = diagnostics(population, vectors, quality, codecs, dataset, config, pairs)
                    all_tables.append(table.assign(dataset=dataset, model_alias=alias, model_uid=spec.model_uid))
        metrics = pd.concat(all_tables, ignore_index=True)
        metrics.to_csv(work / "metrics.csv", index=False)
        pd.DataFrame().to_csv(work / "paired.csv", index=False)
        summary = metrics.groupby(["dataset", "model_alias", "method", "status"], dropna=False).size().reset_index(name="conditions")
        summary.to_csv(work / "overview.csv", index=False)
        guide = Path(__file__).resolve().parents[2] / "notebooks/WORKFLOW_GUIDE.md"
        shutil.copyfile(guide, work / "HOW_TO_READ.md")
        # These are aggregate diagnostics only; no raw scores/identities are published.
        parts = write_summary_parts(work, metrics)
        summary_files = ("HOW_TO_READ.md", "overview.csv")
        write_json(work / "chatgpt_summary.json", dict(run_uid=uid,
            baseline_run_uid=prior["run_uid"], baseline_manifest_sha256=contract["baseline_manifest_sha256"],
            upload_order=["HOW_TO_READ.md", "chatgpt_summary.json", "overview.csv"],
            detailed_upload="Select named model/dataset/group/fold/profile/target summary_parts; full metrics.csv is optional",
            summary_bytes=sum((work / p).stat().st_size for p in summary_files),
            estimated_text_tokens=sum((work / p).stat().st_size for p in summary_files)//3,
            summary_parts=parts, detail_file_count=len(parts),
            detail_bytes=sum(p["bytes"] for p in parts),
            token_estimate_basis="UTF-8 bytes / 3, approximate, not a model limit",
            dependence_note="Overlapping folds/subsets are not independent; row-level Wilson is descriptive",
            status_counts=metrics.status.value_counts().to_dict(),
            formal_fpir_guarantee=False, final_test_used=False, final_safety_used=False))
        files = {p.relative_to(work).as_posix(): sha256_file(p) for p in work.rglob("*")
                 if p.is_file() and p.name not in ("RUNNING.lock", "run_manifest.json", "COMPLETED")}
        expected_jobs = sum(d in (trained, "lfw", "rfw") for _, trained in models.values() for d in selected)
        if len(all_tables) != expected_jobs:
            raise ValueError("Incomplete diagnostic job coverage")
        write_json(work / "run_manifest.json", dict(status="completed", run_uid=uid, contract=contract, files=files,
            completed_jobs=len(all_tables), expected_jobs=expected_jobs, keep_raw_results=keep_raw_results,
            study_stage="calibration_size", final_test_used=False, final_safety_used=False,
            status_counts=metrics.status.value_counts().to_dict(), formal_fpir_guarantee=False,
            raw_retention_scope="aggregate diagnostic only; baseline inputs/codecs preserved"))
        (work / "COMPLETED").write_text("completed\n", encoding="utf-8")
        work.rename(destination)
    finally:
        lock.unlink(missing_ok=True)
        (destination / "RUNNING.lock").unlink(missing_ok=True)
    read_results(destination)
    return dict(report_dir=str(destination), run_uid=uid, reused=False)
