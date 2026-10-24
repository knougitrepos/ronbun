"""Sequential FaceX producer -> 1024D inputs -> dataset-specific evaluation.

SurvFace uses fixed non-mated fit/safety identities, TinyFace and SCFace rank
evaluation, LFW/RFW official test pairs plus an added calibration protocol.
No FR training or threshold fitting on final test is performed here.
"""
from pathlib import Path
import json
import math
import os
import shutil

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from research.protocols.facex_handoff import load_handoff, load_pair_dataset, pair_partition, EVALUATION_DATASETS
from research.experiments.facex_inputs import materialize_inputs, read_inputs, write_json
from research.runtime.hashing import canonical_sha256, sha256_file
from research.runtime.source_fingerprints import source_fingerprints


def _config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("embedding_dim") != 1024:
        raise ValueError("The active FaceX workflow requires 1024D")
    if not config.get("datasets") or not set(config["datasets"]) <= set(EVALUATION_DATASETS):
        raise ValueError("Unknown dataset selection")
    targets = config["target_rates"]
    if not targets or len(set(targets)) != len(targets) or any(not 0 < t < 1 for t in targets):
        raise ValueError("Explicit unique target rates in (0,1) required")
    return config


def _models(config, handoff):
    from research.embeddings.manifests import read_model_spec
    result = {}
    for alias, entry in config["models"].items():
        spec = read_model_spec(Path(config["project_root"]) / entry["spec"])
        if spec.embedding_dim != 1024 or not spec.module_factory.startswith("research.embeddings.pytorch.facex_zoo:"):
            raise ValueError("Register a FaceX 1024D export before selecting it")
        source = Path(spec.checkpoint.path)
        meta = json.loads(source.with_name(source.stem + "_meta.json").read_text(encoding="utf-8"))
        dataset = entry["training_dataset"]
        if dataset not in handoff or meta["dataset"] != dataset:
            raise ValueError("Model training dataset differs from its configured protocol")
        expected = handoff[dataset][0]
        if meta.get("protocol_record", {}).get("roles_sha256") != expected["roles_sha256"]:
            raise ValueError("Export was not trained with this frozen role manifest")
        if meta.get("protocol_record", {}).get("training_identity_sha256") != expected["training_identity_sha256"]:
            raise ValueError("Export training identity hash mismatch")
        from research.experiments.facex_inputs import input_transform_from_metadata
        input_transform_from_metadata(meta)
        result[alias] = (spec, dataset)
    if len({spec.model_uid for spec, _ in result.values()}) != len(result):
        raise ValueError("Duplicate model UID under multiple aliases")
    return result


def _population(config, handoff, dataset):
    if dataset in handoff:
        return handoff[dataset][1], None
    population, pairs = load_pair_dataset(config["project_root"], config["pairs"][dataset], dataset)
    if dataset == "rfw":
        from research.experiments.facex_inputs import archive_order
        population = archive_order(population, config["project_root"])
    return population, pairs


def inspect_workflow(config_path, *, datasets=None):
    config = _config(config_path)
    selected = list(datasets or config["datasets"])
    if not set(selected) <= set(config["datasets"]):
        raise ValueError("Dataset selection is not predeclared")
    handoff = load_handoff(config["handoff"]) if Path(config["handoff"]).is_file() else {}
    models = _models(config, handoff) if config["models"] else {}
    rows = []
    for dataset in selected:
        is_pair = dataset in ("lfw", "rfw")
        ready = (all((Path(config["project_root"]) / v).is_file() for v in config["pairs"][dataset].values())
                 if is_pair else dataset in handoff)
        matched = [alias for alias, (_, trained_on) in models.items() if is_pair or trained_on == dataset]
        row = dict(dataset=dataset, protocol_ready=ready, selected_models=len(matched), model_aliases=matched,
                   task=("pair TAR/FMR" if is_pair else "TPIR@20/FPIR" if dataset == "survface" else "rank"),
                   status="model_specs_required" if not matched else "ready_for_inputs" if ready else "protocol_required")
        if dataset in handoff:
            record = handoff[dataset][0]
            row.update(fr_identities=record["num_classes"], fr_images=record["images"],
                       role_counts=record["role_counts"])
        rows.append(row)
    return pd.DataFrame(rows)


def _workload(config, datasets):
    handoff = load_handoff(config["handoff"])
    models = _models(config, handoff)
    if not models:
        raise ValueError("Configure exact registered model specs in models; no automatic latest checkpoint selection")
    if not set(datasets) <= set(config["datasets"]):
        raise ValueError("Select predeclared datasets")
    missing = [d for d in datasets if not any(d in (trained_on, "lfw", "rfw") for _, trained_on in models.values())]
    if missing or not datasets:
        raise ValueError(f"No applicable registered model for selected datasets: {missing or datasets}")
    models = {a: (s, d) for a, (s, d) in models.items() if set(datasets) & {d, "lfw", "rfw"}}
    populations = {}
    jobs = []
    for alias, (spec, trained_on) in models.items():
        needed = {trained_on} | (set(datasets) & {trained_on, "lfw", "rfw"})
        for dataset in sorted(needed):
            if dataset not in populations:
                populations[dataset] = _population(config, handoff, dataset)
            jobs.append((alias, spec, trained_on, dataset))
    identity = dict(models={a: s.model_uid for a, (s, _) in models.items()}, datasets=sorted(datasets),
                    handoff=sha256_file(config["handoff"]), quality=config["quality"],
                    populations={n: canonical_sha256(f.fillna("").astype(str).to_dict("records"))
                                 for n, (f, _) in populations.items()},
                    input_code=source_fingerprints("research.experiments.facex_workflow",
                                                   "research.embeddings.pytorch.facex_zoo"))
    identity["pair_protocols"] = {n: canonical_sha256(p.fillna("").astype(str).to_dict("records"))
                                  for n, (_, p) in populations.items() if p is not None}
    if set(datasets) & {"survface", "lfw", "rfw"} and config["quality"].get("source") == "cr_fiqa":
        identity["quality_checkpoint_sha256"] = sha256_file(Path(config["project_root"]) / config["quality"]["checkpoint"])
    return handoff, models, populations, jobs, identity


def _root(config, output_root):
    root = Path(output_root or config["output_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    ignore = root / ".gitignore"
    content = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
    missing = [p for p in ("inputs/", "quality/", "reports/**/*.partial/", "reports/**/details/") if p not in content.splitlines()]
    if missing:
        ignore.write_text(content.rstrip("\n") + ("\n" if content else "") + "\n".join(missing) + "\n", encoding="utf-8")
    return root


def _read_optional_csv(path):
    """Read an optional result table, including legacy LF/CRLF-only files.

    Missing files and malformed nonempty tables remain errors. Callers verify
    artifact hashes before reading, so an absent table is not silently accepted.
    """
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def prepare_inputs(config_path, *, datasets=None, output_root=None):
    config = _config(config_path)
    selected = list(datasets or config["datasets"])
    _, _, populations, jobs, identity = _workload(config, selected)
    root = _root(config, output_root)
    required = sum(len(populations[d][0]) * 4104 for _, _, _, d in jobs)
    if shutil.disk_usage(root).free < required * 1.1:
        raise RuntimeError(f"Insufficient free space for input caches; estimate={required} bytes")
    records = {}
    for alias, spec, _, dataset in jobs:
        path = materialize_inputs(config["project_root"], populations[dataset][0], spec, dataset, root,
                                  config["quality"], batch_size=int(config.get("batch_size", 16)),
                                  quality_required=dataset in selected and dataset in ("survface", "lfw", "rfw"))
        records[f"{alias}/{dataset}"] = dict(path=str(path), manifest_sha256=sha256_file(path / "manifest.json"))
    index = root / "input_indexes" / (canonical_sha256(identity) + ".json")
    payload = dict(identity=identity, inputs=records)
    if index.exists() and json.loads(index.read_text(encoding="utf-8")) != payload:
        raise FileExistsError("Pinned input index differs")
    write_json(index, payload)
    return dict(input_index=str(index), input_count=len(records), estimated_input_bytes=required)


def validate_codec_budget(count, source_dim, setting, *, min_points=39):
    family = setting["family"]
    if family == "origin":
        return dict(profile=f"origin_{source_dim}", payload_bytes=source_dim * 4)
    if family == "pca":
        dim = int(setting["dimensions"])
        if not 0 < dim <= min(count - 1, source_dim):
            raise ValueError("PCA exceeds centered sample rank/source dimension")
        return dict(profile=f"pca_{dim}", payload_bytes=dim * 4)
    if family != "pq":
        raise ValueError("Unknown compressor family")
    m, bits = int(setting["m"]), int(setting["nbits"])
    if m < 1 or source_dim % m or not 1 <= bits <= 8:
        raise ValueError("Invalid PQ dimension/subquantizers/bits")
    required = int(min_points) * 2**bits
    if count < required:
        raise ValueError(f"PQ development undersampled: {count} < {required}; choose fewer bits or explicit external development")
    return dict(profile=f"pq_{source_dim}_m{m}_b{bits}", payload_bytes=math.ceil(m * bits / 8),
                fit_count=count, points_per_centroid=count / 2**bits, heuristic_min_points=min_points)


def fit_codecs(vectors, settings, root, *, seed=8972, min_points=39):
    from research.compression import PCACompressor, PQCompressor
    if sum(s["family"] == "origin" for s in settings) != 1:
        raise ValueError("Exactly one Origin reference profile is required")
    codecs = {}
    for setting in settings:
        info = validate_codec_budget(len(vectors), vectors.shape[1], setting, min_points=min_points)
        family, profile = setting["family"], info["profile"]
        if profile in codecs:
            raise ValueError("Duplicate compression profile")
        codec = None
        if family == "pca":
            codec = PCACompressor(setting["dimensions"], random_state=seed).fit(vectors)
        elif family == "pq":
            codec = PQCompressor(source_dim=vectors.shape[1], m=setting["m"], nbits=setting["nbits"], random_state=seed).fit(vectors)
        if codec is not None:
            Path(root).mkdir(parents=True, exist_ok=True)
            saved = codec.save(Path(root) / (profile + ".joblib"))
            info.update(codec_sha256=sha256_file(saved), codec_filename=Path(saved).name)
            info["codec_file_bytes"] = Path(saved).stat().st_size
            info["codec_parameter_bytes"] = (int(codec.model.components_.nbytes + codec.model.mean_.nbytes)
                                             if family == "pca" else vectors.shape[1] * 2**setting["nbits"] * 4)
        codecs[profile] = (family, codec, info)
    return codecs


def normalize(vectors):
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 0):
        raise ValueError("Finite nonzero embeddings required")
    return np.ascontiguousarray(values / norms)


def rank_scores(query, gallery, query_ids, gallery_ids, *, family="origin", codec=None, top_k=20):
    """Return genuine candidate score and rank; never substitute top-1 score."""
    import faiss
    if not len(query) or not len(gallery):
        raise ValueError("Nonempty query/gallery required")
    k = min(top_k, len(gallery))
    if family == "pq":
        codes = codec.encode(gallery)
        distances, indices = codec.search_adc(query, codes, top_k=k)
        scores, space = -distances, "negative_squared_l2_adc"
    else:
        if family == "pca":
            query, gallery = normalize(codec.transform(query)), normalize(codec.transform(gallery))
        index = faiss.IndexFlatIP(query.shape[1])
        index.add(np.ascontiguousarray(gallery, dtype=np.float32))
        scores, indices = index.search(np.ascontiguousarray(query, dtype=np.float32), k)
        space = "cosine" if family == "origin" else "pca_cosine"
    gallery_ids, query_ids = np.asarray(gallery_ids), np.asarray(query_ids)
    matches = gallery_ids[indices] == query_ids[:, None]
    mated = np.isin(query_ids, gallery_ids) & (query_ids != "")
    found = matches.any(axis=1) & mated
    positions = matches.argmax(axis=1)
    rank = np.where(found, positions + 1, np.nan)
    genuine = np.where(found, scores[np.arange(len(query)), positions], np.nan)
    return pd.DataFrame(dict(score=scores[:, 0], is_mated=mated, top_k=top_k,
                             true_identity_rank=rank, true_identity_score=genuine,
                             top_k_correct=found)), space


def _gallery(population, vectors, role, *, templates):
    mask = population.role.eq(role).to_numpy()
    frame, matrix = population.loc[mask].reset_index(drop=True), np.asarray(vectors[mask])
    if templates:
        ids, codes = np.unique(frame.identity_id, return_inverse=True)
        means = np.zeros((len(ids), matrix.shape[1]), dtype=np.float32)
        np.add.at(means, codes, matrix)
        means /= np.bincount(codes)[:, None]
        return normalize(means), ids
    return matrix, frame.identity_id.to_numpy()


def _events_summary(frame, accepted, genuine_accepted):
    from research.evaluation.metrics import wilson_score_interval
    mated = frame.is_mated.to_numpy(dtype=bool)
    correct = frame.top_k_correct.to_numpy(dtype=bool)
    tp = mated & correct & genuine_accepted
    fp = ~mated & accepted
    npos, nneg = int(mated.sum()), int((~mated).sum())
    result = dict(mated_count=npos, non_mated_count=nneg, true_positive_count=int(tp.sum()),
                  false_positive_count=int(fp.sum()), tpir=float(tp.sum() / npos),
                  realized_fpir=float(fp.sum() / nneg), rank_failure_count=int((mated & ~correct).sum()),
                  threshold_failure_count=int((mated & correct & ~genuine_accepted).sum()))
    result["fpir_wilson95_low"], result["fpir_wilson95_high"] = wilson_score_interval(int(fp.sum()), nneg)
    result["fpir_ci_scope"] = "probe-level descriptive Wilson; repeated identity dependence not modeled"
    return result, tp[mated]


def evaluate_survface(population, vectors, quality, codecs, config, *, details=None):
    from research.calibration.conditional import fit_global_threshold
    from research.calibration.continuous import fit_continuous_threshold
    from research.evaluation.cluster_bootstrap import cluster_rate_draws
    from research.evaluation.lfw_verification import thresholds
    cal_mask = population.role.isin(["calibration_fit", "calibration_safety", "calibration_registered_probe"]).to_numpy()
    test_mask = population.role.isin(["test_registered_probe", "test_unknown_probe"]).to_numpy()
    cal_rows, test_rows = population.loc[cal_mask].reset_index(drop=True), population.loc[test_mask].reset_index(drop=True)
    labels = cal_rows.role.map({"calibration_fit": "fit", "calibration_safety": "safety", "calibration_registered_probe": "fit"})
    cg, cids = _gallery(population, vectors, "calibration_gallery", templates=True)
    tg, tids = _gallery(population, vectors, "test_gallery", templates=True)
    metrics, paired, models, curves = [], [], [], []
    for profile, (family, codec, info) in codecs.items():
        cal, space = rank_scores(vectors[cal_mask], cg, cal_rows.identity_id, cids, family=family, codec=codec)
        test, _ = rank_scores(vectors[test_mask], tg, test_rows.identity_id, tids, family=family, codec=codec)
        for frame, source, q in ((cal, cal_rows, quality[cal_mask]), (test, test_rows, quality[test_mask])):
            frame["sample_id"], frame["identity_id"], frame["fiqa_score"] = source.image_id, source.identity_id, q
        for target in config["target_rates"]:
            curve_config = config.get("sample_efficiency", {})
            if curve_config.get("enabled"):
                from research.calibration.sample_efficiency import quantile_learning_curve
                curve = quantile_learning_curve(cal, labels, target=target, score_space=space,
                    fractions=curve_config.get("fractions", [.25, .5, .75, 1.]),
                    folds=curve_config.get("folds", 5), seed=config.get("partition_seed", 8972))
                curves.extend(curve.assign(compression_profile=profile).to_dict("records"))
            common = dict(target_fpir=target, score_space=space, partition_labels=labels.to_numpy())
            fitted = {"global_safe": fit_global_threshold(cal, **common),
                      "continuous_fiqa": fit_continuous_threshold(cal, **common)}
            events, rows = [], []
            for method, model in fitted.items():
                tau = thresholds(model, test)
                summary, event = _events_summary(test, test.score.to_numpy() >= tau,
                                                test.true_identity_score.to_numpy() >= tau)
                row = dict(compression_profile=profile, score_space=space, method=method, target_fpir=target,
                           target_met_on_test=bool(summary["realized_fpir"] <= target), **summary)
                rows.append(row)
                events.append(event)
                models.append(dict(compression_profile=profile, target=target, method=method, model=model.as_dict()))
                if details:
                    out = test.assign(applied_threshold=tau, accepted=test.score.to_numpy() >= tau,
                                      genuine_accepted=test.true_identity_score.to_numpy() >= tau)
                    out.to_csv(Path(details) / f"{profile}_{method}_{target}.csv", index=False)
            ids = test.loc[test.is_mated, "identity_id"]
            draws = cluster_rate_draws(ids, np.column_stack(events), resamples=config.get("bootstrap_resamples", 500))
            for j, row in enumerate(rows):
                row["tpir_cluster95_low"], row["tpir_cluster95_high"] = map(float, np.quantile(draws[:, j], [.025, .975]))
            delta = draws[:, 1] - draws[:, 0]
            low, high = np.quantile(delta, [.025, .975])
            paired.append(dict(compression_profile=profile, target_fpir=target,
                               continuous_minus_global_tpir=rows[1]["tpir"] - rows[0]["tpir"],
                               continuous_minus_global_fpir=rows[1]["realized_fpir"] - rows[0]["realized_fpir"],
                               delta_tpir_cluster95_low=float(low), delta_tpir_cluster95_high=float(high),
                               both_target_met=all(r["target_met_on_test"] for r in rows),
                               ci_scope="paired genuine-identity bootstrap, fixed gallery/threshold; not an FPIR guarantee"))
            metrics.extend(rows)
    return pd.DataFrame(metrics), pd.DataFrame(paired), dict(models=models, sample_efficiency=curves)


def evaluate_rank(population, vectors, codecs, dataset, *, details=None):
    probes = population.role.eq("test_registered_probe").to_numpy()
    rows = []
    conditions = (False, True) if dataset == "tinyface" else (False,)
    for distractors in conditions:
        mask = population.role.eq("test_gallery") | (population.role.eq("test_gallery_distractor") & distractors)
        gallery = population.loc[mask]
        for profile, (family, codec, info) in codecs.items():
            frame, space = rank_scores(vectors[probes], vectors[mask], population.loc[probes, "identity_id"],
                                       gallery.identity_id, family=family, codec=codec)
            distance = population.loc[probes, "distance"].astype(str).to_numpy() if "distance" in population else np.full(probes.sum(), "")
            if details is not None:
                Path(details).mkdir(parents=True, exist_ok=True)
                detail = frame.assign(sample_id=population.loc[probes, "image_id"].to_numpy(),
                    identity_id=population.loc[probes, "identity_id"].to_numpy(), distance=distance,
                    compression_profile=profile, score_space=space, distractors=distractors,
                    rank_search_limit=20)
                detail.to_csv(Path(details) / f"{profile}_distractors_{int(distractors)}_probes.csv", index=False)
            for label in (["all", "1", "2", "3"] if dataset == "scface" else ["all"]):
                part = frame if label == "all" else frame.loc[distance == label]
                if part.empty:
                    raise ValueError("Missing SCFace distance cohort")
                for rank in (1, 5, 10, 20):
                    success = int((part.true_identity_rank <= rank).sum())
                    rows.append(dict(compression_profile=profile, score_space=space, distance=label,
                                     distractors=distractors, rank=rank, success_count=success,
                                     probe_count=len(part), recognition_rate=success / len(part),
                                     payload_bytes_per_vector=info["payload_bytes"]))
    return pd.DataFrame(rows), pd.DataFrame(), []


def evaluate_pairs(population, vectors, quality, pairs, codecs, config, *, details=None):
    from research.evaluation.lfw_verification import pair_scores, evaluate_fold
    from research.experiments.origin_vs_pq_calibration import OriginPQSettings
    settings = OriginPQSettings(target_fpirs=tuple(config["target_rates"]), resamples=config.get("bootstrap_resamples", 500))
    q = pd.Series(quality, index=population.image_id)
    metrics, paired, inventories, model_rows = [], [], [], []
    for group, group_pairs in pairs.groupby("group", sort=True):
        scored = []
        for profile, (family, codec, _) in codecs.items():
            values = normalize(codec.transform(vectors)) if family == "pca" else vectors
            scores = pair_scores(values, group_pairs, population.image_id, codec=codec if family == "pq" else None)
            scored.append(group_pairs.assign(compression_profile=profile, pair_score=scores,
                          score_space="negative_squared_l2_adc" if family == "pq" else "pca_cosine" if family == "pca" else "cosine"))
        scored = pd.concat(scored, ignore_index=True)
        for fold in config["fold_ids"]:
            cal, test, inventory = pair_partition(group_pairs, fold, seed=config.get("partition_seed", 8972))
            origin = next(p for p, (family, _, _) in codecs.items() if family == "origin")
            tables = evaluate_fold(scored, q, cal, test, seed=config.get("partition_seed", 8972),
                                   settings=settings, origin_profile=origin, keep_decisions=details is not None)
            if details is not None:
                Path(details).mkdir(parents=True, exist_ok=True)
                columns = ["pair_id", "left_image_id", "right_image_id", "left_identity", "right_identity"]
                decisions = tables["decisions"].merge(test[columns], on="pair_id", how="left", validate="many_to_one")
                decisions["fiqa_score"] = decisions.left_image_id.map(q)
                decisions.assign(group=group, fold=fold).to_csv(
                    Path(details) / f"{group}_fold_{int(fold):02d}_decisions.csv", index=False)
            metrics.append(tables["operating"].assign(group=group, fold=fold))
            paired.append(tables["paired"].assign(group=group, fold=fold))
            inventories.append(dict(group=group, **inventory))
            model_rows.extend(tables["models"].assign(group=group, fold=fold).to_dict("records"))
    return pd.concat(metrics, ignore_index=True), pd.concat(paired, ignore_index=True), dict(models=model_rows, partitions=inventories)


def _science_hash():
    return source_fingerprints("research.experiments.facex_workflow", "research.embeddings.pytorch.facex_zoo")


def write_analysis_summary(root, metrics):
    """Pool trial counts within declared conditions; never pool identity CIs."""
    dimensions = [c for c in ("dataset", "model_alias", "model_uid", "trained_on", "group",
                  "compression_profile", "method", "target_fpir", "target_fmr", "rank", "distance", "distractors")
                  if c in metrics]
    counts = [c for c in ("mated_count", "non_mated_count", "true_positive_count", "false_positive_count",
              "genuine_pairs", "impostor_pairs", "true_accepts", "false_accepts", "probe_count", "success_count") if c in metrics]
    records = []
    for names, part in metrics.groupby(dimensions, dropna=False, sort=False):
        row = dict(zip(dimensions, names))
        row.update({c: float(part[c].sum()) for c in counts if part[c].notna().any()})
        row.update(evaluation_rows=len(part),
                   dependence_note="pooled trial counts only; repeated identities/folds are not independent replications")
        for numerator, denominator, rate in (("true_positive_count", "mated_count", "pooled_tpir"),
                ("false_positive_count", "non_mated_count", "pooled_fpir"),
                ("true_accepts", "genuine_pairs", "pooled_tar"), ("false_accepts", "impostor_pairs", "pooled_fmr"),
                ("success_count", "probe_count", "recognition_rate")):
            if row.get(denominator, 0) > 0:
                row[rate] = row[numerator] / row[denominator]
        if "target_met_on_test" in part and part.target_met_on_test.notna().any():
            row["all_evaluation_rows_target_met"] = bool(part.target_met_on_test.dropna().all())
        records.append(row)
    pd.DataFrame(records).to_csv(Path(root) / "overview.csv", index=False)


def run_workflow(config_path, *, datasets=None, output_root=None, keep_raw_results=None):
    config = _config(config_path)
    selected = list(datasets or config["datasets"])
    handoff, models, populations, jobs, identity = _workload(config, selected)
    root = _root(config, output_root)
    index_path = root / "input_indexes" / (canonical_sha256(identity) + ".json")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index["identity"] != identity:
        raise ValueError("Input index lineage mismatch")
    keep = config.get("keep_raw_results", True) if keep_raw_results is None else bool(keep_raw_results)
    contract = dict(config=config, inputs=index, keep_raw_results=keep, science=_science_hash(), datasets=selected)
    uid = "facex-" + canonical_sha256(contract)[:24]
    destination = root / "reports" / uid
    if destination.exists():
        read_results(destination)
        return dict(report_dir=str(destination), reused=True)
    work = destination.with_name(uid + ".partial")
    work.mkdir(parents=True, exist_ok=True)
    lock = work / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), run_uid=uid)))
    write_json(work / "run_manifest.json", dict(status="running", run_uid=uid, contract=contract))
    try:
        all_metrics, all_paired = [], []
        import faiss
        faiss.omp_set_num_threads(2)
        with threadpool_limits(limits=2):
            for alias, (spec, trained_on) in models.items():
                item = index["inputs"][f"{alias}/{trained_on}"]
                if sha256_file(Path(item["path"]) / "manifest.json") != item["manifest_sha256"]:
                    raise ValueError("Input manifest changed")
                _, train_pop, train_vectors, _, _ = read_inputs(item["path"], expected_uid=spec.model_uid)
                dev = train_pop.role.eq("pq_development").to_numpy()
                codecs = fit_codecs(train_vectors[dev], config["compression"][trained_on], work / "codecs" / alias,
                                    seed=config.get("codec_seed", 8972), min_points=config.get("pq_min_points_per_centroid", 39))
                write_json(work / f"{alias}_codec_inventory.json", {p: info for p, (_, _, info) in codecs.items()})
                for dataset in selected:
                    if dataset not in (trained_on, "lfw", "rfw"):
                        continue
                    job = f"{alias}_{dataset}"
                    item = index["inputs"][f"{alias}/{dataset}"]
                    if sha256_file(Path(item["path"]) / "manifest.json") != item["manifest_sha256"]:
                        raise ValueError("Input manifest changed")
                    _, population, vectors, _, quality = read_inputs(item["path"], expected_uid=spec.model_uid)
                    metrics_file, paired_file = work / f"{job}_metrics.csv", work / f"{job}_paired.csv"
                    done = work / f"{job}_COMPLETED.json"
                    if done.exists():
                        record = json.loads(done.read_text(encoding="utf-8"))
                        for name, digest in record.items():
                            if sha256_file(work / name) != digest:
                                raise ValueError("Interrupted result changed")
                        metrics = pd.read_csv(metrics_file)
                        paired = _read_optional_csv(paired_file)
                    else:
                        details = work / "details" / job
                        if keep:
                            details.mkdir(parents=True, exist_ok=True)
                        if dataset == "survface":
                            metrics, paired, fitted = evaluate_survface(population, vectors, quality, codecs, config,
                                                                      details=details if keep else None)
                        elif dataset in ("tinyface", "scface"):
                            metrics, paired, fitted = evaluate_rank(population, vectors, codecs, dataset,
                                                                   details=details if keep else None)
                        else:
                            metrics, paired, fitted = evaluate_pairs(population, vectors, quality, populations[dataset][1], codecs, config,
                                                                    details=details if keep else None)
                        metrics.to_csv(metrics_file, index=False)
                        paired.to_csv(paired_file, index=False)
                        model_file = work / f"{job}_models.json"
                        write_json(model_file, fitted)
                        job_files = [metrics_file, paired_file, model_file]
                        if keep:
                            job_files.extend(p for p in details.rglob("*") if p.is_file())
                        write_json(done, {p.relative_to(work).as_posix(): sha256_file(p) for p in job_files})
                    all_metrics.append(metrics.assign(dataset=dataset, model_alias=alias, model_uid=spec.model_uid, trained_on=trained_on))
                    if not paired.empty:
                        all_paired.append(paired.assign(dataset=dataset, model_alias=alias, model_uid=spec.model_uid))
        metrics = pd.concat(all_metrics, ignore_index=True)
        paired = pd.concat(all_paired, ignore_index=True) if all_paired else pd.DataFrame()
        metrics.to_csv(work / "metrics.csv", index=False)
        paired.to_csv(work / "paired.csv", index=False)
        write_analysis_summary(work, metrics)
        curves = []
        for path in work.glob("*_models.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("sample_efficiency"):
                curves.append(pd.DataFrame(payload["sample_efficiency"]).assign(source_job=path.stem.removesuffix("_models")))
        if curves:
            pd.concat(curves, ignore_index=True).to_csv(work / "sample_efficiency.csv", index=False)
        guide = Path(__file__).resolve().parents[2] / "notebooks/WORKFLOW_GUIDE.md"
        shutil.copyfile(guide, work / "HOW_TO_READ.md")
        if len(pd.read_csv(work / "metrics.csv")) != len(metrics):
            raise ValueError("Summary round-trip count mismatch")
        # Only this run's managed raw outputs are eligible for cleanup.
        if not keep and (work / "details").exists():
            shutil.rmtree(work / "details")
        files = {p.relative_to(work).as_posix(): sha256_file(p) for p in work.rglob("*")
                 if p.is_file() and p.name not in ("RUNNING.lock", "run_manifest.json", "COMPLETED")}
        summary_files = {p: files[p] for p in ("overview.csv", "HOW_TO_READ.md")}
        parts = [{"path": p, "bytes": (work / p).stat().st_size,
                  "estimated_text_tokens": (work / p).stat().st_size // 3}
                 for p in sorted(files) if p.endswith(("_metrics.csv", "_paired.csv"))]
        write_json(work / "chatgpt_summary.json", dict(run_uid=uid, summary_files=summary_files,
                   summary_bytes=sum((work / p).stat().st_size for p in summary_files),
                   estimated_text_tokens=sum((work / p).stat().st_size for p in summary_files) // 3,
                   token_estimate_basis="UTF-8 byte count / 3; rough estimate, not a model limit",
                   summary_parts=parts,
                   upload_order=["HOW_TO_READ.md", "chatgpt_summary.json", "overview.csv"],
                   detailed_upload="Choose the model/dataset summary_parts of interest; full metrics.csv/paired.csv are optional"))
        files["chatgpt_summary.json"] = sha256_file(work / "chatgpt_summary.json")
        write_json(work / "run_manifest.json", dict(status="completed", run_uid=uid, contract=contract, files=files,
                   completed_jobs=len(all_metrics), expected_jobs=sum(d in (trained, "lfw", "rfw")
                       for _, trained in models.values() for d in selected),
                   keep_raw_results=keep, formal_fpir_guarantee=False,
                   gallery_aggregation="SurvFace: mean of unit image embeddings then L2; TinyFace/SCFace: image gallery",
                    raw_retention_scope="details: SurvFace query scores/decisions, TinyFace/SCFace probe ranks, LFW/RFW pair decisions; input/resume caches and codecs retained",
                   protocol_scope="official test roles/pairs; researcher-defined calibration and cosine/PQ score spaces"))
        (work / "COMPLETED").write_text("completed\n", encoding="utf-8")
        work.rename(destination)
    finally:
        lock.unlink(missing_ok=True)
        (destination / "RUNNING.lock").unlink(missing_ok=True)
    read_results(destination)
    return dict(report_dir=str(destination), run_uid=uid, reused=False)


def read_results(directory):
    root = Path(directory)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or not (root / "COMPLETED").is_file():
        raise ValueError("Result is incomplete; inspect the explicit .partial directory")
    for name, digest in manifest["files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Published result changed: {name}")
    return dict(manifest=manifest, metrics=pd.read_csv(root / "metrics.csv"),
                paired=_read_optional_csv(root / "paired.csv"))
