"""Read-only source checks and restartable origin controls for notebook 03.

PQ score artifacts and FIQA are reused. Origin test scores come from the
verified Step-4 ledger; only origin calibration retrieval is replayed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from research.calibration.conditional import (
    IDENTIFICATION_METRIC_CONTRACT, validate_identification_scores,
)
from research.evaluation.compression_characterization import _batched_cosine_top_k, _row_normalize
from research.experiments.calibration_matrix import (
    DEFAULT_RUN_MATRIX, MODEL_UIDS, PQ_PROFILES, inspect_calibration_matrix,
)
from research.experiments.calibration_protocols import OPEN_SET_DATASETS, calibration_protocol
from research.experiments.fiqa_retrieval_features import _atomic_json
from research.experiments.fiqa_split_stability import _frame_hash
from research.experiments.fiqa_threshold_calibration import (
    ConditionScoreTables, _completed_run, _frozen_pq_codec, _ledger_condition,
    _read_json, _verified_table, load_condition_score_artifact,
)
from research.experiments.step2_compression import open_set_protocol_arrays, prepared_population_frame
from research.explainability.gradcam.artifacts import (
    read_prepared_population_artifact,
)
from research.fiqa import CRFIQA_VARIANTS, load_fiqa_score_artifact
from research.runtime.hashing import canonical_sha256, sha256_file

ORIGIN_PROFILE = "origin_512"
ORIGIN_SPACE = "cosine_similarity"
ORIGIN_MODE = "origin_exact_cosine"
COHORT_COLUMNS = ("sample_id", "identity_id", "evaluation_split", "is_mated", "top_k",
                  "aligned_content_sha256")
SOURCE_KEYS = ("source_run_id", "source_run_manifest_sha256", "source_freeze_manifest_sha256",
               "selected_manifest_sha256", "prepared_population_manifest_sha256",
               "aligned_bundle_manifest_sha256", "dataset_id", "model_uid", "extraction_uid",
               "origin_embedding_artifact_uid", "calibration_seed", "protocol_uid", "top_k")
ORIGIN_FIELDS = {
    "origin_top1_score": "score", "origin_true_identity_score": "true_identity_score",
    "origin_true_identity_rank": "true_identity_rank", "origin_rank1_correct": "rank1_correct",
    "origin_top_k_correct": "top_k_correct",
}


def input_code_hashes():
    root = Path(__file__).parents[1]
    names = ("experiments/origin_pq_inputs.py", "experiments/calibration_protocols.py",
             "experiments/step2_compression.py", "experiments/fiqa_threshold_calibration.py",
             "evaluation/compression_characterization.py", "protocols/open_set.py",
             "templates/aggregation.py", "datasets/rfw_custom.py", "explainability/gradcam/artifacts.py")
    return {name: sha256_file(root / name) for name in names}


def condition_directory(result_root, run_id, profile):
    return Path(result_root) / "condition_scores" / run_id / f"{profile}__pq_adc_exhaustive" / IDENTIFICATION_METRIC_CONTRACT


def inspect_origin_pq_experiment(project_root, *, run_matrix=None, datasets=OPEN_SET_DATASETS,
                                 models=tuple(MODEL_UIDS), profiles=PQ_PROFILES, variant="L",
                                 calibration_root=None):
    """Check every requested source, existing PQ input and FIQA without fitting.

    Missing PQ/FIQA inputs fail with their exact paths; use the existing input
    preparation workflow first. No new inference or saliency is triggered.
    """
    project = Path(project_root).resolve()
    result_root = Path(calibration_root or project / "results/calibration")
    if variant not in CRFIQA_VARIANTS:
        raise ValueError("unknown FIQA variant")
    plan = inspect_calibration_matrix(project, run_matrix or DEFAULT_RUN_MATRIX,
                                     datasets=datasets, models=models, profiles=profiles)
    directories, fiqa_dirs, hashes, fiqa_hashes = [], [], [], []
    verified_fiqa = {}
    for row in plan.itertuples(index=False):
        directory = condition_directory(result_root, row.source_run_id, row.compression_profile)
        cm = _read_json(directory / "manifest.json")
        expected = dict(status="completed", schema_version=2, metric_contract=IDENTIFICATION_METRIC_CONTRACT,
                        source_run_id=row.source_run_id, source_run_manifest_sha256=row.source_manifest_sha256,
                        model_uid=row.model_uid, dataset_id=row.dataset_id, compression_profile=row.compression_profile,
                        score_space="negative_squared_l2_adc", persisted_test_core_sha256=row.test_core_sha256)
        for key, value in expected.items():
            if cm.get(key) != value:
                raise ValueError(f"PQ condition mismatch: {directory}: {key}")
        for name in ("calibration_scores.parquet", "test_scores.parquet"):
            if sha256_file(directory / name) != cm["files"][name]["sha256"]:
                raise ValueError(f"PQ input hash mismatch: {directory / name}")
        fd = result_root / "fiqa_scores" / row.dataset_id / CRFIQA_VARIANTS[variant].model_uid
        if str(fd) not in verified_fiqa:
            verified_fiqa[str(fd)] = load_fiqa_score_artifact(fd).manifest
        fm = verified_fiqa[str(fd)]
        if (fm["aligned_bundle_manifest_sha256"] != cm["aligned_bundle_manifest_sha256"]
                or fm["checkpoint_sha256"] != CRFIQA_VARIANTS[variant].expected_sha256):
            raise ValueError("FIQA alignment/checkpoint mismatch")
        core = pd.read_parquet(_verified_table(Path(row.source_run_dir) / "artifacts/step2_workflow/retrieval_ledger",
                              _ledger_condition(_read_json(Path(row.source_run_dir) / "artifacts/step2_workflow/retrieval_ledger/manifest.json"),
                                                compression_profile=row.compression_profile, search_mode="pq_adc_exhaustive")["core"]),
                               columns=["origin_score_space", *ORIGIN_FIELDS])
        if not core.origin_score_space.eq(ORIGIN_SPACE).all():
            raise ValueError("source ledger is not origin cosine")
        directories.append(str(directory.resolve()))
        fiqa_dirs.append(str(fd.resolve()))
        hashes.append(canonical_sha256(cm))
        fiqa_hashes.append(canonical_sha256(fm))
    return plan.assign(condition_dir=directories, fiqa_dir=fiqa_dirs,
                       condition_sha256=hashes, fiqa_sha256=fiqa_hashes)


def assert_same_cohort(left, right):
    """Order, identities, labels, K and alignment must agree, not just row counts."""
    for split in ("calibration", "test"):
        a, b = getattr(left, split), getattr(right, split)
        if not a[list(COHORT_COLUMNS)].reset_index(drop=True).equals(b[list(COHORT_COLUMNS)].reset_index(drop=True)):
            raise ValueError(f"{split} cohort/order/identity/alignment mismatch")
    for key in SOURCE_KEYS:
        if left.manifest.get(key) != right.manifest.get(key):
            raise ValueError(f"source lineage mismatch: {key}")


def origin_rows_from_ledger(raw, template):
    """Use genuine origin scores, including NaN when the genuine rank exits K."""
    ids = raw.query_id.astype(str)
    if ids.duplicated().any() or set(ids) != set(template.sample_id.astype(str)):
        raise ValueError("origin ledger query cohort mismatch")
    raw = raw.set_index(ids).loc[template.sample_id.astype(str)].reset_index(drop=True)
    if (not np.array_equal(raw.query_identity_id.astype(str), template.identity_id.astype(str))
            or not np.array_equal(raw.is_mated, template.is_mated)):
        raise ValueError("origin ledger identity/mated mismatch")
    if "origin_score_space" in raw and not raw.origin_score_space.eq(ORIGIN_SPACE).all():
        raise ValueError("origin ledger score space mismatch")
    if "top_k" in raw and not np.array_equal(raw.top_k, template.top_k):
        raise ValueError("origin ledger top_k mismatch")
    result = template.copy().reset_index(drop=True)
    for source, target in ORIGIN_FIELDS.items():
        result[target] = raw[source].to_numpy()
    result["compression_profile"] = ORIGIN_PROFILE
    result["search_mode"] = ORIGIN_MODE
    result["score_space"] = ORIGIN_SPACE
    validate_identification_scores(result)
    if not result.score.between(-1.000001, 1.000001).all():
        raise ValueError("origin cosine scores outside [-1,1]")
    return result


def exact_origin_rows(queries, gallery, query_ids, query_identities, gallery_identities, *, top_k):
    """Canonical NumPy CPU cosine search with genuine Top-K rank/score fields."""
    q = _row_normalize(np.asarray(queries, dtype=np.float32), name="queries")
    g = _row_normalize(np.asarray(gallery, dtype=np.float32), name="gallery")
    gi, qi = np.asarray(gallery_identities).astype(str), np.asarray(query_identities).astype(str)
    if len(set(gi)) != len(gi) or not 1 <= top_k <= len(g):
        raise ValueError("unique identity templates and valid K required")
    indices, scores = _batched_cosine_top_k(q, g, top_k=top_k, query_batch_size=128, gallery_batch_size=4096)
    matches = gi[indices] == qi[:, None]
    found, first = matches.any(axis=1), matches.argmax(axis=1)
    return pd.DataFrame(dict(query_id=np.asarray(query_ids).astype(str), query_identity_id=qi,
                             is_mated=np.isin(qi, gi), top_k=top_k, origin_score_space=ORIGIN_SPACE,
                             origin_top1_score=scores[:, 0], origin_rank1_correct=matches[:, 0],
                             origin_top_k_correct=found,
                             origin_true_identity_rank=np.where(found, first + 1, np.nan),
                             origin_true_identity_score=np.where(found, scores[np.arange(len(q)), first], np.nan)))


def _load_origin(directory, spec):
    manifest = _read_json(directory / "manifest.json")
    if (manifest.get("status") != "completed" or manifest.get("spec") != spec
            or manifest.get("artifact_type") != "origin_calibration_test_score_tables"):
        raise ValueError("origin cache contract mismatch")
    frames = {}
    for split in ("calibration", "test"):
        path = directory / f"{split}_scores.parquet"
        entry = manifest["files"][path.name]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("origin cache hash mismatch")
        frames[split] = pd.read_parquet(path)
        if len(frames[split]) != entry["rows"]:
            raise ValueError("origin cache row count mismatch")
        validate_identification_scores(frames[split])
    return ConditionScoreTables(frames["calibration"], frames["test"], manifest)


def prepare_origin_control(run_dir, anchor, output_root, *, shard_size=4096, progress=print):
    """Resume calibration score shards; never alter any completed source files."""
    if isinstance(shard_size, bool) or not isinstance(shard_size, int) or shard_size < 1:
        raise ValueError("positive integer shard_size required")
    root, run, workflow = _completed_run(run_dir)
    output = Path(output_root).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("origin output must be separate from the completed run")
    cm = anchor.manifest
    paths = {root / "run_manifest.json": "source_run_manifest_sha256",
             workflow / "freeze_manifest.json": "source_freeze_manifest_sha256",
             workflow / "selected_manifest.csv": "selected_manifest_sha256",
             workflow / "prepared_population/manifest.json": "prepared_population_manifest_sha256"}
    if cm.get("status") != "completed" or cm["source_run_id"] != run["run_id"]:
        raise ValueError("origin requires exact completed PQ anchor")
    for path, key in paths.items():
        if sha256_file(path) != cm[key]:
            raise ValueError(f"source changed: {key}")
    ledger_root = workflow / "retrieval_ledger"
    ledger = _read_json(ledger_root / "manifest.json")
    entry = _ledger_condition(ledger, compression_profile=cm["compression_profile"], search_mode="pq_adc_exhaustive")
    core_path = _verified_table(ledger_root, entry["core"])
    if entry["core"]["sha256"] != cm["persisted_test_core_sha256"]:
        raise ValueError("anchor ledger changed")
    _, codec, bundle = _frozen_pq_codec(root, workflow, compression_profile=cm["compression_profile"])
    if codec["artifact_sha256"] != cm["frozen_codec"]["sha256"] or bundle["fit_seed"] != cm["calibration_seed"]:
        raise ValueError("anchor codec/seed mismatch")
    spec = dict(anchor_manifest_sha256=canonical_sha256(cm), implementation=input_code_hashes(),
                shard_size=shard_size, numpy_version=np.__version__, pandas_version=pd.__version__,
                source_ledger_manifest_sha256=sha256_file(ledger_root / "manifest.json"),
                score_space=ORIGIN_SPACE, search_mode=ORIGIN_MODE)
    uid = "origin-scores-" + canonical_sha256(spec)[:24]
    destination = output / uid
    if (destination / "manifest.json").exists():
        result = _load_origin(destination, spec)
        assert_same_cohort(anchor, result)
        return result
    destination.mkdir(parents=True, exist_ok=True)
    journal_path = destination / "progress.json"
    journal = _read_json(journal_path) if journal_path.exists() else dict(spec=spec, shards=[])
    if journal["spec"] != spec:
        raise ValueError("origin replay journal spec mismatch")
    prepared = read_prepared_population_artifact(workflow / "prepared_population")
    selected = pd.read_csv(workflow / "selected_manifest.csv", low_memory=False)
    population = prepared_population_frame(prepared, selected)
    arrays = {}
    signatures = {}
    for split in ("calibration", "test"):
        a = open_set_protocol_arrays(calibration_protocol(run, population, split, int(cm["calibration_seed"])), population)
        rows = getattr(anchor, split)
        ids = pd.Index(a["query_ids"].astype(str))
        if ids.has_duplicates or set(ids) != set(rows.sample_id):
            raise ValueError(f"{split} reconstructed query cohort mismatch")
        order = ids.get_indexer(rows.sample_id)
        if (not np.array_equal(a["query_identity_ids"][order].astype(str), rows.identity_id.astype(str))
                or not np.array_equal(np.isin(a["query_identity_ids"][order], a["gallery_identity_ids"]), rows.is_mated)):
            raise ValueError("reconstructed identity/mated contract mismatch")
        arrays[split] = (a, order)
        signatures[split] = dict(gallery_identities=list(a["gallery_identity_ids"].astype(str)),
                                 gallery_shape=list(a["gallery"].shape), gallery_dtype=str(a["gallery"].dtype),
                                 gallery_vectors_sha256=hashlib.sha256(a["gallery"].tobytes()).hexdigest())
    a, order = arrays["calibration"]
    frames = []
    for shard, start in enumerate(range(0, len(anchor.calibration), shard_size)):
        template = anchor.calibration.iloc[start:start + shard_size].reset_index(drop=True)
        name = f"calibration-{shard:06d}.parquet"
        path = destination / name
        if shard < len(journal["shards"]):
            receipt = journal["shards"][shard]
            if receipt["name"] != name or sha256_file(path) != receipt["sha256"]:
                raise ValueError("origin replay shard hash mismatch")
            frame = pd.read_parquet(path)
            if not frame[list(COHORT_COLUMNS)].equals(template[list(COHORT_COLUMNS)]):
                raise ValueError("origin replay shard cohort mismatch")
        else:
            ix = order[start:start + shard_size]
            raw = exact_origin_rows(a["queries"][ix], a["gallery"], a["query_ids"][ix],
                                    a["query_identity_ids"][ix], a["gallery_identity_ids"], top_k=int(cm["top_k"]))
            frame = origin_rows_from_ledger(raw, template)
            temp = path.with_suffix(".tmp.parquet")
            frame.to_parquet(temp, index=False)
            temp.replace(path)
            journal["shards"].append(dict(name=name, sha256=sha256_file(path), rows=len(frame)))
            _atomic_json(journal_path, journal)
        validate_identification_scores(frame)
        frames.append(frame)
        if progress:
            progress(dict(stage="origin_calibration", source_run_id=run["run_id"],
                          completed=min(start + shard_size, len(anchor.calibration)), total=len(anchor.calibration)))
    calibration = pd.concat(frames, ignore_index=True)
    test = origin_rows_from_ledger(pd.read_parquet(core_path), anchor.test)
    # Publish the completion manifest last. Partial shards remain resumable.
    files = {}
    for split, frame in (("calibration", calibration), ("test", test)):
        validate_identification_scores(frame)
        path = destination / f"{split}_scores.parquet"
        temp = path.with_suffix(".tmp.parquet")
        frame.to_parquet(temp, index=False)
        temp.replace(path)
        files[path.name] = dict(sha256=sha256_file(path), rows=len(frame))
    manifest = {key: cm[key] for key in SOURCE_KEYS}
    manifest.update(artifact_type="origin_calibration_test_score_tables", schema_version=2,
                    metric_contract=IDENTIFICATION_METRIC_CONTRACT, status="completed",
                    condition_uid=uid, compression_profile=ORIGIN_PROFILE, compression_family="origin",
                    search_mode=ORIGIN_MODE, score_space=ORIGIN_SPACE, threshold_comparator=">=",
                    spec=spec, files=files, gallery_signatures=signatures,
                    retrieval_backend="numpy_cpu_exact_cosine", origin_test_reused=True,
                    source_test_core_sha256=entry["core"]["sha256"],
                    calibration_frame_sha256=_frame_hash(calibration), test_frame_sha256=_frame_hash(test))
    _atomic_json(destination / "manifest.json", manifest)
    result = _load_origin(destination, spec)
    assert_same_cohort(anchor, result)
    return result


def load_run_inputs(group, output_root, *, shard_size=4096, progress=print):
    """Load one model/dataset at a time, checking all PQ budgets share the origin."""
    conditions = {}
    origin_test = None
    anchor = None
    for row in group.itertuples(index=False):
        condition = load_condition_score_artifact(row.condition_dir)
        if canonical_sha256(condition.manifest) != row.condition_sha256:
            raise ValueError("condition changed since preflight")
        if anchor is None:
            anchor = condition
        else:
            assert_same_cohort(anchor, condition)
        workflow = Path(row.source_run_dir) / "artifacts/step2_workflow"
        cm = condition.manifest
        for name, key in (("freeze_manifest.json", "source_freeze_manifest_sha256"),
                          ("selected_manifest.csv", "selected_manifest_sha256"),
                          ("prepared_population/manifest.json", "prepared_population_manifest_sha256")):
            if sha256_file(workflow / name) != cm[key]:
                raise ValueError(f"PQ frozen source changed: {name}")
        _, codec, bundle = _frozen_pq_codec(Path(row.source_run_dir), workflow, compression_profile=row.compression_profile)
        if codec["artifact_sha256"] != cm["frozen_codec"]["sha256"] or bundle["fit_seed"] != cm["calibration_seed"]:
            raise ValueError("PQ frozen codec/seed changed")
        lr = workflow / "retrieval_ledger"
        entry = _ledger_condition(_read_json(lr / "manifest.json"), compression_profile=row.compression_profile,
                                  search_mode="pq_adc_exhaustive")
        raw = pd.read_parquet(_verified_table(lr, entry["core"]))
        current = origin_rows_from_ledger(raw, condition.test)
        if origin_test is not None and not current.equals(origin_test):
            raise ValueError("PQ budgets have different origin test scores/ranks")
        origin_test = current
        conditions[row.compression_profile] = condition
    first = group.iloc[0]
    fiqa = load_fiqa_score_artifact(first.fiqa_dir)
    if any(group.fiqa_sha256 != canonical_sha256(fiqa.manifest)):
        raise ValueError("FIQA changed since preflight")
    origin = prepare_origin_control(first.source_run_dir, anchor, output_root, shard_size=shard_size, progress=progress)
    return {ORIGIN_PROFILE: origin, **conditions}, fiqa
