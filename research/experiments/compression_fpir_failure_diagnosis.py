"""Read-only, fixed-cohort compression FPIR diagnostics from completed ledgers.

No inference, codec fitting, threshold selection or test-driven model selection.
Rates are fractions; paired CIs condition on the recorded gallery/threshold/codec.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from research.evaluation.metrics import (
    paired_binary_rate_difference_bootstrap_interval,
    wilson_score_interval,
)
from research.evaluation.retrieval_ledger import load_retrieval_ledger_manifest
from research.evaluation.search_conditions import (
    search_condition_metadata, threshold_policies_for_search_mode,
)
from research.runtime.hashing import sha256_file


SOURCE_RUNS = {
    "arcface": "survface_20260902/20260902-R001-61915edf_step4_survface_arcface-7972a704552df378345f",
    "adaface": "survface_20260830/20260830-R001-ec6e5d4a_step4_survface_adaface-4df25b75e065b0b9ed43",
    "magface": "survface_20260831/20260831-R001-6695386d_step4_survface_magface-6931178ad2025e1b3799",
    "edgeface": "survface_20260901/20260901-R001-56c2f3ed_step4_survface_edgeface-a348c305af33c223b337",
}
PQ_PROFILES = tuple(f"pq_512_m{m}_b8" for m in (128, 64, 32, 16, 8))
PQ_MODES = ("pq_reconstruction_cosine", "pq_one_sided_cosine", "pq_adc_exhaustive")
LIMITATIONS = (
    "Descriptive diagnostic, not a causal identification of quantization geometry. "
    "FPIR Wilson and paired bootstrap are query-level, fixed-gallery/threshold/codec; "
    "identity dependence, calibration uncertainty and multiple comparisons are not covered. "
    "TPIR20 is genuine-score threshold AND rank<=20, not maximum-score acceptance. "
    "ADC and cosine scores are not subtracted. Recalibration is not recognition recovery. "
    "Test results must not select thresholds, seeds or compression profiles."
)


def verified_path(root: Path, entry: dict) -> Path:
    """Resolve only in-root files and require both recorded size and SHA-256."""
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact escapes root")
    if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise ValueError(f"artifact size/SHA-256 mismatch: {path}")
    return path


def _one(frame: pd.DataFrame, column: str):
    values = frame[column].drop_duplicates()
    if len(values) != 1 or pd.isna(values.iloc[0]):
        raise ValueError(f"expected one non-null {column}")
    return values.iloc[0]


def _bool(frame: pd.DataFrame, column: str) -> np.ndarray:
    if frame[column].isna().any() or not frame[column].isin([True, False]).all():
        raise ValueError(f"invalid boolean: {column}")
    return frame[column].to_numpy(dtype=bool)


def event_counts(reference, candidate, *, seed=8972, resamples=2000) -> dict:
    """Four mutually exclusive states; candidate minus reference paired CI."""
    reference, candidate = np.asarray(reference, bool), np.asarray(candidate, bool)
    if reference.ndim != 1 or reference.shape != candidate.shape or not len(reference):
        raise ValueError("paired nonempty vectors required")
    both = int((reference & candidate).sum())
    new, lost = int((~reference & candidate).sum()), int((reference & ~candidate).sum())
    n = len(reference)
    low, high = paired_binary_rate_difference_bootstrap_interval(
        int(reference.sum()), int(candidate.sum()), both, n,
        random_seed=seed, resamples=resamples,
    )
    return dict(n=n, reference_fa=int(reference.sum()), candidate_fa=int(candidate.sum()),
                both_fa=both, neither_fa=int((~reference & ~candidate).sum()),
                new_fa=new, lost_fa=lost, delta_fpir=(new-lost)/n,
                delta_ci_low=low, delta_ci_high=high)


def diagnose_decision(frame: pd.DataFrame, *, seed=8972, resamples=2000):
    """Validate a complete condition/decision and return summary + tail/winner rows."""
    if frame.empty or frame.query_id.isna().any() or frame.query_id.duplicated().any():
        raise ValueError("query_id must be complete and unique")
    meta = {c: _one(frame, c) for c in (
        "compression_profile", "search_mode", "threshold_policy", "target_fpir",
        "protocol_uid", "model_uid", "gallery_template_count", "top_k",
        "origin_score_space", "compressed_score_space",
    )}
    if meta["top_k"] != 20:
        raise ValueError("this diagnostic requires stored top_k=20")
    for column, expected in search_condition_metadata(meta["search_mode"]).items():
        if _one(frame, column) != expected:
            raise ValueError(f"search mode contract mismatch: {column}")
    mated = _bool(frame, "is_mated")
    if not mated.any() or mated.all():
        raise ValueError("both mated and non-mated queries required")
    if _bool(frame, "origin_fallback_used").any():
        raise ValueError("origin fallback invalidates compression comparison")
    comparable = bool(_one(frame, "score_spaces_comparable"))
    cosine = meta["origin_score_space"] == meta["compressed_score_space"] == "cosine_similarity"
    if comparable != cosine:
        raise ValueError("unsupported or inconsistent score-space contract")
    policy = meta["threshold_policy"]
    if policy not in ("frozen_origin", "recalibrated_compressed"):
        raise ValueError(f"unsupported threshold policy: {policy}")
    if policy == "frozen_origin" and not (cosine and bool(_one(frame, "frozen_origin_threshold_applicable"))):
        raise ValueError("frozen cosine threshold is not applicable")
    decisions = {}
    for side in ("origin", "compressed"):
        scores = frame[f"{side}_top1_score"].to_numpy(float)
        if not np.isfinite(scores).all():
            raise ValueError("non-finite top1 score")
        threshold = float(_one(frame, f"{side}_decision_threshold"))
        if np.isnan(threshold):
            raise ValueError("NaN threshold")
        accepted = scores >= threshold
        if not np.array_equal(accepted, _bool(frame, f"{side}_accepted")):
            raise ValueError("stored acceptance disagrees with scores")
        correct = _bool(frame, f"{side}_top_k_correct")
        rank = frame[f"{side}_true_identity_rank"].to_numpy(float)
        if not np.array_equal(correct[mated], (rank[mated] >= 1) & (rank[mated] <= 20)):
            raise ValueError("stored top20 correctness disagrees with genuine rank")
        genuine = frame[f"{side}_true_identity_score"].to_numpy(float)
        if np.isnan(genuine[mated & correct]).any():
            raise ValueError("missing genuine score for a top20 hit")
        tpir = mated & correct & (genuine >= threshold)
        if not np.array_equal(tpir[mated], _bool(frame, f"{side}_tpir_at_rank_k")[mated]):
            raise ValueError("stored TPIR violates genuine-score-topk metric")
        unknown = accepted[~mated]
        low, high = wilson_score_interval(int(unknown.sum()), len(unknown))
        meta.update({f"{side}_threshold": threshold,
                     f"{side}_fpir": float(unknown.mean()),
                     f"{side}_fpir_ci_low": low, f"{side}_fpir_ci_high": high,
                     f"{side}_tpir20_count": int(tpir.sum()),
                     f"{side}_tpir20": float(tpir.sum()/mated.sum()),
                     f"{side}_rank20": float(correct[mated].mean())})
        decisions[side] = unknown
    if policy == "frozen_origin" and meta["origin_threshold"] != meta["compressed_threshold"]:
        raise ValueError("frozen thresholds differ")
    meta.update(event_counts(decisions["origin"], decisions["compressed"], seed=seed, resamples=resamples))
    meta["mated_count"] = int(mated.sum())
    meta["target_met_on_test"] = meta["compressed_fpir"] <= meta["target_fpir"]
    unknown = frame.loc[~mated]
    new = ~decisions["origin"] & decisions["compressed"]
    same = unknown.origin_top1_gallery_id.to_numpy() == unknown.compressed_top1_gallery_id.to_numpy()
    meta["new_fa_same_winner"] = int((new & same).sum())
    meta["new_fa_changed_winner"] = int((new & ~same).sum())
    tails = []
    for side in ("origin", "compressed"):
        values = unknown[f"{side}_top1_score"].to_numpy(float)
        for q in (.5, .95, .99, .999, 1.):
            tails.append(dict(side=side, quantile=q, score=float(np.quantile(values, q)),
                              threshold=meta[f"{side}_threshold"], score_space=meta[f"{side}_score_space"]))
    winners = []
    if cosine:
        original = unknown.origin_top1_score.to_numpy(float)
        fixed = unknown.compressed_score_at_origin_top1.to_numpy(float)
        maximum = unknown.compressed_top1_score.to_numpy(float)
        a, b = fixed-original, maximum-fixed
        residual = a+b-(maximum-original)
        if not np.isfinite(fixed).all() or b.min() < -1e-6 or np.abs(residual).max() > 1e-6:
            raise ValueError("cosine winner decomposition invariant failed")
        for label, mask in (("all_nonmated", np.ones(len(a), bool)), ("new_fa", new),
                            ("new_fa_same_winner", new & same), ("new_fa_changed_winner", new & ~same)):
            winners.append(dict(cohort=label, count=int(mask.sum()),
                                fixed_winner_drift_mean=float(a[mask].mean()) if mask.any() else np.nan,
                                selection_gain_mean=float(b[mask].mean()) if mask.any() else np.nan,
                                maximum_score_drift_mean=float((a+b)[mask].mean()) if mask.any() else np.nan,
                                decomposition_max_abs_residual=float(np.abs(residual).max()),
                                selection_gain_min_raw=float(b.min())))
        at_old = float((maximum >= meta["origin_threshold"]).mean())
        meta["score_effect_at_origin_threshold"] = at_old-meta["origin_fpir"]
        meta["threshold_effect_after_score_change"] = meta["compressed_fpir"]-at_old
    else:
        meta["score_effect_at_origin_threshold"] = np.nan
        meta["threshold_effect_after_score_change"] = np.nan
    return meta, tails, winners


def run_diagnosis(project_root, *, model="arcface", profiles=PQ_PROFILES,
                  modes=PQ_MODES, targets=(.01, .05, .1, .2, .3), seed=8972,
                  resamples=2000) -> dict:
    """Read one explicitly pinned run. Return compact tables without writing files."""
    root = Path(project_root).resolve()
    profiles, modes = tuple(profiles), tuple(modes)
    if not profiles or not modes or len(set(profiles)) != len(profiles) or len(set(modes)) != len(modes):
        raise ValueError("unique nonempty profiles and modes required")
    run = root / "runs" / SOURCE_RUNS[model]
    run_manifest = run / "run_manifest.json"
    run_data = json.loads(run_manifest.read_text(encoding="utf-8"))
    if run_data["status"] != "completed" or not (run / "COMPLETED").is_file():
        raise ValueError("source run must be completed")
    run_id = run.name.split("_step4_")[0]
    summary_dir = root / "results/paper/survface" / run_id / "search_space_v6_query_gallery_conditions"
    summary_path = summary_dir / "summary_manifest.json"
    sm = json.loads(summary_path.read_text(encoding="utf-8"))
    if sm.get("artifact_type") != "step4_search_space_query_gallery_conditions_v6" or sm.get("schema_version") != 6:
        raise ValueError("requires v6 query/gallery summary")
    if verified_path(root, sm["source_files"]["run_manifest.json"]) != run_manifest.resolve():
        raise ValueError("summary source run mismatch")
    compact = pd.read_csv(verified_path(root, sm["output_files"]["retrieval_summary.csv"]))
    ledger_path = run / "artifacts/step2_workflow/retrieval_ledger/manifest.json"
    ledger = load_retrieval_ledger_manifest(ledger_path)
    selected = [c for c in ledger["conditions"] if c["condition"]["compression_profile"] in profiles
                and c["condition"]["search_mode"] in modes]
    expected = {(p, m) for p in profiles for m in modes}
    observed = [(c["condition"]["compression_profile"], c["condition"]["search_mode"]) for c in selected]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("requested profile/mode grid missing or duplicated")
    if not targets or len(set(targets)) != len(targets) or any(t <= 0 or t >= 1 for t in targets):
        raise ValueError("unique target FPIRs in (0,1) required")
    rows, tails, winners, adjacent = [], [], [], []
    previous = {}
    baseline = None
    inventory = []
    selected.sort(key=lambda c: (modes.index(c["condition"]["search_mode"]), profiles.index(c["condition"]["compression_profile"])))
    for condition in selected:
        core_path = verified_path(ledger_path.parent, condition["core"])
        core = pd.read_parquet(core_path)
        if len(core) != condition["row_count"] or core.core_row_id.duplicated().any():
            raise ValueError("invalid core row count/keys")
        for col, value in condition["condition"].items():
            if _one(core, col) != value:
                raise ValueError(f"core lineage mismatch: {col}")
        if _one(core, "threshold_source_split") != "calibration" or _one(core, "evaluation_split") != "test":
            raise ValueError("requires calibration thresholds and test evaluation")
        if _one(core, "model_uid") != sm["model_uid"]:
            raise ValueError("summary/ledger model mismatch")
        identity_cols = ["query_id", "query_identity_id", "is_mated", "origin_top1_gallery_id", "origin_top1_score",
                         "origin_top_k_correct", "origin_true_identity_score", "origin_true_identity_rank",
                         "protocol_uid", "extraction_uid", "origin_embedding_artifact_uid", "gallery_template_count"]
        aligned = core[identity_cols].sort_values("query_id").reset_index(drop=True)
        if baseline is None:
            baseline = aligned
        elif not baseline.equals(aligned):
            raise ValueError("cross-condition origin/cohort/lineage mismatch")
        applicable = threshold_policies_for_search_mode(_one(core, "search_mode"))
        ds = [d for d in condition["decisions"] if d["target_fpir"] in targets]
        if {(d["target_fpir"], d["threshold_policy"]) for d in ds} != {(t,p) for t in targets for p in applicable} or len(ds) != len(targets)*len(applicable):
            raise ValueError("decision grid missing/duplicated or illegal")
        for decision in ds:
            decision_path = verified_path(ledger_path.parent, decision["artifact"])
            values = pd.read_parquet(decision_path)
            if len(values) != len(core) or set(values.core_row_id) != set(core.core_row_id):
                raise ValueError("decision/core row keys differ")
            frame = core.merge(values, on="core_row_id", validate="one_to_one").sort_values("query_id")
            for col in ("target_fpir", "threshold_policy"):
                if _one(frame, col) != decision[col]:
                    raise ValueError("decision metadata mismatch")
            row, tail, winner = diagnose_decision(frame, seed=seed, resamples=resamples)
            match = compact.loc[(compact.compression_profile == row["compression_profile"]) &
                                (compact.search_mode == row["search_mode"]) &
                                (compact.target_fpir == row["target_fpir"]) &
                                (compact.threshold_policy == row["threshold_policy"])]
            if len(match) != 1:
                raise ValueError("ambiguous compact summary row")
            stored = match.iloc[0]
            signature = sm["origin_calibration_signatures"][str(float(row["target_fpir"]))]
            if (signature["protocol_uid"] != row["protocol_uid"]
                    or signature["test_non_mated_count"] != row["n"]
                    or signature["test_false_accept_count"] != row["reference_fa"]
                    or signature["decision_threshold"] != row["origin_threshold"]):
                raise ValueError("origin calibration signature mismatch")
            for column in ("protocol_uid", "extraction_uid", "origin_embedding_artifact_uid", "gallery_template_count", "model_uid"):
                if stored[column] != _one(frame, column):
                    raise ValueError(f"compact/ledger lineage mismatch: {column}")
            for side in ("origin", "compressed"):
                if not np.isclose(stored[f"{side}_decision_threshold"], row[f"{side}_threshold"], rtol=0, atol=1e-10):
                    raise ValueError("compact/ledger threshold mismatch")
            for field, calculated in (("origin_false_accept_count", row["reference_fa"]),
                                      ("compressed_false_accept_count", row["candidate_fa"]),
                                      ("non_mated_count", row["n"]), ("mated_count", row["mated_count"]),
                                      ("origin_tpir20_count", row["origin_tpir20_count"]),
                                      ("compressed_tpir20_count", row["compressed_tpir20_count"])):
                if stored[field] != calculated:
                    raise ValueError(f"compact/ledger metric mismatch: {field}")
            keys = {k: row[k] for k in ("compression_profile", "search_mode", "threshold_policy", "target_fpir")}
            rows.append(row)
            tails.extend({**keys, **t} for t in tail)
            winners.extend({**keys, **w} for w in winner)
            mask = ~frame.is_mated.to_numpy(bool)
            accepted = frame.compressed_accepted.to_numpy(bool)[mask]
            key = (row["search_mode"], row["threshold_policy"], row["target_fpir"])
            if key in previous:
                prev_profile, prev = previous[key]
                adjacent.append({**keys, "reference_profile": prev_profile,
                                 **event_counts(prev, accepted, seed=seed, resamples=resamples)})
            previous[key] = (row["compression_profile"], accepted)
        inventory.append(dict(condition_id=condition["condition_id"], core_sha256=condition["core"]["sha256"],
                              profile=condition["condition"]["compression_profile"], mode=condition["condition"]["search_mode"]))
    provenance = dict(model=model, source_run=str(run), run_id=run_id,
                      generated_at_utc=datetime.now(timezone.utc).isoformat(),
                      run_manifest_sha256=sha256_file(run_manifest),
                      summary_manifest_sha256=sha256_file(summary_path),
                      ledger_manifest_sha256=sha256_file(ledger_path),
                      implementation_sha256=sha256_file(Path(__file__)),
                      metrics_sha256=sha256_file(root / "research/evaluation/metrics.py"),
                      search_conditions_sha256=sha256_file(root / "research/evaluation/search_conditions.py"),
                      numpy_version=np.__version__, pandas_version=pd.__version__,
                      source_evaluator_git=sm.get("evaluator_git"),
                      origin_calibration_signatures=sm["origin_calibration_signatures"],
                      source_claim_status=sm.get("claim_status"), seed=seed, resamples=resamples,
                      profiles=list(profiles), modes=list(modes), targets=list(targets),
                      ci_contract="fixed-condition-query-paired-bootstrap-v1", limitations=LIMITATIONS,
                      source_artifacts_preserved=True, inventory=inventory)
    return dict(summary=pd.DataFrame(rows), tails=pd.DataFrame(tails), winners=pd.DataFrame(winners),
                adjacent=pd.DataFrame(adjacent), provenance=provenance)


def write_diagnosis(result: dict, output_root) -> Path:
    """Explicit opt-in export, new directory only; never overwrite completed results."""
    destination = Path(output_root) / ("diagnostic-" + uuid.uuid4().hex[:16])
    destination.mkdir(parents=True, exist_ok=False)
    manifest = dict(result["provenance"], artifact_type="compression_fpir_failure_diagnosis", schema_version=1,
                    status="completed", output_files={})
    for name in ("summary", "tails", "winners", "adjacent"):
        path = destination / f"{name}.csv"
        result[name].to_csv(path, index=False)
        manifest["output_files"][path.name] = dict(path=path.name, bytes=path.stat().st_size, sha256=sha256_file(path))
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return destination
