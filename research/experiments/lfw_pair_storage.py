"""Transactional pair checkpoints and immutable compact, verified publications."""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import sqlite3
import shutil
import uuid

import pandas as pd

from research.runtime.hashing import canonical_sha256, sha256_file


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)


class PairStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, spec TEXT, payload BLOB, sha TEXT)")
        self.connection.commit()

    def close(self):
        self.connection.close()

    def get(self, spec):
        row = self.connection.execute("SELECT spec,payload,sha FROM records WHERE key=?",
                                      (canonical_sha256(spec),)).fetchone()
        if row is None:
            return None
        if row[0] != json_text(spec) or hashlib.sha256(row[1]).hexdigest() != row[2]:
            raise ValueError("pair checkpoint integrity failure")
        with ZipFile(BytesIO(row[1])) as archive:
            return {name[:-8]: pd.read_parquet(BytesIO(archive.read(name))) for name in archive.namelist()}

    def put(self, spec, tables):
        existing = self.get(spec)
        if existing is not None:
            if set(existing) != set(tables):
                raise ValueError("cannot replace completed pair checkpoint")
            for name in tables:
                pd.testing.assert_frame_equal(existing[name], tables[name], check_exact=True)
            return
        stream = BytesIO()
        with ZipFile(stream, "w") as archive:
            for name, frame in tables.items():
                if not name.isidentifier():
                    raise ValueError("unsafe table name")
                archive.writestr(name + ".parquet", frame.to_parquet(index=False, compression="zstd"))
        payload = stream.getvalue()
        with self.connection:
            self.connection.execute("INSERT INTO records VALUES (?,?,?,?)",
                (canonical_sha256(spec), json_text(spec), payload, hashlib.sha256(payload).hexdigest()))


def seed_summary(frame, keys):
    """Preserve model/fold/profile/method/target; seeds are descriptive, not new subjects."""
    if frame.empty:
        return frame
    grouped = frame.groupby(keys, dropna=False, sort=True)
    result = grouped.size().rename("seed_count").to_frame()
    for col in frame.select_dtypes(include="number").columns:
        if col in keys or col == "partition_seed":
            continue
        for stat in ("min", "median", "max"):
            result[f"{col}_{stat}"] = getattr(grouped[col], stat)()
    if "target_met_on_test" in frame:
        result["target_met_seed_count"] = grouped.target_met_on_test.sum()
    return result.reset_index()


def compact_tables(tables):
    accuracy = tables["accuracy_cv"].drop(columns="partition_seed").drop_duplicates().reset_index(drop=True)
    grouped = accuracy.groupby(["model", "compression_profile", "score_space"])
    accuracy_summary = grouped.agg(evaluated_folds=("fold", "nunique"),
        correct_count=("correct_count", "sum"), test_pairs=("test_pairs", "sum"),
        accuracy_fold_mean=("accuracy", "mean"), accuracy_fold_std=("accuracy", "std")).reset_index()
    accuracy_summary["pooled_accuracy"] = accuracy_summary.correct_count / accuracy_summary.test_pairs
    accuracy_summary["all_ten_folds_complete"] = accuracy_summary.evaluated_folds.eq(10)
    result = {
        "operating": seed_summary(tables["operating"], ["model", "fold", "compression_profile", "score_space", "method", "target_fmr"]),
        "diagnostic": seed_summary(tables["diagnostic"], ["model", "fold", "compression_profile", "method", "fitted_target_fmr", "requested_fmr"]),
        "paired": seed_summary(tables["paired"], ["model", "fold", "compression_profile", "method", "target_fmr", "comparison"]),
        "accuracy_cv": accuracy,
        "accuracy_summary": accuracy_summary,
        "inventory": tables["inventory"].copy(),
    }
    catalog = tables.get("baseline_catalog", pd.DataFrame())
    if not catalog.empty:
        result["baseline_catalog"] = catalog
        contrasts = backbone_contrasts(tables["operating"], catalog)
        if not contrasts.empty:
            result["backbone_contrasts"] = seed_summary(contrasts, ["model", "parent_baseline", "fold",
                "compression_profile", "score_space", "method", "target_fmr"])
    return result


def backbone_contrasts(operating, catalog):
    """Within each fixed cohort compare checkpoints separately from FIQA gains.

    These are descriptive rate differences, not a paired CI or causal claim.
    The existing paired table estimates FIQA/compression contrasts WITHIN a model.
    """
    keys = ["fold", "partition_seed", "compression_profile", "score_space", "method", "target_fmr"]
    output = []
    for baseline in catalog.itertuples(index=False):
        if baseline.parent_baseline == "none":
            continue
        candidate = operating.loc[operating.model.eq(baseline.model)]
        reference = operating.loc[operating.model.eq(baseline.parent_baseline)]
        joined = candidate.merge(reference, on=keys, suffixes=("_candidate", "_reference"), validate="one_to_one")
        if len(joined) != len(candidate) or len(joined) != len(reference):
            raise ValueError("checkpoint comparison requires identical fold/seed/profile/method/target cohorts")
        for count in ("test_pairs", "genuine_pairs", "impostor_pairs"):
            if not joined[count+"_candidate"].eq(joined[count+"_reference"]).all():
                raise ValueError("checkpoint comparison denominators differ")
        frame = joined[keys].copy()
        frame["model"], frame["parent_baseline"] = baseline.model, baseline.parent_baseline
        for metric in ("tar", "realized_fmr"):
            frame["candidate_"+metric] = joined[metric+"_candidate"]
            frame["reference_"+metric] = joined[metric+"_reference"]
            frame[metric+"_difference"] = joined[metric+"_candidate"] - joined[metric+"_reference"]
        # Difference of the FIQA gain, holding compression fixed in each backbone.
        global_keys = [k for k in keys if k != "method"]
        globals_ = frame.loc[frame.method.eq("global_safe"), global_keys+["tar_difference"]].rename(columns={"tar_difference":"global_backbone_tar_difference"})
        frame = frame.merge(globals_, on=global_keys, validate="many_to_one")
        frame["fiqa_gain_difference"] = frame.tar_difference-frame.global_backbone_tar_difference
        output.append(frame)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def read_report(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if not (root / "_SUCCESS").is_file() or manifest.get("artifact_type") != "lfw_pair_verification_summary":
        raise ValueError("incomplete pair report")
    if manifest.get("status") not in ("partial", "completed"):
        raise ValueError("invalid pair report status")
    expected_uid = "lfw-pairs-report-" + canonical_sha256(dict(campaign=manifest["spec"],
        completed_jobs=manifest["completed_jobs"], expected_jobs=manifest["expected_jobs"]))[:24]
    if (manifest["report_uid"] != expected_uid or (root / "_SUCCESS").read_text(encoding="utf8") != expected_uid
            or (manifest["status"] == "completed") != (manifest["completed_jobs"] == manifest["expected_jobs"])):
        raise ValueError("pair report completion/spec mismatch")
    for name, entry in manifest["files"].items():
        if Path(name).name != name or sha256_file(root / name) != entry["sha256"]:
            raise ValueError("pair report integrity failure")
    tables = {}
    with ZipFile(root / "analysis.zip") as archive:
        if set(archive.namelist()) != set(manifest["members"]):
            raise ValueError("pair report member inventory mismatch")
        for name, entry in manifest["members"].items():
            payload = archive.read(name)
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError("pair report table hash mismatch")
            frame = pd.read_csv(BytesIO(payload), float_precision="round_trip")
            if len(frame) != entry["rows"]:
                raise ValueError("pair report table count mismatch")
            tables.setdefault(entry["table"], []).append(frame)
    return {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}, manifest


def publish(root, tables, *, spec, completed_jobs, expected_jobs, raw_path, guide):
    compact = compact_tables(tables)
    publication_spec = dict(campaign=spec, completed_jobs=completed_jobs, expected_jobs=expected_jobs)
    uid = "lfw-pairs-report-" + canonical_sha256(publication_spec)[:24]
    dest = Path(root) / "reports" / uid
    if dest.exists():
        read_report(dest)
        return dest
    staging = dest.parent / (".staging-" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    members = {}
    with ZipFile(staging / "analysis.zip", "w", compression=ZIP_DEFLATED) as archive:
        for table, frame in compact.items():
            # Keep potentially large diagnostics usable in chat without dropping
            # folds, seeds or targets: model/profile/fitted-target are upload units.
            keys = [k for k in ("model", "compression_profile") if k in frame]
            if table == "diagnostic":
                keys.append("fitted_target_fmr")
            groups = frame.groupby(keys, sort=True) if keys else [("all", frame)]
            for group, chunk in groups:
                payload = chunk.to_csv(index=False).encode("utf8")
                parts = group if isinstance(group, tuple) else (group,)
                name = table + "_" + "_".join(str(v) for v in parts) + ".csv"
                archive.writestr(name, payload)
                members[name] = dict(table=table, rows=len(chunk), bytes=len(payload),
                    estimated_tokens_upper=len(payload), estimated_tokens_typical=(len(payload)+3)//4,
                    sha256=hashlib.sha256(payload).hexdigest())
    shutil.copyfile(guide, staging / "INTERPRETATION.md")
    complete = completed_jobs == expected_jobs
    status = "completed" if complete else "partial"
    retention = "retained" if spec["keep_raw_results"] or not complete else "not_retained_after_verified_summary"
    start = (f"# LFW 1:1 verification: {status}\n\n"
        f"Completed jobs: {completed_jobs}/{expected_jobs}. This is not a 1:N identification result.\n\n"
        "Read INTERPRETATION.md first. Upload manifest.json and operating_<model>_<profile>.csv, then "
        "accuracy_summary_<model>_<profile>.csv and paired_<model>_<profile>.csv. "
        "Diagnostics are split additionally by fitted target FMR and are separate test-curve descriptions. "
        "analysis.zip contains only aggregates. If too large for the chat, unpack and upload the selected logical CSVs.\n\n"
        "Repeated partition seeds share test pairs. Min/median/max and CI endpoint ranges are descriptive, "
        "not combined confidence intervals. Counts and denominators remain in each fold/target/profile/method.\n\n"
        f"Raw status: {retention}. Raw/checkpoint location: {raw_path}. "
        "Without raw data, new cutoffs, pair decisions and new bootstrap analyses require recomputation.\n")
    (staging / "START_HERE.md").write_text(start, encoding="utf8")
    files = {p.name: dict(sha256=sha256_file(p), bytes=p.stat().st_size) for p in staging.iterdir()}
    manifest = dict(artifact_type="lfw_pair_verification_summary", schema_version=1, status=status,
        report_uid=uid, spec=spec, completed_jobs=completed_jobs, expected_jobs=expected_jobs,
        files=files, members=members, raw_results_status=retention, raw_checkpoint=str(raw_path),
        raw_checkpoint_sha256=sha256_file(raw_path),
        uncertainty="genuine identity cluster bootstrap; impostor Wilson is nominal pair-independence only",
        token_estimation="UTF-8 bytes/4 typical; bytes upper estimate, not a model-specific guarantee",
        extracted_table_bytes=sum(v["bytes"] for v in members.values()),
        formal_fmr_guarantee=False, threshold_fit_on_test=False, failed_jobs=0,
        checkpoint_training_overlap_verified=False)
    (staging / "manifest.json").write_text(json_text(manifest), encoding="utf8")
    (staging / "_SUCCESS").write_text(uid, encoding="utf8")
    reloaded, _ = read_report(staging)
    for name, frame in compact.items():
        # CSV reload must preserve all rows and numerical values before raw cleanup.
        keys = [c for c in frame.columns if c in ("model", "fold", "partition_seed", "compression_profile", "method", "target_fmr", "fitted_target_fmr", "requested_fmr", "comparison")]
        a = frame.sort_values(keys).reset_index(drop=True) if keys else frame.reset_index(drop=True)
        b = reloaded[name].sort_values(keys).reset_index(drop=True) if keys else reloaded[name]
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-15)
    staging.rename(dest)
    return dest


def cleanup_checkpoint(path, campaign_root, report):
    """Remove only this completed campaign's explicit SQLite file after verification."""
    _, meta = read_report(report)
    root, path = Path(campaign_root).resolve(), Path(path).resolve()
    if (meta["status"] != "completed" or meta["spec"]["keep_raw_results"]
            or path != root / "raw" / "checkpoint.sqlite3"
            or Path(meta["raw_checkpoint"]).resolve() != path):
        raise ValueError("raw cleanup not authorized by this completed campaign")
    if path.exists():
        if sha256_file(path) != meta["raw_checkpoint_sha256"]:
            raise ValueError("checkpoint changed after report verification")
        path.unlink()
