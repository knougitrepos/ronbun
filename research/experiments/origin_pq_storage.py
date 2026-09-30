"""Few-file, lossless checkpoints and reports for the origin/PQ experiment.

One SQLite file holds completed split checkpoints. Each split is a single
transaction, so interruption cannot publish a partly written result. Final
reports contain only a manifest and a ZIP of compressed Parquet tables.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import sqlite3
from time import sleep
from uuid import uuid4
from zipfile import ZIP_STORED, ZipFile

import pandas as pd

from research.runtime.hashing import canonical_sha256, sha256_file

_TABLE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SCHEMA_VERSION = 2


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=str)


def split_uid(spec):
    return "origin-pq-split-" + canonical_sha256(spec)[:24]


def _manifest(uid, spec, entries, *, storage):
    return dict(
        artifact_type="origin_vs_pq_calibration", schema_version=_SCHEMA_VERSION,
        status="completed", result_uid=uid, spec=json.loads(_json(spec)),
        storage=storage, tables=entries, formal_fpir_guarantee=False,
        uncertainty=dict(
            tpir="query_weighted_mated_identity_cluster", fpir="query_bootstrap_and_wilson",
            threshold_uncertainty_included=False, multiple_comparison_adjustment="none",
            between_splits="descriptive_not_independent_replications",
            diagnostic_curves="test_interpolation_only_no_ci_no_deployment_threshold"),
        checkpoint_training_overlap_verified=False,
        unseen_identity_claim_for_rfw_edgeface=False)


def _names(tables):
    names = list(tables)
    if not names or any(not isinstance(n, str) or not _TABLE_NAME.fullmatch(n) for n in names):
        raise ValueError("non-empty, safe table names required")
    return sorted(names)


def _encode_frame(frame):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("checkpoint tables must be pandas DataFrames")
    # Preserve index metadata and numeric values; no CSV float round trip.
    stream = BytesIO()
    frame.to_parquet(stream, engine="pyarrow", compression="zstd", index=None)
    return stream.getvalue()


def _entry(frame, blob):
    return dict(sha256=sha256(blob).hexdigest(), bytes=len(blob), rows=len(frame),
                columns=list(frame.columns), format="parquet", compression="zstd")


def _decode_frame(blob, entry):
    if len(blob) != entry["bytes"] or sha256(blob).hexdigest() != entry["sha256"]:
        raise ValueError("table blob hash mismatch")
    frame = pd.read_parquet(BytesIO(blob), engine="pyarrow", use_threads=False)
    if len(frame) != entry["rows"] or list(frame.columns) != entry["columns"]:
        raise ValueError("table shape mismatch")
    return frame


def _validate_manifest(manifest, *, uid, spec=None, storage):
    if (manifest.get("artifact_type") != "origin_vs_pq_calibration"
            or manifest.get("schema_version") != _SCHEMA_VERSION
            or manifest.get("status") != "completed"
            or manifest.get("storage") != storage
            or manifest.get("result_uid") != uid):
        raise ValueError("invalid completed storage manifest")
    if spec is not None and _json(manifest.get("spec")) != _json(spec):
        raise ValueError("checkpoint spec mismatch")
    _names(manifest.get("tables", {}))


class SplitStore:
    """An immutable split cache with one persistent file and a transient journal.

    The SQLite rollback journal uses FULL synchronous writes. A crash leaves
    either the previous database or the complete transaction; an unfinished
    split is never returned by ``get``. It is safe to retry that split.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            self.connection.execute("PRAGMA journal_mode=DELETE")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA cache_size=-8192")
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("unsupported split database version")
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    uid TEXT PRIMARY KEY, spec_json TEXT NOT NULL,
                    manifest_json TEXT, manifest_sha256 TEXT,
                    status TEXT NOT NULL CHECK(status IN ('writing', 'completed'))
                );
                CREATE TABLE IF NOT EXISTS tables (
                    job_uid TEXT NOT NULL REFERENCES jobs(uid) ON DELETE CASCADE,
                    name TEXT NOT NULL, payload BLOB NOT NULL,
                    PRIMARY KEY(job_uid, name)
                );
                PRAGMA user_version=1;
            """)
        except BaseException:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def has(self, spec):
        """Check completion and verify the stored tables, never trust a row alone."""
        return self.get(spec) is not None

    def get(self, spec):
        uid = split_uid(spec)
        row = self.connection.execute(
            "SELECT spec_json, manifest_json, manifest_sha256, status FROM jobs WHERE uid=?",
            (uid,)).fetchone()
        if row is None or row[3] != "completed":
            return None
        if row[0] != _json(spec):
            raise ValueError("checkpoint spec mismatch")
        if not row[1] or sha256(row[1].encode("utf8")).hexdigest() != row[2]:
            raise ValueError("checkpoint manifest hash mismatch")
        manifest = json.loads(row[1])
        _validate_manifest(manifest, uid=uid, spec=spec, storage="sqlite-parquet-v1")
        stored_names = {r[0] for r in self.connection.execute(
            "SELECT name FROM tables WHERE job_uid=?", (uid,))}
        if stored_names != set(manifest["tables"]):
            raise ValueError("checkpoint table inventory mismatch")
        tables = {}
        for name, entry in manifest["tables"].items():
            blob = self.connection.execute(
                "SELECT payload FROM tables WHERE job_uid=? AND name=?", (uid, name)).fetchone()[0]
            tables[name] = _decode_frame(blob, entry)
        return tables, manifest

    def put(self, spec, tables):
        """Commit all tables together, refusing changes to any completed split."""
        names, uid = _names(tables), split_uid(spec)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.get(spec)
            if existing is not None:
                existing_tables, manifest = existing
                if set(existing_tables) != set(names):
                    raise ValueError("cannot replace completed split tables")
                for name in names:
                    try:
                        pd.testing.assert_frame_equal(existing_tables[name], tables[name], check_exact=True)
                    except AssertionError as error:
                        raise ValueError("cannot replace completed split tables") from error
                self.connection.execute("COMMIT")
                return manifest
            # Only unpublished rows may be removed; completed rows are immutable.
            self.connection.execute("DELETE FROM jobs WHERE uid=? AND status='writing'", (uid,))
            self.connection.execute("INSERT INTO jobs(uid,spec_json,status) VALUES(?,?,'writing')",
                                    (uid, _json(spec)))
            entries = {}
            for name in names:
                blob = _encode_frame(tables[name])
                entries[name] = _entry(tables[name], blob)
                self.connection.execute("INSERT INTO tables(job_uid,name,payload) VALUES(?,?,?)",
                                        (uid, name, sqlite3.Binary(blob)))
            manifest = _manifest(uid, spec, entries, storage="sqlite-parquet-v1")
            payload = _json(manifest)
            self.connection.execute(
                "UPDATE jobs SET manifest_json=?, manifest_sha256=?, status='completed' WHERE uid=?",
                (payload, sha256(payload.encode("utf8")).hexdigest(), uid))
            self.connection.execute("COMMIT")
            return manifest
        except BaseException:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise


def write_report(root, tables, spec):
    """Publish a complete report as two files without replacing completed output."""
    names = _names(tables)
    uid = "origin-pq-report-" + canonical_sha256(spec)[:24]
    destination = Path(root) / uid
    if destination.exists():
        existing, manifest = read_report(destination)
        if _json(manifest["spec"]) != _json(spec) or set(existing) != set(names):
            raise ValueError("cannot replace completed report")
        for name in names:
            try:
                pd.testing.assert_frame_equal(existing[name], tables[name], check_exact=True)
            except AssertionError as error:
                raise ValueError("cannot replace completed report") from error
        return destination
    staging = Path(root) / (".staging-" + uuid4().hex)
    staging.mkdir(parents=True)
    entries = {}
    archive = staging / "results.zip"
    # Parquet already uses Zstandard; storing entries avoids another compression pass.
    with ZipFile(archive, mode="w", compression=ZIP_STORED, allowZip64=True) as zipped:
        for name in names:
            blob = _encode_frame(tables[name])
            entries[name] = {**_entry(tables[name], blob), "path": f"{name}.parquet"}
            zipped.writestr(entries[name]["path"], blob)
    manifest = _manifest(uid, spec, entries, storage="zip-parquet-v1")
    manifest["files"] = {"results.zip": sha256_file(archive)}
    (staging / "manifest.json").write_text(_json(manifest), encoding="utf8")
    # Same-parent rename is atomic. Preserve staging on any publication failure.
    for attempt in range(4):
        if destination.exists():
            raise FileExistsError(f"completed report appeared during write: {destination}")
        try:
            staging.rename(destination)
            break
        except PermissionError:
            if attempt == 3:
                raise
            sleep(.05 * 2**attempt)
    return destination


def read_report(directory, names=None):
    """Validate the whole archive, then load all or selected Parquet tables."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    uid = "origin-pq-report-" + canonical_sha256(manifest.get("spec"))[:24]
    _validate_manifest(manifest, uid=uid, storage="zip-parquet-v1")
    if directory.name != uid:
        raise ValueError("report directory identity mismatch")
    archive = directory / "results.zip"
    if (set(manifest.get("files", {})) != {"results.zip"}
            or sha256_file(archive) != manifest["files"]["results.zip"]):
        raise ValueError("report archive hash mismatch")
    selected = list(manifest["tables"]) if names is None else list(names)
    if len(set(selected)) != len(selected) or not set(selected).issubset(manifest["tables"]):
        raise ValueError("unknown or duplicate report table selection")
    entries = {name: entry["path"] for name, entry in manifest["tables"].items()}
    if any(path != f"{name}.parquet" for name, path in entries.items()):
        raise ValueError("unsafe report table path")
    with ZipFile(archive) as zipped:
        if sorted(zipped.namelist()) != sorted(entries.values()):
            raise ValueError("report table inventory mismatch")
        tables = {name: _decode_frame(zipped.read(entries[name]), manifest["tables"][name])
                  for name in selected}
    return tables, manifest
