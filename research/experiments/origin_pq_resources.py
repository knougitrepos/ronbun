"""Resource checkpoints and removal of verified, redundant origin replay scratch."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import psutil

from research.runtime.hashing import sha256_file


class ResourceBudgetExceeded(RuntimeError):
    """Stop at a checkpoint; this is a guard, not an operating-system memory cap."""


def check_resources(*, minimum_available_gb=8., maximum_process_gb=16.):
    if (not all(math.isfinite(v) for v in (minimum_available_gb, maximum_process_gb))
            or minimum_available_gb < 0 or maximum_process_gb <= 0):
        raise ValueError("invalid memory guard settings")
    available = psutil.virtual_memory().available / 2**30
    used = psutil.Process().memory_info().rss / 2**30
    if available < minimum_available_gb or used > maximum_process_gb:
        raise ResourceBudgetExceeded(
            f"Memory checkpoint: available={available:.2f} GiB, process={used:.2f} GiB; "
            "completed jobs are retained. Restart the kernel before continuing.")
    return dict(available_gb=available, process_gb=used)


def prune_origin_replay_scratch(directory):
    """Delete only journal-listed shards duplicated in verified final score tables.

    Completed manifest and its two referenced score tables are immutable.
    Missing shards are allowed so an interrupted cleanup can finish on restart.
    No recursive directory deletion or broad glob cleanup is performed.
    """
    root = Path(directory).resolve()
    journal_path = root / "progress.json"
    manifest_path = root / "manifest.json"
    if not journal_path.exists() or not manifest_path.exists():
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    journal = json.loads(journal_path.read_text(encoding="utf8"))
    if (manifest.get("status") != "completed"
            or manifest.get("artifact_type") != "origin_calibration_test_score_tables"
            or manifest.get("spec") != journal.get("spec")):
        raise ValueError("origin scratch cleanup requires matching completed scores")
    for name in ("calibration_scores.parquet", "test_scores.parquet"):
        entry = manifest["files"][name]
        if sha256_file(root / name) != entry["sha256"]:
            raise ValueError("origin final score hash mismatch before cleanup")
    shards = journal["shards"]
    if (sum(x["rows"] for x in shards) != manifest["files"]["calibration_scores.parquet"]["rows"]
            or len({x["name"] for x in shards}) != len(shards)):
        raise ValueError("origin scratch coverage mismatch")
    paths = []
    for entry in shards:
        name = entry["name"]
        if not re.fullmatch(r"calibration-\d{6}\.parquet", name):
            raise ValueError("invalid origin scratch name")
        path = (root / name).resolve()
        if path.parent != root or name in manifest["files"]:
            raise ValueError("origin scratch path outside its owned directory")
        if path.exists():
            if sha256_file(path) != entry["sha256"]:
                raise ValueError("origin scratch hash mismatch")
            paths.append(path)
    # Every surviving file is verified before the first unlink.
    for path in paths:
        path.unlink()
    journal_path.unlink()
    return len(paths) + 1
