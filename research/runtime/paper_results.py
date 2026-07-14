from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Mapping
from uuid import uuid4

from research.runtime.hashing import sha256_file
from research.runtime.run_store import RunStore


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")


def _validated_component(value: str, *, field: str) -> str:
    normalized = value.strip().lower()
    if not normalized or not _SAFE_COMPONENT.fullmatch(normalized):
        raise ValueError(f"{field} must contain only letters, numbers, '_' or '-': {value!r}")
    return normalized


def _manifest_payload(
    *,
    run: RunStore,
    dataset: str,
    source_phase: str,
    source_attempt: str,
    files: Mapping[str, Path],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": dataset,
        "run_id": run.run_id,
        "config_hash": run.config_hash,
        "source": {
            "run_directory": str(run.run_dir.resolve()),
            "phase": source_phase,
            "attempt": source_attempt,
        },
        "files": {
            name: {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in sorted(files.items())
        },
    }


def export_paper_results(
    *,
    run: RunStore,
    dataset: str,
    source_phase: str,
    source_attempt: str,
    files: Mapping[str, str | Path],
    output_root: str | Path,
) -> dict[str, object]:
    """Atomically export a compact, Git-trackable result bundle.

    The immutable run remains the provenance source. Re-exporting the exact same
    bundle is safe, while a conflicting bundle for the same run is never
    overwritten silently.
    """
    dataset_name = _validated_component(dataset, field="dataset")
    if not files:
        raise ValueError("at least one result file is required")

    sources: dict[str, Path] = {}
    for destination_name, source in files.items():
        name = Path(destination_name).name
        if name != destination_name or name == "result_manifest.json":
            raise ValueError(f"result file name must be a simple file name: {destination_name!r}")
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        sources[name] = source_path

    root = Path(output_root).resolve()
    dataset_dir = root / dataset_name
    destination = dataset_dir / run.run_id
    manifest = _manifest_payload(
        run=run,
        dataset=dataset_name,
        source_phase=source_phase,
        source_attempt=source_attempt,
        files=sources,
    )

    existing_manifest_path = destination / "result_manifest.json"
    if destination.exists():
        if existing_manifest_path.is_file():
            existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
            same_content = all(
                existing.get(key) == manifest[key]
                for key in ("schema_version", "dataset", "run_id", "config_hash", "files")
            )
            intact_files = all(
                (destination / name).is_file()
                and sha256_file(destination / name) == manifest["files"][name]["sha256"]
                for name in sources
            )
            if same_content and intact_files:
                return {
                    "status": "unchanged",
                    "directory": str(destination),
                    "manifest": str(existing_manifest_path),
                    "files": sorted(sources),
                }
        raise FileExistsError(
            f"paper result bundle already exists with different contents: {destination}"
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    temporary = dataset_dir / f".{run.run_id}.{uuid4().hex}.tmp"
    temporary.mkdir(exist_ok=False)
    try:
        for name, source in sources.items():
            shutil.copy2(source, temporary / name)
        (temporary / "result_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "status": "created",
        "directory": str(destination),
        "manifest": str(destination / "result_manifest.json"),
        "files": sorted(sources),
    }
