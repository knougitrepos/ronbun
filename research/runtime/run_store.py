from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading
import traceback
from typing import Any
from zoneinfo import ZoneInfo

from research.runtime.hashing import canonical_sha256, sha256_file
from research.runtime.redaction import redact


KST = ZoneInfo("Asia/Seoul")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WRITE_LOCK = threading.Lock()
ACTIVE_RUN_POINTER = "active_run.json"


def _iso_utc(now: datetime | None = None) -> str:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-")
    return normalized or "experiment"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(redact(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_output(repo_root: Path, *args: str, binary: bool = False):
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=not binary,
    )
    if result.returncode != 0:
        return b"" if binary else ""
    return result.stdout


def _git_provenance(repo_root: Path) -> dict[str, object]:
    commit = str(_git_output(repo_root, "rev-parse", "HEAD")).strip() or None
    branch = str(_git_output(repo_root, "branch", "--show-current")).strip() or None
    status = str(_git_output(repo_root, "status", "--porcelain=v1"))
    diff = _git_output(repo_root, "diff", "--binary", "HEAD", binary=True)
    untracked_output = str(
        _git_output(repo_root, "ls-files", "--others", "--exclude-standard")
    )
    untracked_paths = [line for line in untracked_output.splitlines() if line]
    untracked_digest = hashlib.sha256()
    for relative_path in sorted(untracked_paths):
        path = repo_root / relative_path
        untracked_digest.update(relative_path.encode("utf-8"))
        untracked_digest.update(b"\0")
        if path.is_file():
            untracked_digest.update(sha256_file(path).encode("ascii"))
        untracked_digest.update(b"\0")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status.strip()),
        "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_paths": untracked_paths,
        "untracked_content_sha256": untracked_digest.hexdigest(),
    }


def _package_versions() -> dict[str, str | None]:
    packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "sqlalchemy",
        "pgvector",
        "faiss-cpu",
        "insightface",
        "onnxruntime",
        "opencv-python",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _gpu_info() -> dict[str, str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    first = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    if len(first) != 3:
        return {"raw": result.stdout.splitlines()[0].strip()}
    return {"name": first[0], "driver": first[1], "memory_mib": first[2]}


@dataclass
class PhaseContext(AbstractContextManager):
    run: "RunStore"
    phase_name: str
    attempt: int
    attempt_dir: Path
    started_at: str
    details: dict[str, Any] = field(default_factory=dict)
    status: str = "running"

    @property
    def artifact_dir(self) -> Path:
        directory = self.run.run_dir / "artifacts" / self.phase_name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def record(self, event: str, **data: Any) -> None:
        self.run.record_event(event, phase=self.phase_name, attempt=self.attempt, **data)

    def record_counts(self, **counts: int) -> None:
        self.details.setdefault("counts", {}).update({key: int(value) for key, value in counts.items()})
        self.record("phase_counts", counts=counts)

    def publish_artifact(self, source: str | Path, *, name: str | None = None) -> Path:
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination = self.artifact_dir / (name or source_path.name)
        if destination.exists():
            raise FileExistsError(
                f"artifact already exists and will not be overwritten: {destination}; "
                "use an attempt-qualified name"
            )
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        shutil.copyfile(source_path, temporary)
        os.replace(temporary, destination)
        artifact = {
            "path": str(destination.relative_to(self.run.run_dir)),
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        }
        self.details.setdefault("artifacts", []).append(artifact)
        self.record("artifact_published", artifact=artifact)
        return destination

    def _write_manifest(self, *, error: BaseException | None = None) -> None:
        payload = {
            "phase": self.phase_name,
            "attempt": self.attempt,
            "status": self.status,
            "started_at_utc": self.started_at,
            "finished_at_utc": _iso_utc(),
            "details": self.details,
        }
        if error is not None:
            payload["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            }
        _atomic_write_json(self.attempt_dir / "phase_manifest.json", payload)

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            self.status = "completed"
            self._write_manifest()
            self.run.record_event(
                "phase_completed", phase=self.phase_name, attempt=self.attempt
            )
            return False
        self.status = "failed"
        self._write_manifest(error=exc)
        self.run.record_event(
            "phase_failed",
            phase=self.phase_name,
            attempt=self.attempt,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return False


@dataclass
class RunStore:
    root: Path
    run_dir: Path
    run_id: str
    run_name: str
    sequence: int
    config_hash: str
    config: dict[str, Any]
    created_at_utc: str

    @classmethod
    def create(
        cls,
        *,
        experiment_name: str,
        config: dict[str, Any],
        root: str | Path = "runs",
        now: datetime | None = None,
        repo_root: str | Path = PROJECT_ROOT,
    ) -> "RunStore":
        safe_config = redact(config)
        config_hash = canonical_sha256(safe_config)
        local_now = now or datetime.now(KST)
        if local_now.tzinfo is None:
            local_now = local_now.replace(tzinfo=KST)
        local_now = local_now.astimezone(KST)
        root_path = Path(root)
        date_dir = root_path / local_now.strftime("%Y") / local_now.strftime("%m") / local_now.strftime("%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        slug = _slug(experiment_name)

        for sequence in range(1, 10000):
            run_id = f"{local_now:%Y%m%d}-R{sequence:03d}-{config_hash[:8]}"
            run_dir = date_dir / f"{run_id}_{slug}"
            try:
                run_dir.mkdir(parents=False, exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError(f"daily run sequence exhausted: {date_dir}")

        for child in ("logs", "phases", "artifacts", "figures", "models"):
            (run_dir / child).mkdir()

        created_at = _iso_utc(local_now)
        run = cls(
            root=root_path,
            run_dir=run_dir,
            run_id=run_id,
            run_name=experiment_name,
            sequence=sequence,
            config_hash=config_hash,
            config=safe_config,
            created_at_utc=created_at,
        )
        manifest = {
            "run_id": run_id,
            "run_name": experiment_name,
            "sequence": sequence,
            "status": "created",
            "created_at_utc": created_at,
            "created_at_kst": local_now.isoformat(),
            "config_hash": config_hash,
            "config": safe_config,
            "git": _git_provenance(Path(repo_root)),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "gpu": _gpu_info(),
                "packages": _package_versions(),
            },
        }
        _atomic_write_json(run_dir / "run_manifest.json", manifest)
        run.record_event("run_created", sequence=sequence, config_hash=config_hash)
        _write_active_run_pointer(run)
        return run

    @classmethod
    def open(cls, run_dir: str | Path) -> "RunStore":
        """Attach to an incomplete run so a failed phase can be retried explicitly."""
        directory = Path(run_dir)
        manifest_path = directory / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"run manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "run_id",
            "run_name",
            "sequence",
            "config_hash",
            "config",
            "created_at_utc",
        }
        missing = sorted(required.difference(manifest))
        if missing:
            raise ValueError(f"run manifest is missing fields: {missing}")
        if canonical_sha256(manifest["config"]) != manifest["config_hash"]:
            raise ValueError("run manifest config hash mismatch")
        root = directory.parents[3] if len(directory.parents) >= 4 else directory.parent
        return cls(
            root=root,
            run_dir=directory,
            run_id=str(manifest["run_id"]),
            run_name=str(manifest["run_name"]),
            sequence=int(manifest["sequence"]),
            config_hash=str(manifest["config_hash"]),
            config=dict(manifest["config"]),
            created_at_utc=str(manifest["created_at_utc"]),
        )

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    def _read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _assert_mutable(self) -> None:
        manifest = self._read_manifest()
        if manifest.get("status") == "completed" or (self.run_dir / "COMPLETED").exists():
            raise RuntimeError("completed runs are immutable")

    def _update_manifest(self, **changes: Any) -> None:
        manifest = self._read_manifest()
        manifest.update(redact(changes))
        _atomic_write_json(self.manifest_path, manifest)

    def record_event(self, event: str, **data: Any) -> None:
        payload = {
            "timestamp_utc": _iso_utc(),
            "run_id": self.run_id,
            "level": str(data.pop("level", "INFO")),
            "event": event,
            **redact(data),
        }
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        with _WRITE_LOCK:
            with (self.run_dir / "logs" / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def record_input(self, path: str | Path, *, role: str) -> dict[str, object]:
        self._assert_mutable()
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        entry: dict[str, object] = {
            "role": role,
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
        }
        manifest = self._read_manifest()
        inputs = list(manifest.get("inputs", []))
        for existing in inputs:
            if existing.get("role") == role and existing.get("path") == entry["path"]:
                if existing.get("sha256") != entry["sha256"]:
                    raise ValueError(f"registered input changed for {role}: {source}")
                return dict(existing)
        inputs.append(entry)
        self._update_manifest(inputs=inputs)
        self.record_event("input_registered", input=entry)
        return entry

    def verify_inputs(self, *, roles: set[str] | None = None) -> list[dict[str, object]]:
        """Fail fast when a frozen upstream input was removed or changed in place."""
        manifest = self._read_manifest()
        checked: list[dict[str, object]] = []
        for entry in manifest.get("inputs", []):
            if roles is not None and str(entry.get("role")) not in roles:
                continue
            path = Path(str(entry["path"]))
            if not path.is_file():
                raise FileNotFoundError(f"frozen input is missing: {path}")
            actual = sha256_file(path)
            expected = str(entry["sha256"])
            if actual != expected:
                raise ValueError(
                    f"frozen input hash mismatch for {entry.get('role')}: {path}"
                )
            checked.append(dict(entry))
        self.record_event(
            "inputs_verified",
            roles=sorted(roles) if roles is not None else "all",
            count=len(checked),
        )
        return checked

    def verify_phase_artifacts(
        self,
        phase_name: str,
        *,
        attempt: int | None = None,
    ) -> list[dict[str, object]]:
        """Verify checksums published by a completed upstream phase attempt."""
        attempts_dir = self.run_dir / "phases" / _slug(phase_name) / "attempts"
        candidates: list[tuple[int, dict[str, Any]]] = []
        for path in sorted(attempts_dir.glob("A*/phase_manifest.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            number = int(payload["attempt"])
            if payload.get("status") == "completed" and (attempt is None or number == attempt):
                candidates.append((number, payload))
        if not candidates:
            suffix = "latest" if attempt is None else f"A{attempt:03d}"
            raise RuntimeError(f"no completed {suffix} attempt for phase {phase_name}")
        number, payload = max(candidates, key=lambda item: item[0])
        verified: list[dict[str, object]] = []
        run_root = self.run_dir.resolve()
        for artifact in payload.get("details", {}).get("artifacts", []):
            path = (self.run_dir / str(artifact["path"])).resolve()
            if run_root not in path.parents:
                raise ValueError(f"artifact escapes run directory: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"phase artifact is missing: {path}")
            if sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"phase artifact hash mismatch: {path}")
            verified.append(dict(artifact))
        self.record_event(
            "phase_artifacts_verified",
            phase=phase_name,
            attempt=number,
            count=len(verified),
        )
        return verified

    def phase(self, phase_name: str) -> PhaseContext:
        self._assert_mutable()
        manifest = self._read_manifest()
        phase_slug = _slug(phase_name)
        attempts_dir = self.run_dir / "phases" / phase_slug / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, 10000):
            attempt_dir = attempts_dir / f"A{attempt:03d}"
            try:
                attempt_dir.mkdir(exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError(f"phase attempt sequence exhausted: {attempts_dir}")
        started_at = _iso_utc()
        _atomic_write_json(
            attempt_dir / "phase_manifest.json",
            {
                "phase": phase_name,
                "attempt": attempt,
                "status": "running",
                "started_at_utc": started_at,
            },
        )
        self._update_manifest(status="running", started_at_utc=manifest.get("started_at_utc", started_at))
        self.record_event("phase_started", phase=phase_name, attempt=attempt)
        return PhaseContext(self, phase_name, attempt, attempt_dir, started_at)

    def complete(self) -> None:
        if (self.run_dir / "COMPLETED").exists():
            raise RuntimeError("run is already completed")
        completed_at = _iso_utc()
        self._update_manifest(status="completed", finished_at_utc=completed_at)
        (self.run_dir / "COMPLETED").write_text(completed_at + "\n", encoding="utf-8")
        self.record_event("run_completed")

    def fail(self, error: BaseException) -> None:
        self._assert_mutable()
        self._update_manifest(
            status="failed",
            finished_at_utc=_iso_utc(),
            failure={"type": type(error).__name__, "message": str(error)},
        )
        self.record_event(
            "run_failed", level="ERROR", error_type=type(error).__name__, error_message=str(error)
        )


def _write_active_run_pointer(run: RunStore) -> Path:
    pointer_path = Path(run.root).expanduser().resolve() / ACTIVE_RUN_POINTER
    _atomic_write_json(
        pointer_path,
        {
            "run_dir": str(run.run_dir.resolve()),
            "run_id": run.run_id,
            "config_hash": run.config_hash,
            "updated_at_utc": _iso_utc(),
        },
    )
    return pointer_path


def _validate_active_run(
    run_dir: str | Path,
    *,
    expected_run_id: str | None = None,
    expected_config_hash: str | None = None,
) -> Path:
    directory = Path(run_dir).expanduser().resolve()
    run = RunStore.open(directory)
    manifest = run._read_manifest()
    if manifest.get("status") == "completed" or (directory / "COMPLETED").exists():
        raise RuntimeError(f"active run is already completed: {directory}")
    if expected_run_id is not None and run.run_id != expected_run_id:
        raise ValueError(
            f"active run pointer run_id mismatch: expected={expected_run_id}, actual={run.run_id}"
        )
    if expected_config_hash is not None and run.config_hash != expected_config_hash:
        raise ValueError(
            "active run pointer config_hash mismatch: "
            f"expected={expected_config_hash}, actual={run.config_hash}"
        )
    return directory


def resolve_active_run(
    run_root: str | Path,
    *,
    environment_variable: str = "RONBUN_RUN_DIR",
) -> Path:
    """Resolve the run shared by notebooks 01-05.

    An explicit environment-variable override has highest priority. Otherwise
    the pointer written by notebook 00/RunStore.create is validated. For runs
    created before the pointer feature existed, exactly one incomplete run may
    be discovered as a safe fallback; ambiguous candidates are never guessed.
    """

    explicit = os.environ.get(environment_variable, "").strip()
    if explicit:
        return _validate_active_run(explicit)

    root = Path(run_root).expanduser().resolve()
    pointer_path = root / ACTIVE_RUN_POINTER
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        required = {"run_dir", "run_id", "config_hash"}
        missing = sorted(required.difference(pointer))
        if missing:
            raise ValueError(f"active run pointer is missing fields: {missing}")
        return _validate_active_run(
            str(pointer["run_dir"]),
            expected_run_id=str(pointer["run_id"]),
            expected_config_hash=str(pointer["config_hash"]),
        )

    candidates: list[Path] = []
    for manifest_path in sorted(root.rglob("run_manifest.json")) if root.is_dir() else []:
        directory = manifest_path.parent
        try:
            candidates.append(_validate_active_run(directory))
        except RuntimeError as exc:
            if "already completed" not in str(exc):
                raise

    if not candidates:
        raise FileNotFoundError(
            f"no active run found under {root}; execute notebook 00 first"
        )
    if len(candidates) > 1:
        choices = ", ".join(str(path) for path in candidates[:5])
        raise RuntimeError(
            "multiple active runs found and none is selected by active_run.json; "
            f"rerun notebook 00 or set {environment_variable} explicitly: {choices}"
        )
    return candidates[0]
