from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterable

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from research.database.cleanup import (
    CleanupExecutionReport,
    CleanupPlan,
    SCOPE_EXACT_RUN_UID,
    build_cleanup_plan,
    execute_cleanup_plan,
)


RESET_KIND_COMPLETE = "complete_run_reset"
RESET_PLAN_VERSION = 1
RUN_MANIFEST_NAME = "run_manifest.json"
RESULT_MANIFEST_NAME = "result_manifest.json"
ACTIVE_RUN_POINTER_NAME = "active_run.json"

PRESERVED_RESOURCES = (
    "PostgreSQL images table",
    "data/raw",
    "shared data/interim dataset manifests and aligned crops",
    "pretrained checkpoints and model registries",
    "other run_uid lineages",
    "runs/database_cleanup audits and quarantine payloads",
    "local paths without an exact owner manifest",
)


class RunResetError(RuntimeError):
    """Base exception for guarded DB and local-artifact reset."""


class RunResetSelectionError(RunResetError, ValueError):
    """The requested run or a discovered local target is unsafe."""


class RunResetConfirmationError(RunResetError):
    """The reset confirmation token is missing, blocked, or stale."""


class RunResetPlanChangedError(RunResetError):
    """DB or local state changed after the reset preview."""


class RunResetRollbackError(RunResetError):
    """A failed reset could not restore every quarantined local target."""


@dataclass(frozen=True)
class LocalResetTarget:
    kind: str
    relative_path: str
    file_count: int
    byte_count: int
    state_digest: str
    owner_manifest: str
    owner_manifest_sha256: str
    promoted: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "relative_path": self.relative_path,
            "file_count": self.file_count,
            "byte_count": self.byte_count,
            "state_digest": self.state_digest,
            "owner_manifest": self.owner_manifest,
            "owner_manifest_sha256": self.owner_manifest_sha256,
            "promoted": self.promoted,
        }


@dataclass(frozen=True)
class QuarantinedTarget:
    kind: str
    source_relative_path: str
    quarantine_relative_path: str

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "source_relative_path": self.source_relative_path,
            "quarantine_relative_path": self.quarantine_relative_path,
        }


@dataclass(frozen=True)
class RunResetPlan:
    reset_kind: str
    run_uid: str
    project_root: str
    database_plan: CleanupPlan
    local_targets: tuple[LocalResetTarget, ...]
    allow_completed_run: bool
    allow_promoted_results: bool
    allow_unverified_lineage: bool
    preserved_resources: tuple[str, ...]
    warnings: tuple[str, ...]
    blockers: tuple[str, ...]
    plan_digest: str
    confirmation_token: str | None

    @property
    def total_database_rows(self) -> int:
        return self.database_plan.total_rows

    @property
    def total_files(self) -> int:
        return sum(item.file_count for item in self.local_targets)

    @property
    def total_bytes(self) -> int:
        return sum(item.byte_count for item in self.local_targets)

    @property
    def executable(self) -> bool:
        return self.confirmation_token is not None and not self.blockers

    def as_dict(self) -> dict[str, object]:
        return {
            "reset_plan_version": RESET_PLAN_VERSION,
            "reset_kind": self.reset_kind,
            "run_uid": self.run_uid,
            "project_root": self.project_root,
            "database_plan": self.database_plan.as_dict(),
            "local_targets": [item.as_dict() for item in self.local_targets],
            "total_database_rows": self.total_database_rows,
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
            "allow_completed_run": self.allow_completed_run,
            "allow_promoted_results": self.allow_promoted_results,
            "allow_unverified_lineage": self.allow_unverified_lineage,
            "preserved_resources": list(self.preserved_resources),
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
            "plan_digest": self.plan_digest,
            "confirmation_token": self.confirmation_token,
            "executable": self.executable,
        }


@dataclass(frozen=True)
class RunResetExecutionReport:
    completed_at_utc: str
    run_uid: str
    plan_digest: str
    database_report: CleanupExecutionReport
    quarantined_targets: tuple[QuarantinedTarget, ...]
    quarantine_dir: str
    audit_path: str | None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "completed_at_utc": self.completed_at_utc,
            "run_uid": self.run_uid,
            "plan_digest": self.plan_digest,
            "database_report": self.database_report.as_dict(),
            "quarantined_targets": [
                item.as_dict() for item in self.quarantined_targets
            ],
            "quarantine_dir": self.quarantine_dir,
            "audit_path": self.audit_path,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _TargetCandidate:
    kind: str
    path: Path
    owner_manifest: Path
    promoted: bool = False


def build_run_reset_plan(
    session: Session,
    *,
    run_uid: str,
    project_root: str | Path,
    allow_completed_run: bool = False,
    allow_promoted_results: bool = False,
    allow_unverified_lineage: bool = False,
) -> RunResetPlan:
    normalized_run_uid = str(run_uid).strip()
    if not normalized_run_uid:
        raise RunResetSelectionError(
            "run_uid is required for complete_run_reset"
        )
    if len(normalized_run_uid) > 96:
        raise RunResetSelectionError(
            "run_uid exceeds the database limit of 96 characters"
        )

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise RunResetSelectionError(
            f"project_root is not a directory: {root}"
        )

    database_plan = build_cleanup_plan(
        session,
        scope_kind=SCOPE_EXACT_RUN_UID,
        run_uid=normalized_run_uid,
        table_groups=("all_run_scoped_data",),
        include_research_run_record=True,
        allow_completed_run=allow_completed_run,
        allow_empty_scope=True,
        project_root=root,
    )

    candidates, discovery_warnings, discovery_blockers = (
        _discover_local_candidates(
            root,
            normalized_run_uid,
            allow_promoted_results=allow_promoted_results,
        )
    )
    local_targets: list[LocalResetTarget] = []
    blockers = list(database_plan.blockers)
    blockers.extend(discovery_blockers)
    warnings = list(database_plan.warnings)
    warnings.extend(discovery_warnings)
    for candidate in candidates:
        try:
            local_targets.append(_snapshot_candidate(candidate, root))
        except (OSError, RunResetSelectionError) as error:
            blockers.append(str(error))

    has_run_manifest = any(
        item.kind == "run_workspace" for item in local_targets
    )
    has_result_manifest = any(
        item.kind == "result_bundle" for item in local_targets
    )
    has_database_lineage = database_plan.research_run_status is not None
    has_owned_lineage = (
        has_run_manifest or has_result_manifest or has_database_lineage
    )
    has_any_state = bool(local_targets) or database_plan.total_rows > 0
    if has_any_state and not has_owned_lineage:
        message = (
            "the selected state has no research_runs record, run manifest, or "
            "result manifest proving its lineage"
        )
        if allow_unverified_lineage:
            warnings.append(
                "unverified lineage protection was explicitly overridden: "
                + message
            )
        else:
            blockers.append(
                message + "; set allow_unverified_lineage=True only after review"
            )
    if not has_any_state:
        blockers.append(
            "the exact run_uid matches neither DB rows nor owned local artifacts"
        )

    ordered_targets = tuple(
        sorted(local_targets, key=lambda item: (item.relative_path, item.kind))
    )
    digest_payload = {
        "reset_plan_version": RESET_PLAN_VERSION,
        "reset_kind": RESET_KIND_COMPLETE,
        "run_uid": normalized_run_uid,
        "project_root": str(root),
        "database_plan_digest": database_plan.plan_digest,
        "local_targets": [item.as_dict() for item in ordered_targets],
        "allow_completed_run": bool(allow_completed_run),
        "allow_promoted_results": bool(allow_promoted_results),
        "allow_unverified_lineage": bool(allow_unverified_lineage),
        "preserved_resources": list(PRESERVED_RESOURCES),
    }
    plan_digest = _json_digest(digest_payload)
    total_files = sum(item.file_count for item in ordered_targets)
    total_bytes = sum(item.byte_count for item in ordered_targets)
    confirmation_token = None
    if not blockers:
        confirmation_token = (
            f"RESET {normalized_run_uid} "
            f"DB_ROWS={database_plan.total_rows} "
            f"FILES={total_files} BYTES={total_bytes} "
            f"{plan_digest[:12]}"
        )

    return RunResetPlan(
        reset_kind=RESET_KIND_COMPLETE,
        run_uid=normalized_run_uid,
        project_root=str(root),
        database_plan=database_plan,
        local_targets=ordered_targets,
        allow_completed_run=bool(allow_completed_run),
        allow_promoted_results=bool(allow_promoted_results),
        allow_unverified_lineage=bool(allow_unverified_lineage),
        preserved_resources=PRESERVED_RESOURCES,
        warnings=tuple(_deduplicate(warnings)),
        blockers=tuple(_deduplicate(blockers)),
        plan_digest=plan_digest,
        confirmation_token=confirmation_token,
    )


def execute_run_reset_plan(
    engine: Engine,
    plan: RunResetPlan,
    *,
    confirmation_token: str,
    project_root: str | Path,
) -> RunResetExecutionReport:
    if plan.reset_kind != RESET_KIND_COMPLETE:
        raise RunResetConfirmationError(
            f"unsupported reset kind: {plan.reset_kind}"
        )
    if not plan.executable:
        raise RunResetConfirmationError(
            "run reset plan is blocked and cannot be executed"
        )
    if confirmation_token != plan.confirmation_token:
        raise RunResetConfirmationError(
            "confirmation token does not exactly match the latest reset preview"
        )

    root = Path(project_root).resolve()
    if str(root) != plan.project_root:
        raise RunResetConfirmationError(
            "project_root does not match the reset preview"
        )

    with Session(engine) as session:
        refreshed = build_run_reset_plan(
            session,
            run_uid=plan.run_uid,
            project_root=root,
            allow_completed_run=plan.allow_completed_run,
            allow_promoted_results=plan.allow_promoted_results,
            allow_unverified_lineage=plan.allow_unverified_lineage,
        )
    if refreshed.plan_digest != plan.plan_digest:
        raise RunResetPlanChangedError(
            "DB rows or local artifact state changed after preview; build a new "
            "complete_run_reset plan"
        )
    if confirmation_token != refreshed.confirmation_token:
        raise RunResetPlanChangedError(
            "confirmation token is stale after the reset re-preview"
        )

    operation_timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    operation_name = (
        f"{operation_timestamp}_{_safe_component(plan.run_uid)}_"
        f"{plan.plan_digest[:12]}"
    )
    operation_dir = (
        root / "runs" / "database_cleanup" / "quarantine" / operation_name
    )
    if operation_dir.exists():
        raise RunResetSelectionError(
            f"reset operation directory already exists: {operation_dir}"
        )
    payload_dir = operation_dir / "payload"
    payload_dir.mkdir(parents=True, exist_ok=False)
    _write_json(operation_dir / "plan.json", refreshed.as_dict())

    moved: list[QuarantinedTarget] = []

    def quarantine_before_commit() -> None:
        current_targets: list[LocalResetTarget] = []
        for target in refreshed.local_targets:
            candidate = _candidate_from_target(target, root)
            current_targets.append(_snapshot_candidate(candidate, root))
        if tuple(current_targets) != refreshed.local_targets:
            raise RunResetPlanChangedError(
                "local artifact state changed immediately before quarantine"
            )

        for target in refreshed.local_targets:
            source = _resolve_relative_target(
                root,
                target.relative_path,
                kind=target.kind,
            )
            destination = payload_dir / Path(target.relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise RunResetSelectionError(
                    f"quarantine destination already exists: {destination}"
                )
            _rename_path(source, destination)
            moved.append(
                QuarantinedTarget(
                    kind=target.kind,
                    source_relative_path=target.relative_path,
                    quarantine_relative_path=_relative_path(
                        destination,
                        root,
                    ),
                )
            )
        _write_json(
            operation_dir / "quarantine_state.json",
            {
                "status": "quarantined_before_db_commit",
                "run_uid": refreshed.run_uid,
                "plan_digest": refreshed.plan_digest,
                "moved_targets": [item.as_dict() for item in moved],
            },
        )

    try:
        database_report = execute_cleanup_plan(
            engine,
            refreshed.database_plan,
            confirmation_token=(
                refreshed.database_plan.confirmation_token or ""
            ),
            project_root=root,
            before_commit=quarantine_before_commit,
        )
    except BaseException as error:
        restore_errors = _restore_quarantined_targets(
            moved,
            root=root,
        )
        rollback_payload = {
            "status": (
                "rollback_restore_failed"
                if restore_errors
                else "database_rollback_and_local_restore_completed"
            ),
            "run_uid": refreshed.run_uid,
            "plan_digest": refreshed.plan_digest,
            "error_type": type(error).__name__,
            "error": str(error),
            "restore_errors": restore_errors,
            "moved_targets": [item.as_dict() for item in moved],
        }
        try:
            _write_json(operation_dir / "rollback.json", rollback_payload)
        except OSError:
            pass
        if restore_errors:
            raise RunResetRollbackError(
                "DB transaction failed and one or more local targets could not "
                f"be restored: {restore_errors}"
            ) from error
        raise

    completed_at = datetime.now(timezone.utc).isoformat()
    quarantine_relative = _relative_path(operation_dir, root)
    report = RunResetExecutionReport(
        completed_at_utc=completed_at,
        run_uid=refreshed.run_uid,
        plan_digest=refreshed.plan_digest,
        database_report=database_report,
        quarantined_targets=tuple(moved),
        quarantine_dir=quarantine_relative,
        audit_path=None,
    )
    completion_path = operation_dir / "completion.json"
    warnings: tuple[str, ...] = ()
    try:
        payload = report.as_dict()
        payload["audit_path"] = _relative_path(completion_path, root)
        _write_json(completion_path, payload)
        report = RunResetExecutionReport(
            completed_at_utc=completed_at,
            run_uid=refreshed.run_uid,
            plan_digest=refreshed.plan_digest,
            database_report=database_report,
            quarantined_targets=tuple(moved),
            quarantine_dir=quarantine_relative,
            audit_path=_relative_path(completion_path, root),
        )
    except OSError as error:
        warnings = (
            "reset committed but completion.json could not be written: "
            + str(error),
        )
        report = RunResetExecutionReport(
            completed_at_utc=completed_at,
            run_uid=refreshed.run_uid,
            plan_digest=refreshed.plan_digest,
            database_report=database_report,
            quarantined_targets=tuple(moved),
            quarantine_dir=quarantine_relative,
            audit_path=None,
            warnings=warnings,
        )
    return report


def _discover_local_candidates(
    root: Path,
    run_uid: str,
    *,
    allow_promoted_results: bool,
) -> tuple[list[_TargetCandidate], list[str], list[str]]:
    candidates: list[_TargetCandidate] = []
    warnings: list[str] = []
    blockers: list[str] = []
    cleanup_root = (root / "runs" / "database_cleanup").resolve()

    run_candidates: list[_TargetCandidate] = []
    runs_root = root / "runs"
    if runs_root.is_dir():
        for manifest_path in sorted(runs_root.rglob(RUN_MANIFEST_NAME)):
            if _is_within(manifest_path, cleanup_root):
                continue
            payload, error = _read_json(manifest_path)
            if error is not None:
                if _path_name_matches_run(manifest_path.parent, run_uid):
                    blockers.append(
                        f"exact-named run manifest is unreadable: "
                        f"{_relative_path(manifest_path, root)}"
                    )
                continue
            if str(payload.get("run_id", "")).strip() != run_uid:
                continue
            run_candidates.append(
                _TargetCandidate(
                    kind="run_workspace",
                    path=manifest_path.parent,
                    owner_manifest=manifest_path,
                )
            )
    if len(run_candidates) > 1:
        blockers.append(
            "multiple run_manifest.json files claim the exact run_uid: "
            + ", ".join(
                _relative_path(item.owner_manifest, root)
                for item in run_candidates
            )
        )
    candidates.extend(run_candidates)

    results_root = root / "results"
    if results_root.is_dir():
        promoted_root = (results_root / "paper").resolve()
        for manifest_path in sorted(results_root.rglob(RESULT_MANIFEST_NAME)):
            payload, error = _read_json(manifest_path)
            if error is not None:
                if _path_name_matches_run(manifest_path.parent, run_uid):
                    blockers.append(
                        f"exact-named result manifest is unreadable: "
                        f"{_relative_path(manifest_path, root)}"
                    )
                continue
            owner = payload.get("run_uid", payload.get("run_id", ""))
            if str(owner).strip() != run_uid:
                continue
            promoted = _is_within(manifest_path, promoted_root)
            candidates.append(
                _TargetCandidate(
                    kind="result_bundle",
                    path=manifest_path.parent,
                    owner_manifest=manifest_path,
                    promoted=promoted,
                )
            )
            if promoted and not allow_promoted_results:
                blockers.append(
                    "promoted result reset is blocked: "
                    f"{_relative_path(manifest_path.parent, root)}; set "
                    "allow_promoted_results=True only after publication review"
                )
            elif promoted:
                warnings.append(
                    "promoted result protection was explicitly overridden: "
                    + _relative_path(manifest_path.parent, root)
                )

    if runs_root.is_dir():
        for pointer_path in sorted(runs_root.rglob(ACTIVE_RUN_POINTER_NAME)):
            if _is_within(pointer_path, cleanup_root):
                continue
            payload, error = _read_json(pointer_path)
            if error is not None:
                warnings.append(
                    "unreadable active run pointer was preserved: "
                    + _relative_path(pointer_path, root)
                )
                continue
            if str(payload.get("run_id", "")).strip() != run_uid:
                continue
            candidates.append(
                _TargetCandidate(
                    kind="active_run_pointer",
                    path=pointer_path,
                    owner_manifest=pointer_path,
                )
            )

    try:
        candidates = _validate_and_prune_candidates(candidates, root)
    except RunResetSelectionError as error:
        blockers.append(str(error))
        candidates = []
    return candidates, warnings, blockers


def _validate_and_prune_candidates(
    candidates: Iterable[_TargetCandidate],
    root: Path,
) -> list[_TargetCandidate]:
    unique: dict[str, _TargetCandidate] = {}
    for candidate in candidates:
        resolved = candidate.path.resolve()
        if candidate.kind in {"run_workspace", "active_run_pointer"}:
            allowed_root = (root / "runs").resolve()
        elif candidate.kind == "result_bundle":
            allowed_root = (root / "results").resolve()
        else:
            raise RunResetSelectionError(
                f"unsupported local reset target kind: {candidate.kind}"
            )
        if resolved == allowed_root or not _is_within(resolved, allowed_root):
            raise RunResetSelectionError(
                f"local reset target escapes or equals an allowlisted root: "
                f"{candidate.path}"
            )
        cleanup_root = (root / "runs" / "database_cleanup").resolve()
        if _is_within(resolved, cleanup_root):
            raise RunResetSelectionError(
                f"cleanup audit/quarantine cannot be reset: {candidate.path}"
            )
        relative = _relative_path(resolved, root)
        existing = unique.get(relative)
        if existing is not None and existing != candidate:
            raise RunResetSelectionError(
                f"conflicting local reset targets resolve to {relative}"
            )
        unique[relative] = _TargetCandidate(
            kind=candidate.kind,
            path=resolved,
            owner_manifest=candidate.owner_manifest.resolve(),
            promoted=candidate.promoted,
        )

    ordered = sorted(
        unique.values(),
        key=lambda item: (len(item.path.parts), str(item.path).lower()),
    )
    pruned: list[_TargetCandidate] = []
    for candidate in ordered:
        if any(
            parent.path.is_dir()
            and _is_within(candidate.path, parent.path)
            for parent in pruned
        ):
            continue
        pruned.append(candidate)
    return pruned


def _snapshot_candidate(
    candidate: _TargetCandidate,
    root: Path,
) -> LocalResetTarget:
    path = candidate.path
    if not path.exists():
        raise RunResetSelectionError(
            f"local reset target is missing: {_relative_path(path, root)}"
        )
    if path.is_symlink():
        raise RunResetSelectionError(
            f"symbolic links are not accepted as reset targets: "
            f"{_relative_path(path, root)}"
        )
    manifest = candidate.owner_manifest
    if not manifest.is_file() or manifest.is_symlink():
        raise RunResetSelectionError(
            f"owner manifest is missing or unsafe: "
            f"{_relative_path(manifest, root)}"
        )

    entries: list[str] = []
    file_count = 0
    byte_count = 0
    if path.is_file():
        stat = path.stat()
        file_count = 1
        byte_count = int(stat.st_size)
        entries.append(
            f"F\t.\t{stat.st_size}\t{stat.st_mtime_ns}"
        )
    elif path.is_dir():
        for directory, directory_names, file_names in os.walk(
            path,
            topdown=True,
            followlinks=False,
        ):
            directory_path = Path(directory)
            directory_names.sort()
            file_names.sort()
            for name in directory_names:
                child = directory_path / name
                if child.is_symlink():
                    raise RunResetSelectionError(
                        "symbolic links inside a reset target are blocked: "
                        + _relative_path(child, root)
                    )
                relative = child.relative_to(path).as_posix()
                stat = child.stat()
                entries.append(
                    f"D\t{relative}\t0\t{stat.st_mtime_ns}"
                )
            for name in file_names:
                child = directory_path / name
                if child.is_symlink():
                    raise RunResetSelectionError(
                        "symbolic links inside a reset target are blocked: "
                        + _relative_path(child, root)
                    )
                relative = child.relative_to(path).as_posix()
                stat = child.stat()
                file_count += 1
                byte_count += int(stat.st_size)
                entries.append(
                    f"F\t{relative}\t{stat.st_size}\t{stat.st_mtime_ns}"
                )
    else:
        raise RunResetSelectionError(
            f"unsupported local reset target type: {path}"
        )

    return LocalResetTarget(
        kind=candidate.kind,
        relative_path=_relative_path(path, root),
        file_count=file_count,
        byte_count=byte_count,
        state_digest=hashlib.sha256(
            "\n".join(entries).encode("utf-8")
        ).hexdigest(),
        owner_manifest=_relative_path(manifest, root),
        owner_manifest_sha256=_sha256_file(manifest),
        promoted=candidate.promoted,
    )


def _candidate_from_target(
    target: LocalResetTarget,
    root: Path,
) -> _TargetCandidate:
    return _TargetCandidate(
        kind=target.kind,
        path=_resolve_relative_target(
            root,
            target.relative_path,
            kind=target.kind,
        ),
        owner_manifest=_resolve_relative_target(
            root,
            target.owner_manifest,
            kind=target.kind,
            owner_manifest=True,
        ),
        promoted=target.promoted,
    )


def _resolve_relative_target(
    root: Path,
    relative_path: str,
    *,
    kind: str,
    owner_manifest: bool = False,
) -> Path:
    candidate = (root / Path(relative_path)).resolve()
    if kind in {"run_workspace", "active_run_pointer"}:
        allowed_root = (root / "runs").resolve()
    elif kind == "result_bundle":
        allowed_root = (root / "results").resolve()
    else:
        raise RunResetSelectionError(
            f"unsupported local reset target kind: {kind}"
        )
    if not _is_within(candidate, allowed_root):
        raise RunResetSelectionError(
            f"stored reset path escapes its allowlisted root: {relative_path}"
        )
    if not owner_manifest and candidate == allowed_root:
        raise RunResetSelectionError(
            f"stored reset path equals a broad allowlisted root: {relative_path}"
        )
    cleanup_root = (root / "runs" / "database_cleanup").resolve()
    if _is_within(candidate, cleanup_root):
        raise RunResetSelectionError(
            f"stored reset path enters cleanup state: {relative_path}"
        )
    return candidate


def _restore_quarantined_targets(
    moved: Iterable[QuarantinedTarget],
    *,
    root: Path,
) -> list[str]:
    errors: list[str] = []
    for item in reversed(tuple(moved)):
        try:
            source = (root / Path(item.quarantine_relative_path)).resolve()
            destination = (root / Path(item.source_relative_path)).resolve()
            if destination.exists():
                raise FileExistsError(
                    f"restore destination already exists: {destination}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            _rename_path(source, destination)
        except Exception as error:
            errors.append(str(error))
    return errors


def _rename_path(source: Path, destination: Path) -> None:
    source.rename(destination)


def _read_json(path: Path) -> tuple[dict[str, object], Exception | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {}, error
    if not isinstance(payload, dict):
        return {}, ValueError("JSON root is not an object")
    return payload, None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise RunResetSelectionError(
            f"path escapes project root: {resolved}"
        ) from error


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _path_name_matches_run(path: Path, run_uid: str) -> bool:
    return path.name == run_uid or path.name.startswith(run_uid + "_")


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not normalized:
        raise RunResetSelectionError(
            "run_uid cannot be converted to a safe operation name"
        )
    return normalized[:96]


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
