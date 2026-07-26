from datetime import datetime
import hashlib
import json
from zoneinfo import ZoneInfo

import pytest

from research.runtime.hashing import canonical_sha256
from research.runtime.redaction import redact
from research.runtime.run_store import RunStore


KST = ZoneInfo("Asia/Seoul")


def _manifest(run):
    return json.loads(run.manifest_path.read_text(encoding="utf-8"))


def test_run_store_allocates_dated_daily_sequences_and_redacts_manifest(tmp_path):
    now = datetime(2026, 7, 14, 9, 30, tzinfo=KST)
    config = {
        "database": {
            "user": "researcher",
            "password": "never-write-this",
            "dsn": "postgresql://researcher:also-secret@localhost/postgres",
        },
        "experiment": {"seed": 7},
    }

    first = RunStore.create(
        experiment_name="face search",
        config=config,
        root=tmp_path / "runs",
        now=now,
        repo_root=tmp_path,
    )
    second = RunStore.create(
        experiment_name="face search",
        config=config,
        root=tmp_path / "runs",
        now=now,
        repo_root=tmp_path,
    )

    assert first.run_dir.parent == tmp_path / "runs" / "2026" / "07" / "14"
    assert first.run_id.startswith("20260714-R001-")
    assert second.run_id.startswith("20260714-R002-")
    assert first.config_hash == canonical_sha256(redact(config))

    manifest_text = first.manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["config_hash"] == first.config_hash
    assert manifest["config"]["database"]["password"] == "***"
    assert manifest["config"]["database"]["dsn"] == "***"
    assert "never-write-this" not in manifest_text
    assert "also-secret" not in manifest_text

    first.record_event(
        "redaction_check",
        connection="postgresql://researcher:event-secret@localhost/postgres",
        api_key="key-secret",
    )
    events = (first.run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert "event-secret" not in events
    assert "key-secret" not in events


def test_run_store_records_input_sha256_and_phase_retry_attempts(tmp_path):
    run = RunStore.create(
        experiment_name="retry-test",
        config={"seed": 11},
        root=tmp_path / "runs",
        now=datetime(2026, 7, 14, 10, 0, tzinfo=KST),
        repo_root=tmp_path,
    )
    source = tmp_path / "manifest.csv"
    source.write_bytes(b"image_id,identity_id\na,a\n")

    entry = run.record_input(source, role="dataset_manifest")

    assert entry["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert _manifest(run)["inputs"] == [entry]
    assert run.verify_inputs() == [entry]

    with pytest.raises(RuntimeError, match="intentional retry"):
        with run.phase("embedding extraction") as attempt:
            assert attempt.attempt == 1
            raise RuntimeError("intentional retry")

    with run.phase("embedding extraction") as attempt:
        assert attempt.attempt == 2
        attempt.record_counts(images=1)

    attempts = run.run_dir / "phases" / "embedding-extraction" / "attempts"
    first = json.loads((attempts / "A001" / "phase_manifest.json").read_text(encoding="utf-8"))
    second = json.loads((attempts / "A002" / "phase_manifest.json").read_text(encoding="utf-8"))
    assert first["status"] == "failed"
    assert second["status"] == "completed"
    assert second["details"]["counts"] == {"images": 1}

    source.write_bytes(b"image_id,identity_id\nb,b\n")
    with pytest.raises(ValueError, match="frozen input hash mismatch"):
        run.verify_inputs()
    with pytest.raises(ValueError, match="registered input changed"):
        run.record_input(source, role="dataset_manifest")


def test_phase_artifacts_are_never_overwritten(tmp_path):
    run = RunStore.create(
        experiment_name="artifact-test",
        config={"seed": 13},
        root=tmp_path / "runs",
        now=datetime(2026, 7, 14, 10, 30, tzinfo=KST),
        repo_root=tmp_path,
    )
    first_source = tmp_path / "first.json"
    first_source.write_text('{"attempt": 1}', encoding="utf-8")
    with run.phase("materialize") as phase:
        published = phase.publish_artifact(first_source, name="summary.json")

    verified = run.verify_phase_artifacts("materialize")
    assert verified[0]["path"] == str(published.relative_to(run.run_dir))

    second_source = tmp_path / "second.json"
    second_source.write_text('{"attempt": 2}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        with run.phase("materialize") as phase:
            phase.publish_artifact(second_source, name="summary.json")

    published.write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="phase artifact hash mismatch"):
        run.verify_phase_artifacts("materialize", attempt=1)


def test_completed_run_is_immutable_and_open_validates_config_hash(tmp_path):
    run = RunStore.create(
        experiment_name="immutable-test",
        config={"seed": 17},
        root=tmp_path / "runs",
        now=datetime(2026, 7, 14, 11, 0, tzinfo=KST),
        repo_root=tmp_path,
    )
    run.complete()

    opened = RunStore.open(run.run_dir)
    assert opened.run_id == run.run_id
    assert opened.config_hash == run.config_hash
    with pytest.raises(RuntimeError, match="completed runs are immutable"):
        opened.phase("search")
    immutable_input = tmp_path / "late.csv"
    immutable_input.write_text("late", encoding="utf-8")
    with pytest.raises(RuntimeError, match="completed runs are immutable"):
        opened.record_input(immutable_input, role="late_input")

    manifest = _manifest(run)
    manifest["config"]["seed"] = 999
    run.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="config hash mismatch"):
        RunStore.open(run.run_dir)


def test_create_or_reuse_active_keeps_one_incomplete_result(tmp_path):
    root = tmp_path / "runs"
    first = RunStore.create_or_reuse_active(
        experiment_name="step2-arcface",
        config={"model_uid": "arcface-1", "seed": 42},
        root=root,
        repo_root=tmp_path,
    )
    reopened = RunStore.create_or_reuse_active(
        experiment_name="step2-arcface",
        config={"model_uid": "arcface-1", "seed": 42},
        root=root,
        repo_root=tmp_path,
    )
    assert reopened.run_dir == first.run_dir

    with pytest.raises(RuntimeError, match="different incomplete run"):
        RunStore.create_or_reuse_active(
            experiment_name="step2-adaface",
            config={"model_uid": "adaface-1", "seed": 42},
            root=root,
            repo_root=tmp_path,
        )

    first.complete()
    second = RunStore.create_or_reuse_active(
        experiment_name="step2-adaface",
        config={"model_uid": "adaface-1", "seed": 42},
        root=root,
        repo_root=tmp_path,
    )
    assert second.run_dir != first.run_dir
