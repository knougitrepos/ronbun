from io import StringIO
import threading

import pytest

from research.runtime import ProgressReporter


def test_progress_reporter_prints_start_and_done_messages():
    output = StringIO()
    reporter = ProgressReporter("phase-02", heartbeat_seconds=60, stream=output)

    with reporter.step("fit PCA", expected="10s-2m"):
        reporter.emit("matrix ready", rows=100)

    text = output.getvalue()
    assert "phase-02 | START fit PCA" in text
    assert "expected=10s-2m" in text
    assert "matrix ready" in text
    assert "rows=100" in text
    assert "phase-02 | DONE fit PCA" in text


def test_progress_reporter_reports_failed_step_and_reraises():
    output = StringIO()
    reporter = ProgressReporter("phase-03", heartbeat_seconds=60, stream=output)

    with pytest.raises(RuntimeError, match="boom"):
        with reporter.step("DB insert"):
            raise RuntimeError("boom")

    text = output.getvalue()
    assert "phase-03 | FAILED DB insert" in text
    assert "error_type=RuntimeError" in text


def test_progress_reporter_prints_heartbeat_for_long_step():
    output = StringIO()
    reporter = ProgressReporter("phase-04", heartbeat_seconds=0.01, stream=output)

    with reporter.step("search"):
        threading.Event().wait(0.04)

    assert "phase-04 | RUNNING search" in output.getvalue()


def test_progress_reporter_heartbeat_includes_dynamic_details():
    output = StringIO()
    reporter = ProgressReporter("phase-01", heartbeat_seconds=0.01, stream=output)
    state = {"processed": 0, "current_file": "a.jpg"}

    with reporter.step("extract", heartbeat_details=lambda: dict(state)):
        state["processed"] = 7
        state["current_file"] = "person/b.jpg"
        threading.Event().wait(0.04)

    text = output.getvalue()
    assert "processed=7" in text
    assert "current_file=person/b.jpg" in text


def test_progress_reporter_rejects_non_positive_heartbeat():
    with pytest.raises(ValueError, match="heartbeat_seconds must be positive"):
        ProgressReporter("invalid", heartbeat_seconds=0)


def test_progress_reporter_emits_only_ten_percent_milestones():
    output = StringIO()
    reporter = ProgressReporter(
        "SurvFace embedding",
        heartbeat_seconds=None,
        milestone_percent=10,
        stream=output,
    )

    for completed in range(1, 101):
        reporter.milestone(
            "embedding extraction",
            completed=completed,
            total=100,
        )

    lines = output.getvalue().splitlines()
    assert len(lines) == 10
    assert [f"progress={value}%" in line for value, line in zip(range(10, 101, 10), lines)] == [
        True
    ] * 10


def test_progress_reporter_large_jump_emits_one_latest_boundary():
    output = StringIO()
    reporter = ProgressReporter(
        "SurvFace search",
        heartbeat_seconds=None,
        stream=output,
    )

    assert reporter.milestone("search", completed=4, total=100) is False
    assert reporter.milestone("search", completed=27, total=100) is True
    assert reporter.milestone("search", completed=29, total=100) is False

    text = output.getvalue()
    assert text.count("\n") == 1
    assert "progress=20%" in text


def test_progress_callback_uses_processed_or_scanned_totals():
    output = StringIO()
    reporter = ProgressReporter(
        "SurvFace materialization",
        heartbeat_seconds=None,
        stream=output,
    )
    callback = reporter.callback()

    callback("scan", {"scanned": 10, "total": 100, "matched": 4})
    callback("scan", {"scanned": 15, "total": 100, "matched": 6})
    callback("scan", {"scanned": 20, "total": 100, "matched": 8})

    lines = output.getvalue().splitlines()
    assert len(lines) == 2
    assert "progress=10%" in lines[0]
    assert "progress=20%" in lines[1]
    assert "matched=8" in lines[1]


def test_progress_reporter_rejects_invalid_milestone_percentage():
    with pytest.raises(ValueError, match="positive divisor of 100"):
        ProgressReporter("invalid", milestone_percent=7)
