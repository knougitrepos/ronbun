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
