from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import sys
import threading
from time import perf_counter
from typing import Iterator, TextIO


def _duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


class ProgressReporter:
    """Print notebook-friendly stage messages and periodic heartbeats."""

    def __init__(
        self,
        label: str,
        *,
        heartbeat_seconds: float = 30.0,
        stream: TextIO | None = None,
    ) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self.label = str(label)
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.stream = stream or sys.stdout
        self.started_at = perf_counter()
        self._lock = threading.Lock()

    def emit(self, message: str, **details: object) -> None:
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        elapsed = _duration(perf_counter() - self.started_at)
        detail_text = " ".join(
            f"{key}={value}" for key, value in details.items() if value is not None
        )
        suffix = f" | {detail_text}" if detail_text else ""
        with self._lock:
            print(
                f"[{timestamp}] {self.label} | {message} | elapsed={elapsed}{suffix}",
                file=self.stream,
                flush=True,
            )

    @contextmanager
    def step(self, name: str, *, expected: str | None = None) -> Iterator[None]:
        step_started = perf_counter()
        self.emit(f"START {name}", expected=expected)
        stop = threading.Event()

        def heartbeat() -> None:
            while not stop.wait(self.heartbeat_seconds):
                self.emit(
                    f"RUNNING {name}",
                    step_elapsed=_duration(perf_counter() - step_started),
                    expected=expected,
                )

        worker = threading.Thread(
            target=heartbeat,
            name=f"progress-{self.label}-{name}",
            daemon=True,
        )
        worker.start()
        try:
            yield
        except BaseException as exc:
            stop.set()
            worker.join(timeout=1.0)
            self.emit(
                f"FAILED {name}",
                step_elapsed=_duration(perf_counter() - step_started),
                error_type=type(exc).__name__,
            )
            raise
        else:
            stop.set()
            worker.join(timeout=1.0)
            self.emit(
                f"DONE {name}",
                step_elapsed=_duration(perf_counter() - step_started),
            )
