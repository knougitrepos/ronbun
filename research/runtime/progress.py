from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import math
import sys
import threading
from time import perf_counter
from typing import Callable, Iterator, TextIO


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
    """Print notebook-friendly stage messages and coarse progress milestones.

    Long dataset loops should call :meth:`milestone` (or ``callback()``) for
    every batch. The reporter itself suppresses those calls and prints only
    when the configured percentage boundary is crossed. Heartbeats remain
    available for operations that cannot expose a meaningful total.
    """

    def __init__(
        self,
        label: str,
        *,
        heartbeat_seconds: float | None = 30.0,
        milestone_percent: int = 10,
        stream: TextIO | None = None,
    ) -> None:
        if heartbeat_seconds is not None and heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        if (
            isinstance(milestone_percent, bool)
            or int(milestone_percent) != milestone_percent
            or not 1 <= int(milestone_percent) <= 100
            or 100 % int(milestone_percent) != 0
        ):
            raise ValueError(
                "milestone_percent must be a positive divisor of 100"
            )
        self.label = str(label)
        self.heartbeat_seconds = (
            float(heartbeat_seconds) if heartbeat_seconds is not None else None
        )
        self.milestone_percent = int(milestone_percent)
        self.stream = stream or sys.stdout
        self.started_at = perf_counter()
        self._lock = threading.Lock()
        self._milestones: dict[str, int] = {}

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

    def milestone(
        self,
        message: str,
        *,
        completed: int,
        total: int,
        key: str | None = None,
        **details: object,
    ) -> bool:
        """Emit at most once per configured percentage boundary.

        Returns ``True`` when a line was emitted. Batch callbacks may therefore
        call this method freely without flooding a notebook output cell.
        """

        if isinstance(completed, bool) or isinstance(total, bool):
            raise ValueError("completed and total must be integers")
        completed_value = int(completed)
        total_value = int(total)
        if completed_value != completed or total_value != total:
            raise ValueError("completed and total must be integers")
        if total_value <= 0:
            raise ValueError("total must be positive")
        if completed_value < 0 or completed_value > total_value:
            raise ValueError("completed must be in [0, total]")

        percent = 100.0 * completed_value / total_value
        boundary = (
            100
            if completed_value == total_value
            else int(math.floor(percent / self.milestone_percent))
            * self.milestone_percent
        )
        if boundary < self.milestone_percent:
            return False

        milestone_key = str(key or message)
        with self._lock:
            previous = self._milestones.get(milestone_key, 0)
            if boundary <= previous:
                return False
            self._milestones[milestone_key] = boundary

        elapsed_seconds = max(perf_counter() - self.started_at, 1e-9)
        rate = completed_value / elapsed_seconds
        remaining = total_value - completed_value
        eta = _duration(remaining / rate) if rate > 0.0 else None
        self.emit(
            message,
            progress=f"{boundary}%",
            processed=completed_value,
            total=total_value,
            rate=f"{rate:.2f}/s",
            eta=eta,
            **details,
        )
        return True

    def callback(
        self,
        *,
        key_prefix: str = "",
    ) -> Callable[[str, dict[str, object]], None]:
        """Adapt experiment callbacks to milestone-suppressed notebook logs."""

        def report(message: str, details: dict[str, object]) -> None:
            payload = dict(details)
            total = payload.pop("total", None)
            completed_name = next(
                (
                    name
                    for name in ("processed", "scanned", "written", "completed")
                    if name in payload
                ),
                None,
            )
            if total is not None and completed_name is not None:
                completed = payload.pop(completed_name)
                self.milestone(
                    message,
                    completed=int(completed),
                    total=int(total),
                    key=f"{key_prefix}{message}",
                    **payload,
                )
                return
            self.emit(message, **payload)

        return report

    @contextmanager
    def step(
        self,
        name: str,
        *,
        expected: str | None = None,
        heartbeat_details: Callable[[], dict[str, object]] | None = None,
    ) -> Iterator[None]:
        step_started = perf_counter()
        self.emit(f"START {name}", expected=expected)
        stop = threading.Event()

        def heartbeat() -> None:
            assert self.heartbeat_seconds is not None
            while not stop.wait(self.heartbeat_seconds):
                dynamic_details = heartbeat_details() if heartbeat_details else {}
                self.emit(
                    f"RUNNING {name}",
                    step_elapsed=_duration(perf_counter() - step_started),
                    expected=expected,
                    **dynamic_details,
                )

        worker = None
        if self.heartbeat_seconds is not None:
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
            if worker is not None:
                worker.join(timeout=1.0)
            self.emit(
                f"FAILED {name}",
                step_elapsed=_duration(perf_counter() - step_started),
                error_type=type(exc).__name__,
            )
            raise
        else:
            stop.set()
            if worker is not None:
                worker.join(timeout=1.0)
            self.emit(
                f"DONE {name}",
                step_elapsed=_duration(perf_counter() - step_started),
            )
