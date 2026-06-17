"""Helper functions for timing operations."""

import time


class Timer:
    """Simple wall-clock timer using :func:`time.perf_counter`."""

    def __init__(self) -> None:
        self.start_time: float | None = None
        self.end_time: float | None = None

    def enter(self) -> Timer:
        self.start_time = time.perf_counter()
        return self

    def exit(self) -> None:
        self.end_time = time.perf_counter()

    def elapsed_s(self) -> float:
        if self.start_time is None or self.end_time is None:
            raise RuntimeError("Timer not started or not stopped")
        return self.end_time - self.start_time
