"""Wall time, CPU time, memory, CPU %, and load average logging for ROSE workflow steps."""

from __future__ import annotations

import json
import logging
import os
import resource
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import psutil

_ENV_JSONL = "ROSE_RESOURCES_JSONL"


def configure_resources_log(path: str | os.PathLike[str]) -> Path:
    """Set ``ROSE_RESOURCES_JSONL`` so step scripts append to the same file."""
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    os.environ[_ENV_JSONL] = str(resolved)
    return resolved


def resources_log_path() -> Path | None:
    raw = os.environ.get(_ENV_JSONL)
    return Path(raw) if raw else None


@dataclass
class ResourceRecord:
    step: str
    wall_s: float
    cpu_user_s: float
    cpu_sys_s: float
    rss_mb: float
    peak_rss_mb: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    iteration: int | None = None
    cpu_percent_process: float | None = None
    cpu_percent_system: float | None = None
    load_avg_1m: float | None = None
    load_avg_5m: float | None = None
    load_avg_15m: float | None = None
    load_per_cpu_1m: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in (
            "iteration",
            "cpu_percent_process",
            "cpu_percent_system",
            "load_avg_1m",
            "load_avg_5m",
            "load_avg_15m",
            "load_per_cpu_1m",
        ):
            if out.get(key) is None:
                out.pop(key, None)
        if not out.get("extra"):
            out.pop("extra", None)
        return out


def _cpu_times() -> tuple[float, float]:
    proc = psutil.Process(os.getpid())
    ct = proc.cpu_times()
    return ct.user, ct.system


def _memory_mb() -> tuple[float, float]:
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    # Linux/WSL: ru_maxrss is kilobytes
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return rss_mb, peak_mb


def _prime_cpu_counters(proc: psutil.Process) -> None:
    """First call returns 0; primes process and system CPU percent counters."""
    proc.cpu_percent(None)
    psutil.cpu_percent(None)


def _cpu_load_snapshot(proc: psutil.Process) -> dict[str, float]:
    """Process/system CPU % since last prime, plus Unix load averages."""
    snap: dict[str, float] = {
        "cpu_percent_process": float(proc.cpu_percent()),
        "cpu_percent_system": float(psutil.cpu_percent(None)),
    }
    try:
        l1, l5, l15 = os.getloadavg()
        snap["load_avg_1m"] = float(l1)
        snap["load_avg_5m"] = float(l5)
        snap["load_avg_15m"] = float(l15)
        n_cpu = psutil.cpu_count(logical=True) or 1
        snap["load_per_cpu_1m"] = float(l1 / n_cpu)
    except (AttributeError, OSError):
        pass
    return snap


def append_resource_record(path: Path, record: ResourceRecord) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


def log_resource_record(logger: logging.Logger, record: ResourceRecord) -> None:
    cpu_part = ""
    if record.cpu_percent_process is not None:
        cpu_part = f" proc_cpu={record.cpu_percent_process:.1f}%"
    if record.cpu_percent_system is not None:
        cpu_part += f" sys_cpu={record.cpu_percent_system:.1f}%"
    load_part = ""
    if record.load_avg_1m is not None:
        load_part = (
            f" load(1/5/15)={record.load_avg_1m:.2f}"
            f"/{record.load_avg_5m:.2f}/{record.load_avg_15m:.2f}"
        )
        if record.load_per_cpu_1m is not None:
            load_part += f" load/cpu={record.load_per_cpu_1m:.2f}"

    logger.info(
        "resources %s: wall=%.2fs cpu_user=%.2fs cpu_sys=%.2fs "
        "rss=%.1fMB peak_rss=%.1fMB%s%s%s",
        record.step,
        record.wall_s,
        record.cpu_user_s,
        record.cpu_sys_s,
        record.rss_mb,
        record.peak_rss_mb,
        cpu_part,
        load_part,
        f" iter={record.iteration}" if record.iteration is not None else "",
    )
    if record.extra:
        logger.info("resources %s extra: %s", record.step, record.extra)


def _record_from_snapshots(
    step: str,
    *,
    wall_s: float,
    cpu_user_s: float,
    cpu_sys_s: float,
    rss_mb: float,
    peak_mb: float,
    iteration: int | None,
    load_snap: dict[str, float],
    extra: dict[str, Any],
) -> ResourceRecord:
    return ResourceRecord(
        step=step,
        wall_s=wall_s,
        cpu_user_s=cpu_user_s,
        cpu_sys_s=cpu_sys_s,
        rss_mb=rss_mb,
        peak_rss_mb=peak_mb,
        iteration=iteration,
        cpu_percent_process=load_snap.get("cpu_percent_process"),
        cpu_percent_system=load_snap.get("cpu_percent_system"),
        load_avg_1m=load_snap.get("load_avg_1m"),
        load_avg_5m=load_snap.get("load_avg_5m"),
        load_avg_15m=load_snap.get("load_avg_15m"),
        load_per_cpu_1m=load_snap.get("load_per_cpu_1m"),
        extra=extra,
    )


@contextmanager
def record_resources(
    logger: logging.Logger,
    step: str,
    *,
    iteration: int | None = None,
    jsonl_path: Path | os.PathLike[str] | None = None,
    **extra: Any,
) -> Iterator[None]:
    """Log wall/CPU/memory/CPU%/load for a block (sim, train, workflow, etc.)."""
    path = Path(jsonl_path) if jsonl_path else resources_log_path()
    proc = psutil.Process(os.getpid())
    _prime_cpu_counters(proc)
    cpu_u0, cpu_s0 = _cpu_times()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        wall_s = time.perf_counter() - t0
        cpu_u1, cpu_s1 = _cpu_times()
        rss_mb, peak_mb = _memory_mb()
        load_snap = _cpu_load_snapshot(proc)
        record = _record_from_snapshots(
            step,
            wall_s=wall_s,
            cpu_user_s=cpu_u1 - cpu_u0,
            cpu_sys_s=cpu_s1 - cpu_s0,
            rss_mb=rss_mb,
            peak_mb=peak_mb,
            iteration=iteration,
            load_snap=load_snap,
            extra=extra,
        )
        log_resource_record(logger, record)
        if path is not None:
            append_resource_record(path, record)


def log_checkpoint(
    logger: logging.Logger,
    step: str,
    *,
    iteration: int | None = None,
    jsonl_path: Path | os.PathLike[str] | None = None,
    **extra: Any,
) -> None:
    """Instant snapshot (e.g. end of each AL iteration in the driver)."""
    proc = psutil.Process(os.getpid())
    _prime_cpu_counters(proc)
    cpu_u, cpu_s = _cpu_times()
    rss_mb, peak_mb = _memory_mb()
    load_snap = _cpu_load_snapshot(proc)
    record = _record_from_snapshots(
        step,
        wall_s=0.0,
        cpu_user_s=cpu_u,
        cpu_sys_s=cpu_s,
        rss_mb=rss_mb,
        peak_mb=peak_mb,
        iteration=iteration,
        load_snap=load_snap,
        extra=extra,
    )
    log_resource_record(logger, record)
    path = Path(jsonl_path) if jsonl_path else resources_log_path()
    if path is not None:
        append_resource_record(path, record)
