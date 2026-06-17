"""Shared logger setup for the ROSE pipeline.

Two log scopes coexist in a campaign:

- **global** — ``{run_dir}/rose.log``: campaign-level events (driver start,
  per-iteration progress, timing). Written by the ``rose.run`` logger.
- **sub-workflow** — ``{wf_dir}/rose.log``: everything specific to one
  ``wf_*`` learner and the task subprocesses it spawns (sim, train, AL,
  check_mse). Written by a per-sub ``rose.wf.<wf_ID>`` logger and, in the task
  subprocesses, by ``get_logger(__name__)``.

The driver process holds *several* of these files open simultaneously (one
global + one per sub), so handlers are kept in a registry keyed by resolved
path — opening the same file twice reuses the handler instead of closing it.

Environment:
- ``ROSE_LOG_FILE``: the *implicit* active file for ``get_logger`` calls that
  pass no explicit ``log_file`` (set by drivers / ``SubRunContext.load``).
- ``ROSE_LOG_LEVEL``: default ``INFO``.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any

_DEFAULT_LEVEL = "INFO"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"

# Resolved-path -> open file handler. Lets one process write to many log files
# concurrently (global + each sub) without ever closing a handler in use.
_file_handlers: dict[str, logging.FileHandler] = {}
_stream_handler: logging.StreamHandler | None = None


def _resolve_level() -> int:
    raw = os.environ.get("ROSE_LOG_LEVEL", _DEFAULT_LEVEL).upper()
    return getattr(logging, raw, logging.INFO)


def _short_name(name: str | None) -> str:
    if not name or name == "__main__":
        return "rose"
    return name.rsplit(".", 1)[-1]


def _formatter() -> logging.Formatter:
    return logging.Formatter(_FORMAT, datefmt=_DATEFMT)


def _handler_for_file(path: str | os.PathLike[str], *, mode: str = "a") -> logging.FileHandler:
    """Get (or create) the single shared file handler for ``path``.

    ``mode`` only applies the first time a path is opened in this process; later
    calls reuse the existing handler (so a campaign truncates its log once on
    creation, and everything afterwards appends).
    """
    resolved = str(Path(path).resolve())
    handler = _file_handlers.get(resolved)
    if handler is None:
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(resolved, mode=mode)
        handler.setLevel(_resolve_level())
        handler.setFormatter(_formatter())
        _file_handlers[resolved] = handler
    return handler


def _shared_stream_handler() -> logging.StreamHandler:
    global _stream_handler
    if _stream_handler is None:
        _stream_handler = logging.StreamHandler(sys.stderr)
        _stream_handler.setLevel(_resolve_level())
        _stream_handler.setFormatter(_formatter())
    return _stream_handler


def configure_run_log_file(
    path: str | os.PathLike[str],
    *,
    mode: str = "a",
) -> Path:
    """Set the implicit active log file (``ROSE_LOG_FILE``) for this process.

    Used by task subprocesses (one file per process) and as the driver's
    campaign default, so ``get_logger(name)`` with no explicit file routes here.
    Does not disturb other open handlers.
    """
    resolved = Path(path).resolve()
    _handler_for_file(resolved, mode=mode)  # ensure it exists (and truncate if mode="w")
    os.environ["ROSE_LOG_FILE"] = str(resolved)
    return resolved


def configure_run_log_dir(path: str | os.PathLike[str]) -> Path:
    """Backward-compatible alias: log to ``<dir>/rose.log``."""
    return configure_run_log_file(Path(path) / "rose.log")


def get_logger(
    name: str | None = None,
    *,
    log_file: str | os.PathLike[str] | None = None,
    mode: str = "a",
    stream: bool = True,
) -> logging.Logger:
    """Return a logger bound to a log file (and, by default, stderr).

    Pass ``log_file`` to bind this logger to a *specific* file (the driver uses
    this for the global ``rose.run`` logger and each per-sub logger). With no
    ``log_file`` the target is the implicit active file: ``ROSE_LOG_FILE`` if
    set, else ``./logs/<name>.log`` — this is what task subprocesses use after
    :meth:`SubRunContext.load` pins the sub file.

    Logger names should be unique per file (e.g. ``rose.wf.wf_0``); two loggers
    sharing a name share handlers, which would funnel both to one file.
    """
    short = _short_name(name)
    logger = logging.getLogger(short)
    logger.setLevel(_resolve_level())
    logger.propagate = False

    if log_file is not None:
        target = _handler_for_file(log_file, mode=mode)
    else:
        env_file = os.environ.get("ROSE_LOG_FILE")
        if env_file:
            target = _handler_for_file(env_file)
        else:
            fallback = Path.cwd() / "logs" / f"{short}.log"
            target = _handler_for_file(fallback)

    if target not in logger.handlers:
        logger.addHandler(target)
    if stream:
        sh = _shared_stream_handler()
        if sh not in logger.handlers:
            logger.addHandler(sh)
    return logger


def log_section(logger: logging.Logger, title: str, lines: dict[str, Any]) -> None:
    """Emit a bordered block of key/value lines."""
    logger.info("%s", "=" * 60)
    logger.info("%s", title)
    logger.info("%s", "=" * 60)
    for key, value in lines.items():
        logger.info("  %s: %s", key, value)


def log_dataclass(
    logger: logging.Logger,
    title: str,
    obj: Any,
    *,
    skip: frozenset[str] = frozenset(),
) -> None:
    """Log public dataclass fields (skips callables and private names)."""
    lines: dict[str, Any] = {}
    for f in dc_fields(obj):
        if f.name in skip or f.name.startswith("_"):
            continue
        lines[f.name] = getattr(obj, f.name)
    log_section(logger, title, lines)
