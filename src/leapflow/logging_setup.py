# Copyright (c) Alibaba, Inc. and its affiliates.
"""Centralized process logging setup — the single owner of log configuration.

Every LeapFlow process surface initializes logging through this module so that
format, secret redaction, and level policy stay consistent and are defined in
exactly one place:

- CLI / TUI process       -> ``init_cli_logging(settings)``    (runtime.log_level)
- leapd daemon process    -> ``init_daemon_logging(settings)`` (daemon.log_level)
- LHP edge node process   -> ``init_node_logging(settings)``   (hardware.node_log_*)
- auxiliary processes     -> ``init_logging(level)``           (explicit level)

Design rules:

- **Single handler owner.** ``init_logging`` installs one stderr handler tagged
  as LeapFlow-owned. Re-initialization is idempotent: it only updates levels,
  never stacks duplicate handlers.
- **Redaction is not optional.** The handler always carries
  ``RedactingFormatter`` so secrets never reach stdout/stderr or leapd.log.
- **No import-time side effects.** Nothing here runs at import; process entry
  points call an init function explicitly.
- **A long-running process owns its file.** A process that runs unattended for
  days (the edge node) writes a size-rotated file itself rather than relying on
  a launcher's redirect: a redirect grows without bound, and a launcher that
  truncates on restart destroys the record of the run that needed diagnosing.

Scoped, temporary level changes on third-party loggers (e.g. muting a noisy
library during shutdown) are intentionally out of scope — they are local
concerns, not process configuration.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Attribute tags identifying the handlers owned by this module.
_HANDLER_TAG = "_leapflow_log_handler"
_FILE_HANDLER_TAG = "_leapflow_file_handler"


def resolve_level(name: str, *, default: int = logging.WARNING) -> int:
    """Map a level name to a logging constant, falling back to ``default``."""
    resolved = getattr(logging, str(name or "").strip().upper(), None)
    return resolved if isinstance(resolved, int) else default


def _build_formatter() -> logging.Formatter:
    try:
        from leapflow.security.redact import RedactingFormatter
        return RedactingFormatter(LOG_FORMAT)
    except ImportError:  # pragma: no cover - redact is a first-party module
        return logging.Formatter(LOG_FORMAT)


def _owned_handler(root: logging.Logger, tag: str = _HANDLER_TAG) -> logging.Handler | None:
    for handler in root.handlers:
        if getattr(handler, tag, False):
            return handler
    return None


def init_logging(level: str, *, default: int = logging.WARNING) -> None:
    """Initialize (or re-level) process logging. Idempotent.

    First call attaches one redacting stderr handler to the root logger and
    sets the root level. Subsequent calls only adjust the level — handlers are
    never duplicated, so this is safe to call from any entry point.
    """
    resolved = resolve_level(level, default=default)
    root = logging.getLogger()
    handler = _owned_handler(root)
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_build_formatter())
        setattr(handler, _HANDLER_TAG, True)
        root.addHandler(handler)
    root.setLevel(resolved)


def init_cli_logging(settings: Any) -> None:
    """Configure logging for the interactive CLI/TUI process.

    Driven by ``runtime.log_level`` (default WARNING: the interactive surface
    stays quiet; verbose diagnostics belong to the daemon file log).
    """
    init_logging(str(getattr(settings, "log_level", "") or "WARNING"))


def init_daemon_logging(settings: Any) -> None:
    """Configure logging for the leapd daemon process.

    Driven by ``daemon.log_level`` (default INFO: stdout/stderr land in
    leapd.log, so INFO field evidence — deferred init progress, turn usage,
    empty-response warnings — is captured for diagnosis).
    """
    init_logging(str(getattr(settings, "daemon_log_level", "") or "INFO"), default=logging.INFO)


def attach_rotating_file(
    path: str | Path, *, max_bytes: int, backup_count: int
) -> logging.Handler:
    """Attach (or retarget) the single size-rotated file handler. Idempotent.

    Appends, never truncates: restarting the process continues the record, and
    rotation is the only thing that ever retires old lines. Re-attaching the same
    path keeps the existing handler; a different path replaces it.
    """
    target = Path(path).expanduser().resolve()
    root = logging.getLogger()
    existing = _owned_handler(root, _FILE_HANDLER_TAG)
    if existing is not None:
        if Path(getattr(existing, "baseFilename", "")) == target:
            return existing
        root.removeHandler(existing)
        existing.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        target,
        mode="a",
        maxBytes=max(0, int(max_bytes)),
        backupCount=max(0, int(backup_count)),
        encoding="utf-8",
    )
    handler.setFormatter(_build_formatter())
    setattr(handler, _FILE_HANDLER_TAG, True)
    root.addHandler(handler)
    return handler


def init_node_logging(settings: Any, *, log_file: str = "", verbose: bool = False) -> Path | None:
    """Configure logging for an LHP edge-node process; return the file in use.

    Driven by ``hardware.node_log_level`` (default INFO: a node's halts, stops,
    and run lifecycle are field evidence) and ``hardware.node_log_file``, which
    ``log_file`` overrides. With a file, the node writes and rotates it itself
    (``hardware.node_log_max_mb`` x ``hardware.node_log_backups``) and stderr is
    kept to warnings, so a launcher redirect stays small and the full record
    lives in one bounded place. Without one, everything goes to stderr.
    """
    level = "DEBUG" if verbose else str(getattr(settings, "hardware_node_log_level", "") or "INFO")
    init_logging(level, default=logging.INFO)
    path = str(log_file or getattr(settings, "hardware_node_log_file", "") or "").strip()
    if not path:
        return None
    max_mb = float(getattr(settings, "hardware_node_log_max_mb", 16.0) or 0.0)
    handler = attach_rotating_file(
        path,
        max_bytes=int(max_mb * 1024 * 1024),
        backup_count=int(getattr(settings, "hardware_node_log_backups", 5) or 0),
    )
    console = _owned_handler(logging.getLogger())
    if console is not None:
        console.setLevel(logging.WARNING)
    return Path(getattr(handler, "baseFilename", path))
