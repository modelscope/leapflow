# Copyright (c) Alibaba, Inc. and its affiliates.
"""Centralized process logging setup — the single owner of log configuration.

Every LeapFlow process surface initializes logging through this module so that
format, secret redaction, and level policy stay consistent and are defined in
exactly one place:

- CLI / TUI process       -> ``init_cli_logging(settings)``    (runtime.log_level)
- leapd daemon process    -> ``init_daemon_logging(settings)`` (daemon.log_level)
- auxiliary processes     -> ``init_logging(level)``           (explicit level)

Design rules:

- **Single handler owner.** ``init_logging`` installs one stderr handler tagged
  as LeapFlow-owned. Re-initialization is idempotent: it only updates levels,
  never stacks duplicate handlers.
- **Redaction is not optional.** The handler always carries
  ``RedactingFormatter`` so secrets never reach stdout/stderr or leapd.log.
- **No import-time side effects.** Nothing here runs at import; process entry
  points call an init function explicitly.

Scoped, temporary level changes on third-party loggers (e.g. muting a noisy
library during shutdown) are intentionally out of scope — they are local
concerns, not process configuration.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Attribute tag identifying the handler owned by this module.
_HANDLER_TAG = "_leapflow_log_handler"


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


def _owned_handler(root: logging.Logger) -> logging.Handler | None:
    for handler in root.handlers:
        if getattr(handler, _HANDLER_TAG, False):
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
