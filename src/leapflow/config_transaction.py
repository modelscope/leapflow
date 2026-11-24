# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded cross-process config transactions; consent happens before entry."""
from __future__ import annotations

import errno
import os
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

from leapflow.utils.file_lock import lock_fd, unlock_fd


@contextmanager
def config_transaction(path: Path, lock_path: Path, *, timeout_s: float = 5.0) -> Iterator[dict[str, Any]]:
    """Lock a stable layout-owned file, then reread durable YAML fail-closed.

    Atomic rename protects readers from torn files, but only this lock protects
    cooperating read/modify/write operations from losing unrelated changes.
    Call from a worker thread in async paths; never hold this across approval.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Config lock must be a regular file")
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                lock_fd(fd, blocking=False)
                locked = True
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError("Configuration is busy; no change was saved") from exc
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            loaded = {}
        except (yaml.YAMLError, UnicodeError) as exc:
            raise ValueError("Existing configuration is unreadable; refusing to overwrite it") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError("Existing configuration must be a mapping; refusing to overwrite it")
        loaded.setdefault("version", 1)
        yield loaded
    finally:
        try:
            if locked:
                unlock_fd(fd)
        finally:
            os.close(fd)
