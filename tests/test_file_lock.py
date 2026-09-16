# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the cross-platform lock_fd / unlock_fd pair."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from leapflow.utils.file_lock import lock_fd, unlock_fd

_HOLD_SCRIPT = """
import os, sys, time
from leapflow.utils.file_lock import lock_fd, unlock_fd

fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR)
lock_fd(fd, blocking=True)
print("HELD", flush=True)
time.sleep(float(sys.argv[2]))
unlock_fd(fd)
os.close(fd)
"""


def _start_holder(lock_path, hold_seconds: float) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLD_SCRIPT, str(lock_path), str(hold_seconds)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Wait until the child confirms it holds the lock.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if "HELD" in line:
            return proc
    proc.kill()
    raise AssertionError(f"holder never acquired: {proc.stderr.read()}")


@pytest.fixture
def lock_path(tmp_path):
    return tmp_path / "test.lock"


@pytest.fixture
def lock_fd_open(lock_path):
    """An open fd on the lock file, closed automatically at teardown."""
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    yield fd
    os.close(fd)


def test_lock_unlock_roundtrip(lock_fd_open):
    lock_fd(lock_fd_open)
    unlock_fd(lock_fd_open)
    # Re-acquire after release must succeed.
    lock_fd(lock_fd_open)
    unlock_fd(lock_fd_open)


def test_nonblocking_succeeds_when_free(lock_fd_open):
    lock_fd(lock_fd_open, blocking=False)
    unlock_fd(lock_fd_open)


def test_accepts_file_object_like_flock(lock_path):
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        lock_fd(lock_file)
        unlock_fd(lock_file)


def test_nonblocking_raises_while_other_process_holds(lock_path):
    holder = _start_holder(lock_path, 1.5)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        try:
            with pytest.raises(OSError):
                lock_fd(fd, blocking=False)
        finally:
            os.close(fd)
    finally:
        holder.wait(timeout=10)


def test_blocking_waits_for_release(lock_path):
    holder = _start_holder(lock_path, 0.8)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        try:
            lock_fd(fd, blocking=True)
            unlock_fd(fd)
        finally:
            os.close(fd)
    finally:
        holder.wait(timeout=10)
