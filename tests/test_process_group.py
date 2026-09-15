# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for leapflow.utils.process_group — cross-platform tree termination."""
from __future__ import annotations

import subprocess
import sys
import time

from leapflow.daemon.lifecycle import _process_alive
from leapflow.utils.process_group import ProcessGroup

_SLEEP = "import time; time.sleep(60)"


def _spawn_sleeper() -> subprocess.Popen:
    # start_new_session gives the child its own POSIX process group; without it
    # attach() would record pytest's group and terminate() would kill the runner.
    return subprocess.Popen([sys.executable, "-c", _SLEEP], start_new_session=True)  # noqa: S603 - trusted argv


def _spawn_parent_then_child() -> subprocess.Popen:
    """Spawn a parent that sleeps, then (after 2s) spawns a child and reports its pid."""
    script = (
        "import subprocess, sys, time\n"
        "time.sleep(2)\n"
        f"p = subprocess.Popen([sys.executable, '-c', {_SLEEP!r}])\n"
        "print(p.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    return subprocess.Popen(  # noqa: S603 - trusted argv
        [sys.executable, "-c", script], stdout=subprocess.PIPE, start_new_session=True
    )


def test_terminate_kills_attached_tree() -> None:
    """A process attached before it spawns children takes its whole tree down."""
    proc = _spawn_parent_then_child()
    group = ProcessGroup()
    assert group.attach(proc.pid) is True
    assert proc.stdout is not None
    grand_pid = int(proc.stdout.readline())  # child spawned after attach
    time.sleep(0.3)

    assert group.terminate() is True
    time.sleep(0.5)
    assert _process_alive(proc.pid) is False
    assert _process_alive(grand_pid) is False


def test_terminate_kills_attached_process() -> None:
    proc = _spawn_sleeper()
    group = ProcessGroup()
    assert group.attach(proc.pid) is True
    assert group.terminate() is True
    time.sleep(0.5)
    assert _process_alive(proc.pid) is False


def test_attach_dead_pid_is_false_or_harmless() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603 - trusted argv
    proc.wait()
    time.sleep(0.2)  # let the process object vanish (Windows handle teardown)
    group = ProcessGroup()
    # Windows: OpenProcess fails -> False. POSIX: an unreaped child still has a
    # pgid, so True is also correct; neither may raise.
    assert group.attach(proc.pid) in (True, False)


def test_second_terminate_is_false() -> None:
    proc = _spawn_sleeper()
    group = ProcessGroup()
    group.attach(proc.pid)
    assert group.terminate() is True
    proc.wait(timeout=5)  # reap on POSIX: a zombie keeps the group alive
    assert group.terminate() is False  # job handle released / pgid already dead


def test_terminate_without_attach_is_false() -> None:
    group = ProcessGroup()
    assert group.terminate() is False
