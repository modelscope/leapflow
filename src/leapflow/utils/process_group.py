# Copyright (c) Alibaba, Inc. and its affiliates.
"""Cross-platform process-group termination.

POSIX process groups and Windows job objects play the same role: a
membership inherited by child processes, so the whole tree spawned behind
a shell can be terminated at once. Killing only the direct child leaves
its descendants running as orphans.

Usage::

    group = ProcessGroup()
    proc = subprocess.Popen(cmd, start_new_session=True)  # POSIX makes it a group leader
    group.attach(proc.pid)
    ...
    group.terminate(signal.SIGTERM)

On POSIX ``attach`` records the target's process group and ``terminate``
signals every member. On Windows ``attach`` assigns the process to a job
object (descendants inherit membership automatically) and ``terminate``
kills every member; there is no graceful equivalent there, so the signal
argument is accepted for API parity only.
"""
from __future__ import annotations

import logging
import os
import signal
import sys

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    import ctypes

    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001

    class ProcessGroup:
        """Process group backed by a Windows job object."""

        def __init__(self) -> None:
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._handle = self._kernel32.CreateJobObjectW(None, None)
            self._attached = False
            if not self._handle:
                raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")

        def attach(self, pid: int) -> bool:
            """Assign a process to the job; its descendants inherit membership."""
            kernel32 = self._kernel32
            proc = kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
            if not proc:
                logger.debug("process group: cannot open pid=%s", pid)
                return False
            try:
                ok = bool(kernel32.AssignProcessToJobObject(self._handle, proc))
                self._attached = self._attached or ok
                return ok
            finally:
                kernel32.CloseHandle(proc)

        def terminate(self, sig: int = signal.SIGTERM) -> bool:
            """Terminate every job member (always forced on Windows)."""
            if not self._handle or not self._attached:
                return False
            ok = bool(self._kernel32.TerminateJobObject(self._handle, 1))
            self._kernel32.CloseHandle(self._handle)
            self._handle = None
            return ok

else:

    class ProcessGroup:
        """Process group backed by the POSIX pgid (set via start_new_session)."""

        def __init__(self) -> None:
            self._pgid: int | None = None

        def attach(self, pid: int) -> bool:
            """Record the target's current process group."""
            try:
                self._pgid = os.getpgid(pid)
                return True
            except OSError:
                logger.debug("process group: cannot resolve pgid for pid=%s", pid)
                return False

        def terminate(self, sig: int = signal.SIGTERM) -> bool:
            """Signal every member of the recorded process group."""
            if self._pgid is None:
                return False
            try:
                os.killpg(self._pgid, sig)
                self._pgid = None
                return True
            except OSError:
                return False
