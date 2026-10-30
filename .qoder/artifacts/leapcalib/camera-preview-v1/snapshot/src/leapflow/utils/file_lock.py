# Copyright (c) Alibaba, Inc. and its affiliates.
"""Cross-platform exclusive advisory locking on open files.

POSIX systems provide ``fcntl.flock`` (advisory, whole-file, released when
the fd closes); Windows has no ``fcntl``, so this module uses
``msvcrt.locking`` (byte-range, released on unlock or handle close). These
two functions unify both backends behind one API with identical semantics:

- ``lock_fd(fd, blocking=True)`` waits until the lock is free;
- ``lock_fd(fd, blocking=False)`` probes once and raises ``OSError`` if
  another process holds the lock (``BlockingIOError`` on POSIX,
  ``PermissionError`` on Windows);
- ``unlock_fd(fd)`` releases the lock; closing the fd also releases it on
  both platforms, so a crashed holder never strands the lock.

Guarantee scope: single-machine, inter-process mutual exclusion only. The
lock is advisory on POSIX and byte-range based on Windows, so it only
coordinates callers that lock the same file through these functions.

Callers own fd lifecycle and directory layout::

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        lock_fd(fd, blocking=False)
    except OSError:
        os.close(fd)
        ...  # another process holds the lock
"""

from __future__ import annotations

import os
import sys
from typing import Protocol, runtime_checkable


@runtime_checkable
class _FilenoHolder(Protocol):
    def fileno(self) -> int: ...


# Same accept-anything contract as fcntl.flock's FileDescriptorLike:
# a raw descriptor or any object exposing fileno() (e.g. an open() file).
FdLike = "int | _FilenoHolder"

if sys.platform == "win32":
    import msvcrt

    def lock_fd(fd: FdLike, blocking: bool = True) -> None:
        """Acquire an exclusive lock on an open file.

        Raises ``OSError`` if ``blocking`` is False and another process
        holds the lock.
        """
        # msvcrt.locking() only accepts a raw descriptor, unlike
        # fcntl.flock(), so resolve file-like objects here.
        if isinstance(fd, _FilenoHolder):
            fd = fd.fileno()
        # Lock a single byte from the file start. msvcrt.locking() is
        # positioned-relative, so rewind before every operation.
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)

    def unlock_fd(fd: FdLike) -> None:
        """Release the lock held on an open file."""
        if isinstance(fd, _FilenoHolder):
            fd = fd.fileno()
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def lock_fd(fd: FdLike, blocking: bool = True) -> None:
        """Acquire an exclusive lock on an open file.

        Raises ``OSError`` if ``blocking`` is False and another process
        holds the lock.
        """
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        fcntl.flock(fd, flags)

    def unlock_fd(fd: FdLike) -> None:
        """Release the lock held on an open file."""
        fcntl.flock(fd, fcntl.LOCK_UN)
