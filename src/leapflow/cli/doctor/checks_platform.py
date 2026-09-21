# Copyright (c) Alibaba, Inc. and its affiliates.
"""Platform-level diagnostic checks (Python version, OS, disk space)."""
from __future__ import annotations

import os
import platform
import shutil
import sys

from leapflow.cli.doctor.protocol import Finding


class PythonVersionCheck:
    """Verify the running Python version is within the supported range."""

    name = "Python version"
    section = "platform"

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        major, minor = sys.version_info[:2]
        version_str = f"{major}.{minor}.{sys.version_info[2]}"
        if major != 3 or minor < 11:
            f.error(f"Python >= 3.11 required, found {version_str}")
        elif minor >= 14:
            f.warn(f"Python {version_str} is newer than tested range (3.11–3.13)")
        else:
            f.pass_()
        return f


class OSCompatibilityCheck:
    """Check that the host OS is a known-supported platform."""

    name = "OS compatibility"
    section = "platform"

    _SUPPORTED = {"Darwin", "Linux"}

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        system = platform.system()
        if system in self._SUPPORTED:
            f.pass_()
        elif system == "Windows":
            f.warn("Windows is not officially supported; expect rough edges")
        else:
            f.warn(f"Unknown OS '{system}'; LeapFlow is tested on macOS and Linux")
        return f


class DiskSpaceCheck:
    """Ensure there is sufficient free space on the data partition."""

    name = "Disk space"
    section = "platform"

    _MIN_FREE_MB = 100

    def __init__(self, data_dir: str | os.PathLike[str] | None = None) -> None:
        self._data_dir = data_dir

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        target = str(self._data_dir) if self._data_dir else os.path.expanduser("~/.leapflow")
        try:
            usage = shutil.disk_usage(target)
            free_mb = usage.free / (1024 * 1024)
            if free_mb < self._MIN_FREE_MB:
                f.error(f"Low disk space: {free_mb:.0f} MB free on {target} (min {self._MIN_FREE_MB} MB)")
            else:
                f.pass_()
        except OSError as exc:
            f.warn(f"Cannot check disk space for {target}: {exc}")
        return f
