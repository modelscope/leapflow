"""macOS LaunchAgent (launchd) registration helper.

Generates and installs a LaunchAgent plist into ``~/Library/LaunchAgents``
and toggles its runtime state via ``launchctl``. Falls back from the modern
``bootstrap``/``bootout`` API (macOS 11+) to the legacy ``load``/``unload``
API automatically.
"""

from __future__ import annotations

import logging
import os
import plistlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"


class LaunchdError(RuntimeError):
    """Raised when a launchctl invocation fails irrecoverably."""


class LaunchdService:
    """Manage a per-user LaunchAgent.

    All paths are accepted as ``Path`` and rendered into the plist as their
    expanded absolute string form so launchd can resolve them without shell
    interpolation.
    """

    def __init__(
        self,
        *,
        label: str,
        executable: Path,
        socket_path: Path,
        pid_file: Path,
        log_file: Path,
        extra_args: Optional[List[str]] = None,
        environment: Optional[Dict[str, str]] = None,
        keep_alive: bool = True,
        run_at_load: bool = True,
        throttle_interval: int = 5,
    ) -> None:
        if not label or "/" in label:
            raise ValueError(f"invalid launchd label: {label!r}")
        self.label = label
        self.executable = Path(executable).expanduser()
        self.socket_path = Path(socket_path).expanduser()
        self.pid_file = Path(pid_file).expanduser()
        self.log_file = Path(log_file).expanduser()
        self.extra_args = list(extra_args or [])
        self.environment = dict(environment or {})
        self.keep_alive = keep_alive
        self.run_at_load = run_at_load
        self.throttle_interval = max(1, int(throttle_interval))

    # ── plist ────────────────────────────────────────────────────────

    @property
    def plist_path(self) -> Path:
        return LAUNCH_AGENTS_DIR / f"{self.label}.plist"

    def generate_plist(self) -> Dict[str, Any]:
        """Build the plist dictionary (deterministic, no I/O)."""
        program_arguments: List[str] = [
            str(self.executable),
            "--daemon",
            "--socket", str(self.socket_path),
            "--pid-file", str(self.pid_file),
            "--log-file", str(self.log_file),
        ]
        program_arguments.extend(self.extra_args)

        plist: Dict[str, Any] = {
            "Label": self.label,
            "ProgramArguments": program_arguments,
            "RunAtLoad": bool(self.run_at_load),
            "KeepAlive": bool(self.keep_alive),
            "ThrottleInterval": int(self.throttle_interval),
            "ProcessType": "Background",
            "StandardOutPath": str(self.log_file),
            "StandardErrorPath": str(self.log_file),
            "WorkingDirectory": str(self.executable.parent),
        }
        if self.environment:
            plist["EnvironmentVariables"] = dict(self.environment)
        return plist

    def write_plist(self) -> Path:
        """Serialize the plist to disk; create parent dirs as needed."""
        try:
            LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LaunchdError(f"cannot create {LAUNCH_AGENTS_DIR}: {exc}") from exc

        path = self.plist_path
        try:
            with open(path, "wb") as fh:
                plistlib.dump(self.generate_plist(), fh)
        except OSError as exc:
            raise LaunchdError(f"cannot write {path}: {exc}") from exc
        logger.info("Wrote LaunchAgent plist → %s", path)
        return path

    # ── launchctl ────────────────────────────────────────────────────

    def is_registered(self) -> bool:
        """True if the plist exists and is currently known to launchd."""
        if not self.plist_path.exists():
            return False
        try:
            res = subprocess.run(
                ["launchctl", "list", self.label],
                capture_output=True, text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return res.returncode == 0

    def register(self) -> bool:
        """Write plist and load it into launchd. Idempotent."""
        path = self.write_plist()

        # Best-effort unload first to make this idempotent.
        self._launchctl_unload(path, ignore_errors=True)

        # Modern API (macOS 11+): bootstrap into the user's GUI domain.
        uid = os.getuid()
        modern = self._run_launchctl(
            ["bootstrap", f"gui/{uid}", str(path)],
            ignore_errors=True,
        )
        if modern is True:
            logger.info("launchd bootstrap OK: %s", self.label)
            return True

        # Legacy fallback.
        legacy = self._run_launchctl(["load", "-w", str(path)], ignore_errors=False)
        if legacy:
            logger.info("launchd load OK: %s", self.label)
            return True
        return False

    def unregister(self) -> bool:
        """Unload from launchd and delete the plist file. Idempotent."""
        path = self.plist_path
        ok = self._launchctl_unload(path, ignore_errors=True)
        if path.exists():
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("cannot delete %s: %s", path, exc)
                return False
        logger.info("launchd unregister done: %s", self.label)
        return ok

    # ── internals ────────────────────────────────────────────────────

    def _launchctl_unload(self, path: Path, *, ignore_errors: bool) -> bool:
        uid = os.getuid()
        modern = self._run_launchctl(
            ["bootout", f"gui/{uid}/{self.label}"],
            ignore_errors=True,
        )
        if modern is True:
            return True
        if not path.exists():
            return True
        return self._run_launchctl(
            ["unload", "-w", str(path)],
            ignore_errors=ignore_errors,
        )

    def _run_launchctl(self, args: List[str], *, ignore_errors: bool) -> bool:
        cmd = ["launchctl", *args]
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            if ignore_errors:
                logger.debug("launchctl %s failed: %s", args, exc)
                return False
            raise LaunchdError(f"launchctl {' '.join(args)}: {exc}") from exc
        if res.returncode != 0:
            stderr = (res.stderr or "").strip()
            msg = f"launchctl {' '.join(args)} → rc={res.returncode}: {stderr}"
            if ignore_errors:
                logger.debug(msg)
                return False
            raise LaunchdError(msg)
        return True
