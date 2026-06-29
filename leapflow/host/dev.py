"""OS Host development mode with auto-rebuild on file changes."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from leapflow.config import Settings, load_config

# ── ANSI colors ──────────────────────────────────────────────────────────

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[1;36m"
_BLUE = "\033[34m"

_PREFIX = f"{_DIM}[dev]{_RESET}"
_SEP = "──────────────────────────────────────"


def _dev_print(msg: str, *, color: str = "") -> None:
    """Print a dev-mode message with [dev] prefix."""
    if color:
        print(f"{_PREFIX} {color}{msg}{_RESET}")
    else:
        print(f"{_PREFIX} {msg}")


class HostDevServer:
    """Watch Swift source files and auto-rebuild/restart OS Host.

    Core design:
    - File monitoring via mtime polling (zero external dependencies)
    - Debounce: coalesces rapid saves into a single rebuild
    - Subprocess management: child process stdout/stderr piped to terminal
    - Graceful exit: Ctrl+C stops child, exits loop cleanly
    """

    def __init__(
        self,
        *,
        source_dir: Path,
        package_file: Path,
        build_dir: Path,
        socket_path: Path,
        pid_file: Optional[Path] = None,
        build_config: str = "debug",
        poll_interval: float = 1.0,
        debounce_delay: float = 0.5,
        on_rebuild: Optional[Callable[[], None]] = None,
    ) -> None:
        self.source_dir = source_dir
        self.package_file = package_file
        self.build_dir = build_dir
        self.socket_path = socket_path
        self.pid_file = pid_file
        self.build_config = build_config
        self.poll_interval = poll_interval
        self.debounce_delay = debounce_delay
        self.on_rebuild = on_rebuild

        self._host_process: Optional[subprocess.Popen[bytes]] = None
        self._file_mtimes: Dict[Path, float] = {}
        self._running = False

    # ── Public API ────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main loop: watch → build → run → repeat."""
        self._running = True
        self._setup_signal_handlers()

        _dev_print(f"Watching {self.source_dir.relative_to(self.build_dir.parent)}/ for changes...", color=_CYAN)

        # Initial build + start
        self._snapshot_files()
        success = await self._build()
        if success:
            self._start_host()

        # Watch loop
        try:
            while self._running:
                await asyncio.sleep(self.poll_interval)
                changed = self._detect_changes()
                if changed:
                    # Report changed files
                    for f in changed[:5]:
                        rel = self._relative_path(f)
                        _dev_print(f"File changed: {rel}", color=_YELLOW)
                    if len(changed) > 5:
                        _dev_print(f"  ... and {len(changed) - 5} more", color=_DIM)

                    # Debounce: wait briefly for more saves
                    await asyncio.sleep(self.debounce_delay)
                    # Re-snapshot to catch any further changes during debounce
                    self._snapshot_files()

                    # Rebuild
                    _dev_print("Rebuilding...", color=_CYAN)
                    success = await self._build()
                    if success:
                        _dev_print("Restarting OS Host...", color=_CYAN)
                        self._stop_host()
                        self._start_host()
                    else:
                        _dev_print("Build failed — waiting for next change...", color=_RED)
        except asyncio.CancelledError:
            pass
        finally:
            self._stop_host()
            _dev_print("Stopped.", color=_DIM)

    # ── File watching ─────────────────────────────────────────────────

    def _collect_swift_files(self) -> list[Path]:
        """Collect all .swift files under source_dir + Package.swift."""
        files: list[Path] = []
        if self.package_file.exists():
            files.append(self.package_file)
        if self.source_dir.exists():
            for root, _dirs, filenames in os.walk(self.source_dir):
                for name in filenames:
                    if name.endswith(".swift"):
                        files.append(Path(root) / name)
        return files

    def _snapshot_files(self) -> None:
        """Take a snapshot of all tracked files' mtimes."""
        self._file_mtimes = {}
        for f in self._collect_swift_files():
            try:
                self._file_mtimes[f] = f.stat().st_mtime
            except OSError:
                continue

    def _detect_changes(self) -> list[Path]:
        """Compare current mtimes against snapshot; update snapshot.

        Returns list of changed/new/deleted files.
        """
        changed: list[Path] = []
        current_files = self._collect_swift_files()
        current_mtimes: Dict[Path, float] = {}

        for f in current_files:
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            current_mtimes[f] = mtime
            prev = self._file_mtimes.get(f)
            if prev is None or mtime != prev:
                changed.append(f)

        # Detect deletions
        for f in self._file_mtimes:
            if f not in current_mtimes:
                changed.append(f)

        if changed:
            self._file_mtimes = current_mtimes

        return changed

    # ── Build ─────────────────────────────────────────────────────────

    async def _build(self) -> bool:
        """Run swift build and return True on success."""
        _dev_print(f"Building ({self.build_config})...", color=_BLUE)
        start = time.monotonic()

        cmd = ["swift", "build", "-c", self.build_config]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.build_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
        except FileNotFoundError:
            _dev_print("'swift' not found in PATH. Install Xcode Command Line Tools.", color=_RED)
            return False

        elapsed = time.monotonic() - start

        if proc.returncode == 0:
            _dev_print(f"Build succeeded ({elapsed:.1f}s)", color=_GREEN)
            if self.on_rebuild:
                self.on_rebuild()
            return True
        else:
            _dev_print(f"Build failed ({elapsed:.1f}s):", color=_RED)
            # Print compiler output
            if stdout:
                output = stdout.decode("utf-8", errors="replace")
                for line in output.splitlines():
                    print(f"  {_DIM}{line}{_RESET}")
            return False

    # ── Host process ──────────────────────────────────────────────────

    @property
    def _executable_path(self) -> Path:
        return self.build_dir / ".build" / self.build_config / "OSHost"

    def _start_host(self) -> None:
        """Launch OS Host as a child process with output to terminal."""
        exe = self._executable_path
        if not exe.exists():
            _dev_print(f"Executable not found: {exe}", color=_RED)
            return

        # Ensure socket parent directory exists
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [str(exe), "--socket", str(self.socket_path)]
        if self.pid_file:
            self.pid_file.parent.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--pid-file", str(self.pid_file)])

        _dev_print(f"Starting OS Host on {self.socket_path}", color=_GREEN)
        print(f"{_PREFIX} {_DIM}{_SEP}{_RESET}")

        try:
            self._host_process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                cwd=str(exe.parent),
                # Keep in same process group so Ctrl+C propagates
            )
        except OSError as exc:
            _dev_print(f"Cannot start OS Host: {exc}", color=_RED)
            self._host_process = None

    def _stop_host(self) -> None:
        """Stop the running OS Host child process."""
        proc = self._host_process
        if proc is None:
            return

        print(f"{_PREFIX} {_DIM}{_SEP}{_RESET}")

        if proc.poll() is not None:
            # Already exited
            self._host_process = None
            return

        # Send SIGTERM for graceful shutdown
        try:
            proc.terminate()
        except OSError:
            pass

        # Wait briefly for exit
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            # Escalate to SIGKILL
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                pass

        self._host_process = None

        # Clean up stale socket and pid file
        for path in (self.socket_path, self.pid_file):
            try:
                if path and path.exists():
                    path.unlink()
            except OSError:
                pass

    # ── Signal handling ───────────────────────────────────────────────

    def _setup_signal_handlers(self) -> None:
        """Register signal handlers for graceful exit."""
        loop = asyncio.get_running_loop()

        def _handle_signal() -> None:
            self._running = False

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _handle_signal)

    # ── Utils ─────────────────────────────────────────────────────────

    def _relative_path(self, path: Path) -> str:
        """Return a short relative path for display."""
        try:
            return str(path.relative_to(self.build_dir))
        except ValueError:
            return str(path)


# ── Factory ──────────────────────────────────────────────────────────────


def create_dev_server(settings: Optional[Settings] = None) -> HostDevServer:
    """Create a HostDevServer from project configuration.

    Automatically locates os_host/ relative to the project root.
    """
    if settings is None:
        settings = load_config()

    # Locate os_host directory relative to this package
    project_root = Path(__file__).resolve().parents[2]
    os_host_dir = project_root / "os_host"

    return HostDevServer(
        source_dir=os_host_dir / "Sources",
        package_file=os_host_dir / "Package.swift",
        build_dir=os_host_dir,
        socket_path=settings.host_socket,
        pid_file=settings.host_pid_file,
        build_config="debug",
    )


async def cmd_host_dev() -> int:
    """Entry point for `leap host dev` command."""
    settings = load_config()
    server = create_dev_server(settings)
    await server.run()
    return 0
