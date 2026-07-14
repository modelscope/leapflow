"""Reusable backend event source implementations."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping

from leapflow.gateway.connectors.protocol import BackendEvent, EventSourceStatus

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# UnavailableEventSource — sentinel for unconfigured platforms
# ═══════════════════════════════════════════════════════════════

class UnavailableEventSource:
    """Event source that explicitly reports an unsupported or unconfigured inbound path."""

    def __init__(
        self,
        *,
        platform_id: str,
        backend_kind: str,
        detail: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.platform_id = platform_id
        self.backend_kind = backend_kind
        self._detail = detail
        self._metadata = dict(metadata or {})
        self._started = False

    async def start(self, *, checkpoint: str = "") -> EventSourceStatus:
        self._started = False
        return EventSourceStatus(
            ok=False,
            backend_kind=self.backend_kind,
            detail=self._detail,
            checkpoint=checkpoint,
            metadata={**self._metadata, "available": False},
        )

    async def stop(self) -> EventSourceStatus:
        self._started = False
        return await self.status()

    async def events(self) -> AsyncIterator[BackendEvent]:
        if False:
            yield BackendEvent(event_id="", event_type="", platform_id=self.platform_id)
        return

    async def status(self) -> EventSourceStatus:
        return EventSourceStatus(
            ok=False,
            backend_kind=self.backend_kind,
            detail=self._detail,
            metadata={**self._metadata, "available": False, "started": self._started},
        )


# ═══════════════════════════════════════════════════════════════
# CliNdjsonEventSource — generic long-lived CLI subprocess
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CliEventSourceConfig:
    """Configuration for a CLI-backed NDJSON event source."""

    binary: str
    args: tuple[str, ...]
    platform_id: str
    env: Mapping[str, str] = field(default_factory=dict)
    ready_pattern: str = ""
    error_pattern: str = ""
    ready_timeout_s: float = 30.0
    restart_backoff_base_s: float = 5.0
    max_restart_backoff_s: float = 300.0
    max_restarts: int = 20


class CliNdjsonEventSource:
    """Long-lived CLI subprocess that yields NDJSON events on stdout.

    Satisfies the ``BackendEventSource`` protocol.  The subprocess
    lifecycle follows the lark-cli event consume contract:

    - stdin is kept open (closing it triggers graceful exit)
    - stderr is monitored for a ready marker and error patterns
    - stdout is read line-by-line as NDJSON

    This class is platform-agnostic.  Platform-specific wrappers
    (e.g. ``LarkCliEventSource``) provide the ``CliEventSourceConfig``
    with platform-tuned binary, args, and stderr patterns.
    """

    backend_kind = "cli"

    def __init__(self, config: CliEventSourceConfig) -> None:
        self._config = config
        self.platform_id = config.platform_id
        self._proc: asyncio.subprocess.Process | None = None
        self._running = False
        self._ready = asyncio.Event()
        self._restart_count = 0
        self._last_error = ""
        self._stderr_task: asyncio.Task[None] | None = None
        self._ready_re = re.compile(config.ready_pattern) if config.ready_pattern else None
        self._error_re = re.compile(config.error_pattern) if config.error_pattern else None

    async def start(self, *, checkpoint: str = "") -> EventSourceStatus:
        """Spawn the subprocess and wait for the ready marker."""
        if self._running:
            return await self.status()
        self._restart_count = 0
        self._last_error = ""
        try:
            await self._spawn_process()
        except FileNotFoundError:
            return EventSourceStatus(
                ok=False,
                backend_kind=self.backend_kind,
                detail=f"CLI binary not found: {self._config.binary}",
                metadata={"available": False},
            )
        except OSError as exc:
            return EventSourceStatus(
                ok=False,
                backend_kind=self.backend_kind,
                detail=f"Failed to spawn CLI process: {exc}",
                metadata={"available": False},
            )

        if self._ready_re:
            try:
                await asyncio.wait_for(
                    self._ready.wait(), timeout=self._config.ready_timeout_s,
                )
            except asyncio.TimeoutError:
                await self._kill_process()
                return EventSourceStatus(
                    ok=False,
                    backend_kind=self.backend_kind,
                    detail=(
                        f"Event source did not become ready within "
                        f"{self._config.ready_timeout_s}s"
                    ),
                    metadata={"available": True, "started": False,
                              "last_error": self._last_error},
                )

        if self._last_error:
            await self._kill_process()
            return EventSourceStatus(
                ok=False,
                backend_kind=self.backend_kind,
                detail=self._last_error,
                metadata={"available": True, "started": False},
            )

        self._running = True
        return EventSourceStatus(
            ok=True,
            backend_kind=self.backend_kind,
            detail="Event source started",
            checkpoint=checkpoint,
            metadata={"available": True, "started": True},
        )

    async def stop(self) -> EventSourceStatus:
        """Gracefully stop the subprocess."""
        self._running = False
        self._ready.clear()
        await self._kill_process()
        return EventSourceStatus(
            ok=True,
            backend_kind=self.backend_kind,
            detail="Event source stopped",
            metadata={"available": True, "started": False},
        )

    async def events(self) -> AsyncIterator[BackendEvent]:
        """Yield parsed BackendEvent objects from stdout NDJSON lines."""
        while self._running:
            proc = self._proc
            if proc is None or proc.stdout is None:
                if not self._running:
                    return
                await self._restart_with_backoff()
                continue

            try:
                line = await proc.stdout.readline()
            except asyncio.CancelledError:
                return

            if not line:
                exit_code = await proc.wait()
                logger.info(
                    "Event source %s process exited (code=%s)",
                    self.platform_id, exit_code,
                )
                if not self._running:
                    return
                await self._restart_with_backoff()
                continue

            event = self._parse_ndjson_line(line)
            if event is not None:
                self._restart_count = 0
                yield event

    async def status(self) -> EventSourceStatus:
        """Return current event source health."""
        alive = self._proc is not None and self._proc.returncode is None
        return EventSourceStatus(
            ok=self._running and alive,
            backend_kind=self.backend_kind,
            detail=self._last_error or ("running" if alive else "stopped"),
            metadata={
                "available": True,
                "started": self._running,
                "process_alive": alive,
                "restart_count": self._restart_count,
            },
        )

    # ── Internal ─────────────────────────────────────────────

    async def _spawn_process(self) -> None:
        """Spawn the CLI subprocess with stdin/stdout/stderr pipes."""
        if self._stderr_task is not None and not self._stderr_task.done():
            self._stderr_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None

        argv = [self._config.binary, *self._config.args]
        env = dict(self._config.env) if self._config.env else None
        self._ready.clear()
        self._last_error = ""

        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._stderr_task = asyncio.create_task(
            self._monitor_stderr(),
            name=f"stderr-{self.platform_id}",
        )

    async def _monitor_stderr(self) -> None:
        """Read stderr for ready markers and error signals."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                if self._ready_re and self._ready_re.search(text):
                    self._ready.set()

                if text.startswith("{"):
                    self._handle_stderr_json(text)
                elif self._error_re and self._error_re.search(text):
                    self._last_error = text
                    logger.warning(
                        "Event source %s stderr error: %s",
                        self.platform_id, text,
                    )
                    if not self._ready.is_set():
                        self._ready.set()
                else:
                    logger.debug("Event source %s stderr: %s", self.platform_id, text)
        except asyncio.CancelledError:
            pass

    def _handle_stderr_json(self, text: str) -> None:
        """Parse structured JSON error from stderr (lark-cli error envelope)."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return
        if isinstance(data, dict) and data.get("ok") is False:
            error = data.get("error", {})
            if isinstance(error, dict):
                msg = str(error.get("message", ""))
                hint = str(error.get("hint", ""))
                self._last_error = f"{msg} ({hint})" if hint else msg
            elif error:
                self._last_error = str(error)
            if not self._ready.is_set():
                self._ready.set()

    async def _restart_with_backoff(self) -> None:
        """Restart the subprocess with exponential backoff."""
        self._restart_count += 1
        if self._restart_count > self._config.max_restarts:
            self._running = False
            logger.error(
                "Event source %s exceeded max restarts (%d), stopping",
                self.platform_id, self._config.max_restarts,
            )
            return
        backoff = min(
            self._config.restart_backoff_base_s * (2 ** (self._restart_count - 1)),
            self._config.max_restart_backoff_s,
        )
        logger.warning(
            "Event source %s restarting in %.1fs (attempt %d/%d)",
            self.platform_id, backoff,
            self._restart_count, self._config.max_restarts,
        )
        await asyncio.sleep(backoff)
        if not self._running:
            return
        try:
            await self._spawn_process()
            if self._ready_re:
                await asyncio.wait_for(
                    self._ready.wait(), timeout=self._config.ready_timeout_s,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "Event source %s ready timeout on restart attempt %d",
                self.platform_id, self._restart_count,
            )
        except (FileNotFoundError, OSError) as exc:
            logger.error("Event source %s spawn failed: %s", self.platform_id, exc)

    async def _kill_process(self) -> None:
        """Terminate the subprocess gracefully, then forcefully."""
        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None

        proc = self._proc
        self._proc = None
        if proc is None or proc.returncode is not None:
            return

        if proc.stdin and not proc.stdin.is_closing():
            proc.stdin.close()
            with suppress(Exception):
                await proc.stdin.drain()

        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                with suppress(ProcessLookupError):
                    proc.kill()
                with suppress(ProcessLookupError):
                    await proc.wait()

    def _parse_ndjson_line(self, line: bytes) -> BackendEvent | None:
        """Parse one NDJSON line into a BackendEvent."""
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.debug(
                "Event source %s: skipping non-JSON line: %.100s",
                self.platform_id, text,
            )
            return None
        if not isinstance(data, dict):
            return None
        return BackendEvent(
            event_id=str(data.get("event_id", "")),
            event_type=str(data.get("type") or data.get("event_type", "")),
            platform_id=self.platform_id,
            payload=data,
        )
