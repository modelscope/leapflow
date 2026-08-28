"""Restricted Node subprocess host for executable DSH plugin bridges."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
import sys
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from leapflow.plugins.dsh.capabilities import DshCapabilityBroker, DshCapabilityError
from leapflow.plugins.dsh.protocol import (
    CapabilityRequest,
    DshProtocolError,
    DshRequest,
    DshResponse,
    MAX_LINE_BYTES,
    capability_response,
    parse_message,
)
from leapflow.utils.process_group import ProcessGroup

logger = logging.getLogger(__name__)
_NODE_VERSION = re.compile(r"^v(?P<major>\d+)(?:\.\d+){2}$")
_MIN_NODE_MAJOR = 22


class DshRuntimeUnavailable(RuntimeError):
    """The restricted Node runtime cannot start in this environment."""


class DshNodeHost:
    """Own one restricted Node worker and proxy versioned NDJSON requests.

    The host serializes requests per process. Capability requests are serviced
    while the outer tool request is pending, then the matching response resumes.
    stdout is protocol-only; stderr is drained independently into a bounded tail.
    """

    def __init__(
        self,
        source_root: str | Path,
        *,
        source_kind: str,
        entry_point: str,
        broker: DshCapabilityBroker | None = None,
        invoke_timeout_s: float = 30.0,
        discovery_timeout_s: float = 10.0,
        max_line_bytes: int = MAX_LINE_BYTES,
        max_stderr_bytes: int = 64_000,
        max_memory_mb: int = 128,
    ) -> None:
        self._source_root = Path(source_root).expanduser().resolve()
        self._source_kind = str(source_kind)
        self._entry_point = str(entry_point)
        self._broker = broker or DshCapabilityBroker()
        self._invoke_timeout_s = max(0.1, float(invoke_timeout_s))
        self._discovery_timeout_s = max(0.1, float(discovery_timeout_s))
        self._max_line_bytes = max(1024, int(max_line_bytes))
        self._max_stderr_bytes = max(1024, int(max_stderr_bytes))
        self._max_memory_mb = max(32, int(max_memory_mb))
        self._proc: asyncio.subprocess.Process | None = None
        self._group: ProcessGroup | None = None
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_chunks: deque[bytes] = deque()
        self._stderr_size = 0

    @property
    def stderr_tail(self) -> str:
        return b"".join(self._stderr_chunks).decode("utf-8", errors="replace")

    async def start(self) -> None:
        if self._proc is not None:
            return
        node = shutil.which("node")
        if not node:
            raise DshRuntimeUnavailable("Node.js >=22 is required for DSH plugins")
        major = await _node_major(node)
        if major < _MIN_NODE_MAJOR:
            raise DshRuntimeUnavailable(
                f"Node.js >={_MIN_NODE_MAJOR} is required; found major version {major}"
            )
        worker = Path(__file__).with_name("node_worker.js").resolve()
        if not self._source_root.is_dir():
            raise DshRuntimeUnavailable(f"DSH source root is not a directory: {self._source_root}")
        entry = (self._source_root / self._entry_point).resolve()
        try:
            entry.relative_to(self._source_root)
        except ValueError as exc:
            raise DshRuntimeUnavailable("DSH entry point escapes the source root") from exc
        if not entry.is_file():
            raise DshRuntimeUnavailable(f"DSH runtime entry does not exist: {entry}")

        env = _minimal_environment()
        env["LEAPFLOW_DSH_MAX_LINE_BYTES"] = str(self._max_line_bytes)
        env["LEAPFLOW_DSH_CAPABILITY_TIMEOUT_MS"] = str(
            int(max(self._invoke_timeout_s, 1.0) * 1000)
        )
        args = [
            node,
            "--permission",
            f"--allow-fs-read={worker}",
            f"--allow-fs-read={self._source_root}",
            "--disable-sigusr1",
            f"--max-old-space-size={self._max_memory_mb}",
            str(worker),
            str(self._source_root),
            self._source_kind,
            self._entry_point,
        ]
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":  # pragma: no cover - Windows-specific
            import subprocess

            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(self._source_root),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=self._max_line_bytes + 1,
                **kwargs,
            )
        except OSError as exc:
            raise DshRuntimeUnavailable(f"Cannot start restricted Node worker: {exc}") from exc
        self._group = ProcessGroup()
        self._group.attach(self._proc.pid)
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def discover(self) -> DshResponse:
        return await self._request("discover", {}, timeout_s=self._discovery_timeout_s)

    async def invoke(self, tool_name: str, arguments: dict[str, Any]) -> DshResponse:
        return await self._request(
            "invoke",
            {"tool_name": tool_name, "arguments": dict(arguments)},
            timeout_s=self._invoke_timeout_s,
        )

    async def _request(
        self, method: str, payload: dict[str, Any], *, timeout_s: float
    ) -> DshResponse:
        if self._proc is None:
            await self.start()
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            return DshResponse(request_id="", ok=False, error="DSH worker is not running")
        request_id = uuid.uuid4().hex
        request = DshRequest(request_id=request_id, method=method, payload=payload)
        encoded_request = (request.to_json() + "\n").encode("utf-8")
        if len(encoded_request) > self._max_line_bytes:
            return self._failure(
                request_id,
                "request_too_large",
                f"DSH request exceeds {self._max_line_bytes} bytes",
            )
        async with self._lock:
            try:
                proc.stdin.write(encoded_request)
                await proc.stdin.drain()
                deadline = asyncio.get_running_loop().time() + timeout_s
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                    if not line:
                        return self._failure(
                            request_id,
                            "worker_closed",
                            f"DSH worker closed unexpectedly. {self.stderr_tail}".strip(),
                        )
                    message = parse_message(line, max_bytes=self._max_line_bytes)
                    message_type = message.get("type")
                    if message_type == "capability_request":
                        capability = CapabilityRequest.from_message(message)
                        if capability.parent_request_id != request_id:
                            raise DshProtocolError(
                                "capability request does not belong to the active tool invocation"
                            )
                        await asyncio.wait_for(
                            self._handle_capability(capability, proc),
                            timeout=remaining,
                        )
                        continue
                    response = DshResponse.from_message(message)
                    if response.request_id != request_id:
                        raise DshProtocolError(
                            f"response id mismatch: expected {request_id}, got {response.request_id}"
                        )
                    return response
            except TimeoutError:
                await self._terminate()
                return self._failure(
                    request_id,
                    "timeout",
                    f"DSH {method} timed out after {timeout_s}s",
                )
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                await self._terminate()
                return self._failure(request_id, "communication_error", str(exc))
            except (DshProtocolError, ValueError) as exc:
                await self._terminate()
                return self._failure(request_id, "protocol_error", str(exc))

    async def _handle_capability(
        self,
        capability: CapabilityRequest,
        proc: asyncio.subprocess.Process,
    ) -> None:
        if proc.stdin is None:
            return
        try:
            result = await self._broker.dispatch(
                capability.capability, capability.arguments
            )
            encoded = capability_response(
                capability.request_id, ok=True, result=result
            )
        except DshCapabilityError as exc:
            encoded = capability_response(
                capability.request_id,
                ok=False,
                error=str(exc),
                error_type="capability_denied",
            )
        except Exception as exc:  # noqa: BLE001 - capability boundary fails closed
            logger.warning("DSH capability failed", exc_info=True)
            encoded = capability_response(
                capability.request_id,
                ok=False,
                error=str(exc),
                error_type="capability_error",
            )
        proc.stdin.write((encoded + "\n").encode("utf-8"))
        await proc.stdin.drain()

    async def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while True:
            chunk = await proc.stderr.read(4096)
            if not chunk:
                return
            self._stderr_chunks.append(chunk)
            self._stderr_size += len(chunk)
            while self._stderr_size > self._max_stderr_bytes and self._stderr_chunks:
                overflow = self._stderr_size - self._max_stderr_bytes
                oldest = self._stderr_chunks[0]
                if len(oldest) <= overflow:
                    self._stderr_chunks.popleft()
                    self._stderr_size -= len(oldest)
                else:
                    self._stderr_chunks[0] = oldest[overflow:]
                    self._stderr_size -= overflow

    async def stop(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.returncode is None and proc.stdin is not None and proc.stdout is not None:
            try:
                await self._request("shutdown", {}, timeout_s=2.0)
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except (TimeoutError, OSError):
                await self._terminate()
        if proc.returncode is None:
            await self._terminate()
        if self._stderr_task is not None:
            try:
                await asyncio.wait_for(self._stderr_task, timeout=1.0)
            except TimeoutError:
                self._stderr_task.cancel()
                try:
                    await self._stderr_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
        self._proc = None
        self._group = None
        self._stderr_task = None

    async def _terminate(self) -> None:
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        group = self._group
        if group is not None and group.terminate(signal.SIGTERM):
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
                return
            except TimeoutError:
                if sys.platform != "win32":
                    try:
                        # The worker starts a fresh session, so its pid is also
                        # the process-group id. Escalate the whole group rather
                        # than orphaning descendants after a graceful timeout.
                        os.killpg(proc.pid, signal.SIGKILL)
                        await proc.wait()
                        return
                    except (ProcessLookupError, OSError):
                        pass
        try:
            proc.kill()
            await proc.wait()
        except (ProcessLookupError, OSError):
            pass

    def _failure(self, request_id: str, error_type: str, error: str) -> DshResponse:
        return DshResponse(
            request_id=request_id,
            ok=False,
            error=error,
            error_type=error_type,
        )


async def _node_major(node: str) -> int:
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            node,
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
    except TimeoutError as exc:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()
        raise DshRuntimeUnavailable("Node.js version preflight timed out") from exc
    except OSError as exc:
        raise DshRuntimeUnavailable(f"Cannot run Node.js preflight: {exc}") from exc
    version = stdout.decode("utf-8", errors="replace").strip()
    match = _NODE_VERSION.fullmatch(version)
    if proc.returncode != 0 or match is None:
        raise DshRuntimeUnavailable(f"Cannot determine Node.js version: {version!r}")
    return int(match.group("major"))


def _minimal_environment() -> dict[str, str]:
    allowed = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT")
    return {key: os.environ[key] for key in allowed if key in os.environ}
