# Copyright (c) Alibaba, Inc. and its affiliates.
"""Cross-platform daemon IPC transport.

Unix (macOS/Linux): Unix Domain Socket via asyncio.start_unix_server / open_unix_connection
Windows: TCP loopback (127.0.0.1:dynamic-port) via asyncio.start_server / open_connection
"""
from __future__ import annotations

import abc
import asyncio
import socket
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Tuple


RPC_STREAM_LIMIT = 4 * 1024 * 1024


class DaemonTransport(abc.ABC):
    """Abstract base for daemon IPC transport."""

    @abc.abstractmethod
    async def start_server(
        self,
        client_connected_cb: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Any],
        runtime_dir: Path,
    ) -> asyncio.AbstractServer:
        """Start listening for client connections. Returns the server instance."""

    @abc.abstractmethod
    async def connect(self, runtime_dir: Path) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open a client connection to the daemon."""

    @abc.abstractmethod
    def probe_healthy(self, runtime_dir: Path) -> bool:
        """Synchronous quick health check — returns True if the daemon is reachable."""

    @abc.abstractmethod
    def readiness_path(self, runtime_dir: Path) -> Path:
        """Artifact file the daemon publishes when it is ready to serve."""

    @abc.abstractmethod
    def cleanup(self, runtime_dir: Path) -> None:
        """Remove transport artifacts (socket file, port file, etc.)."""


class UnixSocketTransport(DaemonTransport):
    """Unix Domain Socket transport (macOS/Linux)."""

    def _sock_path(self, runtime_dir: Path) -> Path:
        return runtime_dir / "leapd.sock"

    async def start_server(
        self,
        client_connected_cb: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Any],
        runtime_dir: Path,
    ) -> asyncio.AbstractServer:
        sock_path = self._sock_path(runtime_dir)
        return await asyncio.start_unix_server(
            client_connected_cb,
            path=str(sock_path),
            limit=RPC_STREAM_LIMIT,
        )

    async def connect(self, runtime_dir: Path) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        sock_path = self._sock_path(runtime_dir)
        return await asyncio.open_unix_connection(str(sock_path), limit=RPC_STREAM_LIMIT)

    def probe_healthy(self, runtime_dir: Path) -> bool:
        sock_path = self._sock_path(runtime_dir)
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect(str(sock_path))
            s.close()
            return True
        except (OSError, socket.timeout):
            return False

    def readiness_path(self, runtime_dir: Path) -> Path:
        return self._sock_path(runtime_dir)

    def cleanup(self, runtime_dir: Path) -> None:
        self._sock_path(runtime_dir).unlink(missing_ok=True)


class TcpLoopbackTransport(DaemonTransport):
    """TCP loopback transport for Windows (127.0.0.1:dynamic-port)."""

    def _port_path(self, runtime_dir: Path) -> Path:
        return runtime_dir / "leapd.port"

    def _read_port(self, runtime_dir: Path) -> int:
        port_path = self._port_path(runtime_dir)
        try:
            port = int(port_path.read_text(encoding="utf-8").strip())
        except ValueError as exc:
            raise OSError(f"invalid daemon port file: {port_path}") from exc
        if port <= 0 or port > 65535:
            raise OSError(f"invalid daemon port value: {port}")
        return port

    async def start_server(
        self,
        client_connected_cb: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Any],
        runtime_dir: Path,
    ) -> asyncio.AbstractServer:
        server = await asyncio.start_server(
            client_connected_cb,
            host="127.0.0.1",
            port=0,
            limit=RPC_STREAM_LIMIT,
        )
        # Extract the assigned port from the server socket and persist it.
        addr = server.sockets[0].getsockname()
        port = addr[1]
        port_path = self._port_path(runtime_dir)
        port_path.write_text(str(port), encoding="utf-8")
        return server

    async def connect(self, runtime_dir: Path) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        port = self._read_port(runtime_dir)
        return await asyncio.open_connection("127.0.0.1", port, limit=RPC_STREAM_LIMIT)

    def probe_healthy(self, runtime_dir: Path) -> bool:
        try:
            port = self._read_port(runtime_dir)
        except (OSError, ValueError):
            return False
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect(("127.0.0.1", port))
            s.close()
            return True
        except (OSError, socket.timeout):
            return False

    def readiness_path(self, runtime_dir: Path) -> Path:
        return self._port_path(runtime_dir)

    def cleanup(self, runtime_dir: Path) -> None:
        self._port_path(runtime_dir).unlink(missing_ok=True)


def get_transport() -> DaemonTransport:
    """Return the platform-appropriate daemon transport."""
    if sys.platform == "win32":
        return TcpLoopbackTransport()
    return UnixSocketTransport()
