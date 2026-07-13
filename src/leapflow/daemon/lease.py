"""Client lease files for leapd multi-client lifecycle tracking."""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

_CLIENTS_DIR = "clients"
_DEFAULT_LEASE_TTL_S = 120.0
_DEFAULT_TOUCH_INTERVAL_S = 30.0


@dataclass(frozen=True)
class ClientLeaseSnapshot:
    """Immutable view of one active client lease."""

    client_id: str
    pid: int
    kind: str
    state: str
    session_id: str
    started_at: float
    last_seen_at: float
    path: Path


def default_lease_ttl_s() -> float:
    """Return the maximum age for a live client lease."""
    raw = os.getenv("LEAPFLOW_CLIENT_LEASE_TTL_S", str(_DEFAULT_LEASE_TTL_S)).strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_LEASE_TTL_S


def read_active_client_leases(
    run_dir: Path,
    *,
    now: float | None = None,
    ttl_s: float | None = None,
) -> list[ClientLeaseSnapshot]:
    """Return currently active client leases and remove stale entries."""
    clients_dir = run_dir / _CLIENTS_DIR
    if not clients_dir.exists():
        return []
    current = time.time() if now is None else now
    max_age = default_lease_ttl_s() if ttl_s is None else max(1.0, ttl_s)
    active: list[ClientLeaseSnapshot] = []
    for path in clients_dir.glob("*.json"):
        snapshot = _read_lease(path)
        if snapshot is None:
            path.unlink(missing_ok=True)
            continue
        if current - snapshot.last_seen_at > max_age or not _process_alive(snapshot.pid):
            path.unlink(missing_ok=True)
            continue
        active.append(snapshot)
    return active


def has_active_client_leases(
    run_dir: Path,
    *,
    now: float | None = None,
    ttl_s: float | None = None,
) -> bool:
    """Return True when any live client lease exists."""
    return bool(read_active_client_leases(run_dir, now=now, ttl_s=ttl_s))


class ClientLease:
    """Maintain one client lease file while a TUI client is alive."""

    def __init__(
        self,
        run_dir: Path,
        *,
        kind: str,
        session_id: str = "",
        state: str = "idle",
        touch_interval_s: float = _DEFAULT_TOUCH_INTERVAL_S,
    ) -> None:
        self._run_dir = run_dir
        self._client_id = uuid.uuid4().hex
        self._kind = kind
        self.session_id = session_id
        self.state = state
        self._started_at = time.time()
        self._touch_interval_s = max(1.0, touch_interval_s)
        self._task: asyncio.Task[None] | None = None

    @property
    def client_id(self) -> str:
        """Return this lease's stable client id."""
        return self._client_id

    @property
    def path(self) -> Path:
        """Return the lease file path."""
        return self._run_dir / _CLIENTS_DIR / f"{self._client_id}.json"

    async def start(self) -> None:
        """Create the lease and start periodic touch updates."""
        await self.touch()
        self._task = asyncio.create_task(self._touch_loop())

    async def stop(self) -> None:
        """Stop updating and remove the lease."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await asyncio.to_thread(self.path.unlink, True)

    async def touch(
        self,
        *,
        state: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Update the lease timestamp and optional mutable state."""
        if state is not None:
            self.state = state
        if session_id is not None:
            self.session_id = session_id
        await asyncio.to_thread(self._write_sync)

    async def _touch_loop(self) -> None:
        while True:
            await asyncio.sleep(self._touch_interval_s)
            await self.touch()

    def _write_sync(self) -> None:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "client_id": self._client_id,
            "pid": os.getpid(),
            "kind": self._kind,
            "state": self.state,
            "session_id": self.session_id,
            "cwd": os.getcwd(),
            "started_at": self._started_at,
            "last_seen_at": time.time(),
        }
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)


def _read_lease(path: Path) -> ClientLeaseSnapshot | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ClientLeaseSnapshot(
            client_id=str(payload["client_id"]),
            pid=int(payload["pid"]),
            kind=str(payload.get("kind") or "unknown"),
            state=str(payload.get("state") or "idle"),
            session_id=str(payload.get("session_id") or ""),
            started_at=float(payload.get("started_at") or 0.0),
            last_seen_at=float(payload.get("last_seen_at") or 0.0),
            path=path,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
