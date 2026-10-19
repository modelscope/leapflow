# Copyright (c) Alibaba, Inc. and its affiliates.
"""Node-local leased sessions: one lease/deadman mechanism, two kinds of body.

A *session* is the node's record that a host connection holds some devices and
keeps proving it is alive. What runs under the lease is the session *body*:

- ``ControlLoopBody`` (``control.open``) -- a ControlBus loop running on the
  node at a fixed frequency.
- a skill run (``skill.execute``) -- see ``lhp/skill_methods.py``.

Both share the same rules, owned here exactly once:

- **Lease / deadman.** ``lease_s`` is clamped to [0.2, 10] s (default 1.0). The
  owner renews before expiry; on expiry or disconnect the node calls the body's
  ``lose`` (a control loop applies its ``comm_loss_policy``; a skill run stops
  and halts whatever it moved).
- **Device mutex.** A device belongs to at most one live session, and to none
  while a node-local holder outside any lease (``device_busy``) has it.
- **Ownership.** Only the connection that created a session may renew, update,
  or close it; any connection may read status.

A body that ends on its own (a skill run finishing) releases its devices at
once: ``live`` is false the moment ``body.finished`` is, so the mutex never
outlives the work it protects.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, runtime_checkable

from leapflow.hardware.lhp.protocol import FAILURE_DEVICE_BUSY

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

MIN_LEASE_S = 0.2
MAX_LEASE_S = 10.0
DEFAULT_LEASE_S = 1.0
FINISH_POLL_S = 0.25
"""Upper bound on how long a finished body keeps its lease bookkeeping alive."""
VALID_COMM_LOSS_POLICIES = frozenset({"halt", "hold"})


# ── Session body contract ─────────────────────────────────────────────────


@runtime_checkable
class SessionBody(Protocol):
    """What runs under a lease. The session manager owns lease, mutex, and ownership."""

    @property
    def kind(self) -> str: ...

    @property
    def device_ids(self) -> tuple[str, ...]: ...

    @property
    def finished(self) -> bool:
        """True once the body ended on its own; its devices are free again."""
        ...

    async def close(self, reason: str) -> dict[str, Any]:
        """Orderly end requested by the owner; returns end-of-session stats."""
        ...

    async def lose(self, reason: str) -> None:
        """The owner went silent (lease expiry, disconnect): fail safe."""
        ...

    def describe(self) -> dict[str, Any]:
        """Body-specific status fields."""
        ...


# ── Session record ────────────────────────────────────────────────────────


@dataclass
class LeasedSession:
    """One live (or just-ended) lease on the node."""

    control_id: str
    connection_id: str
    body: SessionBody
    lease_s: float
    started_at: float = field(default_factory=time.monotonic)
    last_renewed_at: float = field(default_factory=time.monotonic)
    ended: bool = False
    _watchdog: asyncio.Task[None] | None = field(default=None, repr=False)

    @property
    def live(self) -> bool:
        return not self.ended and not self.body.finished

    @property
    def expires_in_s(self) -> float:
        return max(0.0, self.last_renewed_at + self.lease_s - time.monotonic())

    def lease(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "lease_s": self.lease_s,
            "expires_in_s": self.expires_in_s,
        }

    def to_status(self) -> dict[str, Any]:
        return {
            **self.lease(),
            "kind": self.body.kind,
            "device_ids": list(self.body.device_ids),
            "running": self.live,
            "uptime_s": time.monotonic() - self.started_at,
            **self.body.describe(),
        }


# ── Control-loop body ─────────────────────────────────────────────────────


class ControlLoopBody:
    """A node-local ControlBus loop on one device (``control.open``)."""

    kind = "control"

    def __init__(
        self,
        registry: Any,
        *,
        operation_id: str,
        device_id: str,
        mode: str,
        frequency_hz: float,
        channels: tuple[str, ...],
        comm_loss_policy: str,
    ) -> None:
        self._registry = registry
        self._operation_id = operation_id
        self._device_id = device_id
        self._mode = mode
        self._frequency_hz = frequency_hz
        self._channels = channels
        self._comm_loss_policy = comm_loss_policy
        self._started_at = time.monotonic()

    @property
    def device_ids(self) -> tuple[str, ...]:
        return (self._device_id,)

    @property
    def finished(self) -> bool:
        return False  # a control loop runs until its owner closes it or goes silent

    def start(self) -> None:
        from leapflow.hardware.control_bus import ControlBusConfig

        self._registry.start_control_loop(
            self._device_id,
            _HoldPolicy(self._mode),
            config=ControlBusConfig(frequency_hz=self._frequency_hz),
        )

    async def close(self, reason: str) -> dict[str, Any]:
        stats = self._registry.stop_control_loop(self._device_id)
        result: dict[str, Any] = {
            "status": "stopped",
            "reason": reason,
            "uptime_s": time.monotonic() - self._started_at,
        }
        if isinstance(stats, dict):
            result["stats"] = stats.get("stats", {})
        await self._record_verdict(reason, result)
        return result

    async def lose(self, reason: str) -> None:
        self._registry.stop_control_loop(self._device_id)
        if self._comm_loss_policy != "halt":
            logger.info(
                "control: hold policy, loop stopped on %s after %s", self._device_id, reason
            )
            return
        try:
            transport = await self._registry.transport(self._device_id)
            await transport.halt()
            logger.warning("control: halted %s after %s", self._device_id, reason)
        except Exception as exc:  # noqa: BLE001 - a failed halt is logged, never raised into the watchdog
            logger.error("control: halt of %s failed: %s", self._device_id, exc, exc_info=True)

    def describe(self) -> dict[str, Any]:
        return {
            "device_id": self._device_id,
            "mode": self._mode,
            "frequency_hz": self._frequency_hz,
            "channels": list(self._channels),
            "comm_loss_policy": self._comm_loss_policy,
        }

    async def _record_verdict(self, reason: str, result: Mapping[str, Any]) -> None:
        """Write a session-level OperationVerdict; only clean ends count toward trust."""
        store = getattr(self._registry, "evidence_store", None)
        if store is None:
            return
        try:
            from leapflow.hardware.verification import (
                EvidenceBundle,
                OperationVerdict,
                VerdictStatus,
            )

            normal = reason in {"close", "shutdown"}
            stats = result.get("stats", {})
            uptime = float(result.get("uptime_s", 0.0))
            detail = (
                f"Control session {self._operation_id[:8]} on {self._device_id}; "
                f"mode={self._mode}, freq={self._frequency_hz}Hz; "
                f"reason={reason}, uptime={uptime:.1f}s"
            )
            if stats:
                detail += f"; cycles={stats.get('cycle_count', 0)}"
            verdict = OperationVerdict(
                status=(VerdictStatus.SUCCESS if normal else VerdictStatus.INCONCLUSIVE).value,
                confidence=0.8 if normal else 0.3,
                detail=detail,
            )
            bundle = EvidenceBundle(
                operation_id=self._operation_id,
                device_id=self._device_id,
                channel_id="control_session",
                metadata={**self.describe(), "reason": reason, "uptime_s": uptime, "stats": stats},
            )
            await store.record(bundle, verdict)
        except Exception as exc:  # noqa: BLE001 - telemetry never fails a session end
            logger.debug("control verdict recording failed: %s", exc)


class _HoldPolicy:
    """Hold-position policy installed when a control loop opens (no-op compute)."""

    def __init__(self, mode: str) -> None:
        self._policy_id = f"hold_{mode}"

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def compute(self, state: Any) -> Any:
        from leapflow.hardware.control_bus import ControlCommand

        return ControlCommand(joint_commands={}, metadata={})

    def reset(self) -> None:
        pass


# ── Session manager ───────────────────────────────────────────────────────


class NodeSessions:
    """Lease, device mutex, and ownership for every session on one node."""

    def __init__(self, registry: Any, *, device_busy: Callable[[str], bool] | None = None) -> None:
        self._registry = registry
        self._device_busy = device_busy
        self._sessions: dict[str, LeasedSession] = {}

    # ── Mutex ──

    def holds_device(self, device_id: str) -> bool:
        """Whether a live session holds *device_id*."""
        return any(
            device_id in session.body.device_ids
            for session in self._sessions.values()
            if session.live
        )

    def busy_device(self, device_ids: Iterable[str]) -> str:
        """The first of *device_ids* that is held by a session or a node-local holder."""
        for device_id in device_ids:
            if self.holds_device(device_id):
                return device_id
            if self._device_busy is not None and self._device_busy(device_id):
                return device_id
        return ""

    # ── Lifecycle ──

    def adopt(self, body: SessionBody, *, lease_s: Any, connection_id: str) -> LeasedSession:
        """Put an already started *body* under a lease owned by *connection_id*.

        The caller checks ``busy_device`` first, with no await in between, so the
        check and the adoption are atomic on the event loop.
        """
        self._purge()
        session = LeasedSession(
            control_id=str(uuid.uuid4()),
            connection_id=connection_id,
            body=body,
            lease_s=parse_lease(lease_s),
        )
        self._sessions[session.control_id] = session
        session._watchdog = asyncio.get_running_loop().create_task(
            self._watchdog(session), name=f"lease-{session.control_id[:8]}"
        )
        logger.info(
            "lease %s opened: %s on %s (lease %.1fs)",
            session.control_id[:8],
            body.kind,
            ",".join(body.device_ids),
            session.lease_s,
        )
        return session

    async def open(self, params: Mapping[str, Any], *, connection_id: str) -> dict[str, Any]:
        """``control.open``: precheck, start a control loop, and lease it."""
        device_id = str(params.get("device_id") or "")
        comm_loss = str(params.get("comm_loss_policy") or "halt")
        if comm_loss not in VALID_COMM_LOSS_POLICIES:
            return _error(
                "invalid_comm_loss_policy",
                f"comm_loss_policy must be one of {sorted(VALID_COMM_LOSS_POLICIES)}",
            )
        try:
            frequency_hz = _parse_frequency(params.get("frequency_hz"))
        except ValueError as exc:
            return _error("invalid_frequency", str(exc))
        context = self._registry.context(device_id)
        if context is None:
            return _error("unknown_device", f"unknown device {device_id!r}")
        if self.busy_device((device_id,)):
            return _error(FAILURE_DEVICE_BUSY, f"device {device_id!r} is held by another session")
        precheck = await self._precheck(context)
        if precheck is not None:
            return precheck
        control_id = str(uuid.uuid4())
        body = ControlLoopBody(
            self._registry,
            operation_id=control_id,
            device_id=device_id,
            mode=str(params.get("mode") or ""),
            frequency_hz=frequency_hz,
            channels=_parse_channels(params.get("channels")),
            comm_loss_policy=comm_loss,
        )
        # Re-check after the awaited precheck: another open may have won the device.
        if self.busy_device((device_id,)):
            return _error(FAILURE_DEVICE_BUSY, f"device {device_id!r} is held by another session")
        try:
            body.start()
        except (RuntimeError, ValueError) as exc:
            return _error("control_start_failed", str(exc))
        session = self.adopt(body, lease_s=params.get("lease_s"), connection_id=connection_id)
        return {"ok": True, **session.lease()}

    async def renew(self, params: Mapping[str, Any], *, connection_id: str) -> dict[str, Any]:
        session, refusal = self._owned(params, connection_id, "renew")
        if refusal is not None:
            return refusal
        session.last_renewed_at = time.monotonic()
        return {"ok": True, **session.lease()}

    async def update(self, params: Mapping[str, Any], *, connection_id: str) -> dict[str, Any]:
        """Targets/waypoints update; it also renews the lease."""
        session, refusal = self._owned(params, connection_id, "update")
        if refusal is not None:
            return refusal
        session.last_renewed_at = time.monotonic()
        return {"ok": True, **session.lease()}

    async def close(self, params: Mapping[str, Any], *, connection_id: str) -> dict[str, Any]:
        control_id = str(params.get("control_id") or "")
        session = self._sessions.get(control_id)
        if session is None:
            return _error("unknown_control_session", f"no session {control_id!r}")
        if session.connection_id != connection_id:
            return _error(
                "not_session_owner", "only the connection that created the session may close it"
            )
        if session.ended:
            return {"ok": True, "control_id": control_id, "status": "already_stopped"}
        self._end(session)
        stats = await session.body.close("close")
        return {"ok": True, "control_id": control_id, **stats}

    async def status(self, params: Mapping[str, Any]) -> dict[str, Any]:
        device_id = str(params.get("device_id") or "")
        sessions = [
            session
            for session in self._sessions.values()
            if session.live and (not device_id or device_id in session.body.device_ids)
        ]
        return {"ok": True, "sessions": [session.to_status() for session in sessions]}

    async def on_disconnect(self, connection_id: str) -> None:
        """A connection dropped: every live session it owned fails safe."""
        for session in list(self._sessions.values()):
            if session.connection_id == connection_id and session.live:
                logger.warning(
                    "lease %s: owner connection dropped; failing safe", session.control_id[:8]
                )
                await self._lose(session, "disconnect")

    async def shutdown(self) -> None:
        """Close every live session (node shutdown)."""
        for session in list(self._sessions.values()):
            if session.live:
                self._end(session)
                await session.body.close("shutdown")
        self._sessions.clear()

    # ── Internals ──

    def _owned(
        self, params: Mapping[str, Any], connection_id: str, verb: str
    ) -> tuple[LeasedSession, None] | tuple[None, dict[str, Any]]:
        control_id = str(params.get("control_id") or "")
        session = self._sessions.get(control_id)
        if session is None:
            return None, _error("unknown_control_session", f"no session {control_id!r}")
        if session.connection_id != connection_id:
            return None, _error(
                "not_session_owner", f"only the connection that created the session may {verb} it"
            )
        if not session.live:
            return None, _error("session_halted", "session has already ended")
        return session, None

    def _end(self, session: LeasedSession) -> None:
        session.ended = True
        watchdog, session._watchdog = session._watchdog, None
        if watchdog is not None and not watchdog.done() and watchdog is not asyncio.current_task():
            watchdog.cancel()

    async def _lose(self, session: LeasedSession, reason: str) -> None:
        if session.ended:
            return
        self._end(session)
        await session.body.lose(reason)

    def _purge(self) -> None:
        for control_id in [cid for cid, s in self._sessions.items() if not s.live]:
            del self._sessions[control_id]

    async def _watchdog(self, session: LeasedSession) -> None:
        try:
            while not session.ended:
                if session.body.finished:
                    self._end(session)
                    return
                remaining = session.expires_in_s
                if remaining <= 0:
                    logger.warning(
                        "lease %s expired on %s",
                        session.control_id[:8],
                        ",".join(session.body.device_ids),
                    )
                    await self._lose(session, "lease_expired")
                    return
                await asyncio.sleep(min(remaining, session.lease_s / 2, FINISH_POLL_S))
        except asyncio.CancelledError:
            return

    async def _precheck(self, context: Any) -> dict[str, Any] | None:
        """Interlocks (fail-closed) and degradation for a control loop."""
        from leapflow.hardware.interlocks import failed_interlocks

        for channel in context.channels:
            if not channel.is_writable:
                continue
            failed = await failed_interlocks(self._registry, context, channel)
            if failed:
                return _error(
                    "interlock_unsatisfied", f"interlocks not satisfied: {', '.join(failed)}"
                )
        is_degraded = getattr(self._registry, "is_device_degraded", None)
        if callable(is_degraded) and is_degraded(context.device_id):
            return _error("device_degraded", f"device {context.device_id!r} is in a degraded state")
        return None


# ── Helpers ───────────────────────────────────────────────────────────────


def parse_lease(value: Any) -> float:
    """Clamp a requested lease to the protocol bounds; junk means the default."""
    try:
        lease = float(value or DEFAULT_LEASE_S)
    except (TypeError, ValueError):
        lease = DEFAULT_LEASE_S
    if not math.isfinite(lease):
        lease = DEFAULT_LEASE_S
    return max(MIN_LEASE_S, min(MAX_LEASE_S, lease))


def _parse_frequency(value: Any) -> float:
    try:
        hz = float(value or 100.0)
    except (TypeError, ValueError):
        hz = 100.0
    if hz <= 0 or not math.isfinite(hz):
        raise ValueError("frequency_hz must be a positive finite number")
    return hz


def _parse_channels(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(c) for c in value if c)
    return ()


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "failure_code": code, "error": message}


__all__ = [
    "DEFAULT_LEASE_S",
    "MAX_LEASE_S",
    "MIN_LEASE_S",
    "ControlLoopBody",
    "LeasedSession",
    "NodeSessions",
    "SessionBody",
    "parse_lease",
]
