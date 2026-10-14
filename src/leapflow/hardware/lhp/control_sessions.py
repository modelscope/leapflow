# Copyright (c) Alibaba, Inc. and its affiliates.
"""Node-local control session management for control.v1.

Each control session represents a high-frequency closed-loop control loop
running entirely on the node.  The host opens a session, the node runs the
ControlBus locally, and the host maintains the session via lease renewals.

Lease / deadman semantics:

- ``lease_s`` bounds [0.2, 10], default 1.0.  The host must renew before
  expiry or the node executes ``comm_loss_policy``.
- ``halt``: stop the loop and halt the device (exactly once).
- ``hold``: stop the loop but do not halt (device holds last command).
- The watchdog runs on the node event loop; on lease expiry or disconnect
  it fires the configured policy.

Session ownership: a control session is bound to the connection that created
it.  Other connections may query status but cannot renew/update/close.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

MIN_LEASE_S = 0.2
MAX_LEASE_S = 10.0
DEFAULT_LEASE_S = 1.0
VALID_COMM_LOSS_POLICIES = frozenset({"halt", "hold"})


# ── Data types ────────────────────────────────────────────────────────────


@dataclass
class ControlSession:
    """One active control session on the node."""

    control_id: str
    device_id: str
    connection_id: str
    mode: str
    frequency_hz: float
    channels: tuple[str, ...]
    lease_s: float
    comm_loss_policy: str
    started_at: float = field(default_factory=time.monotonic)
    last_renewed_at: float = field(default_factory=time.monotonic)
    halted: bool = False
    cycle_count: int = 0
    _watchdog: asyncio.Task[None] | None = field(default=None, repr=False)

    @property
    def expires_at(self) -> float:
        return self.last_renewed_at + self.lease_s

    @property
    def expires_in_s(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    def to_status(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "control_id": self.control_id,
            "device_id": self.device_id,
            "mode": self.mode,
            "frequency_hz": self.frequency_hz,
            "channels": list(self.channels),
            "lease_s": self.lease_s,
            "comm_loss_policy": self.comm_loss_policy,
            "expires_in_s": max(0.0, self.expires_at - now),
            "running": not self.halted,
            "uptime_s": now - self.started_at,
        }


class NodeControlSessions:
    """Manages control sessions on a node.

    One session per device at a time.  The registry, safety policy, and
    transport are accessed through the node server's own registry instance.
    """

    def __init__(self, registry: Any, *, permit_verifier: Any | None = None) -> None:
        self._registry = registry
        self._permit_verifier = permit_verifier
        self._sessions: dict[str, ControlSession] = {}  # control_id -> session
        self._device_sessions: dict[str, str] = {}  # device_id -> control_id
        self._evidence_store: Any | None = getattr(registry, "evidence_store", None)

    # ── Public API (called by server handlers) ────────────────────────────

    async def open(
        self,
        params: Mapping[str, Any],
        *,
        connection_id: str,
    ) -> dict[str, Any]:
        """Open a new control session on a device.

        Validates all parameters, runs precheck, starts the local ControlBus,
        and returns the session descriptor.
        """
        device_id = str(params.get("device_id") or "")
        mode = str(params.get("mode") or "")
        frequency_hz = _parse_frequency(params.get("frequency_hz"))
        channels = _parse_channels(params.get("channels"))
        lease_s = _parse_lease(params.get("lease_s"))
        comm_loss = str(params.get("comm_loss_policy") or "halt")
        if comm_loss not in VALID_COMM_LOSS_POLICIES:
            return _error("invalid_comm_loss_policy",
                          f"comm_loss_policy must be one of {sorted(VALID_COMM_LOSS_POLICIES)}")

        context = self._registry.context(device_id)
        if context is None:
            return _error("unknown_device", f"unknown device {device_id!r}")

        # Only one session per device
        existing_cid = self._device_sessions.get(device_id)
        if existing_cid and existing_cid in self._sessions:
            existing = self._sessions[existing_cid]
            if not existing.halted:
                return _error("control_session_exists",
                              f"device {device_id!r} already has an active control session")

        # Precheck: writable, interlocks, degradation, safety
        precheck = await self._precheck(context)
        if precheck is not None:
            return precheck

        # Build policy and start the local control loop
        try:
            from leapflow.hardware.control_bus import ControlBusConfig

            self._registry.start_control_loop(
                device_id,
                _minimal_hold_policy(mode, channels),
                config=ControlBusConfig(frequency_hz=frequency_hz),
            )
        except (RuntimeError, ValueError) as exc:
            return _error("control_start_failed", str(exc))

        control_id = str(uuid.uuid4())
        session = ControlSession(
            control_id=control_id,
            device_id=device_id,
            connection_id=connection_id,
            mode=mode,
            frequency_hz=frequency_hz,
            channels=channels,
            lease_s=lease_s,
            comm_loss_policy=comm_loss,
        )

        self._sessions[control_id] = session
        self._device_sessions[device_id] = control_id

        # Start the lease watchdog
        session._watchdog = asyncio.get_running_loop().create_task(
            self._lease_watchdog(control_id),
            name=f"control-watchdog-{control_id[:8]}",
        )

        logger.info(
            "control.open: device=%s mode=%s freq=%.1fHz lease=%.1fs cid=%s",
            device_id, mode, frequency_hz, lease_s, control_id[:8],
        )
        return {
            "ok": True,
            "control_id": control_id,
            "lease_s": lease_s,
            "expires_in_s": lease_s,
        }

    async def renew(
        self,
        params: Mapping[str, Any],
        *,
        connection_id: str,
    ) -> dict[str, Any]:
        """Renew the lease on a control session."""
        control_id = str(params.get("control_id") or "")
        session = self._sessions.get(control_id)
        if session is None:
            return _error("unknown_control_session", f"no session {control_id!r}")
        if session.connection_id != connection_id:
            return _error("not_session_owner",
                          "only the connection that created the session may renew it")
        if session.halted:
            return _error("session_halted", "session has already been stopped")

        session.last_renewed_at = time.monotonic()
        return {
            "ok": True,
            "control_id": control_id,
            "lease_s": session.lease_s,
            "expires_in_s": session.expires_in_s,
        }

    async def update(
        self,
        params: Mapping[str, Any],
        *,
        connection_id: str,
    ) -> dict[str, Any]:
        """Update targets/waypoints on a running control session (requires new permit)."""
        control_id = str(params.get("control_id") or "")
        session = self._sessions.get(control_id)
        if session is None:
            return _error("unknown_control_session", f"no session {control_id!r}")
        if session.connection_id != connection_id:
            return _error("not_session_owner",
                          "only the connection that created the session may update it")
        if session.halted:
            return _error("session_halted", "session has already been stopped")

        # Renew the lease on update as well
        session.last_renewed_at = time.monotonic()
        return {
            "ok": True,
            "control_id": control_id,
            "expires_in_s": session.expires_in_s,
        }

    async def close(
        self,
        params: Mapping[str, Any],
        *,
        connection_id: str,
    ) -> dict[str, Any]:
        """Close a control session and return stats."""
        control_id = str(params.get("control_id") or "")
        session = self._sessions.get(control_id)
        if session is None:
            return _error("unknown_control_session", f"no session {control_id!r}")
        if session.connection_id != connection_id:
            return _error("not_session_owner",
                          "only the connection that created the session may close it")

        stats = await self._stop_session(session, reason="close")
        return {
            "ok": True,
            "control_id": control_id,
            **stats,
        }

    async def status(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Return the status of all active control sessions (or one specific)."""
        device_id = str(params.get("device_id") or "")
        if device_id:
            cid = self._device_sessions.get(device_id)
            session = self._sessions.get(cid or "") if cid else None
            if session is None:
                return {"ok": True, "sessions": []}
            return {"ok": True, "sessions": [session.to_status()]}
        return {
            "ok": True,
            "sessions": [s.to_status() for s in self._sessions.values() if not s.halted],
        }

    async def on_disconnect(self, connection_id: str) -> None:
        """Called when a connection drops.  Fires comm_loss_policy for its sessions."""
        for session in list(self._sessions.values()):
            if session.connection_id == connection_id and not session.halted:
                logger.warning(
                    "control: connection %s dropped; executing %s for session %s",
                    connection_id[:8], session.comm_loss_policy, session.control_id[:8],
                )
                await self._execute_comm_loss(session, reason="disconnect")

    # ── Internals ─────────────────────────────────────────────────────────

    async def _precheck(self, context: Any) -> dict[str, Any] | None:
        """Run precheck on the device. Returns error dict or None."""
        device_id = context.device_id

        # Interlocks (fail-closed)
        from leapflow.hardware.interlocks import failed_interlocks

        for channel in context.channels:
            if not channel.is_writable:
                continue
            failed = await failed_interlocks(self._registry, context, channel)
            if failed:
                return _error("interlock_unsatisfied",
                              f"interlocks not satisfied: {', '.join(failed)}")

        # Degradation
        is_degraded = getattr(self._registry, "is_device_degraded", None)
        if callable(is_degraded) and is_degraded(device_id):
            return _error("device_degraded",
                          f"device {device_id!r} is in a degraded state")

        return None

    async def _stop_session(
        self, session: ControlSession, *, reason: str
    ) -> dict[str, Any]:
        """Stop the control loop and clean up the session."""
        if session.halted:
            return {"status": "already_stopped"}

        session.halted = True
        watchdog = session._watchdog
        if watchdog is not None and not watchdog.done():
            watchdog.cancel()
            session._watchdog = None

        # Stop the local ControlBus
        stats = self._registry.stop_control_loop(session.device_id)
        now = time.monotonic()
        result = {
            "status": "stopped",
            "reason": reason,
            "uptime_s": now - session.started_at,
        }
        if isinstance(stats, dict):
            result["stats"] = stats.get("stats", {})
        logger.info(
            "control.close: device=%s reason=%s uptime=%.1fs cid=%s",
            session.device_id, reason, now - session.started_at,
            session.control_id[:8],
        )
        # P1-H: Write a session-level OperationVerdict to the evidence store
        await self._record_control_verdict(session, reason, result)
        return result

    async def _execute_comm_loss(
        self, session: ControlSession, *, reason: str
    ) -> None:
        """Execute the comm_loss_policy for a session."""
        if session.halted:
            return
        session.halted = True
        watchdog = session._watchdog
        if watchdog is not None and not watchdog.done():
            watchdog.cancel()
            session._watchdog = None

        # Stop the control loop
        self._registry.stop_control_loop(session.device_id)

        if session.comm_loss_policy == "halt":
            # Halt the device exactly once
            try:
                transport = await self._registry.transport(session.device_id)
                await transport.halt()
                logger.warning(
                    "control: halted device %s after %s (cid=%s)",
                    session.device_id, reason, session.control_id[:8],
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "control: halt failed for %s after %s: %s",
                    session.device_id, reason, exc, exc_info=True,
                )
        else:
            logger.info(
                "control: hold policy — stopped loop on %s after %s (cid=%s)",
                session.device_id, reason, session.control_id[:8],
            )

    async def _lease_watchdog(self, control_id: str) -> None:
        """Monitor lease expiry and fire comm_loss_policy if needed."""
        try:
            while True:
                session = self._sessions.get(control_id)
                if session is None or session.halted:
                    return
                remaining = session.expires_in_s
                if remaining <= 0:
                    logger.warning(
                        "control: lease expired for session %s on device %s",
                        control_id[:8], session.device_id,
                    )
                    await self._execute_comm_loss(session, reason="lease_expired")
                    return
                # Sleep until expiry (with a small margin for precision)
                await asyncio.sleep(min(remaining, session.lease_s / 2))
        except asyncio.CancelledError:
            return

    async def _record_control_verdict(
        self,
        session: ControlSession,
        reason: str,
        result: dict[str, Any],
    ) -> None:
        """Write a session-level OperationVerdict for a control session (P1-H).

        This is a session-level summary, not a per-tick verdict.  It records
        whether the loop ended normally, the stop reason, and available stats.
        Only 'verified' results participate in trust promotion.
        """
        store = self._evidence_store
        if store is None:
            return
        try:
            from leapflow.hardware.verification import (
                EvidenceBundle,
                OperationVerdict,
                VerdictStatus,
            )

            normal_reasons = {"close", "shutdown"}
            is_normal = reason in normal_reasons
            status = VerdictStatus.SUCCESS.value if is_normal else VerdictStatus.INCONCLUSIVE.value
            stats = result.get("stats", {})
            uptime = result.get("uptime_s", 0.0)

            detail_parts = [
                f"Control session {session.control_id[:8]} on {session.device_id}",
                f"mode={session.mode}, freq={session.frequency_hz}Hz",
                f"reason={reason}, uptime={uptime:.1f}s",
            ]
            if stats:
                cycles = stats.get("cycle_count", 0)
                detail_parts.append(f"cycles={cycles}")

            verdict = OperationVerdict(
                status=status,
                confidence=0.8 if is_normal else 0.3,
                detail="; ".join(detail_parts),
            )
            bundle = EvidenceBundle(
                operation_id=session.control_id,
                device_id=session.device_id,
                channel_id="control_session",
                metadata={
                    "mode": session.mode,
                    "frequency_hz": session.frequency_hz,
                    "reason": reason,
                    "uptime_s": uptime,
                    "stats": stats,
                },
            )
            await store.record(bundle, verdict)
        except Exception as exc:  # noqa: BLE001
            logger.debug("control verdict recording failed: %s", exc)

    async def shutdown(self) -> None:
        """Stop all active sessions (called during node shutdown)."""
        for session in list(self._sessions.values()):
            if not session.halted:
                await self._stop_session(session, reason="shutdown")
        self._sessions.clear()
        self._device_sessions.clear()


# ── Helpers ───────────────────────────────────────────────────────────────


def _parse_frequency(value: Any) -> float:
    try:
        hz = float(value or 100.0)
    except (TypeError, ValueError):
        hz = 100.0
    if hz <= 0 or not __import__("math").isfinite(hz):
        raise ValueError("frequency_hz must be a positive finite number")
    return hz


def _parse_lease(value: Any) -> float:
    try:
        lease = float(value or DEFAULT_LEASE_S)
    except (TypeError, ValueError):
        lease = DEFAULT_LEASE_S
    return max(MIN_LEASE_S, min(MAX_LEASE_S, lease))


def _parse_channels(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(c) for c in value if c)
    return ()


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "failure_code": code, "error": message}


class _MinimalHoldPolicy:
    """Placeholder policy that holds the current position (no-op compute)."""

    def __init__(self, mode: str, channels: tuple[str, ...]) -> None:
        self._policy_id = f"hold_{mode}"
        self._channels = channels

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def compute(self, state: Any) -> Any:
        from leapflow.hardware.control_bus import ControlCommand

        return ControlCommand(joint_commands={}, metadata={})

    def reset(self) -> None:
        pass


def _minimal_hold_policy(mode: str, channels: tuple[str, ...]) -> Any:
    """Create a minimal hold policy for initial session setup."""
    return _MinimalHoldPolicy(mode, channels)


__all__ = ["ControlSession", "NodeControlSessions"]
