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
import inspect
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Mapping, Protocol, TypeVar, runtime_checkable

from leapflow.hardware.lhp.protocol import FAILURE_DEVICE_BUSY
from leapflow.hardware.transport import WriteOutcome

logger = logging.getLogger(__name__)
_Result = TypeVar("_Result")

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

    def start(self, *, reservation_token: str) -> None:
        from leapflow.hardware.control_bus import ControlBusConfig

        kwargs: dict[str, Any] = {}
        if getattr(self._registry, "node_sessions", None) is not None:
            kwargs["reservation_token"] = reservation_token
        self._registry.start_control_loop(
            self._device_id,
            _HoldPolicy(self._mode),
            config=ControlBusConfig(frequency_hz=self._frequency_hz),
            **kwargs,
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

        return ControlCommand(joint_commands={})

    def reset(self) -> None:
        pass


# ── Session manager ───────────────────────────────────────────────────────


class NodeSessions:
    """Lease, device mutex, and ownership for every session on one node."""

    def __init__(self, registry: Any, *, device_busy: Callable[[str], bool] | None = None,
                 transaction_run_owner: Callable[[str], str] | None = None) -> None:
        self._registry = registry
        self._device_busy = device_busy
        self._transaction_run_owner = transaction_run_owner
        self._mutex = threading.RLock()
        self._controls: dict[str, tuple[str, Callable[[], None]]] = {}
        self._sessions: dict[str, LeasedSession] = {}
        self._reservations: dict[str, tuple[str, ...]] = {}
        self._dispatches: dict[str, str] = {}
        self._uncertain_devices: set[str] = set()
        self._reserved_tasks: set[asyncio.Task[Any]] = set()
        self._closing = False

    # ── Mutex ──

    def holds_device(self, device_id: str) -> bool:
        """Whether a live session holds *device_id*."""
        with self._mutex:
            return any(
                device_id in session.body.device_ids
                for session in self._sessions.values()
                if session.live
            )

    def busy_device(self, device_ids: Iterable[str]) -> str:
        """The first of *device_ids* that is held by a session or a node-local holder."""
        with self._mutex:
            for device_id in device_ids:
                if self.unsettled_device(device_id):
                    return device_id
                status = getattr(self._registry, "control_loop_status", lambda _: None)(device_id)
                if status and status.get("running"):
                    return device_id
                if self.holds_device(device_id):
                    return device_id
                if self._device_busy is not None and self._device_busy(device_id):
                    return device_id
            return ""

    def reserved_device(self, device_id: str) -> bool:
        with self._mutex:
            return any(device_id in devices for devices in self._reservations.values())

    def unsettled_device(self, device_id: str) -> bool:
        """Pending or uncertain effects, excluding a caller's ordinary run lease."""
        with self._mutex:
            return (
                self._closing or self.reserved_device(device_id)
                or device_id in self._dispatches.values()
                or device_id in self._uncertain_devices
                or any(device == device_id for device, _ in self._controls.values())
            )

    def reserve(self, device_ids: Iterable[str], *, owner_id: str, run_id: str = "") -> str:
        """Atomically reserve devices/consumer IDs before any async transaction work.

        Reservations are NOT leases. Expiry/cancellation cannot release a DB commit
        whose settlement is still uncertain; only the owner releases the token.
        """
        devices = tuple(dict.fromkeys(device_ids))
        if not devices or len(devices) > 64 or not isinstance(owner_id, str) or not owner_id or any(not isinstance(d, str) or not d for d in devices):
            raise ValueError("bounded explicit reservation identities required")
        with self._mutex:
            if run_id:
                self._check_transaction_run(devices, run_id)
            elif busy := self.busy_device(devices):
                raise ValueError(f"device/consumer {busy!r} is busy or has unsettled effects")
            token = str(uuid.uuid4())
            self._reservations[token] = devices
            return token

    def _check_transaction_run(self, devices: tuple[str, ...], run_id: str) -> None:
        """Exclude only the runtime-verified non-motion run, never a named stranger."""
        if not isinstance(run_id, str) or not run_id or self._transaction_run_owner is None:
            raise ValueError("owner-verified non-motion run required for reservation")
        for device in devices:
            if self._transaction_run_owner(device) != run_id or self.unsettled_device(device):
                raise ValueError("transaction does not own the exact hold or has unsettled effects")
            status = getattr(self._registry, "control_loop_status", lambda _: None)(device)
            if status and status.get("running"):
                raise ValueError("active control blocks a model transaction")
            for session in self._sessions.values():
                if session.live and device in session.body.device_ids:
                    if session.body.kind != "skill" or session.body.describe().get("run_id") != run_id:
                        raise ValueError("another session blocks a model transaction")

    def release_reservation(self, token: str) -> None:
        with self._mutex:
            if token not in self._reservations:
                raise ValueError("unknown reservation token")
            del self._reservations[token]

    async def run_reserved(
        self, device_ids: Iterable[str], *, owner_id: str,
        operation: Callable[[], _Result],
        on_settle: Callable[[_Result], None] | None = None,
        run_id: str = "",
        preflight: Callable[[], Awaitable[None]] | None = None,
    ) -> _Result:
        """Owner-internal admitted persistence, reserved before its first await.

        This grants no admission or actuator authority. The callable runs off-loop;
        cancellation cannot abandon its DB settlement or release its reservation.
        Node shutdown drains the same tasks before storage can close.
        """
        if not callable(operation) or (on_settle is not None and not callable(on_settle)):
            raise ValueError("concrete owner persistence operation and settlement callback required")
        if on_settle is not None and (
            inspect.iscoroutinefunction(on_settle)
            or inspect.iscoroutinefunction(getattr(on_settle, "__call__", None))
        ):
            raise ValueError("settlement publication must be synchronous on the owner loop")
        if preflight is not None and not callable(preflight):
            raise ValueError("owner transaction preflight must be callable")
        token = self.reserve(device_ids, owner_id=owner_id, run_id=run_id)
        caller = asyncio.current_task()
        if caller is not None:
            self._reserved_tasks.add(caller)
        try:
            if preflight is not None:
                await preflight()
        except BaseException:
            # No persistence was dispatched. Cancellation of observational
            # preflight releases only after its own bounded work has settled.
            self.release_reservation(token)
            raise
        finally:
            if caller is not None:
                self._reserved_tasks.discard(caller)
        task = asyncio.create_task(self._settle_reserved(token, operation, on_settle), name="reserved-owner-transaction")
        self._reserved_tasks.add(task)
        task.add_done_callback(self._reserved_tasks.discard)
        cancelled = False
        while True:
            try:
                result = await asyncio.shield(task)
                if cancelled:
                    raise asyncio.CancelledError
                return result
            except asyncio.CancelledError:
                if task.done():
                    # Consume an error even when cancellation wins the caller's result.
                    if not task.cancelled():
                        task.exception()
                    raise
                cancelled = True
            except Exception:
                if cancelled:
                    raise asyncio.CancelledError from None
                raise

    async def _settle_reserved(
        self, token: str, operation: Callable[[], _Result],
        on_settle: Callable[[_Result], None] | None,
    ) -> _Result:
        try:
            try:
                result = await asyncio.to_thread(operation)
            except BaseException:
                # A callable can commit and then fail before returning. An exception
                # alone cannot establish that persistence left the prior state intact.
                with self._mutex:
                    self._uncertain_devices.update(self._reservations[token])
                raise
            # Publish the committed owner snapshot on this loop BEFORE releasing
            # resources, including when the original requesting lease has ended.
            if on_settle is not None:
                try:
                    published = on_settle(result)
                    if published is not None:
                        if inspect.iscoroutine(published):
                            published.close()
                        raise ValueError("settlement publication must finish synchronously and return None")
                except BaseException:  # publication cancellation also leaves a committed projection uncertain
                    # Persistence landed but its safety projection did not. Keep
                    # every affected consumer closed until owner reconciliation.
                    with self._mutex:
                        self._uncertain_devices.update(self._reservations[token])
                    raise
            return result
        finally:
            self.release_reservation(token)

    async def settle_transactions(self) -> None:
        """Drain admitted transactions; never cancel a thread holding the shared DB."""
        cancelled = False
        while self._reserved_tasks:
            tasks = tuple(self._reserved_tasks)
            waiter = asyncio.gather(*tasks, return_exceptions=True)
            while not waiter.done():
                try:
                    await asyncio.shield(waiter)
                except asyncio.CancelledError:
                    cancelled = True
            self._reserved_tasks.difference_update(task for task in tasks if task.done())
        if cancelled:
            raise asyncio.CancelledError

    def begin_dispatch(self, device_id: str) -> str:
        """Cover precheck through transport settlement; caller already owns authority."""
        with self._mutex:
            if (
                self._closing or self.reserved_device(device_id)
                or device_id in self._uncertain_devices
                or any(device == device_id for device, _ in self._controls.values())
            ):
                raise ValueError("device reserved for model transaction, control or uncertain prior action")
            token = str(uuid.uuid4())
            self._dispatches[token] = device_id
            return token

    def finish_dispatch(self, token: str, *, settled: bool) -> None:
        with self._mutex:
            device = self._dispatches.pop(token)
            if not settled:
                self._uncertain_devices.add(device)

    def begin_control(
        self, device_id: str, *, stop: Callable[[], None], reservation_token: str | None = None,
    ) -> str:
        """Claim direct control before transport opening; NOT authority or a lease.

        The short mutex also serializes synchronous controller starts with owner-loop
        reservations. No I/O, await, thread join or callback runs under this mutex.
        The ControlBus releases only after its transport/halt thread settles.
        """
        if not isinstance(device_id, str) or not device_id or not callable(stop):
            raise ValueError("explicit device and synchronous control stop required")
        with self._mutex:
            if reservation_token is not None:
                if self._closing or self._reservations.get(reservation_token) != (device_id,):
                    raise ValueError("exact owner control-start reservation required")
            elif self.busy_device((device_id,)):
                raise ValueError("device is busy or has unsettled effects")
            token = str(uuid.uuid4())
            self._controls[token] = (device_id, stop)
            if reservation_token is not None:
                del self._reservations[reservation_token]
            return token

    def finish_control(self, token: str, *, settled: bool) -> None:
        """Thread settlement, retaining uncertain halt as a shared owner refusal."""
        with self._mutex:
            device, _ = self._controls.pop(token)
            if not settled:
                self._uncertain_devices.add(device)

    # ── Lifecycle ──

    def adopt(self, body: SessionBody, *, lease_s: Any, connection_id: str) -> LeasedSession:
        """Put an already started *body* under a lease owned by *connection_id*.

        The caller checks ``busy_device`` first, with no await in between, so the
        check and the adoption are atomic on the event loop.
        """
        with self._mutex:
            return self._adopt(body, lease_s=lease_s, connection_id=connection_id)

    def _adopt(self, body: SessionBody, *, lease_s: Any, connection_id: str) -> LeasedSession:
        if self._closing:
            raise ValueError("node sessions are shutting down")
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

    async def open(
        self, params: Mapping[str, Any], *, connection_id: str,
        authorize: Callable[[], WriteOutcome | None] | None = None,
    ) -> dict[str, Any]:
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
        from leapflow.hardware.node_gate import control_safety_refusal
        from leapflow.hardware.lhp.protocol import FAILURE_SAFETY_DENIED

        if reason := control_safety_refusal(self._registry, device_id):
            return _error(FAILURE_SAFETY_DENIED, reason)
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
        if self._registry.context(device_id) is not context:
            return _error(FAILURE_SAFETY_DENIED, "device declaration changed during control precheck")
        if reason := control_safety_refusal(self._registry, device_id):
            return _error(FAILURE_SAFETY_DENIED, reason)
        # Claim before authority, atomically against synchronous direct-controller
        # starts as well as owner-loop transactions. The bus converts this exact
        # reservation into its lifetime claim without an unowned interval.
        try:
            token = self.reserve((device_id,), owner_id=control_id)
        except ValueError as exc:
            return _error(FAILURE_DEVICE_BUSY, str(exc))
        try:
            if authorize is not None:
                verdict = authorize()
                if verdict is not None:
                    return verdict.to_dict()
            try:
                body.start(reservation_token=token)
            except (RuntimeError, ValueError) as exc:
                return _error("control_start_failed", str(exc))
            session = self.adopt(body, lease_s=params.get("lease_s"), connection_id=connection_id)
            return {"ok": True, **session.lease()}
        finally:
            with self._mutex:
                if token in self._reservations:
                    self.release_reservation(token)

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
        """Close incoming work, settle leased runs and transactions (node shutdown)."""
        with self._mutex:
            self._closing = True
            controls = tuple(stop for _, stop in self._controls.values())
        for session in list(self._sessions.values()):
            if session.live:
                self._end(session)
                await session.body.close("shutdown")
        with self._mutex:
            self._sessions.clear()
        # Includes independently constructed buses, not just registry._control_buses.
        # A timed-out join retains its claim; storage/device release must refuse.
        for stop in controls:
            await asyncio.to_thread(stop)
        with self._mutex:
            if self._controls:
                raise RuntimeError("control startup/thread has not settled; retain node devices and retry shutdown")
        await self.settle_transactions()

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
        with self._mutex:
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
