# Copyright (c) Alibaba, Inc. and its affiliates.
"""The node-side write gate: one governed path from a command to a device.

Every write a node performs -- an LHP ``write``/``write_batch`` from a host, or a
discrete motion issued by a node-local skill -- passes the same chain, in the same
order: writable → envelope → interlocks → degradation → rate → safety policy, then
the per-device I/O lock, the transport, and the rate baseline. Keeping that chain
in one module is what makes "a skill cannot bypass the gate" a structural property
rather than a review checklist item: there is no second write path to drift.

Authorization (write mode, execution permits) deliberately stays with the caller.
The LHP server checks a per-request permit; the skill runtime checks a per-run
permit once and then drives this gate in-process. The gate itself decides only
whether a value is *physically admissible right now*.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from leapflow.hardware.context import HardwareContext, as_numeric
from leapflow.hardware.lhp.protocol import (
    ERR_UNKNOWN_CHANNEL as FAILURE_UNKNOWN_CHANNEL,
)
from leapflow.hardware.lhp.protocol import (
    ERR_UNKNOWN_DEVICE as FAILURE_UNKNOWN_DEVICE,
)
from leapflow.hardware.lhp.protocol import (
    FAILURE_BATCH_ABORTED,
    FAILURE_DEVICE_DEGRADED,
    FAILURE_INTERLOCK_UNSATISFIED,
    FAILURE_NOT_WRITABLE,
    FAILURE_RATE_LIMITED,
    FAILURE_SAFETY_DENIED,
)
from leapflow.hardware.transport import (
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_PARTIAL,
    BatchTransport,
    TransportError,
    WriteOutcome,
    worst_side_effect_state,
)


class ReadingIdentityError(Exception):
    """A transport answered for a different device or channel than was asked.

    Indistinguishable from success downstream, so it is raised where the request
    identity is still known. Callers map it to their own internal-error surface.
    """


def verify_reading_identity(reading: Any, device_id: str, channel_id: str) -> None:
    """Raise ``ReadingIdentityError`` when *reading* contradicts the request."""
    actual_channel = str(getattr(reading, "channel_id", ""))
    if actual_channel != channel_id:
        raise ReadingIdentityError(
            f"transport returned a reading for channel {actual_channel!r}, not {channel_id!r}"
        )
    actual_device = str(getattr(reading, "device_id", ""))
    if actual_device and actual_device != device_id:
        raise ReadingIdentityError(
            f"transport returned a reading for device {actual_device!r}, not {device_id!r}"
        )


def is_cyclic(channel: Any) -> bool:
    """True for an angle channel whose envelope spans a full turn.

    On such a channel -179 and 179 are 2 degrees apart, not 358: the value wraps.
    Derived from the declaration (unit and span), never from a channel-name list.
    """
    envelope = channel.envelope
    if channel.unit != "deg" or envelope.min_value is None or envelope.max_value is None:
        return False
    return envelope.max_value - envelope.min_value >= 360.0 - 1e-6


def rate_delta(channel: Any, value: float, previous: float) -> float:
    """The physical change from *previous* to *value* (shortest turn when cyclic)."""
    delta = float(value) - float(previous)
    if is_cyclic(channel):
        delta = (delta + 180.0) % 360.0 - 180.0
    return delta


def refusal(error: str, failure_code: str) -> WriteOutcome:
    """Build a provable refusal: nothing was attempted, so no effect landed."""
    return WriteOutcome(
        ok=False, side_effect_state=SIDE_EFFECT_NONE, error=error, failure_code=failure_code
    )


def merged_batch_state(ok: bool, outcomes: Sequence[WriteOutcome]) -> str:
    """Merge per-command verdicts into one batch verdict.

    A vector command that left the device mixed (some channels landed, one provably
    did not) is PARTIAL, not COMMITTED -- "committed" would claim the batch as a
    whole took effect.
    """
    merged = worst_side_effect_state(outcome.side_effect_state for outcome in outcomes)
    if not ok and merged == SIDE_EFFECT_COMMITTED:
        merged = SIDE_EFFECT_PARTIAL
    return merged


class NodeWriteGate:
    """Physical admissibility and execution for writes against one registry.

    The registry contract is duck-typed (``context`` / ``device_io`` / ``transport`` /
    ``check_safety_policy`` and the optional ``is_device_degraded`` / ``last_command``
    / ``record_command``), so the production registry and a test stub are equally
    admissible.
    """

    def __init__(self, registry: Any, *, clock: Any = time.monotonic) -> None:
        self._registry = registry
        self._clock = clock

    # ── Admissibility ──

    async def precheck(
        self, context: HardwareContext, channel: Any, value: Any
    ) -> WriteOutcome | None:
        """Return a provable refusal for *value*, or None when the write may proceed.

        Defense in depth: a host or a skill may already have checked, but the node
        never relies on a caller having done so.
        """
        if not channel.is_writable:
            return refusal(f"channel {channel.channel_id!r} is not writable", FAILURE_NOT_WRITABLE)
        if not channel.envelope.contains(value):
            return refusal(
                f"value {value!r} is outside the admitted envelope for "
                f"{context.device_id}.{channel.channel_id}",
                FAILURE_SAFETY_DENIED,
            )
        from leapflow.hardware.interlocks import failed_interlocks

        failed = await failed_interlocks(self._registry, context, channel)
        if failed:
            return refusal(
                f"interlocks not satisfied: {', '.join(failed)}", FAILURE_INTERLOCK_UNSATISFIED
            )
        is_degraded = getattr(self._registry, "is_device_degraded", None)
        if callable(is_degraded) and is_degraded(context.device_id):
            return refusal(
                f"device {context.device_id!r} is in a degraded state", FAILURE_DEVICE_DEGRADED
            )
        wait_s = self.rate_wait_s(context.device_id, channel, value)
        if wait_s > 0.0:
            return refusal(
                f"command exceeds the declared maximum rate for "
                f"{context.device_id}.{channel.channel_id}; retry after "
                f"approximately {wait_s:.3f}s",
                FAILURE_RATE_LIMITED,
            )
        allowed, reason = self._registry.check_safety_policy(context, channel, value)
        if not allowed:
            return refusal(reason, FAILURE_SAFETY_DENIED)
        return None

    def rate_wait_s(self, device_id: str, channel: Any, value: Any) -> float:
        """Seconds until *value* would satisfy the channel's declared maximum rate.

        Exposed separately so a governed caller (``GovernedMotion``) can wait the
        limit out instead of turning every fast-but-legal sequence into a refusal.
        """
        numeric = as_numeric(value)
        baseline = self.baseline(device_id, channel.channel_id)
        if numeric is None or baseline is None:
            return 0.0
        previous, previous_at = baseline
        return float(
            channel.envelope.rate_wait_s(
                delta=rate_delta(channel, numeric, previous),
                elapsed_s=self._clock() - previous_at,
            )
        )

    def baseline(self, device_id: str, channel_id: str) -> tuple[float, float] | None:
        """The last value known to have reached the device and when, or None."""
        last_command = getattr(self._registry, "last_command", None)
        if not callable(last_command):
            return None
        return last_command(device_id, channel_id)

    # ── Execution ──

    async def write(self, device_id: str, channel_id: str, value: Any) -> WriteOutcome:
        """Precheck and execute one channel write."""
        context, channel = self._resolve(device_id, channel_id)
        if channel is None:
            return refusal(
                f"unknown channel {channel_id!r} on device {device_id!r}", FAILURE_UNKNOWN_CHANNEL
            )
        verdict = await self.precheck(context, channel, value)
        if verdict is not None:
            return verdict
        async with self._registry.device_io(device_id):
            transport = await self._registry.transport(device_id)
            outcome = await transport.write(channel_id, value)
        if outcome.readback is not None:
            verify_reading_identity(outcome.readback, device_id, channel_id)
        if outcome.ok:
            self.record_accepted(device_id, channel_id, value)
        return outcome

    async def write_batch(
        self, device_id: str, commands: Sequence[tuple[str, Any]]
    ) -> tuple[bool, tuple[WriteOutcome, ...]]:
        """Precheck every command atomically, then execute the vector.

        A single refusal refuses the whole batch before anything is attempted. A
        transport without ``BatchTransport`` runs stop-on-failure sequentially.
        """
        commands = tuple((str(cid), value) for cid, value in commands)
        context = self._registry.context(device_id)
        if context is None:
            outcome = refusal(f"unknown device {device_id!r}", FAILURE_UNKNOWN_DEVICE)
            return False, tuple(outcome for _ in commands)
        verdicts: list[WriteOutcome | None] = []
        for channel_id, value in commands:
            channel = context.channel(channel_id)
            if channel is None:
                verdicts.append(
                    refusal(
                        f"unknown channel {channel_id!r} on device {device_id!r}",
                        FAILURE_UNKNOWN_CHANNEL,
                    )
                )
                continue
            verdicts.append(await self.precheck(context, channel, value))
        if any(verdict is not None for verdict in verdicts):
            aborted = refusal(
                "batch refused: another command was rejected before anything was attempted",
                FAILURE_BATCH_ABORTED,
            )
            return False, tuple(v if v is not None else aborted for v in verdicts)
        async with self._registry.device_io(device_id):
            transport = await self._registry.transport(device_id)
            if isinstance(transport, BatchTransport):
                batch = await transport.write_batch(commands)
                outcomes, ok = tuple(batch.outcomes), bool(batch.ok)
            else:
                ok, outcomes = await _sequential(transport, commands)
        for index, outcome in enumerate(outcomes):
            if index >= len(commands):
                break
            channel_id, value = commands[index]
            if outcome.readback is not None:
                verify_reading_identity(outcome.readback, device_id, channel_id)
            if outcome.ok:
                self.record_accepted(device_id, channel_id, value)
        return ok, outcomes

    def record_accepted(self, device_id: str, channel_id: str, value: Any) -> None:
        """Move the rate baseline -- only for a value known to have reached the device."""
        numeric = as_numeric(value)
        record_command = getattr(self._registry, "record_command", None)
        if numeric is not None and callable(record_command):
            record_command(device_id, channel_id, numeric)

    def _resolve(self, device_id: str, channel_id: str) -> tuple[Any, Any]:
        context = self._registry.context(device_id)
        if context is None:
            return None, None
        return context, context.channel(channel_id)


async def _sequential(
    transport: Any, commands: Sequence[tuple[str, Any]]
) -> tuple[bool, tuple[WriteOutcome, ...]]:
    """Stop-on-failure sequential fallback for transports without batch support."""
    outcomes: list[WriteOutcome] = []
    for index, (channel_id, value) in enumerate(commands):
        try:
            result = await transport.write(channel_id, value)
        except TransportError as exc:
            result = refusal(str(exc), exc.failure_code)
        outcomes.append(result)
        if not result.ok:
            outcomes.extend(
                refusal("batch stopped: a preceding command failed", FAILURE_BATCH_ABORTED)
                for _ in range(index + 1, len(commands))
            )
            return False, tuple(outcomes)
    return True, tuple(outcomes)


__all__ = [
    "NodeWriteGate",
    "ReadingIdentityError",
    "is_cyclic",
    "merged_batch_state",
    "rate_delta",
    "refusal",
    "verify_reading_identity",
]
