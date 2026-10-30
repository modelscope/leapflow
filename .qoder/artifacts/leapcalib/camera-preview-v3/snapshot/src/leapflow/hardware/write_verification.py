# Copyright (c) Alibaba, Inc. and its affiliates.
"""Canonical expectations and verdicts for physical writes.

A transport acknowledgement says that a command was accepted. These types describe
what should be observed after settling and whether the physical effect matched that
expectation. They are intentionally independent of tool, plugin, and engine layers so
single writes, batch writes, policy execution, and delayed reads share one vocabulary.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from leapflow.hardware.context import Channel, HardwareContext, as_numeric
from leapflow.hardware.outcome import PhysicalOutcome
from leapflow.hardware.transport import Reading
from leapflow.hardware.verification import OperationVerdict, VerdictStatus

logger = logging.getLogger(__name__)

# How long an observation that contradicts its command is followed before the
# contradiction is reported: a servo commanded far from where it stands keeps
# moving past its declared settling time, and a value in motion is not evidence.
# Progress toward the target means the command is being carried out, so the
# follow continues; a value that stops progressing -- at rest, or moving away --
# for ``_CONVERGENCE_STALL_S`` is a contradiction. The overall budget is
# deployment policy (``HardwareSettings.write_verification_follow_s``): a slow
# arm legitimately travels for seconds, and a value still travelling when the
# budget ends is reported as pending, never as a failure.
_CONVERGENCE_POLL_S = 0.2
_CONVERGENCE_STALL_S = 0.6
_DEFAULT_CONVERGENCE_S = 10.0


def _moved(previous: Any, current: Any, tolerance: float) -> bool:
    """Whether an observation changed by more than half the declared tolerance."""
    left, right = as_numeric(previous), as_numeric(current)
    if left is None or right is None:
        return previous != current
    return abs(right - left) > max(tolerance / 2.0, 0.0)


def _converging(
    previous: Any,
    current: Any,
    previous_deviation: float,
    deviation: float,
    tolerance: float,
) -> bool:
    """Whether the observation is still closing the gap to its commanded value."""
    if not _moved(previous, current, tolerance):
        return False
    return deviation < previous_deviation


class WriteVerdictStatus(str, Enum):
    """Observable result of one commanded channel."""

    VERIFIED = "verified"
    MISMATCH = "mismatch"
    UNOBSERVABLE = "unobservable"
    PENDING = "pending"


@dataclass(frozen=True)
class WriteExpectation:
    """The physical observation expected from one accepted write."""

    operation_id: str
    device_id: str
    command_channel_id: str
    observation_channel_id: str
    commanded_value: Any
    tolerance: float
    settle_after: float
    created_at: float
    identity: str = ""

    @classmethod
    def from_channel(
        cls,
        *,
        device_id: str,
        channel: Channel,
        commanded_value: Any,
        now: float | None = None,
        identity: str = "",
    ) -> WriteExpectation:
        """Build an expectation from a channel's admitted declaration."""
        created_at = time.time() if now is None else now
        observation_channel_id = channel.verification_channel_id or channel.channel_id
        return cls(
            operation_id=f"write-{uuid.uuid4().hex}",
            device_id=device_id,
            command_channel_id=channel.channel_id,
            observation_channel_id=observation_channel_id,
            commanded_value=commanded_value,
            tolerance=max(0.0, channel.envelope.tolerance),
            settle_after=created_at + channel.envelope.effective_settling_s,
            created_at=created_at,
            identity=identity,
        )

    @property
    def settle_delay_s(self) -> float:
        """Declared time between command acceptance and valid observation."""
        return max(0.0, self.settle_after - self.created_at)

    def is_settled(self, *, now: float | None = None) -> bool:
        """Return whether an observation may be treated as final."""
        return (time.time() if now is None else now) >= self.settle_after

    def to_dict(self) -> dict[str, Any]:
        """Return a stable representation for tool results and audit records."""
        return {
            "operation_id": self.operation_id,
            "device_id": self.device_id,
            "command_channel_id": self.command_channel_id,
            "observation_channel_id": self.observation_channel_id,
            "commanded": self.commanded_value,
            "tolerance": self.tolerance,
            "created_at": self.created_at,
            "settle_after": self.settle_after,
            "settle_delay_s": self.settle_delay_s,
            "session_id": self.identity,
        }


@dataclass(frozen=True)
class WriteVerdict:
    """Typed physical-effect verdict for one write expectation."""

    status: str
    expectation: WriteExpectation
    observed_value: Any = None
    deviation: float | None = None
    detail: str = ""
    observed_at: float | None = None

    @property
    def is_verified(self) -> bool:
        """Return whether the expected physical effect was observed."""
        return self.status == WriteVerdictStatus.VERIFIED.value

    @property
    def is_mismatch(self) -> bool:
        """Return whether observation definitively contradicted the expectation."""
        return self.status == WriteVerdictStatus.MISMATCH.value

    @property
    def is_terminal(self) -> bool:
        """Return whether no later observation is needed for this expectation."""
        return self.status != WriteVerdictStatus.PENDING.value

    def to_dict(self) -> dict[str, Any]:
        """Return a backward-compatible tool and audit payload."""
        payload: dict[str, Any] = {
            "status": self.status,
            "operation_id": self.expectation.operation_id,
            "channel_id": self.expectation.command_channel_id,
            "observation_channel_id": self.expectation.observation_channel_id,
            "commanded": self.expectation.commanded_value,
            "readback": self.observed_value,
            "match": self.is_verified,
            "expectation": self.expectation.to_dict(),
        }
        if self.deviation is not None:
            payload["deviation"] = self.deviation
        if self.detail:
            payload["detail"] = self.detail
        if self.observed_at is not None:
            payload["observed_at"] = self.observed_at
        return payload

    @classmethod
    def pending(cls, expectation: WriteExpectation) -> WriteVerdict:
        """Describe an expectation whose settling window has not elapsed."""
        return cls(
            status=WriteVerdictStatus.PENDING.value,
            expectation=expectation,
            detail="The command is awaiting a post-settle observation.",
        )

    @classmethod
    def unobservable(cls, expectation: WriteExpectation, detail: str) -> WriteVerdict:
        """Describe an expectation that cannot currently be observed."""
        return cls(
            status=WriteVerdictStatus.UNOBSERVABLE.value,
            expectation=expectation,
            detail=detail,
        )

    @classmethod
    def from_reading(
        cls,
        expectation: WriteExpectation,
        reading: Reading,
        *,
        now: float | None = None,
    ) -> WriteVerdict:
        """Compare a settled observation with the declared expectation."""
        observed_at = time.time() if now is None else now
        if not expectation.is_settled(now=observed_at):
            return cls.pending(expectation)

        commanded = as_numeric(expectation.commanded_value)
        observed = as_numeric(reading.value)
        if commanded is not None and observed is not None:
            deviation = abs(observed - commanded)
            tolerance = expectation.tolerance or 1e-3
            matched = deviation <= tolerance
        else:
            deviation = None
            matched = reading.value == expectation.commanded_value

        return cls(
            status=(
                WriteVerdictStatus.VERIFIED.value if matched else WriteVerdictStatus.MISMATCH.value
            ),
            expectation=expectation,
            observed_value=reading.value,
            deviation=deviation,
            detail=(
                "Observed value matched the commanded expectation."
                if matched
                else "Observed value did not match the commanded expectation."
            ),
            observed_at=observed_at,
        )

    @classmethod
    def from_physical_outcome(
        cls,
        outcome: PhysicalOutcome,
        *,
        expectation: WriteExpectation | None = None,
    ) -> WriteVerdict:
        """Adapt the learning layer's settled outcome into the canonical verdict."""
        resolved_expectation = expectation or WriteExpectation(
            operation_id=f"outcome-{uuid.uuid4().hex}",
            device_id=outcome.device_id,
            command_channel_id=outcome.channel_id,
            observation_channel_id=outcome.channel_id,
            commanded_value=outcome.commanded,
            tolerance=0.0,
            settle_after=outcome.timestamp,
            created_at=outcome.timestamp,
        )
        if not outcome.settled:
            return cls.pending(resolved_expectation)
        return cls(
            status=(
                WriteVerdictStatus.VERIFIED.value
                if outcome.accurate
                else WriteVerdictStatus.MISMATCH.value
            ),
            expectation=resolved_expectation,
            observed_value=outcome.observed,
            deviation=abs(outcome.residual),
            detail=(
                "Settled physical outcome matched the command."
                if outcome.accurate
                else "Settled physical outcome did not match the command."
            ),
            observed_at=outcome.timestamp,
        )

    @classmethod
    def from_operation_verdict(
        cls,
        expectation: WriteExpectation,
        verdict: OperationVerdict,
    ) -> WriteVerdict:
        """Adapt semantic verifier output without conflating inconclusive evidence."""
        if verdict.status == VerdictStatus.SUCCESS.value:
            status = WriteVerdictStatus.VERIFIED.value
        elif verdict.status == VerdictStatus.FAILURE.value:
            status = WriteVerdictStatus.MISMATCH.value
        else:
            status = WriteVerdictStatus.UNOBSERVABLE.value
        return cls(
            status=status,
            expectation=expectation,
            deviation=verdict.deviation,
            detail=verdict.detail,
            observed_at=time.time(),
        )


def aggregate_write_verdicts(verdicts: list[WriteVerdict]) -> str:
    """Return the conservative aggregate status for a multi-channel write."""
    statuses = {verdict.status for verdict in verdicts}
    if WriteVerdictStatus.MISMATCH.value in statuses:
        return WriteVerdictStatus.MISMATCH.value
    if WriteVerdictStatus.UNOBSERVABLE.value in statuses:
        return WriteVerdictStatus.UNOBSERVABLE.value
    if WriteVerdictStatus.PENDING.value in statuses:
        return WriteVerdictStatus.PENDING.value
    return WriteVerdictStatus.VERIFIED.value


@dataclass(frozen=True)
class WriteVerdictEvent:
    """Terminal write verdict emitted on the shared hardware signal path."""

    verdict: WriteVerdict
    identity: str
    trust_transition: dict[str, Any] | None = None

    @property
    def kind(self) -> str:
        return "write_verdict"

    @property
    def event_type(self) -> str:
        return "hw.write_verdict"

    @property
    def device_id(self) -> str:
        return self.verdict.expectation.device_id

    @property
    def channel_id(self) -> str:
        return self.verdict.expectation.command_channel_id

    def to_payload(self) -> dict[str, Any]:
        """Return the pre-normalized EventBus payload."""
        payload = {
            **self.verdict.to_dict(),
            "kind": self.kind,
            "source": f"{self.device_id}.{self.channel_id}",
            "device_id": self.device_id,
            "channel_id": self.channel_id,
            "session_id": self.identity,
            "ts": self.verdict.observed_at or time.time(),
            "_mono_ts": time.monotonic(),
        }
        if self.trust_transition is not None:
            payload["trust_transition"] = self.trust_transition
        return payload

    def to_detail(self) -> str:
        """Return a compact operator-facing summary."""
        operation_id = self.verdict.expectation.operation_id
        return (
            f"[write_verdict] {self.device_id}.{self.channel_id} "
            f"{self.verdict.status} operation={operation_id}"
        )


class WriteVerificationCoordinator:
    """Own post-write settlement, audit, and verified-only trust updates."""

    def __init__(
        self,
        registry: Any,
        *,
        audit_log: Any,
        trust_gate: Any,
        identity_provider: Callable[[], str],
        follow_s: float = _DEFAULT_CONVERGENCE_S,
    ) -> None:
        self._registry = registry
        self._audit = audit_log
        self._trust_gate = trust_gate
        self._identity_provider = identity_provider
        self._follow_s = max(float(follow_s), 0.0)
        pending = getattr(registry, "pending_write_expectations", None)
        if pending is None:
            pending = getattr(registry, "_leapflow_pending_write_expectations", None)
        if pending is None:
            pending = {}
            setattr(registry, "_leapflow_pending_write_expectations", pending)
        self._pending: dict[tuple[str, str], list[WriteExpectation]] = pending

    def record_physical_outcome(self, outcome: Any) -> None:
        """Update trust only from a settled physical observation."""
        if self._trust_gate is None or not bool(getattr(outcome, "settled", False)):
            return
        if bool(getattr(outcome, "accurate", False)):
            self._trust_gate.record_success(outcome.device_id, outcome.channel_id)
        else:
            self._trust_gate.record_failure(outcome.device_id, outcome.channel_id)

    def drop_pending_outcome(self, device_id: str, channel_id: str) -> None:
        """Discard learning state when no matching same-channel reading will arrive."""
        recorder = self._registry.outcome_recorder
        if recorder is not None:
            recorder.drop_pending(device_id, channel_id)

    def defer_or_refuse(
        self,
        context: HardwareContext,
        expectation: WriteExpectation,
    ) -> WriteVerdict:
        """Track an observable expectation or settle it as unobservable."""
        observation = context.channel(expectation.observation_channel_id)
        if observation is None or not observation.is_readable:
            return self._unobservable(
                expectation,
                "The declared verification channel is absent or not readable.",
            )
        if observation.is_privacy_gated:
            return self._unobservable(
                expectation,
                "The verification channel requires separate read consent.",
            )

        key = (expectation.device_id, expectation.observation_channel_id)
        pending = self._pending.setdefault(key, [])
        pending.append(expectation)
        del pending[:-32]
        verdict = WriteVerdict.pending(expectation)
        self.record_verdict(verdict)
        return verdict

    def settle_pending(
        self,
        device_id: str,
        observation_channel_id: str,
        reading: Reading,
        *,
        resolved_outcome: Any | None,
    ) -> list[WriteVerdict]:
        """Resolve settled expectations associated with one observed channel."""
        key = (device_id, observation_channel_id)
        pending = self._pending.get(key, [])
        if not pending:
            if resolved_outcome is not None:
                self.record_physical_outcome(resolved_outcome)
            return []
        now = time.time()
        ready = [expectation for expectation in pending if expectation.is_settled(now=now)]
        waiting = [expectation for expectation in pending if not expectation.is_settled(now=now)]
        if waiting:
            self._pending[key] = waiting
        else:
            self._pending.pop(key, None)

        verdicts: list[WriteVerdict] = []
        outcome_available = resolved_outcome
        visible_identity = self._identity_provider()
        for expectation in ready:
            if (
                outcome_available is not None
                and expectation.command_channel_id == outcome_available.channel_id
            ):
                verdict = WriteVerdict.from_physical_outcome(
                    outcome_available,
                    expectation=expectation,
                )
                outcome_available = None
            else:
                verdict = WriteVerdict.from_reading(expectation, reading, now=now)
                if expectation.command_channel_id != observation_channel_id:
                    self.drop_pending_outcome(device_id, expectation.command_channel_id)
            self.record_verdict(verdict)
            if (
                not visible_identity
                or not expectation.identity
                or expectation.identity == visible_identity
            ):
                verdicts.append(verdict)
        if outcome_available is not None:
            self.record_physical_outcome(outcome_available)
        return verdicts

    async def verify_batch(
        self,
        device_id: str,
        resolved: list[tuple[Channel, Any]],
        expectations: list[WriteExpectation],
        resolved_outcomes: list[Any | None],
    ) -> list[WriteVerdict]:
        """Settle and verify every accepted channel through typed verdicts."""
        wait_s = max(
            (expectation.settle_after - time.time() for expectation in expectations),
            default=0.0,
        )
        if wait_s > 0.0:
            await asyncio.sleep(wait_s)

        context = self._registry.context(device_id)
        assert context is not None
        results: list[WriteVerdict] = []
        async with self._registry.device_io(device_id):
            transport = await self._registry.transport(device_id)
            for (channel, _value), expectation, resolved_outcome in zip(
                resolved, expectations, resolved_outcomes, strict=True
            ):
                verdict = await self._verify_one(
                    context,
                    transport,
                    channel,
                    expectation,
                    resolved_outcome,
                )
                results.append(verdict)
        return results

    async def _verify_one(
        self,
        context: HardwareContext,
        transport: Any,
        channel: Channel,
        expectation: WriteExpectation,
        resolved_outcome: Any | None,
    ) -> WriteVerdict:
        if resolved_outcome is not None:
            verdict = WriteVerdict.from_physical_outcome(
                resolved_outcome,
                expectation=expectation,
            )
            self.record_verdict(verdict)
            return verdict

        observation = context.channel(expectation.observation_channel_id)
        if observation is None or not observation.is_readable:
            return self._unobservable(
                expectation,
                "The declared verification channel is absent or not readable.",
            )
        if observation.is_privacy_gated:
            return self._unobservable(
                expectation,
                "The verification channel requires separate read consent.",
            )
        verdict, detail = await self._observe(transport, channel, expectation)
        if verdict is None:
            return self._unobservable(expectation, detail)
        if verdict.is_mismatch:
            verdict = await self._follow_convergence(transport, channel, expectation, verdict)
        self.record_verdict(verdict)
        return verdict

    async def _observe(
        self,
        transport: Any,
        channel: Channel,
        expectation: WriteExpectation,
    ) -> tuple[WriteVerdict | None, str]:
        """Read and judge the verification channel once; ``None`` when unreadable."""
        try:
            reading = await transport.read(expectation.observation_channel_id)
        except Exception as exc:  # noqa: BLE001 - verification must stay contained
            logger.warning(
                "Post-write verification failed for %s.%s via %s: %s",
                expectation.device_id,
                channel.channel_id,
                expectation.observation_channel_id,
                exc,
                exc_info=True,
            )
            return None, f"Post-write observation failed: {type(exc).__name__}."

        recorder = self._registry.outcome_recorder
        if recorder is not None and expectation.observation_channel_id == channel.channel_id:
            learned_outcome = recorder.observe(
                device_id=expectation.device_id,
                channel_id=channel.channel_id,
                value=reading.value,
            )
            if learned_outcome is not None:
                return WriteVerdict.from_physical_outcome(
                    learned_outcome, expectation=expectation
                ), ""
        elif expectation.observation_channel_id != channel.channel_id:
            self.drop_pending_outcome(expectation.device_id, channel.channel_id)
        return WriteVerdict.from_reading(expectation, reading), ""

    async def _follow_convergence(
        self,
        transport: Any,
        channel: Channel,
        expectation: WriteExpectation,
        verdict: WriteVerdict,
    ) -> WriteVerdict:
        """Follow a contradicting value while it is still closing the gap.

        The declared settling time says when an observation becomes meaningful,
        not when a servo has arrived. A pose channel commanded 100 mm from where
        it stands is still moving well past a fixed wait, and a single read then
        reports a contradiction of an expectation that is being met -- charging
        the plugin a trust failure for a move that is correct.

        So the value is followed instead of judged once: progress toward the
        command keeps the follow alive, arrival settles it as verified, and a
        value that stops progressing -- at rest, or moving away -- for
        ``_CONVERGENCE_STALL_S`` is the contradiction. A value still travelling
        when the follow budget ends is reported as pending, which is exactly
        what "not observable yet" means and costs no trust.
        """
        started = time.monotonic()
        deadline = started + self._follow_s
        progressing_since = started
        latest = verdict
        previous_value = verdict.observed_value
        previous_deviation = abs(verdict.deviation or 0.0)
        while time.monotonic() < deadline:
            await asyncio.sleep(_CONVERGENCE_POLL_S)
            observed, _detail = await self._observe(transport, channel, expectation)
            if observed is None:
                return latest
            if not observed.is_mismatch:
                return observed
            deviation = abs(observed.deviation or 0.0)
            now = time.monotonic()
            if _converging(
                previous_value,
                observed.observed_value,
                previous_deviation,
                deviation,
                expectation.tolerance,
            ):
                progressing_since = now
            elif now - progressing_since >= _CONVERGENCE_STALL_S:
                return observed
            latest = observed
            previous_value = observed.observed_value
            previous_deviation = deviation
        return WriteVerdict(
            status=WriteVerdictStatus.PENDING.value,
            expectation=expectation,
            observed_value=latest.observed_value,
            deviation=latest.deviation,
            detail=(
                "The observed value was still travelling when the follow budget "
                "ended; read the channel again before drawing a conclusion."
            ),
            observed_at=time.time(),
        )

    def _unobservable(self, expectation: WriteExpectation, detail: str) -> WriteVerdict:
        self.drop_pending_outcome(
            expectation.device_id,
            expectation.command_channel_id,
        )
        verdict = WriteVerdict.unobservable(expectation, detail)
        self.record_verdict(verdict)
        return verdict

    def record_verdict(self, verdict: WriteVerdict) -> None:
        """Persist, publish, and apply trust once using the operation's identity."""
        expectation = verdict.expectation
        identity = expectation.identity or self._identity_provider()
        trust_before = self._trust_snapshot(expectation)
        if self._trust_gate is not None:
            if verdict.status == WriteVerdictStatus.VERIFIED.value:
                self._trust_gate.record_success(
                    expectation.device_id,
                    expectation.command_channel_id,
                )
            elif verdict.status == WriteVerdictStatus.MISMATCH.value:
                self._trust_gate.record_failure(
                    expectation.device_id,
                    expectation.command_channel_id,
                )
        trust_after = self._trust_snapshot(expectation)
        trust_transition = (
            {
                "before": trust_before,
                "after": trust_after,
                "changed": trust_before != trust_after,
            }
            if trust_before is not None or trust_after is not None
            else None
        )
        payload = verdict.to_dict()
        if trust_transition is not None:
            payload["trust_transition"] = trust_transition
        self._audit.record(
            action="write_verification",
            device=expectation.device_id,
            channel=expectation.command_channel_id,
            value=payload,
            outcome=verdict.status,
            identity=identity,
        )
        if verdict.is_terminal:
            publish = getattr(self._registry, "publish_event", None)
            if callable(publish):
                publish(
                    WriteVerdictEvent(
                        verdict=verdict,
                        identity=identity,
                        trust_transition=trust_transition,
                    )
                )

    def _trust_snapshot(self, expectation: WriteExpectation) -> dict[str, Any] | None:
        if self._trust_gate is None:
            return None
        try:
            record = self._trust_gate.trust_record(
                expectation.device_id,
                expectation.command_channel_id,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
        level = getattr(record, "level", None)
        return {
            "level": getattr(level, "name", str(level)).lower(),
            "consecutive_ok": int(getattr(record, "consecutive_ok", 0)),
            "consecutive_fail": int(getattr(record, "consecutive_fail", 0)),
            "frozen": bool(getattr(record, "frozen", False)),
        }


__all__ = [
    "WriteExpectation",
    "WriteVerdict",
    "WriteVerdictStatus",
    "WriteVerdictEvent",
    "WriteVerificationCoordinator",
    "aggregate_write_verdicts",
]
