# Copyright (c) Alibaba, Inc. and its affiliates.
"""Typed feasibility snapshots, approval evaluation, and safe sequential writes.

Single writes, batch writes, and control-loop starts share the approval evaluator.
Feasibility checks and execution orchestration still live in HardwareTools; the
snapshot types establish phase boundaries without claiming a complete extraction.

Trust-based exemptions require every affected channel to qualify. The sequential
fallback stops on failure and preserves preceding effects and unattempted outcomes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_PARTIAL,
    SIDE_EFFECT_UNKNOWN,
    BatchWriteOutcome,
    HardwareTransport,
    TransportError,
    WriteOutcome,
    worst_side_effect_state,
)

logger = logging.getLogger(__name__)


# ── Feasibility result ────────────────────────────────────────────────────


@dataclass(frozen=True)
class WriteFeasibilityResult:
    """Outcome of the pre-approval feasibility evaluation for a single write.

    When ``ok`` is True, the feasibility chain passed and ``context``, ``channel``,
    ``descriptor``, and ``value`` are guaranteed to be non-None. When ``ok`` is
    False, ``error`` carries the refusal payload that should be returned to the
    caller.

    This type exists so that the ordering logic (resolve before writable, writable
    before effect class, effect before describe, etc.) is encoded once and reused
    across single writes, batch writes, and future unified governor pipelines.
    """

    ok: bool
    error: dict[str, Any] | None = None
    context: Any = None  # HardwareContext
    channel: Any = None  # Channel
    descriptor: Any = None  # ActionDescriptor
    in_envelope: bool = True
    interlocks_failed: tuple[str, ...] = ()
    dry_run: bool = False
    value: Any = None
    conditions: str = ""


@dataclass(frozen=True)
class BatchFeasibilityResult:
    """Outcome of the pre-approval feasibility evaluation for a batch write.

    Mirrors ``WriteFeasibilityResult`` for the multi-channel case.  When ``ok``
    is True, ``context``, ``resolved``, ``descriptor``, and ``channel_summaries``
    are guaranteed to be populated.  When ``ok`` is False, ``error`` carries the
    refusal payload.
    """

    ok: bool
    error: dict[str, Any] | None = None
    context: Any = None  # HardwareContext
    resolved: tuple[tuple[Any, Any], ...] = ()  # tuple of (Channel, value)
    descriptor: Any = None  # ActionDescriptor
    interlocks_failed: tuple[str, ...] = ()
    dry_run: bool = False
    verify: bool = False
    any_irreversible: bool = False
    channel_summaries: tuple[dict[str, Any], ...] = ()
    source: str = ""

    def __post_init__(self) -> None:
        # Deep-freeze inner dicts so a frozen snapshot stays truly immutable.
        if self.channel_summaries:
            object.__setattr__(
                self,
                "channel_summaries",
                tuple(MappingProxyType(dict(d)) for d in self.channel_summaries),
            )


@dataclass(frozen=True)
class ControlFeasibilityResult:
    """Outcome of the pre-approval feasibility evaluation for a control session.

    Carries the validated control plan, targets, and waypoints so that
    ``control_start`` becomes a thin orchestrator like ``_write``.
    """

    ok: bool
    error: dict[str, Any] | None = None
    context: Any = None  # HardwareContext
    resolved: tuple[tuple[Any, Any], ...] = ()  # tuple of (Channel, value)
    mode: str = ""
    frequency_hz: float = 0.0
    targets: dict[str, float] = field(default_factory=dict)
    waypoints: tuple[tuple[float, dict[str, float]], ...] = ()
    descriptor: Any = None  # ActionDescriptor
    interlocks_failed: tuple[str, ...] = ()
    dry_run: bool = False

    def __post_init__(self) -> None:
        # Deep-freeze mutable dicts in this frozen snapshot.
        if self.targets:
            object.__setattr__(self, "targets", MappingProxyType(dict(self.targets)))
        if self.waypoints:
            object.__setattr__(
                self,
                "waypoints",
                tuple((t, MappingProxyType(dict(d))) for t, d in self.waypoints),
            )


@dataclass(frozen=True)
class WriteGovernanceRequest:
    """Typed snapshot of a write's pre-approval decision state.

    Built after all feasibility checks pass and before the approval gate is
    consulted. Captures enough context for the approval evaluator, the audit log,
    and the dry-run preview without re-deriving anything from raw params.
    """

    device_id: str
    channel_ids: tuple[str, ...]
    values: tuple[Any, ...]
    descriptor: Any  # ActionDescriptor
    identity: str
    dry_run: bool = False
    verify: bool = False
    in_envelope: bool = True
    reversible: bool = True
    interlocks_failed: tuple[str, ...] = ()
    source: str = ""
    conditions: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Deep-freeze the metadata dict in this frozen snapshot.
        if self.metadata:
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def single(
        cls,
        *,
        device_id: str,
        channel_id: str,
        value: Any,
        descriptor: Any,
        identity: str,
        dry_run: bool = False,
        in_envelope: bool = True,
        reversible: bool = True,
        interlocks_failed: tuple[str, ...] = (),
        conditions: str = "",
    ) -> WriteGovernanceRequest:
        """Build a request for a single-channel write."""
        return cls(
            device_id=device_id,
            channel_ids=(channel_id,),
            values=(value,),
            descriptor=descriptor,
            identity=identity,
            dry_run=dry_run,
            in_envelope=in_envelope,
            reversible=reversible,
            interlocks_failed=interlocks_failed,
            conditions=conditions,
        )

    @classmethod
    def batch(
        cls,
        *,
        device_id: str,
        channel_ids: tuple[str, ...],
        values: tuple[Any, ...],
        descriptor: Any,
        identity: str,
        dry_run: bool = False,
        verify: bool = False,
        in_envelope: bool = True,
        interlocks_failed: tuple[str, ...] = (),
        source: str = "",
        reversible: bool = True,
    ) -> WriteGovernanceRequest:
        """Build a request for a multi-channel batch write."""
        return cls(
            device_id=device_id,
            channel_ids=channel_ids,
            values=values,
            descriptor=descriptor,
            identity=identity,
            dry_run=dry_run,
            verify=verify,
            in_envelope=in_envelope,
            interlocks_failed=interlocks_failed,
            source=source,
            reversible=reversible,
        )


@dataclass(frozen=True)
class GovernanceDecision:
    """Result of the approval evaluation."""

    allowed: bool
    denial: str = ""
    trust_skipped: bool = False
    no_gate: bool = False


class WriteGovernor:
    """Evaluate write approval with trust-based skip and fail-closed semantics.

    Encapsulates the approval gate, trust gate, and audit log interactions that
    were previously inlined in HardwareTools._evaluate. All invariants are
    preserved:

    - Irreversible channels always require full approval, even at PRODUCTION trust.
    - No gate installed, or a gate that raises, both mean deny.
    - Trust-based skip is opt-in and only applies to reversible + VERIFIED+ channels.
    """

    def __init__(
        self,
        *,
        gate: Any = None,
        trust_gate: Any = None,
        audit_log: Any = None,
        trust_skip_enabled: bool = False,
    ) -> None:
        self._gate = gate
        self._trust_gate = trust_gate
        self._audit = audit_log
        self._trust_skip_enabled = trust_skip_enabled

    def set_gate(self, gate: Any) -> None:
        """Replace the installed approval gate."""
        self._gate = gate

    def set_audit_log(self, audit_log: Any) -> None:
        """Replace the audit log reference."""
        self._audit = audit_log

    def set_trust_skip_enabled(self, enabled: bool) -> None:
        """Toggle trust-based approval skip."""
        self._trust_skip_enabled = enabled

    async def evaluate(
        self,
        request: WriteGovernanceRequest,
    ) -> GovernanceDecision:
        """Run approval evaluation. Fail-closed on absence and on exception.

        When trust-based skip is enabled and every channel in the request
        qualifies (reversible + trust >= VERIFIED + config on), the gate is
        bypassed. The invariant is absolute: **irreversible channels always
        require full approval**, even at PRODUCTION trust.
        """
        descriptor = request.descriptor

        # ── Trust-based short-circuit ──
        if self._trust_gate is not None and self._trust_skip_enabled:
            try:
                may_skip = self._can_trust_skip(request)
            except Exception:  # noqa: BLE001 - uncertain trust requires consent
                logger.warning("Hardware trust evaluation failed", exc_info=True)
                may_skip = False
            if may_skip:
                self._audit_trust_skip(request)
                return GovernanceDecision(allowed=True, trust_skipped=True)

        if self._gate is None:
            return GovernanceDecision(
                allowed=False,
                no_gate=True,
                denial=(
                    "No approval gate is installed for hardware commands. "
                    "Either start a TUI session (`leap chat`) or configure an "
                    "edge approval channel (Settings → hardware.approval_channel) "
                    "so writes can be reviewed before they execute."
                ),
            )

        try:
            result = await self._gate.evaluate(descriptor)
        except Exception as exc:  # noqa: BLE001 - a failing gate must deny
            logger.error(
                "Hardware approval gate raised for %s: %s",
                descriptor.kind,
                exc,
                exc_info=True,
            )
            return GovernanceDecision(
                allowed=False,
                denial=(
                    "The approval gate failed while assessing this command, so it was refused."
                ),
            )

        if getattr(result, "approved", False):
            return GovernanceDecision(allowed=True)

        message = getattr(result, "denial_message", "") or getattr(result, "reason", "")
        return GovernanceDecision(
            allowed=False,
            denial=str(message or "The command was not approved."),
        )

    def _can_trust_skip(self, request: WriteGovernanceRequest) -> bool:
        """Return whether all channels qualify for trust-based approval skip."""
        meta = getattr(request.descriptor, "metadata", None) or {}
        device_id = str(meta.get("device_id") or request.device_id)
        if not request.reversible or not bool(meta.get("reversible", True)):
            return False
        # The primary descriptor channel is a display anchor, not batch authority.
        channel_ids = set(request.channel_ids)
        primary = str(meta.get("channel_id") or "")
        if primary:
            channel_ids.add(primary)
        channel_ids.update(str(cid) for cid in meta.get("channels", ()))
        for command in meta.get("commands", ()):
            if not command.get("reversible", True):
                return False
            channel_ids.add(str(command.get("channel_id") or ""))
        return bool(channel_ids) and all(
            cid and self._trust_gate.may_skip_approval(device_id, cid, reversible=True)
            for cid in sorted(channel_ids)
        )

    def _audit_trust_skip(self, request: WriteGovernanceRequest) -> None:
        """Record the trust-based skip decision in the audit log."""
        if self._audit is None:
            return
        meta = getattr(request.descriptor, "metadata", None) or {}
        device_id = str(meta.get("device_id") or request.device_id)
        channel_id = str(meta.get("channel_id") or "")
        if not channel_id and request.channel_ids:
            channel_id = request.channel_ids[0]
        try:
            level = self._trust_gate.level(device_id, channel_id)
            level_name = level.name
        except (AttributeError, RuntimeError, TypeError, ValueError):
            level_name = "unknown"
        logger.info(
            "Trust-based approval skip: device=%s channel=%s level=%s",
            device_id,
            channel_id,
            level_name,
        )
        self._audit.record(
            action="trust_skip",
            device=device_id,
            channel=channel_id,
            outcome=f"trust_skip=True level={level_name}",
            identity=request.identity,
        )


async def execute_sequential_batch(
    transport: HardwareTransport, commands: tuple[tuple[str, Any], ...]
) -> BatchWriteOutcome:
    """Stop on the first failure and preserve evidence of earlier physical effects.

    Exceptions describe only the current call, never preceding successful writes.
    Unattempted commands carry an explicit NONE outcome and must not demote trust.
    """
    outcomes: list[WriteOutcome] = []
    stopped = False
    for channel_id, value in commands:
        if stopped:
            outcomes.append(
                WriteOutcome(
                    ok=False,
                    side_effect_state=SIDE_EFFECT_NONE,
                    failure_code="batch_not_attempted",
                    error="Not attempted because an earlier command in this batch failed.",
                )
            )
            continue
        try:
            outcome = await transport.write(channel_id, value)
        except TransportError as exc:
            logger.warning("Sequential hardware write refused on %s", channel_id, exc_info=True)
            outcome = WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_NONE,
                failure_code=exc.failure_code,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - a driver defect cannot prove no effect
            logger.error("Sequential hardware driver failed on %s", channel_id, exc_info=True)
            outcome = WriteOutcome(
                ok=False,
                side_effect_state=SIDE_EFFECT_UNKNOWN,
                failure_code="driver_contract_violation",
                error=f"Driver raised {type(exc).__name__}; verify the channel before retrying.",
            )
        outcomes.append(outcome)
        stopped = not outcome.ok
    ok = all(outcome.ok for outcome in outcomes)
    effect = worst_side_effect_state(outcome.side_effect_state for outcome in outcomes)
    if not ok and effect not in (SIDE_EFFECT_NONE, SIDE_EFFECT_UNKNOWN):
        effect = SIDE_EFFECT_PARTIAL
    return BatchWriteOutcome(ok=ok, outcomes=tuple(outcomes), side_effect_state=effect)


__all__ = [
    "execute_sequential_batch",
    "BatchFeasibilityResult",
    "ControlFeasibilityResult",
    "GovernanceDecision",
    "WriteFeasibilityResult",
    "WriteGovernanceRequest",
    "WriteGovernor",
]
