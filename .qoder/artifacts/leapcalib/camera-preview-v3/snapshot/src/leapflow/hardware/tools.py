# Copyright (c) Alibaba, Inc. and its affiliates.
"""The eight hardware tools, derived from admitted contexts.

The count is fixed regardless of how many devices exist (nine generic tools plus
three real-time control tools). A rig of seven programs
with six channels each would be 40+ schemas if tools were generated per channel,
which would swamp the tool index; instead the index stays constant and the
per-channel limits arrive through ``hw_describe`` when a model actually intends to
act. That is progressive disclosure applied to hardware, and it is also the reason
an upstream standard's \"reference file\" is a rendered view here rather than a
separate document to keep in sync.

Writes are split into three tools by effect class rather than collapsed into one
``hw_write``. Their risk profiles differ, so each maps to its own ``ActionKind``
and reaches a decision instead of a fallback; the tool name is itself a safety
signal the model cannot overlook; and if channel lookup ever fails, the name still
says what class of thing was attempted.

**Why this file exceeds 800 lines**: the twelve tool handlers and their three
feasibility evaluators (single/batch/control) share a common approval pipeline,
trust-skip path, rate evaluator, safety checker, and interlock evaluator, all of
which live on the ``HardwareTools`` instance.  ``WriteGovernor`` already owns the
approval decision.  ``WriteVerificationCoordinator`` owns the post-write verdict.
The remaining feasibility methods are tightly coupled to instance-local helpers
(``_rate_wait_s``, ``_check_safety``, ``_requires_describe``, ``_grant_band``)
that need the live registry and last-command state.  Extracting them into a
separate file would require either injecting HardwareTools or duplicating its
half-dozen stateful helpers.  The decomposition plan targets a ``FeasibilityChain``
delegate as the next structural step; the present size is the justified interim.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Mapping

from leapflow.hardware.audit import HardwareAuditLog
from leapflow.hardware.context import (
    Channel,
    HardwareContext,
    HardwareEffect,
    as_numeric,
)
from leapflow.hardware.execution_authorization import (
    ExecutionAuthorization,
    approved_authorization,
    execution_authorization,
)
from leapflow.hardware.reference import describe, summarize
from leapflow.hardware.transport import (
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_PARTIAL,
    SIDE_EFFECT_UNKNOWN,
    TransportError,
    WriteOutcome,
)
from leapflow.hardware.write_governor import (
    BatchFeasibilityResult,
    ControlFeasibilityResult,
    WriteFeasibilityResult,
    WriteGovernanceRequest,
    WriteGovernor,
    execute_sequential_batch,
)
from leapflow.hardware.write_verification import (
    WriteExpectation,
    WriteVerdict,
    WriteVerificationCoordinator,
)
from leapflow.plugins.protocol import ToolMetadata
from leapflow.security.actions import ActionDescriptor, ActionKind
from leapflow.security.permission_failures import build_readiness_failure

logger = logging.getLogger(__name__)

# Metadata shared by every write tool. Declared once because the four keys below
# are read by *different* consumers with divergent behaviour, and getting one wrong
# fails open silently:
#
#   risk_level / requires_approval  -> honoured by CapabilityManifest (PCD disclosure)
#   effect_scope / idempotency_scope -> honoured by ToolRegistry.from_definitions,
#                                       which re-infers risk_level from the tool
#                                       *name* and ignores the declared value
#
# ``effect_scope="external"`` is what makes execution_policy_for() return
# external_side_effect, which is what stops a failed physical command from being
# replayed. A physical write is irreversible by default: re-running an aspirate
# dispenses twice, and re-sending a motion command from an unknown pose is not a
# retry. Declaring only risk_level would leave the policy at mutating_idempotent,
# i.e. "safe to repeat", which is exactly the wrong default here.
_WRITE_METADATA: dict[str, Any] = {
    "category": "hardware",
    "risk_level": "external",
    "requires_approval": True,
    "effect_scope": "external",
    "idempotency_scope": "session",
    "schema_cost": "low",
}

_READ_METADATA: dict[str, Any] = {
    "category": "hardware",
    "risk_level": "read_only",
    "requires_approval": False,
    "schema_cost": "low",
}

_STORED_WINDOW_LIMIT = 12
"""How many durable history windows a read discloses.

Small on purpose. The point of persisting history is that a later decision can see what
an earlier run did, and a dozen windows answers that; handing over a full series is what
makes a long-running bench unaffordable in context.
"""

_WRITE_TOOLS: dict[str, tuple[str, str]] = {
    # tool name -> (ActionKind, declared HardwareEffect it may command)
    "hw_configure": (ActionKind.DEVICE_CONFIGURE.value, HardwareEffect.CONFIGURE.value),
    "hw_actuate": (ActionKind.DEVICE_ACTUATE.value, HardwareEffect.ACTUATE.value),
    "hw_dispense": (ActionKind.DEVICE_DISPENSE.value, HardwareEffect.DISPENSE.value),
}

_TARGET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "device_id": {"type": "string", "description": "Device id from hw_list"},
        "channel_id": {"type": "string", "description": "Channel id from hw_describe"},
    },
    "required": ["device_id", "channel_id"],
}


def _write_schema(effect: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "device_id": {"type": "string", "description": "Device id from hw_list"},
            "channel_id": {
                "type": "string",
                "description": f"Channel id declaring effect={effect}, from hw_describe",
            },
            "value": {
                "description": (
                    "Value to command. Must lie inside the channel's declared envelope; "
                    "call hw_describe first to read the allowed range and rate."
                )
            },
            "conditions": {
                "type": "string",
                "description": (
                    "What makes this situation distinctive, in plain words -- the material, "
                    "sample, or setup this value was chosen for (e.g. 'viscous BSA protein, "
                    "foams easily'). Recorded with the outcome so a later run facing the "
                    "same situation can reuse what worked instead of re-deriving it. "
                    "Optional, but omitting it makes the result much harder to recall."
                ),
            },
            "dry_run": {
                "type": "boolean",
                "description": (
                    "Preview only. Run every feasibility check (envelope, rate, "
                    "reachability, interlocks) and build the approval detail, then report "
                    "the command that would be issued without touching the device. No "
                    "physical effect occurs and no approval is requested. Defaults to false."
                ),
            },
        },
        "required": ["device_id", "channel_id", "value"],
    }


class HardwareTools:
    """Handlers for the hardware tool surface.

    Holds the registry and the approval gate rather than reaching for globals, so a
    test drives exactly the same code path as production with its own instances.
    """

    def __init__(
        self,
        registry: Any,
        *,
        gate: Any = None,
        session_id: str = "",
        audit_log: HardwareAuditLog | None = None,
        hardware_trust_gate: Any = None,
    ) -> None:
        self._registry = registry
        self._gate = gate
        self._fallback_session_id = session_id
        self._audit = audit_log or HardwareAuditLog(None)
        self._trust_gate = hardware_trust_gate
        self._write_verifier = WriteVerificationCoordinator(
            registry,
            audit_log=self._audit,
            trust_gate=hardware_trust_gate,
            identity_provider=self._session_identity,
            follow_s=getattr(
                getattr(registry, "settings", None),
                "write_verification_follow_s",
                10.0,
            ),
        )
        self._governor = WriteGovernor(
            gate=gate,
            trust_gate=hardware_trust_gate,
            audit_log=self._audit,
            trust_skip_enabled=False,
        )

    def set_gate(self, gate: Any) -> None:
        """Replace the approval gate used by all write handlers.

        Called by the hardware plugin's ``bind_runtime`` when the daemon's
        ``install_gate`` re-binds the gate after assembly.  This updates the
        live instance so already-registered tool handlers resolve to the new
        orchestrator without re-assembly.
        """
        self._gate = gate
        self._governor.set_gate(gate)

    def _session_identity(self) -> str:
        """Resolve the active request identity without sharing it across sessions."""
        from leapflow.tools.execution_context import current_tool_context

        context = current_tool_context()
        active_session_id = getattr(context, "session_id", "") if context is not None else ""
        return str(active_session_id or self._fallback_session_id)

    def _authorization(self, descriptor: ActionDescriptor) -> ExecutionAuthorization:
        """What an approved *descriptor* authorizes downstream (permits on LHP nodes)."""
        return approved_authorization(descriptor, session_id=self._session_identity())

    # ── Discovery ──

    async def hw_list(self, **_: Any) -> dict[str, Any]:
        """Return the compact device index."""
        contexts = self._registry.contexts()
        return {
            "ok": True,
            "devices": [summarize(context) for context in contexts],
            "count": len(contexts),
            "hint": "Call hw_describe(device_id) for channel limits before commanding a device.",
        }

    async def hw_describe(self, device_id: str = "", **_: Any) -> dict[str, Any]:
        """Return the full reference document for one device."""
        context = self._registry.context(device_id)
        if context is None:
            return self._unknown_device(device_id)
        self._registry.mark_described(self._session_identity(), device_id)
        payload = describe(context)
        payload["ok"] = True
        # Prior outcomes for each writable channel, best-tracking first. This is what turns
        # a reference document into accumulated experience: an optimisation performed once
        # becomes a starting point rather than an experiment to repeat.
        experience = self._prior_experience(context)
        if experience:
            payload["prior_experience"] = experience
        # When the device was last calibrated, if a calibration has ever been recorded.
        # A reference document that states a calibration age lets a reader judge whether
        # the transform behind every reading is still current, rather than assuming it.
        last_calibrated = self._last_calibrated_at(device_id)
        if last_calibrated is not None:
            payload["last_calibrated_at"] = last_calibrated
        # Annotate channels whose provenance has not been verified by a human.
        # A reading from an uncalibrated channel is still a measurement, but a
        # decision that treats it as absolute -- comparing it against a published
        # specification, for example -- would be wrong in a way the agent cannot
        # detect.  Stating it here, once, is cheaper than discovering it after
        # the fact.
        if not context.provenance.verified_by:
            payload["calibration_notice"] = (
                "Readings from this device have not been independently calibrated. "
                "Treat numeric values as relative measurements, not absolute references."
            )
        return payload

    def _prior_experience(self, context: HardwareContext) -> dict[str, Any]:
        """Return recalled outcomes keyed by writable channel, omitting empty entries."""
        recorder = self._registry.outcome_recorder
        if recorder is None:
            return {}
        recalled: dict[str, Any] = {}
        for channel in context.writable_channels:
            rows = recorder.recall(device_id=context.device_id, channel=channel)
            if rows:
                recalled[channel.channel_id] = list(rows)
        return recalled

    def _last_calibrated_at(self, device_id: str) -> float | None:
        """Return the instant of the device's most recent calibration, if any.

        Contained: the calibration store is optional (no instrument database, or a store
        that momentarily cannot be read), and a reference document must render with or
        without it -- so any failure degrades to "no calibration on record", not an error.
        """
        store = getattr(self._registry, "calibration_store", None)
        if store is None:
            return None
        try:
            return store.latest_time(device_id)
        except Exception:  # noqa: BLE001 - describe must not fail on an optional annotation
            return None

    async def hw_status(self, device_id: str = "", **_: Any) -> dict[str, Any]:
        """Return live transport health and recent observations for one device.

        Deliberately not serialised on ``device_io``: see ``HardwareTransport.probe``. This
        is the call somebody makes *because* a device looks stuck, so queueing it behind
        the read that is stuck would withhold the one answer they need.
        """
        context = self._registry.context(device_id)
        if context is None:
            return self._unknown_device(device_id)
        try:
            transport = await self._registry.transport(device_id)
            status = await transport.probe()
        except TransportError as exc:
            return {
                "ok": False,
                "device_id": device_id,
                "error": str(exc),
                "failure_code": exc.failure_code,
            }
        payload: dict[str, Any] = {
            "ok": True,
            "device_id": device_id,
            "status": status.to_dict(),
        }
        # Recent derived events, not raw samples: this is what turns "the device is
        # connected" into "here is what it has been doing", which is the question
        # actually being asked when something looks wrong.
        events = self._registry.recent_events(device_id)
        if events:
            payload["recent_events"] = [event.to_detail() for event in events]
        return payload

    # ── Data plane ──

    async def hw_read(self, device_id: str = "", channel_id: str = "", **_: Any) -> dict[str, Any]:
        """Read one channel."""
        context, channel, error = self._resolve(device_id, channel_id)
        if error is not None:
            return error
        if not channel.is_readable:
            return self._refusal(
                device_id,
                channel_id,
                "channel_not_readable",
                f"Channel {channel_id!r} is not readable on {device_id!r}.",
            )
        consent = await self._consent_for_read(context, channel)
        if consent is not None:
            return consent
        try:
            async with self._registry.device_io(device_id):
                transport = await self._registry.transport(device_id)
                reading = await transport.read(channel_id)
        except TransportError as exc:
            return {
                "ok": False,
                "device_id": device_id,
                "channel_id": channel_id,
                "error": str(exc),
                "failure_code": exc.failure_code,
            }
        payload: dict[str, Any] = {"ok": True, "reading": reading.to_dict()}
        self._audit.record(
            action="read",
            device=device_id,
            channel=channel_id,
            value=reading.value,
            outcome="ok",
            identity=self._session_identity(),
        )
        # A read is a trustworthy observation, so it is also the moment a pending command
        # on this channel can finally be scored -- which is how a channel with settling
        # time gets learned from at all.
        recorder = self._registry.outcome_recorder
        resolved = None
        if recorder is not None:
            resolved = recorder.observe(
                device_id=device_id, channel_id=channel_id, value=reading.value
            )
            if resolved is not None:
                payload["command_outcome"] = resolved.to_dict()
        write_verdicts = self._write_verifier.settle_pending(
            device_id,
            channel_id,
            reading,
            resolved_outcome=resolved,
        )
        if write_verdicts:
            payload["write_verdicts"] = [verdict.to_dict() for verdict in write_verdicts]
            if len(write_verdicts) == 1:
                payload["verification"] = write_verdicts[0].to_dict()
        # Sampled history arrives as a summary, never as the raw series: the series is
        # what makes a long run unaffordable in context, and the summary is what a
        # decision needs -- where the value is, where it has been, whether it drifted.
        summary = self._registry.channel_summary(device_id, channel_id)
        if summary:
            payload["history"] = summary
        # Durable windows from earlier runs, newest last and deliberately few. This is the
        # only path by which a previous session's physical behaviour reaches a decision in
        # this one; before it existed, samples vanished with the process.
        stored = self._registry.channel_history(device_id, channel_id, limit=_STORED_WINDOW_LIMIT)
        if stored:
            payload["stored_windows"] = list(stored)
        return payload

    async def hw_configure(self, **params: Any) -> dict[str, Any]:
        return await self._write("hw_configure", params)

    async def hw_actuate(self, **params: Any) -> dict[str, Any]:
        return await self._write("hw_actuate", params)

    async def hw_dispense(self, **params: Any) -> dict[str, Any]:
        return await self._write("hw_dispense", params)

    async def hw_estop(self, device_id: str = "", **_: Any) -> dict[str, Any]:
        """Halt a device immediately.

        Deliberately ungated: waiting for consent to stop a moving machine is
        physically absurd. It is still audited, because frequent halts are a fault
        signal worth keeping.

        Un-serialised for the same class of reason -- see ``HardwareTransport.halt``.
        Whatever holds the device's I/O lock is often the operation that needs stopping,
        so taking the lock first would delay the halt by the runaway operation's duration.
        """
        context = self._registry.context(device_id)
        if context is None:
            return self._unknown_device(device_id)
        try:
            # The stop has to reach the device, so a link is opened when none is
            # open; the registry then retires it if the driver closed it.
            await self._registry.transport(device_id)
            status = await self._registry.halt(device_id)
        except TransportError as exc:
            logger.error("Emergency stop for %r failed: %s", device_id, exc, exc_info=True)
            return {
                "ok": False,
                "device_id": device_id,
                "error": str(exc),
                "failure_code": exc.failure_code,
            }
        if status is None:  # the link was dropped between the two calls
            return {
                "ok": False,
                "device_id": device_id,
                "error": f"no link is open to {device_id!r}",
                "failure_code": "transport_not_open",
            }
        logger.warning(
            "Emergency stop issued device=%s supported=%s delivered=%s detail=%s",
            device_id,
            status.halt_supported,
            status.halt_delivered,
            status.detail,
        )
        outcome = ("undelivered" if not status.halt_delivered
                   else "ok" if status.halt_supported else "unsupported")
        self._audit.record(
            action="estop",
            device=device_id,
            outcome=outcome,
            identity=self._session_identity(),
        )
        # When the device declares an emergency deceleration time, report it so
        # the caller knows how long the device needs to reach zero velocity.
        # The transport may not enforce the wait itself (some drivers issue a
        # halt command and return immediately), so surfacing the value here
        # lets the agent or the operator decide whether to wait before
        # re-commanding the device.
        result: dict[str, Any] = {
            "ok": status.halt_supported and status.halt_delivered,
            "device_id": device_id,
            "halted": status.halt_supported and status.halt_delivered,
            "status": status.to_dict(),
        }
        safety = context.safety
        if safety is not None and safety.emergency_decel_s > 0:
            result["emergency_decel_s"] = safety.emergency_decel_s
            result["note"] = (
                f"Device declares {safety.emergency_decel_s:.3f}s emergency "
                "deceleration time; wait at least that long before re-commanding."
            )
        return result

    # ── Batch write ──

    async def batch_actuate(self, params: dict) -> dict:
        """Write multiple ACTUATE channels in one atomic bus transaction.

        The structure is a three-phase pipeline mirroring ``_write``:

        1. **Feasibility** (_check_batch_feasibility): per-channel resolve,
           writable, effect, envelope, rate, safety, degraded, reachable,
           interlocks, descriptor construction.  Returns a BatchFeasibilityResult.
        2. **Approval** (_evaluate via WriteGovernor): trust skip or gate.
        3. **Execution** (_execute_batch_write): transport write, per-channel
           audit, trust, verification, result aggregation.
        """
        feasibility = await self._check_batch_feasibility(params)
        if not feasibility.ok:
            return feasibility.error  # type: ignore[return-value]

        device_id = feasibility.context.device_id

        if feasibility.dry_run:
            return _batch_preview_result(feasibility)

        if feasibility.interlocks_failed:
            return self._not_ready(
                feasibility.context,
                feasibility.resolved[0][0],
                feasibility.interlocks_failed,
            )

        allowed, denial, no_gate = await self._evaluate(feasibility.descriptor)
        if not allowed:
            result = self._refusal(device_id, "", "approval_denied", denial)
            if no_gate:
                result["interaction_request"] = no_gate_interaction(device_id)
            return result

        with execution_authorization(self._authorization(feasibility.descriptor)):
            return await self._execute_batch_write(feasibility)

    async def _check_batch_feasibility(self, params: Mapping[str, Any]) -> BatchFeasibilityResult:
        """Run the full pre-approval feasibility chain for a batch write.

        Validates every command in the batch sequentially: resolve, writable,
        effect class, envelope, rate.  Then device-level: safety, degradation,
        reachability, interlocks.  On failure returns a BatchFeasibilityResult
        with ``ok=False`` and the refusal payload in ``error``.
        """
        device_id = str(params.get("device_id") or "")
        commands = params.get("commands") or []
        dry_run = bool(params.get("dry_run"))
        verify = bool(params.get("verify"))
        source = str(params.get("source") or "")

        context = self._registry.context(device_id)
        if context is None:
            return BatchFeasibilityResult(ok=False, error=self._unknown_device(device_id))

        if self._requires_describe(device_id):
            return BatchFeasibilityResult(
                ok=False,
                error=self._refusal(
                    device_id,
                    "",
                    "describe_required",
                    f"Call hw_describe(device_id={device_id!r}) before commanding it. Writing to "
                    "channels whose envelope you have not read risks out-of-range commands; the "
                    "describe result carries the allowed range, rate, and reversibility.",
                ),
            )

        if not commands:
            return BatchFeasibilityResult(
                ok=False,
                error=self._refusal(
                    device_id, "", "empty_commands", "No commands provided in the batch."
                ),
            )

        # ── Validate every channel ──
        resolved: list[tuple[Channel, Any]] = []
        for idx, cmd in enumerate(commands):
            channel_id = str(cmd.get("channel_id") or "")
            value = cmd.get("value")
            channel = context.channel(channel_id)
            if channel is None:
                known = ", ".join(c.channel_id for c in context.channels) or "(none)"
                return BatchFeasibilityResult(
                    ok=False,
                    error=self._refusal(
                        device_id,
                        channel_id,
                        "unknown_channel",
                        f"Command [{idx}]: device {device_id!r} has no channel "
                        f"{channel_id!r}. Declared: {known}.",
                    ),
                )
            if not channel.is_writable:
                return BatchFeasibilityResult(
                    ok=False,
                    error=self._refusal(
                        device_id,
                        channel_id,
                        "channel_not_writable",
                        f"Command [{idx}]: channel {channel_id!r} on {device_id!r} is read-only "
                        "in the admitted declaration.",
                    ),
                )
            if channel.effect != HardwareEffect.ACTUATE.value:
                return BatchFeasibilityResult(
                    ok=False,
                    error=self._refusal(
                        device_id,
                        channel_id,
                        "effect_class_mismatch",
                        f"Command [{idx}]: channel {channel_id!r} declares effect "
                        f"{channel.effect!r}; hw_batch_actuate requires effect=actuate.",
                    ),
                )

            envelope = channel.envelope
            if not envelope.contains(value):
                return BatchFeasibilityResult(
                    ok=False,
                    error=self._refusal(
                        device_id,
                        channel_id,
                        "value_out_of_envelope",
                        f"Command [{idx}]: value {value!r} lies outside the declared "
                        f"envelope for {device_id}.{channel_id}. Call hw_describe to "
                        "read the allowed range.",
                    ),
                )

            wait_s = self._rate_wait_s(device_id, channel, value)
            if wait_s > 0.0:
                return BatchFeasibilityResult(
                    ok=False,
                    error={
                        "ok": False,
                        "device_id": device_id,
                        "channel_id": channel_id,
                        "error": (
                            f"Command [{idx}]: commanding {device_id}.{channel_id} to {value} now "
                            f"would exceed its declared maximum rate of "
                            f"{envelope.max_rate:g} {channel.unit or 'units'}/s. Wait about "
                            f"{wait_s:.2f}s and issue the same command again, or command a "
                            "smaller step."
                        ),
                        "failure_code": "rate_limited",
                        "retry_after_s": round(wait_s, 3),
                        "side_effect_state": SIDE_EFFECT_NONE,
                    },
                )
            resolved.append((channel, value))

        # ── SafetyPolicy (device-level, once per command) ──
        for idx, (channel, value) in enumerate(resolved):
            safety_ok, safety_reason = self._check_safety(context, channel, value)
            if not safety_ok:
                return BatchFeasibilityResult(
                    ok=False,
                    error={
                        "ok": False,
                        "device_id": device_id,
                        "channel_id": channel.channel_id,
                        "error": (f"Command [{idx}]: safety policy violation: {safety_reason}"),
                        "failure_code": "safety_policy_violation",
                        "side_effect_state": SIDE_EFFECT_NONE,
                    },
                )

        # ── DegradationPolicy (device-level, once) ──
        if self._registry.is_device_degraded(device_id):
            return BatchFeasibilityResult(
                ok=False,
                error=self._refusal(
                    device_id,
                    "",
                    "device_degraded",
                    "Device is in degraded state (communication or sensor loss). "
                    "Commands blocked until recovery.",
                ),
            )

        # ── Reachability (once for the device) ──
        unreachable = await self._unreachable(device_id, resolved[0][0])
        if unreachable is not None:
            return BatchFeasibilityResult(ok=False, error=unreachable)

        # ── Interlocks (union of all channels) ──
        all_interlocks_failed: list[str] = []
        for channel, _value in resolved:
            failed = await self._failed_interlocks(context, channel)
            for name in failed:
                if name not in all_interlocks_failed:
                    all_interlocks_failed.append(name)

        # ── Build batch approval descriptor ──
        any_irreversible = any(not ch.envelope.reversible for ch, _ in resolved)
        channel_summaries = tuple(
            {
                "channel_id": ch.channel_id,
                "quantity": ch.quantity,
                "value": v,
                "unit": ch.unit,
                "envelope_band": self._grant_band(ch.envelope, v),
                "reversible": ch.envelope.reversible,
            }
            for ch, v in resolved
        )
        descriptor = ActionDescriptor.device(
            kind=ActionKind.DEVICE_ACTUATE.value,
            device_id=device_id,
            channel_id=resolved[0][0].channel_id,
            quantity="batch",
            value=f"{len(resolved)} channels",
            unit="",
            envelope_band=_batch_grant_identity(channel_summaries),
            location=context.location,
            reversible=not any_irreversible,
            metadata={
                "batch": True,
                "commands": list(channel_summaries),
                "interlocks_satisfied": not all_interlocks_failed,
                "interlocks_failed": list(all_interlocks_failed),
                "session_id": self._session_identity(),
                "grant_identity": _batch_grant_identity(channel_summaries),
                **({"source": source} if source else {}),
            },
        )

        return BatchFeasibilityResult(
            ok=True,
            context=context,
            resolved=tuple(resolved),
            descriptor=descriptor,
            interlocks_failed=tuple(all_interlocks_failed),
            dry_run=dry_run,
            verify=verify,
            any_irreversible=any_irreversible,
            channel_summaries=tuple(channel_summaries),
            source=source,
        )

    async def _execute_batch_write(self, feasibility: BatchFeasibilityResult) -> dict[str, Any]:
        """Execute the approved batch write: transport, audit, trust, verification."""
        from leapflow.hardware.transport import BatchTransport

        device_id = feasibility.context.device_id
        resolved = list(feasibility.resolved)
        context = feasibility.context

        try:
            async with self._registry.device_io(device_id):
                transport = await self._registry.transport(device_id)
                if isinstance(transport, BatchTransport):
                    command_tuples = tuple((ch.channel_id, v) for ch, v in resolved)
                    batch_outcome = await transport.write_batch(command_tuples)
                else:
                    logger.warning(
                        "Transport for %s does not support BatchTransport; "
                        "falling back to sequential writes (non-atomic)",
                        device_id,
                    )
                    batch_outcome = await execute_sequential_batch(
                        transport, tuple((ch.channel_id, v) for ch, v in resolved)
                    )
        except TransportError as exc:
            if self._trust_gate is not None:
                for ch, _ in resolved:
                    self._trust_gate.record_failure(device_id, ch.channel_id)
            return {
                "ok": False,
                "device_id": device_id,
                "error": str(exc),
                "failure_code": exc.failure_code,
                "side_effect_state": SIDE_EFFECT_NONE,
            }
        except Exception as exc:  # noqa: BLE001 - see _write for rationale
            logger.error(
                "Hardware transport raised a non-contract exception on batch write to %s: %s",
                device_id,
                exc,
                exc_info=True,
            )
            if self._trust_gate is not None:
                for ch, _ in resolved:
                    self._trust_gate.record_failure(device_id, ch.channel_id, hard=True)
            return {
                "ok": False,
                "device_id": device_id,
                "error": (
                    f"The device driver failed unexpectedly ({type(exc).__name__}). "
                    "Whether the commands reached the device is unknown; verify each "
                    "channel before attempting anything similar."
                ),
                "failure_code": "driver_contract_violation",
                "side_effect_state": SIDE_EFFECT_UNKNOWN,
                "effect_uncertain": True,
            }

        # ── Post-write bookkeeping per channel ──
        expectation_time = time.time()
        identity = self._session_identity()
        expectations = [
            WriteExpectation.from_channel(
                device_id=device_id,
                channel=channel,
                commanded_value=value,
                now=expectation_time,
                identity=identity,
            )
            for channel, value in resolved
        ]
        resolved_outcomes: list[Any | None] = []
        per_channel_results: list[dict] = []
        for i, (channel, value) in enumerate(resolved):
            outcome = batch_outcome.outcomes[i] if i < len(batch_outcome.outcomes) else None
            ch_ok = outcome.ok if outcome else False
            resolved_outcome = None

            self._audit.record(
                action="batch_write",
                device=device_id,
                channel=channel.channel_id,
                value=value,
                outcome="ok" if ch_ok else (outcome.failure_code if outcome else "error"),
                identity=self._session_identity(),
            )

            not_attempted = outcome is not None and outcome.failure_code == "batch_not_attempted"
            if self._trust_gate is not None and not ch_ok and not not_attempted:
                self._trust_gate.record_failure(
                    device_id,
                    channel.channel_id,
                    hard=outcome is not None
                    and outcome.failure_code == "driver_contract_violation",
                )

            if ch_ok:
                numeric = as_numeric(value)
                if numeric is not None:
                    self._registry.record_command(device_id, channel.channel_id, numeric)
                if outcome:
                    resolved_outcome = self._learn_from_write(
                        device_id, channel, value, "", outcome
                    )
            elif not not_attempted:
                recorder = self._registry.outcome_recorder
                if recorder is not None:
                    recorder.drop_pending(device_id, channel.channel_id)
            resolved_outcomes.append(resolved_outcome)

            ch_result: dict = {
                "channel_id": channel.channel_id,
                "ok": ch_ok,
            }
            if outcome:
                ch_result["side_effect_state"] = outcome.side_effect_state
                if outcome.failure_code:
                    ch_result["failure_code"] = outcome.failure_code
                if not_attempted:
                    ch_result["attempted"] = False
                if outcome.readback is not None:
                    ch_result["readback"] = outcome.readback.to_dict()
                if not ch_ok and outcome.error:
                    ch_result["error"] = outcome.error
            per_channel_results.append(ch_result)

        # ── Typed physical-effect verification ──
        if batch_outcome.ok:
            if feasibility.verify:
                verdicts = await self._write_verifier.verify_batch(
                    device_id,
                    resolved,
                    expectations,
                    resolved_outcomes,
                )
            else:
                verdicts = [
                    self._write_verifier.defer_or_refuse(context, expectation)
                    for expectation in expectations
                ]
            for channel_result, verdict in zip(per_channel_results, verdicts, strict=True):
                channel_result["verification"] = verdict.to_dict()

        payload: dict = {
            "ok": batch_outcome.ok,
            "device_id": device_id,
            "side_effect_state": batch_outcome.side_effect_state,
            "batch_size": len(resolved),
            "channels": per_channel_results,
        }
        if not batch_outcome.ok:
            failed_channels = [r["channel_id"] for r in per_channel_results if not r.get("ok")]
            payload["failed_channels"] = failed_channels
            if batch_outcome.side_effect_state != SIDE_EFFECT_NONE:
                payload["effect_uncertain"] = True
                payload["next_step"] = (
                    "The batch did not complete and some effects may have landed. "
                    "Verify each attempted channel before resuming; do not replay the batch."
                )
            if any(not ch.envelope.reversible for ch, _ in resolved):
                payload["effect_uncertain"] = True
                payload["next_step"] = (
                    "Some commands in the batch failed but may have reached the device. "
                    "Read back each channel before attempting a retry; the batch contains "
                    "irreversible channels."
                )
        return payload

    @staticmethod
    def _worst_side_effect(states: list[str]) -> str:
        """Return the worst-case side-effect state across a list of outcomes.

        Precedence: UNKNOWN > PARTIAL > COMMITTED > NONE.
        """
        _PRECEDENCE = {
            SIDE_EFFECT_NONE: 0,
            SIDE_EFFECT_COMMITTED: 1,
            SIDE_EFFECT_PARTIAL: 2,
            SIDE_EFFECT_UNKNOWN: 3,
        }
        worst = SIDE_EFFECT_NONE
        for s in states:
            if _PRECEDENCE.get(s, 3) > _PRECEDENCE.get(worst, 0):
                worst = s
        return worst

    # ── Write path ──

    # ── Real-time control tools ──

    async def control_start(self, **params: Any) -> dict[str, Any]:
        """Validate, approve, and start a local real-time control session.

        The structure is a three-phase pipeline mirroring ``_write``:

        1. **Feasibility** (_check_control_feasibility): resolve, describe,
           LHP rejection, frequency, control plan validation, degradation,
           reachability, interlocks, descriptor construction.
        2. **Approval** (_evaluate via WriteGovernor): trust skip or gate.
        3. **Execution** (_execute_control_start): policy build, loop start,
           audit.
        """
        feasibility = await self._check_control_feasibility(params)
        if not feasibility.ok:
            return feasibility.error  # type: ignore[return-value]

        device_id = feasibility.context.device_id

        if feasibility.dry_run:
            return _control_preview_result(feasibility)

        if feasibility.interlocks_failed:
            return self._not_ready(
                feasibility.context,
                feasibility.resolved[0][0],
                feasibility.interlocks_failed,
            )

        allowed, denial, no_gate = await self._evaluate(feasibility.descriptor)
        if not allowed:
            result = self._refusal(device_id, "", "approval_denied", denial)
            if no_gate:
                result["interaction_request"] = no_gate_interaction(device_id)
            return result

        with execution_authorization(self._authorization(feasibility.descriptor)):
            return await self._execute_control_start(feasibility, params)

    async def _check_control_feasibility(
        self, params: Mapping[str, Any]
    ) -> ControlFeasibilityResult:
        """Run the full pre-approval feasibility chain for a control session.

        Returns a ``ControlFeasibilityResult`` whose ``ok`` field indicates
        whether the session passed all checks. On failure, ``error`` carries
        the refusal payload; on success all plan fields are populated.
        """
        device_id = str(params.get("device_id") or "")
        mode = str(params.get("mode") or "")
        dry_run = bool(params.get("dry_run"))

        context = self._registry.context(device_id)
        if context is None:
            return ControlFeasibilityResult(ok=False, error=self._unknown_device(device_id))
        if self._requires_describe(device_id):
            return ControlFeasibilityResult(
                ok=False,
                error=self._refusal(
                    device_id,
                    "",
                    "describe_required",
                    f"Call hw_describe(device_id={device_id!r}) before opening a control session.",
                ),
            )
        if context.transport.kind == "lhp":
            # Remote control via control.v1 — route through RemoteControlTransport.
            # We check the *already-opened* transport to avoid forcing a
            # connection during feasibility evaluation.  If the transport
            # hasn't been opened yet (no prior read/write), we can't know
            # whether the node advertises control.v1 -- reject early.
            from leapflow.hardware.transport import RemoteControlTransport

            try:
                existing_transport = self._registry.get_open_transport(device_id)
            except Exception:  # noqa: BLE001
                existing_transport = None
            if existing_transport is None or not isinstance(
                existing_transport, RemoteControlTransport
            ):
                return ControlFeasibilityResult(
                    ok=False,
                    error=self._refusal(
                        device_id,
                        "",
                        "remote_control_unsupported",
                        "Node does not advertise control.v1 capability; "
                        "remote control sessions are not available.",
                    ),
                )
            # Fall through: LHP with RemoteControlTransport is allowed
            pass
        try:
            frequency_hz = float(params.get("frequency_hz", 100.0))
        except (TypeError, ValueError):
            frequency_hz = 0.0
        if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
            return ControlFeasibilityResult(
                ok=False,
                error=self._refusal(
                    device_id,
                    "",
                    "invalid_frequency",
                    "frequency_hz must be a positive finite number.",
                ),
            )

        plan, error = self._control_plan(context, mode, params)
        if error is not None:
            return ControlFeasibilityResult(ok=False, error=error)
        assert plan is not None
        resolved, targets, waypoints = plan

        if self._registry.is_device_degraded(device_id):
            return ControlFeasibilityResult(
                ok=False,
                error=self._refusal(
                    device_id,
                    "",
                    "device_degraded",
                    "Device is degraded; control startup is blocked until recovery.",
                ),
            )
        unreachable = await self._unreachable(device_id, resolved[0][0])
        if unreachable is not None:
            return ControlFeasibilityResult(ok=False, error=unreachable)

        failed_interlocks = await self._control_interlocks(context, resolved)
        descriptor = self._control_descriptor(
            context,
            mode,
            frequency_hz,
            resolved,
            failed_interlocks,
        )

        return ControlFeasibilityResult(
            ok=True,
            context=context,
            resolved=tuple(resolved),
            mode=mode,
            frequency_hz=frequency_hz,
            targets=targets,
            waypoints=tuple(waypoints),
            descriptor=descriptor,
            interlocks_failed=failed_interlocks,
            dry_run=dry_run,
        )

    async def _execute_control_start(
        self,
        feasibility: ControlFeasibilityResult,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the control policy, start the loop, and record audit."""
        device_id = feasibility.context.device_id
        mode = feasibility.mode
        frequency_hz = feasibility.frequency_hz
        resolved = list(feasibility.resolved)

        # Remote device (LHP with control.v1) → delegate to RemoteControlTransport
        if feasibility.context.transport.kind == "lhp":
            return await self._execute_remote_control_start(feasibility, params)

        policy, policy_error = self._build_control_policy(
            mode,
            tuple(channel.channel_id for channel, _ in resolved),
            feasibility.targets,
            feasibility.waypoints,
            params,
        )
        if policy_error is not None:
            return policy_error
        try:
            await self._registry.transport(device_id)
            from leapflow.hardware.control_bus import ControlBusConfig

            self._registry.start_control_loop(
                device_id,
                policy,
                config=ControlBusConfig(frequency_hz=frequency_hz),
            )
        except (RuntimeError, ValueError) as exc:
            return self._refusal(
                device_id,
                "",
                "control_start_failed",
                str(exc),
                side_effect_state=SIDE_EFFECT_UNKNOWN,
            )

        self._audit.record(
            action="control_start",
            device=device_id,
            value=mode,
            outcome="ok",
            identity=self._session_identity(),
        )
        return {
            "ok": True,
            "device_id": device_id,
            "mode": mode,
            "frequency_hz": frequency_hz,
            "policy_id": getattr(policy, "policy_id", mode),
            "side_effect_state": SIDE_EFFECT_COMMITTED,
        }

    async def _execute_remote_control_start(
        self,
        feasibility: ControlFeasibilityResult,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Start a remote control session via RemoteControlTransport (control.v1)."""
        from leapflow.hardware.transport import RemoteControlTransport

        device_id = feasibility.context.device_id
        mode = feasibility.mode
        frequency_hz = feasibility.frequency_hz
        resolved = list(feasibility.resolved)

        transport = await self._registry.transport(device_id)
        if not isinstance(transport, RemoteControlTransport):
            return self._refusal(
                device_id,
                "",
                "remote_control_unsupported",
                "transport does not support remote control sessions",
            )
        lease_s = float(params.get("lease_s", 1.0) or 1.0)
        comm_loss_policy = str(params.get("comm_loss_policy", "halt") or "halt")
        try:
            result = await transport.open_control(
                device_id=device_id,
                mode=mode,
                frequency_hz=frequency_hz,
                channels=[ch.channel_id for ch, _ in resolved],
                targets=feasibility.targets,
                lease_s=lease_s,
                comm_loss_policy=comm_loss_policy,
            )
        except Exception as exc:  # noqa: BLE001
            return self._refusal(
                device_id,
                "",
                "control_start_failed",
                str(exc),
                side_effect_state=SIDE_EFFECT_UNKNOWN,
            )
        self._audit.record(
            action="control_start",
            device=device_id,
            value=f"remote:{mode}",
            outcome="ok",
            identity=self._session_identity(),
        )
        result["ok"] = True
        result["device_id"] = device_id
        result["mode"] = mode
        result["frequency_hz"] = frequency_hz
        result["side_effect_state"] = SIDE_EFFECT_COMMITTED
        return result

    def _control_points(
        self, device_id: str, mode: str, params: Mapping[str, Any]
    ) -> tuple[
        list[tuple[float, Mapping[str, Any]]] | None,
        dict[str, Any] | None,
    ]:
        """Parse control targets without allowing malformed inputs to escape."""
        if mode in {"pid", "impedance"}:
            targets = params.get("targets")
            if not isinstance(targets, Mapping) or not targets:
                return None, self._refusal(
                    device_id,
                    "",
                    "missing_targets",
                    f"targets are required for {mode} mode.",
                )
            return [(0.0, targets)], None
        if mode != "trajectory":
            return None, self._refusal(
                device_id,
                "",
                "unknown_mode",
                f"Unknown control mode {mode!r}; supported: pid, impedance, trajectory.",
            )

        trajectory = params.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            return None, self._refusal(
                device_id,
                "",
                "missing_trajectory",
                "A non-empty trajectory is required for trajectory mode.",
            )
        points: list[tuple[float, Mapping[str, Any]]] = []
        for index, waypoint in enumerate(trajectory):
            if not isinstance(waypoint, Mapping) or not isinstance(
                waypoint.get("positions"), Mapping
            ):
                return None, self._refusal(
                    device_id,
                    "",
                    "invalid_trajectory",
                    f"Waypoint [{index}] must contain a positions object.",
                )
            try:
                point_time = float(waypoint.get("t", index))
            except (TypeError, ValueError):
                return None, self._refusal(
                    device_id,
                    "",
                    "invalid_trajectory_time",
                    f"Waypoint [{index}] time must be a finite number.",
                )
            points.append((point_time, waypoint["positions"]))
        return points, None

    def _control_plan(
        self, context: HardwareContext, mode: str, params: Mapping[str, Any]
    ) -> tuple[
        tuple[
            list[tuple[Channel, Any]],
            dict[str, float],
            tuple[tuple[float, dict[str, float]], ...],
        ]
        | None,
        dict[str, Any] | None,
    ]:
        """Normalize and validate every setpoint before control startup."""
        device_id = context.device_id
        channels = {
            channel.channel_id: channel
            for channel in context.channels
            if channel.is_writable and channel.effect == HardwareEffect.ACTUATE.value
        }
        if not channels:
            return None, self._refusal(
                device_id,
                "",
                "no_actuation_channels",
                f"Device {device_id!r} has no writable effect=actuate channels.",
            )
        raw_points, point_error = self._control_points(device_id, mode, params)
        if point_error is not None:
            return None, point_error
        if raw_points is None:
            return None, self._refusal(
                device_id,
                "",
                "invalid_control_plan",
                "Control targets could not be normalized.",
            )

        resolved_by_id: dict[str, tuple[Channel, Any]] = {}
        normalized_points: list[tuple[float, dict[str, float]]] = []
        previous_time = -1.0
        previous_values: dict[str, tuple[float, float]] = {}
        for point_index, (point_time, values) in enumerate(raw_points):
            if not math.isfinite(point_time) or point_time < previous_time:
                return None, self._refusal(
                    device_id,
                    "",
                    "invalid_trajectory_time",
                    f"Waypoint [{point_index}] time must be finite and nondecreasing.",
                )
            normalized: dict[str, float] = {}
            for channel_id, raw_value in values.items():
                channel = channels.get(str(channel_id))
                if channel is None:
                    return None, self._refusal(
                        device_id,
                        str(channel_id),
                        "invalid_control_channel",
                        f"Control channel {channel_id!r} must be writable with effect=actuate.",
                    )
                numeric = as_numeric(raw_value)
                if numeric is None or not channel.envelope.contains(numeric):
                    return None, self._refusal(
                        device_id,
                        channel.channel_id,
                        "value_out_of_envelope",
                        f"Control value {raw_value!r} is outside {device_id}.{channel.channel_id}'s envelope.",
                    )
                safety_ok, reason = self._check_safety(context, channel, numeric)
                if not safety_ok:
                    return None, self._refusal(
                        device_id,
                        channel.channel_id,
                        "safety_policy_violation",
                        reason,
                    )
                prior = previous_values.get(channel.channel_id)
                if (
                    prior is not None
                    and channel.envelope.rate_wait_s(
                        delta=numeric - prior[1],
                        elapsed_s=point_time - prior[0],
                    )
                    > 0.0
                ):
                    return None, self._refusal(
                        device_id,
                        channel.channel_id,
                        "trajectory_rate_exceeded",
                        f"Waypoint [{point_index}] exceeds {channel.channel_id}'s declared maximum rate.",
                    )
                normalized[channel.channel_id] = numeric
                previous_values[channel.channel_id] = (point_time, numeric)
                resolved_by_id[channel.channel_id] = (channel, numeric)
            if not normalized:
                return None, self._refusal(
                    device_id,
                    "",
                    "empty_control_point",
                    f"Control point [{point_index}] contains no actuator targets.",
                )
            normalized_points.append((point_time, normalized))
            previous_time = point_time
        final_targets = dict(normalized_points[-1][1])
        return (list(resolved_by_id.values()), final_targets, tuple(normalized_points)), None

    async def _control_interlocks(
        self, context: HardwareContext, resolved: list[tuple[Channel, Any]]
    ) -> tuple[str, ...]:
        failed: list[str] = []
        for channel, _ in resolved:
            for name in await self._failed_interlocks(context, channel):
                if name not in failed:
                    failed.append(name)
        return tuple(failed)

    def _control_descriptor(
        self,
        context: HardwareContext,
        mode: str,
        frequency_hz: float,
        resolved: list[tuple[Channel, Any]],
        failed_interlocks: tuple[str, ...],
    ) -> ActionDescriptor:
        primary = resolved[0][0]
        # Per-channel band + mode/frequency superset forms the grant identity.
        channel_bands = sorted(f"{ch.channel_id}:{ch.envelope.band_key()}" for ch, _ in resolved)
        grant_id = f"control:{mode}@{frequency_hz:g}Hz|" + "|".join(channel_bands)
        return ActionDescriptor.device(
            kind=ActionKind.DEVICE_ACTUATE.value,
            device_id=context.device_id,
            channel_id=primary.channel_id,
            quantity="control_session",
            value=f"{mode} at {frequency_hz:g} Hz",
            envelope_band=grant_id,
            location=context.location,
            reversible=all(channel.envelope.reversible for channel, _ in resolved),
            metadata={
                "control_session": True,
                "mode": mode,
                "frequency_hz": frequency_hz,
                "channels": [channel.channel_id for channel, _ in resolved],
                "value_in_envelope": True,
                "interlocks_satisfied": not failed_interlocks,
                "interlocks_failed": list(failed_interlocks),
                "session_id": self._session_identity(),
                "grant_identity": grant_id,
            },
        )

    def _build_control_policy(
        self,
        mode: str,
        joint_ids: tuple[str, ...],
        targets: dict[str, float],
        waypoints: tuple[tuple[float, dict[str, float]], ...],
        params: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any] | None]:
        from leapflow.hardware.control_policies import (
            build_control_policy as _shared_build,
        )

        policy, error_msg = _shared_build(mode, joint_ids, targets, waypoints, params)
        if error_msg is not None:
            return None, self._refusal("", "", "invalid_control_policy", error_msg)
        return policy, None

    async def control_stop(self, **params: Any) -> dict[str, Any]:
        """Stop the real-time control loop on a device."""
        device_id = str(params.get("device_id") or "")
        context = self._registry.context(device_id)
        if context is None:
            return self._unknown_device(device_id)

        result = self._registry.stop_control_loop(device_id)
        self._audit.record(
            action="control_stop",
            device=device_id,
            outcome=result.get("status", "unknown"),
            identity=self._session_identity(),
        )
        if result.get("status") == "no_loop":
            result.update(
                ok=False,
                failure_code="not_running",
                counts_as_failure=False,
                side_effect_state=SIDE_EFFECT_NONE,
            )
            return result
        result["ok"] = True
        return result

    async def control_status(self, **params: Any) -> dict[str, Any]:
        """Return the status of a running control loop."""
        device_id = str(params.get("device_id") or "")
        context = self._registry.context(device_id)
        if context is None:
            return self._unknown_device(device_id)

        status = self._registry.control_loop_status(device_id)
        if status is None:
            return {
                "ok": True,
                "device_id": device_id,
                "status": "no_loop",
            }
        status["ok"] = True
        return status

    # ── Write path (single-channel) ──

    async def _write(self, tool_name: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """Validate, gate, and execute one physical write.

        The structure is a three-phase pipeline:

        1. **Feasibility** (_check_write_feasibility): resolve, writable, effect,
           describe, envelope, rate, safety, degraded, reachable, interlocks,
           descriptor construction. Returns a WriteFeasibilityResult.
        2. **Approval** (_evaluate via WriteGovernor): trust skip or gate.
        3. **Execution** (_execute_single_write): transport write, audit, trust,
           verification.
        """
        feasibility = await self._check_write_feasibility(tool_name, params)
        if not feasibility.ok:
            return feasibility.error  # type: ignore[return-value]

        if feasibility.dry_run:
            return _preview_result(
                feasibility.context.device_id,
                feasibility.channel,
                feasibility.descriptor,
                value=feasibility.value,
                in_envelope=feasibility.in_envelope,
                interlocks_failed=feasibility.interlocks_failed,
            )

        if feasibility.interlocks_failed:
            return self._not_ready(
                feasibility.context,
                feasibility.channel,
                feasibility.interlocks_failed,
            )

        allowed, denial, no_gate = await self._evaluate(feasibility.descriptor)
        if not allowed:
            result = self._refusal(
                feasibility.context.device_id,
                feasibility.channel.channel_id,
                "approval_denied",
                denial,
            )
            if no_gate:
                result["interaction_request"] = no_gate_interaction(feasibility.context.device_id)
            return result

        with execution_authorization(self._authorization(feasibility.descriptor)):
            return await self._execute_single_write(feasibility)

    async def _check_write_feasibility(
        self, tool_name: str, params: Mapping[str, Any]
    ) -> WriteFeasibilityResult:
        """Run the full pre-approval feasibility chain for a single-channel write.

        Order is fixed and matches the platform rule that feasibility precedes
        consent: resolve the target, check that the command *could* succeed, and
        only then present it for approval. Prompting for a command that will be
        refused anyway teaches people to click through prompts.

        Returns a ``WriteFeasibilityResult`` whose ``ok`` field indicates whether
        the command passed all checks. On failure, ``error`` carries the refusal
        payload; on success, ``context``, ``channel``, ``descriptor``, and
        ``value`` are all populated.
        """
        kind, expected_effect = _WRITE_TOOLS[tool_name]
        device_id = str(params.get("device_id") or "")
        channel_id = str(params.get("channel_id") or "")
        value = params.get("value")
        conditions = str(params.get("conditions") or "")
        dry_run = bool(params.get("dry_run"))

        context, channel, error = self._resolve(device_id, channel_id)
        if error is not None:
            return WriteFeasibilityResult(ok=False, error=error)

        # Writability is checked before the effect class, because a channel demoted
        # by an admission rule is the *root* cause and naming anything else sends the
        # reader looking in the wrong place.
        if not channel.is_writable:
            return WriteFeasibilityResult(
                ok=False,
                error=self._refusal(
                    device_id,
                    channel_id,
                    "channel_not_writable",
                    f"Channel {channel_id!r} on {device_id!r} is read-only in the admitted "
                    "declaration. Run hw_describe to see why, and check the device declaration.",
                ),
            )

        if channel.effect != expected_effect:
            return WriteFeasibilityResult(
                ok=False,
                error=self._refusal(
                    device_id,
                    channel_id,
                    "effect_class_mismatch",
                    f"Channel {channel_id!r} declares effect {channel.effect!r}; use "
                    f"{_tool_for_effect(channel.effect)} instead of {tool_name}.",
                ),
            )

        if self._requires_describe(device_id):
            return WriteFeasibilityResult(
                ok=False,
                error=self._refusal(
                    device_id,
                    channel_id,
                    "describe_required",
                    f"Call hw_describe(device_id={device_id!r}) before commanding it. Writing to a "
                    "channel whose envelope you have not read risks an out-of-range command; the "
                    "describe result carries the allowed range, rate, and reversibility.",
                ),
            )

        envelope = channel.envelope
        in_envelope = envelope.contains(value)
        wait_s = self._rate_wait_s(device_id, channel, value) if in_envelope else 0.0
        if wait_s > 0.0:
            return WriteFeasibilityResult(
                ok=False,
                error={
                    "ok": False,
                    "device_id": device_id,
                    "channel_id": channel_id,
                    "error": (
                        f"Commanding {device_id}.{channel_id} to {value} now would exceed its "
                        f"declared maximum rate of {envelope.max_rate:g} {channel.unit or 'units'}"
                        f"/s. Wait about {wait_s:.2f}s and issue the same command again, or "
                        "command a smaller step."
                    ),
                    "failure_code": "rate_limited",
                    "retry_after_s": round(wait_s, 3),
                    "side_effect_state": SIDE_EFFECT_NONE,
                },
            )

        safety_ok, safety_reason = self._check_safety(context, channel, value)
        if not safety_ok:
            return WriteFeasibilityResult(
                ok=False,
                error={
                    "ok": False,
                    "device_id": device_id,
                    "channel_id": channel_id,
                    "error": f"Safety policy violation: {safety_reason}",
                    "failure_code": "safety_policy_violation",
                    "side_effect_state": SIDE_EFFECT_NONE,
                },
            )

        if self._registry.is_device_degraded(device_id):
            return WriteFeasibilityResult(
                ok=False,
                error=self._refusal(
                    device_id,
                    channel_id,
                    "device_degraded",
                    "Device is in degraded state (communication or sensor loss). "
                    "Commands blocked until recovery.",
                ),
            )

        unreachable = await self._unreachable(device_id, channel)
        if unreachable is not None:
            return WriteFeasibilityResult(ok=False, error=unreachable)

        interlocks_failed = await self._failed_interlocks(context, channel)

        descriptor = ActionDescriptor.device(
            kind=kind,
            device_id=device_id,
            channel_id=channel_id,
            quantity=channel.quantity,
            value=value,
            unit=channel.unit,
            envelope_band=self._grant_band(envelope, value),
            location=context.location,
            reversible=envelope.reversible,
            metadata={
                "value_in_envelope": in_envelope,
                "interlocks_satisfied": not interlocks_failed,
                "interlocks_failed": list(interlocks_failed),
                "session_id": self._session_identity(),
            },
        )

        return WriteFeasibilityResult(
            ok=True,
            context=context,
            channel=channel,
            descriptor=descriptor,
            in_envelope=in_envelope,
            interlocks_failed=interlocks_failed,
            dry_run=dry_run,
            value=value,
            conditions=conditions,
        )

    async def _execute_single_write(self, feasibility: WriteFeasibilityResult) -> dict[str, Any]:
        """Execute a single-channel write after approval has been granted.

        Takes a passed WriteFeasibilityResult and performs the physical write via
        the transport, records audit, handles trust updates, and produces the
        typed verification verdict.
        """
        context = feasibility.context
        channel = feasibility.channel
        device_id = context.device_id
        channel_id = channel.channel_id
        value = feasibility.value
        conditions = feasibility.conditions

        try:
            async with self._registry.device_io(device_id):
                transport = await self._registry.transport(device_id)
                outcome = await transport.write(channel_id, value)
        except TransportError as exc:
            if self._trust_gate is not None:
                self._trust_gate.record_failure(device_id, channel_id)
            return {
                "ok": False,
                "device_id": device_id,
                "channel_id": channel_id,
                "error": str(exc),
                "failure_code": exc.failure_code,
                "side_effect_state": SIDE_EFFECT_NONE,
            }
        except Exception as exc:  # noqa: BLE001 - see below
            logger.error(
                "Hardware transport raised a non-contract exception on write to %s.%s: %s",
                device_id,
                channel_id,
                exc,
                exc_info=True,
            )
            if self._trust_gate is not None:
                self._trust_gate.record_failure(device_id, channel_id, hard=True)
            return {
                "ok": False,
                "device_id": device_id,
                "channel_id": channel_id,
                "error": (
                    f"The device driver failed unexpectedly ({type(exc).__name__}). "
                    "Whether the command reached the device is unknown; verify the "
                    "channel before attempting anything similar."
                ),
                "failure_code": "driver_contract_violation",
                "side_effect_state": SIDE_EFFECT_UNKNOWN,
                "effect_uncertain": True,
            }

        self._audit.record(
            action="write",
            device=device_id,
            channel=channel_id,
            value=value,
            outcome="ok" if outcome.ok else (outcome.failure_code or "error"),
            identity=self._session_identity(),
        )

        if self._trust_gate is not None and not outcome.ok:
            self._trust_gate.record_failure(device_id, channel_id)

        resolved_outcome = None
        if outcome.ok:
            numeric = as_numeric(value)
            if numeric is not None:
                self._registry.record_command(device_id, channel_id, numeric)
            resolved_outcome = self._learn_from_write(
                device_id, channel, value, conditions, outcome
            )
        else:
            self._write_verifier.drop_pending_outcome(device_id, channel_id)

        result = _write_result(device_id, channel, outcome)
        if outcome.ok:
            expectation = WriteExpectation.from_channel(
                device_id=device_id,
                channel=channel,
                commanded_value=value,
                identity=self._session_identity(),
            )
            if resolved_outcome is not None:
                verdict = WriteVerdict.from_physical_outcome(
                    resolved_outcome,
                    expectation=expectation,
                )
                self._write_verifier.record_verdict(verdict)
            elif (
                outcome.readback is not None
                and expectation.observation_channel_id == channel.channel_id
                and expectation.is_settled()
            ):
                verdict = WriteVerdict.from_reading(expectation, outcome.readback)
                self._write_verifier.record_verdict(verdict)
            else:
                verdict = self._write_verifier.defer_or_refuse(context, expectation)
            result["verification"] = verdict.to_dict()
        return result

    def _learn_from_write(
        self,
        device_id: str,
        channel: Channel,
        value: Any,
        conditions: str,
        outcome: WriteOutcome,
    ) -> Any | None:
        """Register the command for comparison, resolving it now when possible.

        A channel that reads back and has no settling time can be compared immediately.
        One with inertia cannot: a reading taken before the value stabilises measures the
        transition rather than the result, so it waits for a later observation from a read
        or from the sampling loop.
        """
        recorder = self._registry.outcome_recorder
        if recorder is None:
            return None
        recorder.record_command(
            device_id=device_id, channel=channel, value=value, conditions=conditions
        )
        if outcome.readback is not None and channel.envelope.effective_settling_s <= 0:
            resolved = recorder.observe(
                device_id=device_id,
                channel_id=channel.channel_id,
                value=outcome.readback.value,
            )
            if resolved is not None:
                return resolved
        return None

    def _grant_band(self, envelope: Any, value: Any) -> str:
        """Return the band string that defines this command's grant identity.

        By default the channel's declared band, so one consent covers every in-envelope
        value: prompting per microlitre is how a gate gets disabled by the person it
        protects.

        With ``hardware.envelope_grant`` off it degenerates to a per-value identity, so
        each distinct command is decided on its own. The value has to enter the identity
        for that to work -- ``allow_permanent`` is not sufficient, because the
        orchestrator only withholds the profile-wide "always" choice and still offers a
        session scope regardless.
        """
        band = envelope.band_key()
        settings = getattr(self._registry, "settings", None)
        if getattr(settings, "envelope_grant", True):
            return band
        return f"{band}#{value}"

    def _rate_wait_s(self, device_id: str, channel: Channel, value: Any) -> float:
        """Return how long to wait before this command respects ``max_rate``.

        Zero means it may proceed now. ``max_rate`` constrains consecutive commands:
        the delta from the last value that actually reached the device, over the time
        since it did.

        The first command on a channel has no measured interval and is therefore not
        rate checked -- it is still bounded by ``min_value``/``max_value`` and still
        requires consent. Refusing it would make every rate-limited channel unusable
        from a cold start; letting a *later* one through unchecked would defeat the
        limit entirely, which is the case this guards.

        A non-numeric value on a numeric channel is refused later by
        ``Envelope.contains``, so it is simply not rate checked here.
        """
        envelope = channel.envelope
        if envelope.max_rate is None:
            return 0.0
        numeric = as_numeric(value)
        if numeric is None:
            return 0.0
        baseline = self._registry.last_command(device_id, channel.channel_id)
        if baseline is None:
            return 0.0
        previous_value, previous_ts = baseline
        return envelope.rate_wait_s(
            delta=numeric - previous_value,
            elapsed_s=time.monotonic() - previous_ts,
        )

    def _check_safety(
        self,
        context: HardwareContext,
        channel: Channel,
        value: Any,
    ) -> tuple[bool, str]:
        """Delegate to the registry's device-level safety policy check.

        A thin wrapper so the tools layer does not duplicate the constraint
        logic.  Returns ``(allowed, reason)``; ``allowed=True`` when no
        ``SafetyPolicy`` is declared (hc.v0 backward compatibility).
        """
        return self._registry.check_safety_policy(context, channel, value)

    async def _unreachable(self, device_id: str, channel: Channel) -> dict[str, Any] | None:
        """Return a refusal when the device cannot be reached, else None.

        Opening and probing before consent is safe by the transport's own contract:
        ``open()`` establishes a connection and is idempotent, ``probe()`` is declared
        side-effect free. It also sets no new precedent -- ``hw_read`` already opens a
        transport with no approval at all.

        A probe rather than an open alone, because ``transport()`` caches: a connection
        that was live and then died is returned from the cache without ``open()`` ever
        being called again, and that is the *common* failure for a server-backed device
        whose server restarted. Catching only "could not open" would leave exactly the
        case this exists for. The cost is one round trip per command, which a
        human-paced write can afford far more easily than it can afford commanding a
        device nobody can see.

        A dead transport is dropped from the cache so the next attempt reconnects.
        Without that the refusal would outlive the outage.
        """
        channel_id = channel.channel_id
        try:
            transport = await self._registry.transport(device_id)
            status = await transport.probe()
        except TransportError as exc:
            await self._registry.drop_transport(device_id)
            self._report_unreachable(device_id, channel, str(exc))
            return {
                "ok": False,
                "device_id": device_id,
                "channel_id": channel_id,
                "error": (
                    f"{device_id} cannot be reached, so the command was not attempted and "
                    f"nobody was asked to approve it: {exc}. Fix the connection, then "
                    f"call hw_status(device_id={device_id!r}) to confirm it is back before "
                    "commanding it again."
                ),
                "failure_code": exc.failure_code,
                "side_effect_state": SIDE_EFFECT_NONE,
            }
        except Exception as exc:  # noqa: BLE001 - a broken driver is an unusable device
            # probe() is declared side-effect free, so nothing was commanded whatever
            # the driver did here; NONE is the honest verdict for the write.
            logger.error(
                "Hardware transport raised a non-contract exception probing %s: %s",
                device_id,
                exc,
                exc_info=True,
            )
            await self._registry.drop_transport(device_id)
            self._report_unreachable(device_id, channel, f"the driver raised {type(exc).__name__}")
            return {
                "ok": False,
                "device_id": device_id,
                "channel_id": channel_id,
                "error": (
                    f"The driver for {device_id} failed unexpectedly "
                    f"({type(exc).__name__}) while checking that the device was "
                    "reachable, so the command was not attempted. Check the device "
                    "declaration and the driver."
                ),
                "failure_code": "driver_contract_violation",
                "side_effect_state": SIDE_EFFECT_NONE,
            }
        if status.connected:
            return None
        await self._registry.drop_transport(device_id)
        detail = status.detail or "the transport reports it is not connected"
        self._report_unreachable(device_id, channel, detail)
        return {
            "ok": False,
            "device_id": device_id,
            "channel_id": channel_id,
            "error": (
                f"{device_id} is not connected, so the command was not attempted and "
                f"nobody was asked to approve it: {detail}. Restore the connection, then "
                f"call hw_status(device_id={device_id!r}) to confirm it is back."
            ),
            "failure_code": "device_unreachable",
            "side_effect_state": SIDE_EFFECT_NONE,
        }

    def _report_unreachable(self, device_id: str, channel: Channel, detail: str) -> None:
        """Put the refusal on the signal path, not only in the tool result.

        A tool result is read once, by whoever made the call. Without this the one
        condition that most needs to be visible is the one nothing can see: a bench
        refusing every command looks identical to a bench nobody is using, on the
        board and in every watch. Routed through the registry so it lands in the same
        event ring and on the same bus as the sampling loop's observations.
        """
        from leapflow.hardware.stream import EventKind, HardwareEvent

        try:
            self._registry.publish_event(
                HardwareEvent(
                    kind=EventKind.UNREACHABLE,
                    device_id=device_id,
                    channel_id=channel.channel_id,
                    quantity=channel.quantity,
                    detail=f"command refused before approval: {detail}",
                    unit=channel.unit,
                    observed_at=time.time(),
                )
            )
        except Exception as exc:  # noqa: BLE001 - reporting must not fail the refusal
            logger.warning("Could not report %s as unreachable: %s", device_id, exc, exc_info=True)

    async def _consent_for_read(
        self, context: HardwareContext, channel: Channel
    ) -> dict[str, Any] | None:
        """Gate a read that discloses the surroundings; None means it may proceed.

        Only privacy-gated channels reach the orchestrator. Ordinary reads are not
        merely cheap to approve, they are *worth* not approving: a prompt per
        thermometer read is how a gate gets dismissed by the person it protects.

        The descriptor is built exactly as a write's is, so the same classifier,
        policy, grant store and audit log apply -- the risk classifier reads the
        declared privacy tier and answers SAFE for everything else. Two failure modes
        matter and both deny:

        - **No gate installed.** Normal direct CLI and daemon paths bind gates, but an
          incomplete or non-interactive runtime may not; a camera read there is refused
          rather than performed unguarded. Named as a configuration fault because that is
          what it is.
        - **A gate that raises.** Handled inside ``_evaluate``, which is why this goes
          through it rather than calling the gate directly.

        Denial happens *before* the transport is touched, which is the property that
        matters most here: on macOS, opening the device is what raises the system
        permission dialog, so a refused read must never get that far.
        """
        if not channel.is_privacy_gated:
            return None

        descriptor = ActionDescriptor.device(
            kind=ActionKind.DEVICE_READ.value,
            device_id=context.device_id,
            channel_id=channel.channel_id,
            quantity=channel.quantity,
            unit=channel.unit,
            location=context.location,
            # A read changes nothing, so it is reversible in the only sense the word
            # has here -- but the disclosure it makes is not undone by stopping.
            reversible=True,
            metadata={
                "privacy": channel.privacy,
                "representation": channel.representation,
                "session_id": self._session_identity(),
            },
        )
        allowed, denial, _no_gate = await self._evaluate(descriptor)
        if allowed:
            return None
        self._audit.record(
            action="read_denied",
            device=context.device_id,
            channel=channel.channel_id,
            outcome=f"privacy={channel.privacy}",
            identity=self._session_identity(),
        )
        return self._refusal(
            context.device_id,
            channel.channel_id,
            "consent_required",
            denial
            or (
                f"Reading {context.device_id}.{channel.channel_id} observes the physical "
                "surroundings and was not approved."
            ),
        )

    async def _evaluate(self, descriptor: ActionDescriptor) -> tuple[bool, str, bool]:
        """Run the approval gate, failing closed on absence and on exception.

        Delegates to WriteGovernor.evaluate via a thin WriteGovernanceRequest.
        The invariant is absolute: **irreversible channels always require
        full approval**, even at PRODUCTION trust, even with the switch on.

        Returns ``(allowed, denial_message, no_gate)``.
        """
        self._governor.set_trust_skip_enabled(self._trust_skip_enabled())
        meta = getattr(descriptor, "metadata", None) or {}
        commands = meta.get("commands") or ()
        if meta.get("batch") or meta.get("control_session"):
            channel_ids = (
                tuple(str(command.get("channel_id") or "") for command in commands)
                if commands
                else tuple(str(cid) for cid in meta.get("channels", ()))
            )
            request = WriteGovernanceRequest.batch(
                device_id=str(meta.get("device_id") or ""),
                channel_ids=channel_ids,
                values=tuple(command.get("value") for command in commands),
                descriptor=descriptor,
                identity=self._session_identity(),
                reversible=bool(meta.get("reversible", False)),
            )
        else:
            request = WriteGovernanceRequest.single(
                device_id=str(meta.get("device_id") or ""),
                channel_id=str(meta.get("channel_id") or ""),
                value=meta.get("value"),
                descriptor=descriptor,
                identity=self._session_identity(),
                reversible=bool(meta.get("reversible", False)),
            )
        decision = await self._governor.evaluate(request)
        return decision.allowed, decision.denial, decision.no_gate

    async def _failed_interlocks(
        self, context: HardwareContext, channel: Channel
    ) -> tuple[str, ...]:
        """Delegate to the shared interlock evaluation in ``hardware.interlocks``."""
        from leapflow.hardware.interlocks import failed_interlocks

        return await failed_interlocks(self._registry, context, channel)

    def _not_ready(
        self,
        context: HardwareContext,
        channel: Channel,
        failed: tuple[str, ...],
    ) -> dict[str, Any]:
        """Return a deterministic, actionable refusal for an unmet readiness state.

        Built through ``security.permission_failures`` -- the single authority the
        engine and TUI both consult -- so a device that has not reached its
        declared ready state is reported the same way everywhere: the precondition
        to satisfy is named, the repair instruction is executable, and the turn
        hard-stops instead of asking whether to proceed. Nothing reached the
        device, so the side-effect verdict is NONE.
        """
        payload = build_readiness_failure(
            device_id=context.device_id,
            channel_id=channel.channel_id,
            unmet=self._readiness_requirements(context, failed),
        )
        payload["side_effect_state"] = SIDE_EFFECT_NONE
        return payload

    def _readiness_requirements(
        self, context: HardwareContext, failed: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        """Describe each unmet readiness precondition for an actionable refusal.

        A declared interlock contributes its source channel and the comparison it
        must satisfy; one named on the channel but absent from the device
        declaration is reported as undeclared, because "cannot be checked" and
        "not satisfied" carry the same weight in the write path.
        """
        requirements: list[dict[str, Any]] = []
        for name in failed:
            lock = context.interlock(name)
            if lock is None:
                requirements.append(
                    {
                        "interlock_id": name,
                        "channel_id": "",
                        "operator": "",
                        "value": None,
                        "description": "",
                        "declared": False,
                    }
                )
                continue
            requirements.append(
                {
                    "interlock_id": lock.interlock_id,
                    "channel_id": lock.channel_id,
                    "operator": lock.operator,
                    "value": lock.value,
                    "description": lock.description,
                    "declared": True,
                }
            )
        return requirements

    # ── Helpers ──

    def _resolve(
        self, device_id: str, channel_id: str
    ) -> tuple[HardwareContext, Channel, dict[str, Any] | None]:
        context = self._registry.context(device_id)
        if context is None:
            return None, None, self._unknown_device(device_id)  # type: ignore[return-value]
        channel = context.channel(channel_id)
        if channel is None:
            known = ", ".join(c.channel_id for c in context.channels) or "(none)"
            return (
                None,  # type: ignore[return-value]
                None,  # type: ignore[return-value]
                self._refusal(
                    device_id,
                    channel_id,
                    "unknown_channel",
                    f"Device {device_id!r} has no channel {channel_id!r}. Declared: {known}.",
                ),
            )
        return context, channel, None

    def _trust_skip_enabled(self) -> bool:
        """Whether the operator has opted in to trust-based approval skip.

        Reads ``trust_skip_enabled`` from the hardware settings, defaulting to
        False so the feature is off until deliberately activated.
        """
        settings = getattr(self._registry, "settings", None)
        return bool(getattr(settings, "trust_skip_enabled", False))

    def _requires_describe(self, device_id: str) -> bool:
        settings = getattr(self._registry, "settings", None)
        if not getattr(settings, "require_describe_before_write", False):
            return False
        return not self._registry.was_described(self._session_identity(), device_id)

    def _unknown_device(self, device_id: str) -> dict[str, Any]:
        known = ", ".join(c.device_id for c in self._registry.contexts()) or "(none admitted)"
        return {
            "ok": False,
            "device_id": device_id,
            "error": f"Unknown device {device_id!r}. Admitted devices: {known}.",
            "failure_code": "unknown_device",
            "side_effect_state": SIDE_EFFECT_NONE,
        }

    @staticmethod
    def _refusal(
        device_id: str,
        channel_id: str,
        code: str,
        message: str,
        *,
        side_effect_state: str = SIDE_EFFECT_NONE,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "device_id": device_id,
            "channel_id": channel_id,
            "error": message,
            "failure_code": code,
            "side_effect_state": side_effect_state,
        }


def _batch_grant_identity(channel_summaries: tuple[dict[str, Any], ...]) -> str:
    """Compute a deterministic per-channel band identity for batch writes.

    Sorts channel bands alphabetically so two batch writes targeting the same
    channels with the same envelope bands produce the same grant identity,
    allowing one approval to cover repeated in-envelope batches.
    """
    parts = sorted(
        f"{cs.get('channel_id', '')}:{cs.get('envelope_band', 'unknown')}"
        for cs in channel_summaries
    )
    return "batch:" + "|".join(parts) if parts else "batch:empty"


def no_gate_interaction(device_id: str) -> dict[str, Any]:
    """Build a structured InteractionRequest payload for Mode 3 (no TUI).

    When no approval gate is installed the write is refused, and this request
    gives the user two actionable choices instead of a raw denial.
    """
    return {
        "interaction_type": "approval_required",
        "severity": "warning",
        "title": "Hardware write requires an approval surface",
        "description": (
            f"The command targeting {device_id!r} was refused because no approval "
            "gate is available. Hardware writes are fail-closed: without a surface "
            "to present the approval prompt, no physical effect can occur."
        ),
        "suggested_actions": [
            {
                "label": "Start a TUI session",
                "command": "leap chat",
                "description": (
                    "Launch an interactive session where the approval prompt "
                    "can be presented and answered."
                ),
                "is_default": True,
            },
            {
                "label": "Configure edge approval",
                "command": "leap config set hardware.approval_channel edge",
                "description": (
                    "Set up an edge approval channel so headless/daemon-mode "
                    "writes can be reviewed without a TUI."
                ),
            },
        ],
    }


def _write_result(device_id: str, channel: Channel, outcome: WriteOutcome) -> dict[str, Any]:
    """Shape a write outcome into a tool result.

    A failure whose effect may already have landed says so explicitly, so the next
    turn verifies before repeating it. An error is not proof that nothing happened,
    and this field is the only place that distinction survives into the transcript.
    """
    payload: dict[str, Any] = {
        "ok": outcome.ok,
        "device_id": device_id,
        "channel_id": channel.channel_id,
        **outcome.to_dict(),
    }
    if not outcome.ok and outcome.effect_may_have_landed:
        payload["effect_uncertain"] = True
        payload["next_step"] = (
            "The command failed but its physical effect may already have occurred. Read the "
            "channel back or inspect the device before attempting anything similar; do not "
            "repeat the command on the assumption that nothing happened."
        )
        if not channel.envelope.reversible:
            payload["next_step"] += (
                " This channel is declared irreversible, so a repeat would apply the effect twice."
            )
    if outcome.ok and channel.envelope.settling_time_s > 0 and not outcome.settled:
        payload["settling_time_s"] = channel.envelope.settling_time_s
        payload["next_step"] = (
            f"The value was accepted but needs {channel.envelope.settling_time_s:g}s to stabilise; "
            "read it back before drawing conclusions."
        )
    return payload


def _preview_result(
    device_id: str,
    channel: Channel,
    descriptor: ActionDescriptor,
    *,
    value: Any,
    in_envelope: bool,
    interlocks_failed: tuple[str, ...],
) -> dict[str, Any]:
    """Shape a dry run into a tool result without touching the device.

    Reports the command that *would* be issued together with the outcome of every
    check that precedes consent, so a caller can confirm intent before committing
    an irreversible physical effect. ``ok`` reflects whether the command would pass
    validation: it is true only when the value is inside the envelope and every
    interlock holds -- the same two conditions that would otherwise let it reach
    approval. Nothing was written, so the side-effect verdict is NONE and the
    outcome is marked ``preview``.
    """
    ok = in_envelope and not interlocks_failed
    outcome = WriteOutcome(ok=ok, side_effect_state=SIDE_EFFECT_NONE, preview=True)
    payload: dict[str, Any] = {
        "device_id": device_id,
        "channel_id": channel.channel_id,
        **outcome.to_dict(),
        "plan": {
            "summary": descriptor.summary,
            "detail": descriptor.detail,
            "resource": descriptor.resource,
            "quantity": channel.quantity,
            "value": value,
            "unit": channel.unit,
            "effect": channel.effect,
            "reversible": channel.envelope.reversible,
            "requires_approval": True,
            "value_in_envelope": in_envelope,
            "interlocks_satisfied": not interlocks_failed,
            "interlocks_failed": list(interlocks_failed),
        },
    }
    if ok:
        payload["next_step"] = (
            "Dry run only: nothing was commanded. Re-issue the same call without "
            "dry_run to execute it, which will require approval."
        )
    elif not in_envelope:
        payload["failure_code"] = "value_out_of_envelope"
        payload["error"] = (
            f"{value!r} lies outside the declared envelope for "
            f"{device_id}.{channel.channel_id}, so the real command would be refused "
            "before execution. Call hw_describe to read the allowed range."
        )
    else:
        payload["failure_code"] = "interlocks_unsatisfied"
        payload["error"] = (
            f"Interlocks {list(interlocks_failed)} are not satisfied for "
            f"{device_id}.{channel.channel_id}, so the real command would be refused. "
            "Restore the interlock conditions before commanding it."
        )
    return payload


def _batch_preview_result(feasibility: BatchFeasibilityResult) -> dict[str, Any]:
    """Shape a batch dry-run into a tool result without touching the device."""
    device_id = feasibility.context.device_id
    descriptor = feasibility.descriptor
    interlocks_failed = feasibility.interlocks_failed
    ok = not interlocks_failed
    payload: dict[str, Any] = {
        "ok": ok,
        "device_id": device_id,
        "preview": True,
        "side_effect_state": SIDE_EFFECT_NONE,
        "plan": {
            "summary": descriptor.summary,
            "detail": descriptor.detail,
            "resource": descriptor.resource,
            "batch_size": len(feasibility.resolved),
            "commands": list(feasibility.channel_summaries),
            "any_irreversible": feasibility.any_irreversible,
            "requires_approval": True,
            "interlocks_satisfied": not interlocks_failed,
            "interlocks_failed": list(interlocks_failed),
        },
    }
    if interlocks_failed:
        payload["failure_code"] = "interlocks_unsatisfied"
        payload["error"] = (
            f"Interlocks {list(interlocks_failed)} are not satisfied; "
            "the real command would be refused."
        )
    else:
        payload["next_step"] = (
            "Dry run only: nothing was commanded. Re-issue the same call "
            "without dry_run to execute it, which will require approval."
        )
    return payload


def _control_preview_result(feasibility: ControlFeasibilityResult) -> dict[str, Any]:
    """Shape a control-session dry-run into a tool result without starting the loop."""
    device_id = feasibility.context.device_id
    descriptor = feasibility.descriptor
    interlocks_failed = feasibility.interlocks_failed
    ok = not interlocks_failed
    payload: dict[str, Any] = {
        "ok": ok,
        "device_id": device_id,
        "preview": True,
        "side_effect_state": SIDE_EFFECT_NONE,
        "plan": {
            "summary": descriptor.summary,
            "mode": feasibility.mode,
            "frequency_hz": feasibility.frequency_hz,
            "channels": [channel.channel_id for channel, _ in feasibility.resolved],
            "interlocks_satisfied": not interlocks_failed,
            "interlocks_failed": list(interlocks_failed),
            "requires_approval": True,
        },
    }
    if interlocks_failed:
        payload["failure_code"] = "interlocks_unsatisfied"
        payload["error"] = (
            f"Interlocks {list(interlocks_failed)} are not satisfied; "
            "control startup would be refused."
        )
    return payload


def _tool_for_effect(effect: str) -> str:
    for name, (_, declared) in _WRITE_TOOLS.items():
        if declared == effect:
            return name
    return "hw_read"


def build_hardware_tools(tools: HardwareTools) -> list[ToolMetadata]:
    """Return the twelve tool definitions bound to *tools*."""
    return [
        ToolMetadata(
            name="hw_list",
            description=(
                "List connected hardware devices with their channel counts and measured "
                "quantities. Start here; it does not include operating limits."
            ),
            parameters_schema={"type": "object", "properties": {}},
            handler=tools.hw_list,
            x_leapflow=dict(_READ_METADATA),
            provides_capabilities=("hw.list",),
        ),
        ToolMetadata(
            name="hw_describe",
            description=(
                "Return the full reference for one device: every channel, its unit, its "
                "operating envelope, rate limit, reversibility, and required interlocks, "
                "plus any prior outcomes recorded for its writable channels. Required "
                "before commanding a device."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "description": "Device id from hw_list"}
                },
                "required": ["device_id"],
            },
            handler=tools.hw_describe,
            x_leapflow=dict(_READ_METADATA),
            provides_capabilities=("hw.describe",),
        ),
        ToolMetadata(
            name="hw_read",
            description="Read the current value of one device channel. Has no physical effect.",
            parameters_schema=dict(_TARGET_SCHEMA),
            handler=tools.hw_read,
            x_leapflow=dict(_READ_METADATA),
            provides_capabilities=("hw.read",),
        ),
        ToolMetadata(
            name="hw_status",
            description=(
                "Report connection health, halt capability, and recent notable events "
                "(threshold excursions, lost samples, stalled channels) for one device."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "description": "Device id from hw_list"}
                },
                "required": ["device_id"],
            },
            handler=tools.hw_status,
            x_leapflow=dict(_READ_METADATA),
            provides_capabilities=("hw.status",),
        ),
        ToolMetadata(
            name="hw_configure",
            description=(
                "Set a configuration value or setpoint on a channel declaring effect=configure. "
                "Setpoints often have inertia: the value may need time to stabilise."
            ),
            parameters_schema=_write_schema(HardwareEffect.CONFIGURE.value),
            handler=tools.hw_configure,
            x_leapflow=dict(_WRITE_METADATA),
            mutates_state=True,
            provides_capabilities=("hw.configure",),
        ),
        ToolMetadata(
            name="hw_actuate",
            description=(
                "Command motion or output on a channel declaring effect=actuate. This moves "
                "physical hardware; a repeat from an unknown state is not a safe retry."
            ),
            parameters_schema=_write_schema(HardwareEffect.ACTUATE.value),
            handler=tools.hw_actuate,
            x_leapflow=dict(_WRITE_METADATA),
            mutates_state=True,
            provides_capabilities=("hw.actuate",),
        ),
        ToolMetadata(
            name="hw_dispense",
            description=(
                "Consume an irreversible resource on a channel declaring effect=dispense. "
                "Running this twice dispenses twice; never repeat it after a failure without "
                "first verifying what already happened."
            ),
            parameters_schema=_write_schema(HardwareEffect.DISPENSE.value),
            handler=tools.hw_dispense,
            x_leapflow=dict(_WRITE_METADATA),
            mutates_state=True,
            provides_capabilities=("hw.dispense",),
        ),
        ToolMetadata(
            name="hw_estop",
            description=(
                "Stop all motion and output on a device immediately. Never requires approval; "
                "use it whenever device behaviour is unexpected."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "description": "Device id from hw_list"}
                },
                "required": ["device_id"],
            },
            handler=tools.hw_estop,
            x_leapflow={
                "category": "hardware",
                "risk_level": "external",
                "requires_approval": False,
                "effect_scope": "external",
                "idempotency_scope": "turn",
                "schema_cost": "low",
            },
            provides_capabilities=("hw.estop",),
        ),
        ToolMetadata(
            name="hw_batch_actuate",
            description=(
                "Write multiple actuator channels in one atomic transaction. "
                "Used for coordinated multi-joint control where all positions "
                "must be commanded simultaneously (e.g. robot arm joint-space control)."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "device_id": {
                        "type": "string",
                        "description": "Target device identifier",
                    },
                    "commands": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "channel_id": {"type": "string"},
                                "value": {"type": "number"},
                            },
                            "required": ["channel_id", "value"],
                        },
                        "description": "List of channel commands to write atomically",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Preview mode: validate without executing",
                    },
                    "verify": {
                        "type": "boolean",
                        "description": "Run post-write verification",
                    },
                },
                "required": ["device_id", "commands"],
            },
            handler=tools.batch_actuate,
            x_leapflow={
                **_WRITE_METADATA,
                "mutates_state": True,
                "execution_policy": "external_side_effect",
                "provides_capabilities": ["hardware_batch_control"],
            },
            mutates_state=True,
            execution_policy="external_side_effect",
        ),
        # ── Real-time control tools ──
        ToolMetadata(
            name="hw_control_start",
            description=(
                "Start a real-time control loop on a robot device. "
                "Supports PID position tracking, impedance control, and "
                "trajectory following. Learned policies run as the node skill "
                "policy.run (hw_skill_run), never through this loop."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "description": "Target device"},
                    "mode": {
                        "type": "string",
                        "enum": ["pid", "impedance", "trajectory"],
                        "description": "Control mode",
                    },
                    "targets": {
                        "type": "object",
                        "description": "Joint target positions (for pid/impedance)",
                    },
                    "frequency_hz": {
                        "type": "number",
                        "description": "Control frequency (default 100)",
                    },
                    "trajectory": {
                        "type": "array",
                        "description": "Time-indexed joint trajectory (for trajectory mode)",
                    },
                    "gains": {
                        "type": "object",
                        "description": "PID gains {kp, ki, kd} or impedance {stiffness, damping}",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Validate and preview the control session without starting it",
                    },
                },
                "required": ["device_id", "mode"],
            },
            handler=tools.control_start,
            x_leapflow={
                **_WRITE_METADATA,
                "mutates_state": True,
                "execution_policy": "external_side_effect",
            },
            mutates_state=True,
            execution_policy="external_side_effect",
            provides_capabilities=("hw.control_start",),
        ),
        ToolMetadata(
            name="hw_control_stop",
            description=(
                "Stop the real-time control loop on a device. "
                "Gracefully halts the device and returns session statistics."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "description": "Target device"},
                },
                "required": ["device_id"],
            },
            handler=tools.control_stop,
            x_leapflow={
                "category": "hardware",
                "risk_level": "medium",
                "mutates_state": True,
            },
            mutates_state=True,
            provides_capabilities=("hw.control_stop",),
        ),
        ToolMetadata(
            name="hw_control_status",
            description=(
                "Get the status and statistics of a running control loop. "
                "Returns frequency, jitter, overruns, and policy state."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "description": "Target device"},
                },
                "required": ["device_id"],
            },
            handler=tools.control_status,
            x_leapflow={
                "category": "hardware",
                "risk_level": "low",
                "mutates_state": False,
            },
            provides_capabilities=("hw.control_status",),
        ),
    ]


__all__ = ["HardwareTools", "build_hardware_tools", "no_gate_interaction"]
