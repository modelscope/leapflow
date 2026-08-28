"""The eight hardware tools, derived from admitted contexts.

The count is fixed regardless of how many devices exist. A rig of seven programs
with six channels each would be 40+ schemas if tools were generated per channel,
which would swamp the tool index; instead the index stays constant and the
per-channel limits arrive through ``hw_describe`` when a model actually intends to
act. That is progressive disclosure applied to hardware, and it is also the reason
an upstream standard's "reference file" is a rendered view here rather than a
separate document to keep in sync.

Writes are split into three tools by effect class rather than collapsed into one
``hw_write``. Their risk profiles differ, so each maps to its own ``ActionKind``
and reaches a decision instead of a fallback; the tool name is itself a safety
signal the model cannot overlook; and if channel lookup ever fails, the name still
says what class of thing was attempted.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from leapflow.hardware.context import (
    Channel,
    HardwareContext,
    HardwareEffect,
    as_numeric,
)
from leapflow.hardware.reference import describe, summarize
from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_UNKNOWN,
    TransportError,
    WriteOutcome,
)
from leapflow.plugins.protocol import ToolMetadata
from leapflow.security.actions import ActionDescriptor, ActionKind

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
        },
        "required": ["device_id", "channel_id", "value"],
    }


class HardwareTools:
    """Handlers for the hardware tool surface.

    Holds the registry and the approval gate rather than reaching for globals, so a
    test drives exactly the same code path as production with its own instances.
    """

    def __init__(self, registry: Any, *, gate: Any = None, session_id: str = "") -> None:
        self._registry = registry
        self._gate = gate
        self._session_id = session_id

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
        self._registry.mark_described(self._session_id, device_id)
        payload = describe(context)
        payload["ok"] = True
        # Prior outcomes for each writable channel, best-tracking first. This is what turns
        # a reference document into accumulated experience: an optimisation performed once
        # becomes a starting point rather than an experiment to repeat.
        experience = self._prior_experience(context)
        if experience:
            payload["prior_experience"] = experience
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

    async def hw_status(self, device_id: str = "", **_: Any) -> dict[str, Any]:
        """Return live transport health and recent observations for one device."""
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
        try:
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
        # A read is a trustworthy observation, so it is also the moment a pending command
        # on this channel can finally be scored -- which is how a channel with settling
        # time gets learned from at all.
        recorder = self._registry.outcome_recorder
        if recorder is not None:
            resolved = recorder.observe(
                device_id=device_id, channel_id=channel_id, value=reading.value
            )
            if resolved is not None:
                payload["command_outcome"] = resolved.to_dict()
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
        """
        context = self._registry.context(device_id)
        if context is None:
            return self._unknown_device(device_id)
        try:
            transport = await self._registry.transport(device_id)
            status = await transport.halt()
        except TransportError as exc:
            logger.error(
                "Emergency stop for %r failed: %s", device_id, exc, exc_info=True
            )
            return {
                "ok": False,
                "device_id": device_id,
                "error": str(exc),
                "failure_code": exc.failure_code,
            }
        logger.warning(
            "Emergency stop issued device=%s supported=%s detail=%s",
            device_id,
            status.halt_supported,
            status.detail,
        )
        return {
            "ok": status.halt_supported,
            "device_id": device_id,
            "halted": status.halt_supported,
            "status": status.to_dict(),
        }

    # ── Write path ──

    async def _write(self, tool_name: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """Validate, gate, and execute one physical write.

        Order is fixed and matches the platform rule that feasibility precedes
        consent: resolve the target, check that the command *could* succeed, and
        only then ask a human. Prompting for a command that will be refused anyway
        teaches people to click through prompts.
        """
        kind, expected_effect = _WRITE_TOOLS[tool_name]
        device_id = str(params.get("device_id") or "")
        channel_id = str(params.get("channel_id") or "")
        value = params.get("value")
        conditions = str(params.get("conditions") or "")

        context, channel, error = self._resolve(device_id, channel_id)
        if error is not None:
            return error

        # Writability is checked before the effect class, because a channel demoted
        # by an admission rule is the *root* cause and naming anything else sends the
        # reader looking in the wrong place.
        if not channel.is_writable:
            return self._refusal(
                device_id,
                channel_id,
                "channel_not_writable",
                f"Channel {channel_id!r} on {device_id!r} is read-only in the admitted "
                "declaration. Run hw_describe to see why, and check the device declaration.",
            )

        if channel.effect != expected_effect:
            return self._refusal(
                device_id,
                channel_id,
                "effect_class_mismatch",
                f"Channel {channel_id!r} declares effect {channel.effect!r}; use "
                f"{_tool_for_effect(channel.effect)} instead of {tool_name}.",
            )

        if self._requires_describe(device_id):
            return self._refusal(
                device_id,
                channel_id,
                "describe_required",
                f"Call hw_describe(device_id={device_id!r}) before commanding it. Writing to a "
                "channel whose envelope you have not read risks an out-of-range command; the "
                "describe result carries the allowed range, rate, and reversibility.",
            )

        envelope = channel.envelope
        in_envelope = envelope.contains(value)
        # Pacing applies only to a command that would otherwise be permitted. A value
        # outside the envelope is never valid, so telling the caller to wait and retry
        # would send it to sleep on advice that cannot work; that case must fall
        # through to the hardline denial below, which is terminal for a reason.
        wait_s = self._rate_wait_s(device_id, channel, value) if in_envelope else 0.0
        if wait_s > 0.0:
            # Refused before consent is sought, and refused *retryably*: unlike an
            # out-of-envelope value, this exact command becomes safe once enough time
            # has passed, so the caller is told how long rather than being handed a
            # terminal denial it cannot act on.
            return {
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
            }

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
                "session_id": self._session_id,
            },
        )

        allowed, denial = await self._evaluate(descriptor)
        if not allowed:
            return self._refusal(device_id, channel_id, "approval_denied", denial)

        try:
            transport = await self._registry.transport(device_id)
            outcome = await transport.write(channel_id, value)
        except TransportError as exc:
            # A transport raises this only for "could not attempt", so no effect landed.
            return {
                "ok": False,
                "device_id": device_id,
                "channel_id": channel_id,
                "error": str(exc),
                "failure_code": exc.failure_code,
                "side_effect_state": SIDE_EFFECT_NONE,
            }
        except Exception as exc:  # noqa: BLE001 - see below
            # A driver that raises something other than TransportError has broken
            # the contract, and at that point nothing can be concluded about what
            # reached the device. Reporting it as UNKNOWN rather than letting it
            # propagate is the safe reading: UNKNOWN blocks replay just as
            # COMMITTED does, whereas an escaping exception carries no effect
            # verdict at all and invites the caller to simply try again.
            logger.error(
                "Hardware transport raised a non-contract exception on write to %s.%s: %s",
                device_id,
                channel_id,
                exc,
                exc_info=True,
            )
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

        if outcome.ok:
            numeric = as_numeric(value)
            if numeric is not None:
                self._registry.record_command(device_id, channel_id, numeric)
            self._learn_from_write(device_id, channel, value, conditions, outcome)
        else:
            # A failed command must not later be scored as an outcome: whatever the device
            # settles at is not a measurement of what was asked, and recording it would
            # teach the store something false.
            recorder = self._registry.outcome_recorder
            if recorder is not None:
                recorder.drop_pending(device_id, channel_id)
        return _write_result(device_id, channel, outcome)

    def _learn_from_write(
        self,
        device_id: str,
        channel: Channel,
        value: Any,
        conditions: str,
        outcome: WriteOutcome,
    ) -> None:
        """Register the command for comparison, resolving it now when possible.

        A channel that reads back and has no settling time can be compared immediately.
        One with inertia cannot: a reading taken before the value stabilises measures the
        transition rather than the result, so it waits for a later observation from a read
        or from the sampling loop.
        """
        recorder = self._registry.outcome_recorder
        if recorder is None:
            return
        recorder.record_command(
            device_id=device_id, channel=channel, value=value, conditions=conditions
        )
        if outcome.readback is not None and channel.envelope.settling_time_s <= 0:
            recorder.observe(
                device_id=device_id,
                channel_id=channel.channel_id,
                value=outcome.readback.value,
            )

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

    async def _evaluate(self, descriptor: ActionDescriptor) -> tuple[bool, str]:
        """Run the approval gate, failing closed on absence and on exception.

        No gate installed, or a gate that raises, both mean deny. A broken gate must
        never become an open door, and for a physical device the cost of getting
        that wrong is not measured in data.
        """
        if self._gate is None:
            return False, (
                "No approval gate is installed for hardware commands, so the command was "
                "refused. This is a configuration fault, not a user decision."
            )
        try:
            result = await self._gate.evaluate(descriptor)
        except Exception as exc:  # noqa: BLE001 - a failing gate must deny, not propagate
            logger.error(
                "Hardware approval gate raised for %s: %s", descriptor.kind, exc, exc_info=True
            )
            return False, (
                "The approval gate failed while assessing this command, so it was refused."
            )
        # ``approved`` is the orchestrator's own field name. Reading a plausible
        # synonym here would make every command look denied while the gate reported
        # success, so the attribute is asserted against the real ApprovalResult in
        # tests/test_hardware_governance.py rather than trusted.
        if getattr(result, "approved", False):
            return True, ""
        # The gate's own message states that a human withheld consent and that the
        # outcome must not be pursued another way; substituting a generic tool error
        # here would let the agent reroute around a refusal.
        message = getattr(result, "denial_message", "") or getattr(result, "reason", "")
        return False, str(message or "The command was not approved.")

    async def _failed_interlocks(
        self, context: HardwareContext, channel: Channel
    ) -> tuple[str, ...]:
        """Return the interlocks that do not currently hold.

        Fails closed in every uncertain case: a missing interlock, an unreadable
        source channel, or a read that raises all count as unsatisfied. "Cannot
        check" and "not satisfied" must have the same consequence.
        """
        required = channel.envelope.requires_interlocks
        if not required:
            return ()
        failed: list[str] = []
        for name in required:
            lock = context.interlock(name)
            if lock is None:
                failed.append(name)
                continue
            try:
                transport = await self._registry.transport(context.device_id)
                reading = await transport.read(lock.channel_id)
            except Exception as exc:  # noqa: BLE001 - see below
                # Deliberately wider than TransportError. A driver is *supposed* to
                # raise only that, but this is a safety precondition: if a
                # third-party driver raises anything else, the correct reading is
                # "the interlock could not be checked", not "the turn crashes".
                # Letting it propagate would also lose the distinction between a
                # safety mechanism engaging and the system falling over.
                logger.warning(
                    "Interlock %r could not be evaluated on %s.%s (%s); treating as unsatisfied",
                    name,
                    context.device_id,
                    lock.channel_id,
                    exc,
                    exc_info=True,
                )
                failed.append(name)
                continue
            if not lock.evaluate(reading.value):
                failed.append(name)
        return tuple(failed)

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

    def _requires_describe(self, device_id: str) -> bool:
        settings = getattr(self._registry, "settings", None)
        if not getattr(settings, "require_describe_before_write", False):
            return False
        return not self._registry.was_described(self._session_id, device_id)

    def _unknown_device(self, device_id: str) -> dict[str, Any]:
        known = ", ".join(c.device_id for c in self._registry.contexts()) or "(none admitted)"
        return {
            "ok": False,
            "device_id": device_id,
            "error": f"Unknown device {device_id!r}. Admitted devices: {known}.",
            "failure_code": "unknown_device",
        }

    @staticmethod
    def _refusal(device_id: str, channel_id: str, code: str, message: str) -> dict[str, Any]:
        return {
            "ok": False,
            "device_id": device_id,
            "channel_id": channel_id,
            "error": message,
            "failure_code": code,
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


def _tool_for_effect(effect: str) -> str:
    for name, (_, declared) in _WRITE_TOOLS.items():
        if declared == effect:
            return name
    return "hw_read"


def build_hardware_tools(tools: HardwareTools) -> list[ToolMetadata]:
    """Return the eight tool definitions bound to *tools*."""
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
    ]


__all__ = ["HardwareTools", "build_hardware_tools"]
