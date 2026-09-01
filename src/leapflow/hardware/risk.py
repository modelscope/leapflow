"""Risk assessment for physical device actions.

Every tier below is derived from declared data -- the channel's effect class and
its ``Envelope`` -- never from matching text against a device name or a command
string. A safety limit that depends on interpretation is not a limit.

The classifier is registered for the ``device.`` kind prefix through the neutral
``CompositeRiskClassifier``, because ``ApprovalOrchestrator`` holds a single
classifier slot. Composing keeps ``DefaultRiskClassifier`` the authority for every
kind it already owns: adding hardware must not change how a shell command is
assessed.
"""

from __future__ import annotations

import logging
from typing import Any

from leapflow.hardware.context import Envelope, HardwareEffect
from leapflow.security.actions import ActionDescriptor, ActionKind
from leapflow.security.risk import (
    CompositeRiskClassifier,
    DefaultRiskClassifier,
    RiskAssessment,
    RiskClassifier,
    RiskLevel,
)

logger = logging.getLogger(__name__)

DEVICE_KIND_PREFIX = "device."

_EFFECT_FOR_KIND: dict[str, str] = {
    ActionKind.DEVICE_READ.value: HardwareEffect.READ.value,
    ActionKind.DEVICE_CONFIGURE.value: HardwareEffect.CONFIGURE.value,
    ActionKind.DEVICE_ACTUATE.value: HardwareEffect.ACTUATE.value,
    ActionKind.DEVICE_DISPENSE.value: HardwareEffect.DISPENSE.value,
}

# Effect classes that command a physical change and therefore carry the higher
# tier. EMIT rides with ACTUATE: radiating output and moving mass differ in
# mechanism, not in the fact that a person nearby can be harmed.
_HIGH_TIER_EFFECTS = frozenset(
    {HardwareEffect.ACTUATE.value, HardwareEffect.DISPENSE.value, HardwareEffect.EMIT.value}
)


class HardwareRiskClassifier:
    """Assesses ``device.*`` actions from the declared context of the channel.

    Needs the registry because risk lives in the declaration, not in the request:
    the same numeric value is routine on one channel and out of envelope on
    another. Without a resolvable channel the verdict is a hardline deny -- a
    command we cannot describe is one we must not let a human wave through.
    """

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def assess(self, action: ActionDescriptor) -> RiskAssessment:
        kind = str(action.kind or "")

        # Emergency stop never reaches approval: a tool that waits for consent to
        # halt a moving machine is worse than no tool. Kept here as a hard floor in
        # case a caller builds the descriptor anyway.
        if kind == ActionKind.DEVICE_ESTOP.value:
            return RiskAssessment(
                level=RiskLevel.SAFE,
                score=0.0,
                reasons=("emergency_stop",),
                explanation="Emergency stop is never gated.",
            )

        if kind == ActionKind.DEVICE_READ.value:
            return RiskAssessment(
                level=RiskLevel.SAFE,
                score=0.0,
                reasons=("device_read",),
                explanation="Reading a channel has no physical effect.",
            )

        metadata = action.metadata or {}
        device_id = str(metadata.get("device_id") or "")
        channel_id = str(metadata.get("channel_id") or "")
        context = self._registry.context(device_id) if self._registry is not None else None
        channel = context.channel(channel_id) if context is not None else None

        if context is None or channel is None:
            return self._hardline(
                "unresolvable_target",
                f"No declared channel {device_id}.{channel_id}. A device command that cannot be "
                "described against a declaration cannot be assessed, so it is refused.",
            )

        # A declaration that no human has confirmed cannot authorize a physical
        # change. The registry normally demotes such channels to read-only, so
        # reaching here means policy was relaxed after admission.
        if not channel.is_writable:
            return self._hardline(
                "channel_not_writable",
                f"Channel {device_id}.{channel_id} is not writable in the admitted declaration. "
                "Check the load report for the admission rule that demoted it.",
            )

        # The tool used must match the channel's declared effect class. Defence in
        # depth: the tool name already told the model what class of thing it was
        # doing, and disagreement means one of the two is wrong.
        expected = _EFFECT_FOR_KIND.get(kind, "")
        if expected and channel.effect != expected:
            return self._hardline(
                "effect_class_mismatch",
                f"Channel {device_id}.{channel_id} declares effect {channel.effect!r}, but this "
                f"tool performs {expected!r}. Use the tool matching the declared effect class.",
            )

        envelope = channel.envelope
        if not envelope.declared:
            return self._hardline(
                "envelope_undeclared",
                f"Channel {device_id}.{channel_id} has no declared operating envelope. Physical "
                "limits must be declared before the channel can be commanded.",
            )

        unevaluable = self._unevaluable_interlocks(context, envelope)
        if unevaluable:
            return self._hardline(
                "interlock_unevaluable",
                f"Interlocks {', '.join(unevaluable)} cannot be evaluated for "
                f"{device_id}.{channel_id}. An interlock that cannot be checked is treated as "
                "unsatisfied.",
            )

        if metadata.get("interlocks_satisfied") is False:
            failed = metadata.get("interlocks_failed") or ()
            names = ", ".join(str(item) for item in failed) or "one or more"
            return self._hardline(
                "interlock_unsatisfied",
                f"Interlock {names} is not satisfied for {device_id}.{channel_id}.",
            )

        if metadata.get("value_in_envelope") is False:
            return self._hardline(
                "value_out_of_envelope",
                f"Requested value is outside the declared envelope for {device_id}.{channel_id} "
                f"({_describe_bounds(envelope)}).",
            )

        # Rate limiting is deliberately *not* assessed here. A hardline means "this
        # must never happen"; commanding a channel too quickly means "not yet", and
        # the identical command becomes safe after waiting. Treating pacing as a
        # hardline would spend an unbypassable, terminal refusal on a timing issue
        # and leave the model no actionable next step. It is enforced before consent
        # is sought, in HardwareTools, which can say how long to wait.

        return self._tier_for(channel.effect, envelope, device_id, channel_id)

    def _tier_for(
        self, effect: str, envelope: Envelope, device_id: str, channel_id: str
    ) -> RiskAssessment:
        """Return the in-envelope tier for a permitted command.

        ``allow_permanent`` stays True at HIGH for *reversible* commands, which is a
        deliberate departure from the software default. Refusing reusable consent
        would mean prompting for every single motion, and a person asked to confirm
        hundreds of routine operations stops reading the prompts and disables the
        gate -- which is strictly worse than a scoped grant. Safety comes from the
        scope instead: the grant identity is the channel *and its declared band*,
        and anything outside that band is hardline-denied above, where no grant can
        reach.

        An *irreversible* write, and any DISPENSE (which outputs material into the
        world), is the exception: it forces ``allow_permanent=False`` regardless of
        the setting. Reusable consent must never extend to an effect that cannot be
        undone, because a session-wide bypass earned from a lower-risk approval
        would otherwise silently authorise it (see
        ``SessionAwareGate._bypass_all`` fallthrough). Such writes are confirmed
        every time.

        Setting ``hardware.envelope_grant`` to false narrows reversible writes
        further: the grant identity becomes per-value (see
        ``HardwareTools._grant_band``) and no profile-wide "always" choice is
        offered, so each command is decided on its own. ``allow_permanent`` alone
        would not achieve that -- the orchestrator withholds only the "always"
        choice and still offers a session scope -- which is why the value enters the
        grant identity rather than relying on this flag.
        """
        target = f"{device_id}.{channel_id}"
        reusable = self._reusable_consent_allowed()
        if effect in _HIGH_TIER_EFFECTS:
            irreversible = not envelope.reversible
            # DISPENSE outputs material into the world; a substance that has left
            # the device cannot be un-dispensed even if the declaration marks the
            # channel reversible, so it is treated as irreversible for the purpose
            # of reusable consent.  Consequently, effect=dispense **never** receives
            # session-level or profile-level reusable consent (allow_permanent is
            # always False), regardless of the channel's ``reversible`` flag.
            external_output = effect == HardwareEffect.DISPENSE.value
            reasons = [f"device_{effect}"]
            if irreversible:
                reasons.append("irreversible")
            # A write whose effect cannot be undone is confirmed every time:
            # reusable session/profile consent is withheld so that a session-wide
            # bypass earned from a lower-risk approval cannot silently authorise it.
            # Reversible setpoints keep band-scoped reusable consent, which is what
            # keeps the gate usable for routine motion.
            allow_permanent = reusable and not (irreversible or external_output)
            return RiskAssessment(
                level=RiskLevel.HIGH,
                score=0.8 if irreversible else 0.7,
                reasons=tuple(reasons),
                explanation=(
                    f"This commands {target} within its declared envelope"
                    + (
                        ". The effect cannot be undone by writing the value back."
                        if irreversible
                        else "."
                    )
                ),
                allow_permanent=allow_permanent,
                metadata={"envelope_band": envelope.band_key()},
            )
        return RiskAssessment(
            level=RiskLevel.MEDIUM,
            score=0.5,
            reasons=(f"device_{effect}",),
            explanation=(
                f"This changes a setpoint on {target} within its declared envelope"
                + (
                    f", stabilising after {envelope.settling_time_s:g}s."
                    if envelope.settling_time_s > 0
                    else "."
                )
            ),
            allow_permanent=reusable,
            metadata={"envelope_band": envelope.band_key()},
        )

    def _reusable_consent_allowed(self) -> bool:
        """Return whether a band-scoped grant may be offered.

        Defaults to True when the setting cannot be read: the alternative is prompting
        for every command, which is how a gate gets disabled by the person it protects.
        """
        settings = getattr(self._registry, "settings", None)
        return bool(getattr(settings, "envelope_grant", True))

    @staticmethod
    def _unevaluable_interlocks(context: Any, envelope: Envelope) -> tuple[str, ...]:
        readable = {c.channel_id for c in context.channels if c.is_readable}
        missing: list[str] = []
        for name in envelope.requires_interlocks:
            lock = context.interlock(name)
            if lock is None or lock.channel_id not in readable:
                missing.append(name)
        return tuple(sorted(missing))

    @staticmethod
    def _hardline(reason: str, explanation: str) -> RiskAssessment:
        """Build a verdict that no grant, scope, or bypass can override."""
        return RiskAssessment(
            level=RiskLevel.CRITICAL,
            score=1.0,
            reasons=(reason,),
            explanation=explanation,
            hardline=True,
            allow_permanent=False,
        )


def build_risk_classifier(
    registry: Any | None, *, fallback: RiskClassifier | None = None
) -> RiskClassifier:
    """Return the classifier to install on ``ApprovalOrchestrator``.

    Called from both gate installation sites -- in-process and daemon-side -- so a
    device behaves identically whether or not the daemon is running. With no
    registry it returns the fallback unchanged, keeping behaviour byte-identical
    when hardware is disabled.
    """
    base = fallback or DefaultRiskClassifier()
    if registry is None:
        return base
    return CompositeRiskClassifier(
        fallback=base,
        by_prefix={DEVICE_KIND_PREFIX: HardwareRiskClassifier(registry)},
    )


def _describe_bounds(envelope: Envelope) -> str:
    low = "-inf" if envelope.min_value is None else f"{envelope.min_value:g}"
    high = "+inf" if envelope.max_value is None else f"{envelope.max_value:g}"
    return f"allowed range {low}..{high}"


__all__ = [
    "DEVICE_KIND_PREFIX",
    "HardwareRiskClassifier",
    "build_risk_classifier",
]
