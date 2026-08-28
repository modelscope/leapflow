"""Physical outcome learning: commanded value in, numeric prediction error out.

This is where the physical domain earns its keep. In the UI domain "was the prediction
right" is genuinely ambiguous -- did the window title matching count, did the user get
what they wanted -- so the world model has to ask an LLM to rate the distance. Physics
has no such ambiguity: the command was 37.0, the device settled at 36.8, the error is
0.2. It is the first place the prediction loop can get a clean ground truth, and it costs
no model call at all.

So this module reuses the *learning* half of the world model -- ``ExperienceStore`` for
durable storage and similarity retrieval -- and replaces the LLM predict/compare half
with arithmetic. ``PredictionLoop.record_failure`` is the existing precedent for writing
to the store without a snapshot or a model call.

The point is not to score the agent. It is that an optimisation performed once should not
have to be performed again: discovering that a viscous protein sample needs a slow
aspiration rate is worth remembering, and without this the discovery evaporates with the
turn that made it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from leapflow.hardware.context import Channel, Envelope, as_numeric

logger = logging.getLogger(__name__)

DEFAULT_PENDING_TTL_S = 900.0
"""How long a commanded value waits for an observation before being abandoned.

Bounded because a command whose channel is never read again would otherwise sit in memory
for the life of the process, and an observation arriving fifteen minutes later says more
about the room than about the command.
"""


@dataclass(frozen=True)
class PhysicalOutcome:
    """A commanded value compared against what the device actually did."""

    device_id: str
    channel_id: str
    quantity: str
    unit: str
    commanded: float
    observed: float
    delta: float
    residual: float
    conditions: str = ""
    settled: bool = True
    timestamp: float = 0.0

    @property
    def accurate(self) -> bool:
        """Return whether the device landed close to what was asked of it."""
        return self.delta <= 0.05

    def to_action_description(self) -> str:
        """Return the retrieval key: what was done, under what conditions.

        Conditions lead the text because that is what a later question matches on. "What
        rate worked for a viscous protein sample" is a search for the *situation*, not for
        a channel name, and ``ExperienceStore`` retrieves by keyword.
        """
        parts = [f"{self.quantity or self.channel_id} {self.commanded:g}"]
        if self.unit:
            parts.append(self.unit)
        if self.conditions.strip():
            parts.append(f"conditions: {self.conditions.strip()}")
        return " ".join(parts)

    def to_actual_effect(self) -> str:
        return (
            f"settled at {self.observed:g}{f' {self.unit}' if self.unit else ''} "
            f"(residual {self.residual:g}, normalised delta {self.delta:.3f})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "channel_id": self.channel_id,
            "quantity": self.quantity,
            "unit": self.unit,
            "commanded": self.commanded,
            "observed": self.observed,
            "residual": self.residual,
            "delta": self.delta,
            "conditions": self.conditions,
            "settled": self.settled,
        }


@dataclass
class _PendingCommand:
    """A commanded value awaiting a trustworthy observation."""

    device_id: str
    channel_id: str
    quantity: str
    unit: str
    commanded: float
    envelope: Envelope
    conditions: str
    settle_after: float
    expires_at: float


def normalized_delta(
    *, commanded: float, observed: float, envelope: Envelope
) -> tuple[float, float]:
    """Return ``(normalised_delta, raw_residual)`` for one command.

    Normalisation matters more than it looks. ``ExperienceStore`` is shared across every
    domain and its consumers compare delta against fixed thresholds, so a raw residual
    would make 0.2 degrees and 0.2 microlitres per second the same number while meaning
    entirely different things. Dividing by the declared envelope span makes the error
    dimensionless and comparable -- another use for limits a human already wrote down.

    Without a declared span the residual is scaled against the magnitude of the command
    instead, which keeps the value bounded and meaningful; a command of zero falls back to
    the bare residual, clamped.
    """
    residual = observed - commanded
    magnitude = abs(residual)
    span = _declared_span(envelope)
    if span:
        return min(1.0, magnitude / span), residual
    if commanded:
        return min(1.0, magnitude / abs(commanded)), residual
    return min(1.0, magnitude), residual


class HardwareOutcomeRecorder:
    """Turns physical commands and observations into retrievable experience.

    Holds no model and makes no network call. Every write path is contained: learning is
    valuable, but a failure to learn must never fail the operation that produced the
    observation.
    """

    def __init__(
        self,
        experience_store: Any = None,
        *,
        pending_ttl_s: float = DEFAULT_PENDING_TTL_S,
    ) -> None:
        self._store = experience_store
        self._pending_ttl_s = pending_ttl_s
        self._pending: dict[tuple[str, str], _PendingCommand] = {}
        self._recorded = 0

    @property
    def enabled(self) -> bool:
        return self._store is not None

    @property
    def recorded(self) -> int:
        return self._recorded

    @property
    def pending(self) -> int:
        return len(self._pending)

    # ── Command side ──

    def record_command(
        self,
        *,
        device_id: str,
        channel: Channel,
        value: Any,
        conditions: str = "",
        now: float | None = None,
    ) -> None:
        """Remember a command so a later observation can be compared against it.

        Only numeric commands are tracked: a boolean or enumerated state has no residual
        to compute, and inventing one would put meaningless numbers into a store whose
        delta is read by other subsystems.
        """
        if self._store is None:
            return
        commanded = as_numeric(value)
        if commanded is None:
            return
        moment = now if now is not None else time.monotonic()
        key = (device_id, channel.channel_id)
        self._pending[key] = _PendingCommand(
            device_id=device_id,
            channel_id=channel.channel_id,
            quantity=channel.quantity,
            unit=channel.unit,
            commanded=commanded,
            envelope=channel.envelope,
            conditions=conditions,
            # Settling is respected because a reading taken before the value stabilises
            # measures the transition, not the outcome. Recording that as the error would
            # teach the store something false about the device.
            settle_after=moment + max(0.0, channel.envelope.settling_time_s),
            expires_at=moment + self._pending_ttl_s,
        )

    # ── Observation side ──

    def observe(
        self,
        *,
        device_id: str,
        channel_id: str,
        value: Any,
        now: float | None = None,
    ) -> PhysicalOutcome | None:
        """Compare an observation against a pending command, storing the experience.

        Returns the outcome when one was recorded, or None when there is nothing to
        compare, the value is not numeric, or the channel has not settled yet.
        """
        if self._store is None:
            return None
        key = (device_id, channel_id)
        pending = self._pending.get(key)
        if pending is None:
            return None
        moment = now if now is not None else time.monotonic()
        if moment > pending.expires_at:
            self._pending.pop(key, None)
            return None
        if moment < pending.settle_after:
            return None
        observed = as_numeric(value)
        if observed is None:
            return None

        self._pending.pop(key, None)
        delta, residual = normalized_delta(
            commanded=pending.commanded, observed=observed, envelope=pending.envelope
        )
        outcome = PhysicalOutcome(
            device_id=device_id,
            channel_id=channel_id,
            quantity=pending.quantity,
            unit=pending.unit,
            commanded=pending.commanded,
            observed=observed,
            delta=delta,
            residual=residual,
            conditions=pending.conditions,
            timestamp=time.time(),
        )
        self._store_outcome(outcome)
        return outcome

    def _store_outcome(self, outcome: PhysicalOutcome) -> None:
        try:
            self._store.store(
                action_description=outcome.to_action_description(),
                app_context=outcome.device_id,
                predicted_effect=(
                    f"reach {outcome.commanded:g}{f' {outcome.unit}' if outcome.unit else ''}"
                ),
                actual_effect=outcome.to_actual_effect(),
                delta=outcome.delta,
                pre_state_summary=f"{outcome.device_id}.{outcome.channel_id}",
                post_state_summary=outcome.conditions[:200],
            )
            self._recorded += 1
        except Exception as exc:  # noqa: BLE001 - learning must not fail the operation
            logger.warning(
                "Could not store physical outcome for %s.%s: %s",
                outcome.device_id,
                outcome.channel_id,
                exc,
                exc_info=True,
            )

    def drop_pending(self, device_id: str, channel_id: str) -> None:
        """Forget a command, so a failed write cannot later be scored as an outcome."""
        self._pending.pop((device_id, channel_id), None)

    # ── Recall ──

    def recall(
        self,
        *,
        device_id: str,
        channel: Channel,
        conditions: str = "",
        limit: int = 3,
    ) -> tuple[dict[str, Any], ...]:
        """Return prior outcomes for this channel under similar conditions.

        This is the payoff. An optimisation performed once -- finding the rate a viscous
        sample tolerates -- becomes a starting point instead of an experiment to repeat.

        Ranking is by condition relevance *first* and tracking accuracy second, which is
        not interchangeable with the reverse. Keyword retrieval matches on the channel and
        unit tokens that every record for this channel shares, so ordering by accuracy
        alone lets a perfectly-tracking but unrelated experience -- water, when the
        question is about protein -- displace the one that actually answers the question.
        """
        if self._store is None:
            return ()
        query = " ".join(
            part
            for part in (channel.quantity or channel.channel_id, channel.unit, conditions)
            if part
        )
        try:
            experiences = self._store.retrieve_similar(query, device_id, limit=limit * 4)
        except Exception as exc:  # noqa: BLE001 - recall is an optimisation, not a duty
            logger.debug("Physical outcome recall failed: %s", exc, exc_info=True)
            return ()

        wanted = _condition_tokens(conditions)
        rows: list[dict[str, Any]] = []
        for experience in experiences:
            summary = _summarize_experience(experience)
            if summary is None:
                continue
            summary["relevance"] = _relevance(summary["command"], wanted)
            rows.append(summary)
        rows.sort(key=lambda row: (-row["relevance"], row["delta"]))
        return tuple(rows[:limit])


def _condition_tokens(conditions: str) -> frozenset[str]:
    """Return the distinctive words of a condition string.

    Short tokens are dropped because they carry no discriminating power and would make
    every record look related to every query.
    """
    return frozenset(
        token.strip(",.;:()")
        for token in conditions.lower().split()
        if len(token.strip(",.;:()")) >= 3
    )


def _relevance(command: str, wanted: frozenset[str]) -> float:
    """Return the share of requested condition words this experience shares.

    With no conditions requested every experience is equally relevant, so ranking falls
    through to tracking accuracy alone.
    """
    if not wanted:
        return 1.0
    haystack = command.lower()
    matched = sum(1 for token in wanted if token in haystack)
    return matched / len(wanted)


def _summarize_experience(experience: Any) -> dict[str, Any] | None:
    """Reduce a stored experience to the few fields a decision needs.

    Deliberately lossy. The caller is about to put this in a model's context, and the
    useful content is "this command, under these conditions, tracked this well" -- not the
    full record.
    """
    action = str(getattr(experience, "action_description", "") or "")
    if not action:
        return None
    return {
        "command": action,
        "outcome": str(getattr(experience, "actual_effect", "") or ""),
        "delta": float(getattr(experience, "delta", 1.0) or 0.0),
    }


def _declared_span(envelope: Envelope) -> float:
    if envelope.min_value is None or envelope.max_value is None:
        return 0.0
    span = envelope.max_value - envelope.min_value
    return span if span > 0 else 0.0


__all__ = [
    "DEFAULT_PENDING_TTL_S",
    "HardwareOutcomeRecorder",
    "PhysicalOutcome",
    "normalized_delta",
]
