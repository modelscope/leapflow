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

from leapflow.hardware.context import Channel, Envelope, _declared_span, as_numeric

logger = logging.getLogger(__name__)

DEFAULT_PENDING_TTL_S = 900.0
"""How long a commanded value waits for an observation before being abandoned.

Bounded because a command whose channel is never read again would otherwise sit in memory
for the life of the process, and an observation arriving fifteen minutes later says more
about the room than about the command.
"""

_MAX_PENDING_PER_CHANNEL = 4
"""Maximum pending commands tracked per ``(device_id, channel_id)`` pair.

Prevents unbounded memory growth when commands arrive faster than observations.
When the cap is hit the oldest pending command is evicted -- it was the least
likely to match an incoming observation, and losing it is strictly better than
losing the newest command that is still expected to settle.
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
    expected: float | None = None
    """What this channel was expected to reach, given what it has done before.

    ``None``, not zero, when nothing has been learned yet. A numeric default would be
    indistinguishable from a genuine prediction of that value, and zero in particular
    produced nonsense: an outcome built without it reported "reach 0 (commanded 50,
    prior bias -50)" about a device that behaved perfectly. Read through ``predicted``,
    which supplies the honest starting point instead.
    """
    model_error: float | None = None
    model_offset: float | None = None
    """The same error measured against the prediction rather than the command.

    ``None`` while there is no prediction to measure against, in which case
    ``model_delta`` reports the device-relative figure -- not as a stand-in, but
    because with no prediction the two genuinely are the same measurement.

    The distinction between them is the whole point of keeping both. ``delta`` answers
    "how far off was the device?" and never improves -- a valve that always undershoots
    by eight percent reports the same error forever. ``model_delta`` answers "how far
    off were *we*?", which is the only one of the two a learning loop can drive down.
    """

    @property
    def predicted(self) -> float:
        """The value expected of this channel: the command until evidence says otherwise.

        A device is expected to do what it is told, so the first command to a channel
        is predicted exactly. That is a real prior, not a missing value.
        """
        return self.commanded if self.expected is None else self.expected

    @property
    def model_delta(self) -> float:
        """Normalised error against the prediction."""
        return self.delta if self.model_error is None else self.model_error

    @property
    def model_residual(self) -> float:
        """Raw error against the prediction."""
        return self.residual if self.model_offset is None else self.model_offset

    @property
    def accurate(self) -> bool:
        """Return whether the device landed close to what was asked of it."""
        return self.delta <= 0.05

    @property
    def predictable(self) -> bool:
        """Return whether the outcome matched what prior observations implied.

        A device can be inaccurate and perfectly predictable at the same time -- that
        is the useful case, because a known bias can be compensated. The reverse, an
        accurate device behaving unpredictably, is the one that needs attention.
        """
        return self.model_delta <= 0.05

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

    def to_predicted_effect(self) -> str:
        """Return what was expected, naming the learned correction when there was one.

        Stated separately from the command so a later reader can tell an accurate
        device from a well-understood one.
        """
        unit = f" {self.unit}" if self.unit else ""
        if self.predicted == self.commanded:
            return f"reach {self.commanded:g}{unit}"
        return (
            f"reach {self.predicted:g}{unit} "
            f"(commanded {self.commanded:g}, prior bias {self.predicted - self.commanded:+g})"
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
            "predicted": self.predicted,
            "model_residual": self.model_residual,
            "model_delta": self.model_delta,
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
    predicted: float
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

    **G-1 (confirmed by E3-T0)**: when ``tolerance`` (absolute precision) is declared,
    the delta is normalised against it instead of the span.  Bias is an absolute
    quantity that does not scale with the declared range; span-normalisation makes a
    tight-tolerance channel report a misleadingly small error.

    Without a declared span the residual is scaled against the magnitude of the command
    instead, which keeps the value bounded and meaningful; a command of zero falls back to
    the bare residual, clamped.
    """
    residual = observed - commanded
    magnitude = abs(residual)
    # G-1: tolerance takes precedence over span when declared.
    if envelope.tolerance > 0:
        return min(1.0, magnitude / envelope.tolerance), residual
    span = _declared_span(envelope)
    if span:
        return min(1.0, magnitude / span), residual
    if commanded:
        return min(1.0, magnitude / abs(commanded)), residual
    return min(1.0, magnitude), residual


_BIAS_ALPHA = 0.1
"""Weight given to the newest residual when updating a channel's bias.

An exponential moving average (EMA) forgetting factor. Low enough that one
outlier cannot capture the estimate, high enough that a device whose behaviour
genuinely changed is tracked within a moderate number of commands. Under
continuous drift the estimate converges rather than accumulating without bound,
because each update discounts the prior by ``(1 - alpha)``.

A plain mean would never forget the state the bench was in last week.
"""

_MAX_BIAS_SPAN_FRACTION = 0.25
"""Ceiling on the learned correction, as a fraction of the declared envelope span.

A prediction is allowed to be wrong; it is not allowed to be absurd. Without a cap one
badly-timed reading -- a transient caught just after settling -- could push the expected
value outside the limits a human wrote down, and every subsequent model residual would
be measured against a value the device is not permitted to reach.
"""


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
        self._pending: dict[tuple[str, str], list[_PendingCommand]] = {}
        self._bias: dict[tuple[str, str], tuple[float, int]] = {}
        self._recorded = 0
        self._evicted_pending_total = 0

    @property
    def enabled(self) -> bool:
        return self._store is not None

    @property
    def recorded(self) -> int:
        return self._recorded

    @property
    def pending(self) -> int:
        return sum(len(entries) for entries in self._pending.values())

    @property
    def evicted_pending_total(self) -> int:
        """Total non-expired pending commands evicted because the per-channel cap was full."""
        return self._evicted_pending_total

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
        entry = _PendingCommand(
            device_id=device_id,
            channel_id=channel.channel_id,
            quantity=channel.quantity,
            unit=channel.unit,
            commanded=commanded,
            predicted=commanded + self._bias_for(key, channel.envelope),
            envelope=channel.envelope,
            conditions=conditions,
            # Settling is respected because a reading taken before the value stabilises
            # measures the transition, not the outcome. Recording that as the error would
            # teach the store something false about the device.
            settle_after=moment + max(0.0, channel.envelope.effective_settling_s),
            expires_at=moment + self._pending_ttl_s,
        )
        entries = self._pending.setdefault(key, [])
        entries.append(entry)
        # When the list exceeds the cap, reclaim space in two stages:
        # 1. Purge any entries that have already expired — they would never
        #    match an observation anyway, so removing them loses nothing.
        # 2. Only if still over the limit (all entries are live), FIFO-evict
        #    the oldest non-expired entry and count it so the operator can
        #    tell whether the cap is too small for the write rate.
        if len(entries) > _MAX_PENDING_PER_CHANNEL:
            entries[:] = [e for e in entries if moment <= e.expires_at]
            if len(entries) > _MAX_PENDING_PER_CHANNEL:
                entries.pop(0)
                self._evicted_pending_total += 1
                logger.debug(
                    "Evicted a non-expired pending command on %s.%s "
                    "(pending=%d, cap=%d, evicted_total=%d)",
                    device_id,
                    channel.channel_id,
                    len(entries),
                    _MAX_PENDING_PER_CHANNEL,
                    self._evicted_pending_total,
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
        entries = self._pending.get(key)
        if not entries:
            return None
        moment = now if now is not None else time.monotonic()

        # Purge expired entries before searching for a match.
        entries[:] = [e for e in entries if moment <= e.expires_at]
        if not entries:
            del self._pending[key]
            return None

        # Among settled entries, pick the one with the earliest settle_after
        # (FIFO for equal settling) so the oldest ready command is matched first.
        best_idx: int | None = None
        for idx, entry in enumerate(entries):
            if moment >= entry.settle_after:
                if best_idx is None or entry.settle_after < entries[best_idx].settle_after:
                    best_idx = idx
        if best_idx is None:
            return None

        observed = as_numeric(value)
        if observed is None:
            return None

        pending = entries.pop(best_idx)
        if not entries:
            del self._pending[key]
        delta, residual = normalized_delta(
            commanded=pending.commanded, observed=observed, envelope=pending.envelope
        )
        # Measured against the prediction as well as the command. Only the second of
        # these can be driven down by learning: a device with a fixed bias reports the
        # same command-relative error forever, however well it is understood.
        model_delta, model_residual = normalized_delta(
            commanded=pending.predicted, observed=observed, envelope=pending.envelope
        )
        self._update_bias(key, residual)
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
            expected=pending.predicted,
            model_error=model_delta,
            model_offset=model_residual,
        )
        self._store_outcome(outcome)
        return outcome

    # ── Prediction ──

    def _bias_for(self, key: tuple[str, str], envelope: Envelope) -> float:
        """Return the learned correction for a channel, clamped to its envelope.

        Zero until the channel has been observed once, because a device is expected to
        do what it is told until evidence says otherwise.

        Held in memory for the life of the process and deliberately not persisted. The
        durable record is the experience store, but that store is shared across domains
        and keeps only the *magnitude* of the error -- the sign, which is what makes a
        correction a correction, is not recoverable from it. Extending a cross-domain
        schema for one domain's estimator would be the wrong trade, so the honest
        statement is that this calibration restarts with the process.
        """
        entry = self._bias.get(key)
        if entry is None:
            return 0.0
        bias = entry[0]
        span = _declared_span(envelope)
        if span <= 0:
            return bias
        cap = span * _MAX_BIAS_SPAN_FRACTION
        return max(-cap, min(cap, bias))

    def _update_bias(self, key: tuple[str, str], residual: float) -> None:
        """Fold one observation into the channel's running correction via EMA.

        Uses an exponential moving average with forgetting factor ``_BIAS_ALPHA``.
        Under steady-state drift the estimate converges to the true offset rather
        than accumulating without bound.  ``samples`` is incremented each call so
        ``calibration_for()`` can expose the effective observation window.
        """
        entry = self._bias.get(key)
        if entry is None:
            self._bias[key] = (residual, 1)
            return
        previous, samples = entry
        alpha = _BIAS_ALPHA
        self._bias[key] = (previous + alpha * (residual - previous), samples + 1)

    def calibration_for(self, device_id: str, channel_id: str) -> tuple[float, int] | None:
        """Return ``(bias, samples)`` learned for a channel, or None if untested.

        Exposed so a person can see what the agent concluded about their own bench in
        the units they declared. A correction the operator cannot inspect is one they
        cannot disagree with, and this one is derived from observation rather than
        stated by anyone.

        ``samples`` counts the total observations folded in. Because the EMA uses a
        forgetting factor, only the most recent ~1/alpha observations dominate; the
        "effective window" is approximately ``1 / _BIAS_ALPHA`` samples.
        """
        return self._bias.get((device_id, channel_id))

    def _store_outcome(self, outcome: PhysicalOutcome) -> None:
        try:
            self._store.store(
                action_description=outcome.to_action_description(),
                app_context=outcome.device_id,
                predicted_effect=outcome.to_predicted_effect(),
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
        """Forget all pending commands for a channel.

        Called on the write-failure path so that whatever the device settles at
        is not retroactively scored against a command that never landed.
        Clears every pending entry for the ``(device_id, channel_id)`` pair,
        because after a transport failure none of them can be trusted.
        """
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


__all__ = [
    "DEFAULT_PENDING_TTL_S",
    "HardwareOutcomeRecorder",
    "PhysicalOutcome",
    "normalized_delta",
]
