# Copyright (c) Alibaba, Inc. and its affiliates.
"""Continuous sampling: raw readings in, derived events out.

The layering here is the whole point, and it is a boundary decision rather than an
optimisation. ``SignalBuffer`` holds 50 interaction signals and every signal drives
causal fusion; a single 10 Hz channel would flush that buffer in five seconds and
run fusion at sampling rate, destroying both subsystems at once.

So raw readings stay inside this module, in a bounded per-channel ring. What crosses
into the interaction signal pipeline is *events* -- a threshold crossed, a rate
exceeded, samples lost -- which occur at the rate at which something worth noticing
actually happens.

Hard detection uses the channel's unchanged ``Envelope``. Report-only operational
warnings use the separately validated observation policy and never grant authority.

Clocks are not interchangeable. Staleness uses host ``Reading.monotonic_at``;
physical rates require qualified acquisition intervals (remote host arrival timing
is unavailable for this purpose). Events and persisted windows carry wall-clock
``Reading.observed_at``, matching findings, audit, and the board's time axis.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Deque, Iterable, Iterator, Mapping, Protocol, runtime_checkable

from leapflow.hardware.context import (
    Channel,
    DegradationPolicy,
    HardwareContext,
    Quality,
    as_numeric,
)
from leapflow.hardware.degradation import (
    DEGRADATION_SUPPRESSED,
    DegradationCoordinator,
    degradation_actions_enabled,
    sensor_loss_debounce_s,
)
from leapflow.hardware.transport import Reading
from leapflow.hardware.node_gate import rate_delta
from leapflow.hardware.observation import ObservationJudgment, ObservationPolicy, ObservationSettings

logger = logging.getLogger(__name__)

DEFAULT_RING_CAPACITY = 4096

MIN_EVENT_INTERVAL_S = 1.0
"""Floor on how often one channel may re-report the same event kind.

Edge-triggered kinds rarely reach it. Level-triggered ones -- a slew that stays
above ``max_rate`` for a whole ramp -- would otherwise emit once per sample and
reproduce, on the consumer side, the exact flood the raw/event split prevents on
the producer side. The first occurrence of a kind is never suppressed, and
suppressions are counted rather than discarded silently.
"""

EventSink = Callable[["HardwareEvent"], None]
"""Receives derived events. Must be thread-safe and non-blocking."""


class HardwareEventClass:
    """Behavioural classification for hardware events (D-X4 three-way split).

    Every event kind maps to exactly one class. The classification determines the
    dispatch path:

    ``REFLEX``
        Safety-critical, handled without LLM involvement.  Alert-policy actions
        (hw_estop, degradation) execute immediately via the reflex path.
    ``DELIBERATIVE``
        May require an LLM decision.  If hardware-triggered reentry is enabled,
        these events can open an isolated subagent round (writes still gated by
        normal approval).
    ``RECORD``
        Informational only.  Stored in the reading store, visible on the board
        and in the watch pipeline, but not routed to any decision path.
    """

    REFLEX = "reflex"
    DELIBERATIVE = "deliberative"
    RECORD = "record"


# Canonical mapping: event kind → behavioural class.
# The mapping is intentionally conservative: when in doubt an event is REFLEX
# (fail-closed). Adding a new EventKind without extending this mapping makes it
# REFLEX by default, which is the only safe choice for an unknown condition.
_EVENT_CLASS_MAP: dict[str, str] = {
    # ── Reflex (safety-critical, no LLM) ──
    "threshold_exceeded": HardwareEventClass.REFLEX,
    "rate_exceeded": HardwareEventClass.REFLEX,
    "unreachable": HardwareEventClass.REFLEX,
    # ── Deliberative (may trigger reentry round) ──
    "quality_degraded": HardwareEventClass.DELIBERATIVE,
    "stale": HardwareEventClass.DELIBERATIVE,
    # ── Record (informational, store only) ──
    "settled": HardwareEventClass.RECORD,
    "observation_trial": HardwareEventClass.RECORD,
    "sample_loss": HardwareEventClass.RECORD,
    "calibration_started": HardwareEventClass.RECORD,
    "calibration_completed": HardwareEventClass.RECORD,
    "calibration_failed": HardwareEventClass.RECORD,
    "calibration_expired": HardwareEventClass.RECORD,
    "calibration_candidate": HardwareEventClass.RECORD,
    "calibration_rejected": HardwareEventClass.RECORD,
    "calibration_no_op": HardwareEventClass.RECORD,
    "calibration_activated": HardwareEventClass.RECORD,
    "calibration_rolled_back": HardwareEventClass.RECORD,
    "calibration_invalidated": HardwareEventClass.RECORD,
    "calibration_drift": HardwareEventClass.DELIBERATIVE,
    # A withheld supervisory action is a warning for a person, never a trigger:
    # classing it REFLEX would hand the alert policy the stop just withheld.
    DEGRADATION_SUPPRESSED: HardwareEventClass.RECORD,
    # Advisory limits watch what the device's own firmware already enforces; a
    # breach informs a person and must never drive a reflex (see ``advisory``).
    "advisory_exceeded": HardwareEventClass.RECORD,
    "advisory_cleared": HardwareEventClass.RECORD,
    "observation_warning": HardwareEventClass.RECORD,
    "observation_cleared": HardwareEventClass.RECORD,
    "observation_unknown": HardwareEventClass.RECORD,
}


def classify_hardware_event(kind: str) -> str:
    """Return the behavioural class for the given event kind.

    Unknown kinds default to REFLEX (fail-closed): an unrecognised condition is
    assumed safety-critical until explicitly classified.
    """
    return _EVENT_CLASS_MAP.get(kind, HardwareEventClass.REFLEX)


@runtime_checkable
class HardwareDeliberativeObserver(Protocol):
    """Receives deliberative hardware events for potential reentry triggering.

    Registered on the hardware registry and called from ``publish_event`` when
    a classified event has ``event_class == DELIBERATIVE``.  The observer decides
    whether the event warrants an agent reentry round; the write path inside that
    round still runs through normal approval gates.

    Implementations must be non-blocking: long work should be scheduled rather
    than awaited.
    """

    def on_deliberative_event(self, event: "HardwareEvent") -> None:
        """Accept a deliberative hardware event for evaluation."""
        ...


class EventKind:
    """Observed conditions worth telling somebody about.

    Some are traceable to a declared envelope field -- a threshold, a rate limit, a
    settling time. Others are not: sample loss is read from gaps in the transport's
    own sequence, degraded quality from what the device reports about its reading,
    and unreachability from the connection itself. What they share is that each names
    one specific observable condition rather than a judgement about it.
    """

    THRESHOLD_EXCEEDED = "threshold_exceeded"
    RATE_EXCEEDED = "rate_exceeded"
    STALE = "stale"
    SAMPLE_LOSS = "sample_loss"
    QUALITY_DEGRADED = "quality_degraded"
    SETTLED = "settled"
    OBSERVATION_TRIAL = "observation_trial"
    UNREACHABLE = "unreachable"
    """A command was refused because the device could not be reached.

    Raised from the command path rather than the sampling loop, and it is the one
    condition a board cannot infer from anything else: a bench whose commands are all
    being refused looks exactly like a bench nobody is using.
    """

    # ── Calibration lifecycle (IC-6) ──
    CALIBRATION_STARTED = "calibration_started"
    """A calibration procedure has been initiated on a channel."""
    CALIBRATION_COMPLETED = "calibration_completed"
    """A calibration procedure completed successfully."""
    CALIBRATION_FAILED = "calibration_failed"
    """A calibration procedure failed before producing a valid correction."""
    CALIBRATION_EXPIRED = "calibration_expired"
    """The last successful calibration has exceeded its validity period."""
    CALIBRATION_CANDIDATE = "calibration_candidate"
    CALIBRATION_REJECTED = "calibration_rejected"
    CALIBRATION_NO_OP = "calibration_no_op"
    CALIBRATION_ACTIVATED = "calibration_activated"
    CALIBRATION_ROLLED_BACK = "calibration_rolled_back"
    CALIBRATION_INVALIDATED = "calibration_invalidated"
    CALIBRATION_DRIFT = "calibration_drift"
    DEGRADATION_SUPPRESSED = DEGRADATION_SUPPRESSED
    """A degradation policy triggered while actions are disabled; reported only."""
    ADVISORY_EXCEEDED = "advisory_exceeded"
    """A monitor-only limit the device protects itself against was breached."""
    OBSERVATION_WARNING = "observation_warning"
    OBSERVATION_CLEARED = "observation_cleared"
    OBSERVATION_UNKNOWN = "observation_unknown"
    ADVISORY_CLEARED = "advisory_cleared"
    """A breached advisory limit returned inside its hysteresis band."""


@dataclass(frozen=True)
class HardwareEvent:
    """One notable change in a device's observed state."""

    kind: str
    device_id: str
    channel_id: str
    quantity: str
    detail: str
    value: Any = None
    unit: str = ""
    observed_at: float = 0.0
    """Wall-clock. This crosses module boundaries, so it must be the clock every
    consumer outside ``leapflow.hardware`` already uses."""
    event_class: str = ""
    """Behavioural classification (reflex / deliberative / record).

    Populated by ``classify_hardware_event`` at creation time. An empty string
    means the classification has not been applied yet; consumers that depend on
    it should fall back to ``classify_hardware_event(self.kind)``.
    """

    observation: dict[str, Any] = field(default_factory=dict)
    sample_evidence: bool = False

    @property
    def event_type(self) -> str:
        """Return the EventBus type. ``hw`` is the family every consumer groups on.

        Dotted so ``dashboard.service._event_family`` -- which splits on the first
        separator -- yields ``hw`` without any enumeration being added anywhere.
        """
        return f"hw.{self.kind}"

    @property
    def source(self) -> str:
        """Return the EventBus source: the channel that produced the observation."""
        return f"{self.device_id}.{self.channel_id}"

    def to_detail(self) -> str:
        """Return a compact one-line description for the signal pipeline."""
        where = f"{self.device_id}.{self.channel_id}"
        rendered = (
            ""
            if self.value is None
            else f" value={self.value}{f' {self.unit}' if self.unit else ''}"
        )
        return f"[{self.kind}] {where}{rendered}: {self.detail}"

    @property
    def resolved_event_class(self) -> str:
        """Return the event class, computing it from kind if not set."""
        return self.event_class or classify_hardware_event(self.kind)

    def to_payload(self) -> dict[str, Any]:
        """Return the EventBus payload, using that layer's key names.

        ``ts`` and ``_mono_ts`` are the platform's contract, not this module's
        preference: the pre-normalized pass-through reads ``ts`` for the event's
        instant and ``EventReorderBuffer`` reads ``_mono_ts`` for arrival-order
        correction. Supplying ``observed_at`` under its domain name instead would
        leave both unset -- the event would be stamped with the moment it was
        *normalized* rather than observed, and would sort against other sources by
        nothing at all.

        ``source`` is explicit for the same reason: without it the pass-through
        substitutes the event type, and every view then shows that as the origin
        instead of the channel that produced the observation.
        """
        return {
            "kind": self.kind,
            "event_class": self.resolved_event_class,
            "source": self.source,
            "device_id": self.device_id,
            "channel_id": self.channel_id,
            "quantity": self.quantity,
            "detail": self.detail,
            "value": self.value,
            "unit": self.unit,
            "ts": self.observed_at,
            "_mono_ts": time.monotonic(),
            "observation": dict(self.observation),
        }


class ReadingRing:
    """Bounded per-channel history of raw readings.

    Bounded because an unbounded one is a memory leak with a schedule: eight channels
    at 10 Hz for an overnight run is millions of samples. Losing the oldest readings is
    acceptable; what is not acceptable is losing them *silently*, which is why
    ``gaps()`` exists -- a break in the transport's sequence numbering is the only
    evidence that something was dropped between the device and here.
    """

    def __init__(self, capacity: int = DEFAULT_RING_CAPACITY) -> None:
        self._readings: Deque[Reading] = deque(maxlen=max(2, int(capacity)))
        self._dropped_sequences = 0
        self._expected_sequence: int | None = None
        self._sequence_epoch: str = ""
        self._sequence_scope: str = ""

    def record(self, reading: Reading) -> int:
        """Append *reading*, returning how many samples appear to have been lost."""
        lost = 0
        if (reading.acquisition_epoch, reading.sequence_scope) != (self._sequence_epoch, self._sequence_scope):
            self._expected_sequence = None
        self._sequence_epoch, self._sequence_scope = reading.acquisition_epoch, reading.sequence_scope
        if reading.sequence_scope == "channel":
            if self._expected_sequence is not None and reading.sequence > self._expected_sequence:
                lost = reading.sequence - self._expected_sequence
                self._dropped_sequences += lost
            if self._expected_sequence is None or reading.sequence >= self._expected_sequence:
                self._expected_sequence = reading.sequence + 1
        else:
            self._expected_sequence = None
        self._readings.append(reading)
        return lost

    @property
    def latest(self) -> Reading | None:
        return self._readings[-1] if self._readings else None

    @property
    def dropped(self) -> int:
        """Total samples missing from the sequence over this ring's lifetime."""
        return self._dropped_sequences

    def __len__(self) -> int:
        return len(self._readings)

    def __iter__(self) -> Iterator[Reading]:
        return iter(self._readings)

    def window(self, count: int) -> tuple[Reading, ...]:
        """Return the most recent *count* readings, oldest first."""
        if count <= 0:
            return ()
        items = list(self._readings)
        return tuple(items[-count:])

    def summary(self) -> dict[str, Any]:
        """Return a compact summary suitable for disclosure to a model.

        Deliberately small. A model must never be handed the raw series -- that is
        what makes an overnight run affordable -- so this is the shape that reaches
        it: where the value is now, where it has been, and whether anything was lost.
        """
        latest = self.latest
        if latest is None:
            return {"samples": 0}
        numeric = [n for r in self._readings if (n := as_numeric(r.value)) is not None]
        payload: dict[str, Any] = {
            "samples": len(self._readings),
            "latest": latest.value,
            "unit": latest.unit,
            "quality": latest.quality,
            "dropped": self._dropped_sequences,
        }
        if numeric:
            payload.update(
                {
                    "min": min(numeric),
                    "max": max(numeric),
                    "mean": sum(numeric) / len(numeric),
                    "trend": _trend(numeric),
                }
            )
        return payload


def _observation_now(reading: Reading) -> float:
    """Use the reception clock when source evidence cannot carry a valid clock."""
    at = reading.monotonic_at
    return at if reading.is_trustworthy and math.isfinite(at) and at > 0 else time.monotonic()


class HardwareEventDetector:
    """Judge raw hard evidence and separately configured report-only observations.

    Stateful per channel: staleness and rate need to remember what came before. Kept
    separate from the ring so the detection rules can be tested against a synthetic
    series without a transport in the picture.
    """

    def __init__(self, context: HardwareContext, channel: Channel,
                 policy: ObservationPolicy | None = None) -> None:
        self._context = context
        self._channel = channel
        self._last_numeric: float | None = None
        self._last_monotonic: float | None = None
        self._degraded_streak = 0
        self._breached = False
        self._stale = False
        self._last_acquisition: float | None = None
        self._acquisition_epoch = ""
        self._observation = ObservationJudgment(context.device_id, channel, policy or ObservationPolicy())
        self.safety_evidence: dict[str, HardwareEvent | None] = {}

    @property
    def observation_snapshots(self) -> tuple[dict[str, Any], ...]:
        return self._observation.snapshots()

    def acknowledge_observation(self, checkpoints: Mapping[str, Mapping[str, Any]]) -> None:
        self._observation.acknowledge(checkpoints)

    def reconfigure_observation(self, policy: ObservationPolicy) -> None:
        if policy != self._observation.policy:
            self._observation.reconfigure(policy)

    def invalidate_observation(self, *, identity: str = "") -> None:
        self._last_numeric = self._last_acquisition = None
        if identity:
            self._observation.reset_identity(identity)
        else:
            self._observation.invalidate()

    def observe(self, reading: Reading, *, lost: int = 0, ring: ReadingRing | None = None,
                pending_motion: bool = False, fault_free: bool = False) -> tuple[HardwareEvent, ...]:
        """Return the events this reading produces."""
        events: list[HardwareEvent] = []
        channel = self._channel
        envelope = channel.envelope
        known_before = {name for name, state in self._observation.states.items() if state.known and state.active}
        self._observation.prepare_epoch(reading)
        self.safety_evidence = {EventKind.THRESHOLD_EXCEEDED: None, EventKind.STALE: None,
                                EventKind.RATE_EXCEEDED: None, EventKind.QUALITY_DEGRADED: None}
        if reading.quality == Quality.STALE.value:
            self.safety_evidence[EventKind.STALE] = self._event(EventKind.STALE, reading, "stale source evidence")

        if lost > 0:
            events.append(
                self._event(
                    EventKind.SAMPLE_LOSS,
                    reading,
                    f"{lost} sample(s) missing from the transport sequence",
                )
            )

        if reading.quality != Quality.OK.value:
            self._degraded_streak += 1
            self.safety_evidence[EventKind.QUALITY_DEGRADED] = self._event(
                EventKind.QUALITY_DEGRADED, reading, f"quality is {reading.quality!r}")
            # Three in a row rather than one: a lone suspect sample is noise, a run of
            # them is a fault. Reporting every one would reproduce the sampling-rate
            # flood this layer exists to prevent.
            if self._degraded_streak == 3 and reading.quality != Quality.STALE.value:
                events.append(
                    self._event(
                        EventKind.QUALITY_DEGRADED,
                        reading,
                        f"quality has been {reading.quality!r} for 3 consecutive samples",
                    )
                )
        else:
            self._degraded_streak = 0

        numeric = as_numeric(reading.value)
        if numeric is not None and envelope.declared:
            events.extend(self._numeric_events(reading, numeric, report_continuous=lost == 0))
        elif not reading.is_trustworthy or numeric is None:
            self._observation.invalidate()
        valid = numeric is not None and reading.is_trustworthy
        acquisition = reading.acquisition_at if reading.acquisition_at is not None else reading.monotonic_at
        continuous = (self._last_acquisition is None or acquisition > self._last_acquisition)
        self._last_numeric = numeric if valid and continuous else None
        self._last_acquisition = acquisition if valid and math.isfinite(acquisition) else None
        self._acquisition_epoch = reading.acquisition_epoch
        if reading.is_trustworthy:
            self._last_monotonic = _observation_now(reading)
        notices = self._observation.observe_soft(reading, ring=ring, pending_motion=pending_motion,
                                                 fault_free=fault_free)
        for notice in notices:
            kind = EventKind.OBSERVATION_CLEARED if notice["phase"] == "recovery" else EventKind.OBSERVATION_WARNING
            events.append(HardwareEvent(kind=kind, device_id=self._context.device_id,
                                        channel_id=channel.channel_id, quantity=channel.quantity,
                                        detail=(f"{notice['condition']}: {notice['reason']}; "
                                                f"soft={notice['soft_min']}..{notice['soft_max']}, "
                                                f"hard={notice['hard_min']}..{notice['hard_max']}, "
                                                f"margin={notice['margin']:g}, duration={notice['duration_s']:.2f}s; reported only"),
                                        value=reading.value, unit=channel.unit, observed_at=reading.observed_at,
                                        observation=notice))
        # Bad numeric evidence invalidates soft/rate history, not the explicit
        # quality/freshness condition carried by this same reading.
        now = _observation_now(reading)
        self._observation.condition("quality_degraded", breached=self._degraded_streak >= 3 and reading.quality != Quality.STALE.value,
                                    clear=reading.is_trustworthy, reading=reading,
                                    now=now, reason="sampling quality")
        self._observation.condition("stale", breached=reading.quality == Quality.STALE.value,
                                    clear=reading.is_trustworthy, reading=reading,
                                    now=now, reason="sampling freshness")
        self._stale = not reading.is_trustworthy
        if any(not self._observation.states[name].known for name in known_before):
            events.append(replace(self._event(EventKind.OBSERVATION_UNKNOWN, reading,
                "continuous report evidence is unavailable"), sample_evidence=True,
                observation={"phase": "unknown"}))
        return tuple(events)

    def check_stale(self, *, now: float | None = None) -> tuple[HardwareEvent, ...]:
        """Return a staleness event when sampling has stopped.

        Silence is itself an observation: a channel declared at 10 Hz that has said
        nothing for a second has failed, and the absence of readings is the only way
        that failure shows up.

        ``now`` is monotonic, matching ``_last_monotonic``. Comparing a wall-clock
        instant against a monotonic one yields a difference of the two epochs --
        a number so large or so negative that the deadline is either always or
        never met, and in both cases the check reports nothing useful.
        """
        channel = self._channel
        self.safety_evidence[EventKind.STALE] = None
        if channel.sample_rate_hz <= 0 or self._last_monotonic is None:
            return ()
        deadline = 2.0 / channel.sample_rate_hz
        elapsed = (now if now is not None else time.monotonic()) - self._last_monotonic
        if elapsed <= deadline:
            return ()
        self._stale = True
        self._last_numeric = self._last_acquisition = None
        state = self._observation.states.get("stale")
        if state is None or not state.active:
            self._observation.invalidate()
        reading = Reading(self._context.device_id, channel.channel_id, None, observed_at=time.time())
        notice = self._observation.condition("stale", breached=True, clear=False, reading=reading,
                                             now=now if now is not None else time.monotonic(), reason="sampling silence")
        event = HardwareEvent(kind=EventKind.STALE, device_id=self._context.device_id,
                              channel_id=channel.channel_id, quantity=channel.quantity,
                              detail=f"no sample for {elapsed:.2f}s on a {channel.sample_rate_hz:g} Hz channel",
                              unit=channel.unit, observed_at=reading.observed_at,
                              sample_evidence=True, observation=notice or {})
        self.safety_evidence[EventKind.STALE] = event
        return (event,) if notice else ()

    def _numeric_events(self, reading: Reading, numeric: float, *, report_continuous: bool = True) -> list[HardwareEvent]:
        events: list[HardwareEvent] = []
        envelope = self._channel.envelope

        # Asymmetric on purpose. Leaving the range is judged against the declared
        # limit, because that is the limit a human wrote down. Returning must clear
        # an inward margin, so a value resting on the boundary does not alternate
        # breach and recovery at the sampling rate -- which would bury the one
        # crossing that mattered under thousands of identical events.
        if not self._breached:
            if not envelope.contains(numeric):
                self._breached = True
                events.append(
                    self._event(
                        EventKind.THRESHOLD_EXCEEDED,
                        reading,
                        f"left the declared range ({_bounds(envelope)})",
                    )
                )
        elif reading.is_trustworthy and envelope.contains(numeric, margin=envelope.settle_margin):
            # Re-entering is worth one event too, so a watcher can see recovery
            # instead of inferring it from silence.
            self._breached = False
            events.append(self._event(EventKind.SETTLED, reading, "returned to the declared range"))

        if not envelope.contains(numeric):
            self.safety_evidence[EventKind.THRESHOLD_EXCEEDED] = self._event(
                EventKind.THRESHOLD_EXCEEDED, reading, f"left the declared range ({_bounds(envelope)})")
        at = reading.acquisition_at if reading.acquisition_at is not None else reading.monotonic_at
        elapsed = at - self._last_acquisition if self._last_acquisition is not None else 0.0
        rate_valid = (reading.is_trustworthy and reading.rate_timebase != "unavailable"
                      and self._last_numeric is not None and math.isfinite(at) and elapsed > 0
                      and reading.acquisition_epoch == self._acquisition_epoch
                      and (self._channel.sample_rate_hz <= 0 or elapsed <= 2.0 / self._channel.sample_rate_hz))
        delta = rate_delta(self._channel, numeric, self._last_numeric) if rate_valid else 0.0
        rate_breached = rate_valid and envelope.rate_exceeded(delta=delta, elapsed_s=elapsed)
        if rate_breached:
            event = self._event(EventKind.RATE_EXCEEDED, reading,
                                f"changing at {abs(delta) / elapsed:g}/s, above the declared {envelope.max_rate:g}/s")
            self.safety_evidence[EventKind.RATE_EXCEEDED] = event
            events.append(event)
        now = _observation_now(reading)
        threshold_notice = self._observation.condition(
            "threshold_exceeded", breached=self._breached, clear=not self._breached,
            reading=reading, now=now, valid=reading.is_trustworthy,
            reason=f"declared range ({_bounds(envelope)})")
        if threshold_notice:
            events = [replace(event, observation=threshold_notice)
                      if event.kind in {EventKind.THRESHOLD_EXCEEDED, EventKind.SETTLED} else event
                      for event in events]
        report_valid = rate_valid and report_continuous
        rate_notice = self._observation.condition(
            "rate_exceeded", breached=bool(rate_breached), clear=not rate_breached,
            reading=reading, now=at if report_valid else now, valid=bool(report_valid), reason="declared physical rate",
            extra={"rate_available": bool(rate_valid),
                   "measured_rate": abs(delta) / elapsed if rate_valid else None,
                   "declared_max_rate": envelope.max_rate, "rate_delta": delta if rate_valid else None,
                   "acquisition_interval_s": elapsed if rate_valid else None,
                   "rate_timebase": reading.rate_timebase})
        if rate_notice:
            if rate_notice["phase"] == "recovery":
                events.append(replace(self._event(EventKind.OBSERVATION_CLEARED, reading,
                                                 "physical rate returned within its declaration"),
                                      observation=rate_notice))
            else:
                # Raw evidence already reached the reflex path. This confirmed
                # transition also wakes watches, without sample-rate notification.
                events = [replace(event, observation=rate_notice)
                          if event.kind == EventKind.RATE_EXCEEDED else event for event in events]
                if not any(event.kind == EventKind.RATE_EXCEEDED for event in events):
                    events.append(replace(self._event(EventKind.OBSERVATION_WARNING, reading,
                                                     "confirmed physical-rate episode; reported only"),
                                          sample_evidence=True, observation=rate_notice))
        return events

    def _event(self, kind: str, reading: Reading, detail: str) -> HardwareEvent:
        return HardwareEvent(
            kind=kind,
            device_id=self._context.device_id,
            channel_id=self._channel.channel_id,
            quantity=self._channel.quantity,
            detail=detail,
            value=reading.value,
            unit=reading.unit or self._channel.unit,
            observed_at=reading.observed_at,
            sample_evidence=kind in {EventKind.THRESHOLD_EXCEEDED, EventKind.RATE_EXCEEDED, EventKind.QUALITY_DEGRADED},
        )


def _stale_observation(detector: HardwareEventDetector, alert_policy: Any) -> tuple[HardwareEvent, ...]:
    events = detector.check_stale()
    if alert_policy is not None:
        try:
            alert_policy.evaluate_sample(detector._context.device_id, detector._channel.channel_id,
                {EventKind.STALE: detector.safety_evidence.get(EventKind.STALE),
                 EventKind.THRESHOLD_EXCEEDED: None, EventKind.RATE_EXCEEDED: None,
                 EventKind.QUALITY_DEGRADED: None})
        except Exception:  # noqa: BLE001 - a policy failure cannot stop acquisition
            logger.exception("Hardware silence safety evaluation failed")
    return events


def _observation_policy(registry: Any, context: HardwareContext, channel: Channel) -> ObservationPolicy:
    resolver = getattr(registry, "observation_policy", None)
    if callable(resolver):
        return resolver(context.device_id, channel)
    settings = getattr(registry, "settings", None)
    return ObservationSettings.parse(getattr(settings, "hardware_observation", "{}")).resolve(context.device_id, channel)


def _observe_sample(detector: HardwareEventDetector, reading: Reading, ring: ReadingRing,
                    registry: Any, alert_policy: Any, *, lost: int) -> tuple[HardwareEvent, ...]:
    note = getattr(registry, "note_observation", None)
    if callable(note):
        note(reading)
    pending = getattr(registry, "pending_write_expectations", {})
    pending_motion = isinstance(pending, dict) and any(d == reading.device_id and v for (d, _), v in pending.items())
    evidence = getattr(registry, "observation_fault_free", None)
    fault_free = bool(evidence(reading.device_id)) if callable(evidence) else False
    events = detector.observe(reading, lost=lost, ring=ring, pending_motion=pending_motion, fault_free=fault_free)
    if alert_policy is not None:
        try:
            alert_policy.evaluate_sample(reading.device_id, reading.channel_id, detector.safety_evidence)
        except Exception:  # noqa: BLE001 - report a policy failure without stopping acquisition
            logger.exception("Hardware sample safety evaluation failed")
    return events


class HardwareStreamSource:
    """Samples one channel on a schedule, emitting derived events.

    Satisfies ``ActiveSignalSource`` structurally -- ``source_id`` / ``channel_id`` /
    ``start(emit)`` / ``stop()`` -- but the sink receives a ``HardwareEvent``, not an
    ``InteractionSignal``. Moving this under ``ActiveSourceManager`` therefore needs an
    adapter for that one type, not just a registration: the manager's queue is typed for
    interaction signals. Stated here because "can be handed over unchanged" was true of
    the protocol and false of the payload, and that gap is the kind a reader would only
    discover after wiring it.
    """

    def __init__(
        self,
        registry: Any,
        context: HardwareContext,
        channel: Channel,
        *,
        ring_capacity: int = DEFAULT_RING_CAPACITY,
        event_sink: EventSink | None = None,
        reading_store: Any = None,
        alert_policy: Any = None,
        degradation_policy: DegradationPolicy | None = None,
    ) -> None:
        self._registry = registry
        self._context = context
        self._channel = channel
        self.ring = ReadingRing(ring_capacity)
        self._detector = HardwareEventDetector(context, channel, _observation_policy(registry, context, channel))
        self._event_sink = event_sink
        self._store = reading_store
        self._alert_policy = alert_policy
        self._degradation_policy = degradation_policy
        self._degradation_coordinator = DegradationCoordinator(registry, context)
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._last_emitted: dict[str, float] = {}
        self._paced_out = 0
        self._samples = 0
        self._skipped_slots = 0
        self._started_monotonic: float | None = None
        self._comm_loss_start: float | None = None
        self._degradation_triggered: bool = False
        self._sensor_loss_streak: int = 0
        self._sensor_loss_since: float | None = None
        self._sensor_loss_triggered: bool = False
        self._last_degradation_outcome: Any = None

    @property
    def observation_detectors(self) -> tuple[HardwareEventDetector, ...]:
        return (self._detector,)

    @property
    def source_id(self) -> str:
        return f"hw:{self._context.device_id}:{self._channel.channel_id}"

    @property
    def channel_id(self) -> str:
        """Signal channel used for config-level gating of the whole device."""
        return f"hw.{self._context.device_id}"

    async def start(self, emit: Any) -> None:
        """Begin sampling. Returns promptly; the loop runs as an internal task.

        Returning immediately is a protocol requirement, not a style choice: the
        manager starts every source in sequence, so a source that sampled inline would
        stall the ones after it.
        """
        if self._task is not None:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run(emit), name=self.source_id)

    async def stop(self) -> None:
        """Stop sampling. Idempotent, and must not raise."""
        self._stopping.set()
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception as exc:  # noqa: BLE001 - teardown must not propagate
            logger.warning("Hardware stream %s stop raised: %s", self.source_id, exc)

    async def _run(self, emit: Any) -> None:
        interval = 1.0 / self._channel.sample_rate_hz if self._channel.sample_rate_hz > 0 else 1.0
        consecutive_failures = 0
        self._started_monotonic = time.monotonic()
        # Deadline-based rather than sleep-based. A fixed sleep after each read adds
        # the read's own duration to every period, so a channel declared at 10 Hz
        # runs slower than 10 Hz -- silently, because the window record counts the
        # samples it actually got and nothing compares that against the declaration.
        next_at = self._started_monotonic
        while not self._stopping.is_set():
            try:
                async with self._registry.device_io(self._context.device_id):
                    transport = await self._registry.transport(self._context.device_id)
                    reading = await transport.read(self._channel.channel_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one device must not stop the rest
                consecutive_failures += 1
                # Logged once per streak rather than per sample: a disconnected device
                # would otherwise produce log lines at the sampling rate, burying the
                # first and most useful one.
                if consecutive_failures == 1:
                    self._detector.invalidate_observation()
                    logger.warning(
                        "Hardware stream %s read failed: %s", self.source_id, exc, exc_info=True
                    )
                    # Mark the start of a communication loss window.
                    self._comm_loss_start = time.monotonic()
                self._dispatch(_stale_observation(self._detector, self._alert_policy), emit)
                # DegradationPolicy: comm_loss enforcement.
                await self._maybe_trigger_comm_loss()
                await self._sleep(min(interval * (2**consecutive_failures), 30.0))
                # Backoff deliberately abandons the old cadence; resuming the prior
                # deadline would burst to "catch up" on a device that just recovered.
                next_at = time.monotonic()
                continue

            # Recovery: clear comm-loss state only after a healthy read.
            if consecutive_failures > 0 or self._comm_loss_start is not None:
                self._comm_loss_start = None
                self._degradation_triggered = False
                self._registry.clear_degradation(
                    self._context.device_id, recovery="healthy_stream_read"
                )
            consecutive_failures = 0
            self._samples += 1
            lost = self.ring.record(reading)
            self._dispatch(_observe_sample(self._detector, reading, self.ring, self._registry,
                                           self._alert_policy, lost=lost), emit)
            await self._persist(reading, lost=lost)
            # Sensor quality tracking for degradation policy.
            await self._track_sensor_quality(reading)

            next_at += interval
            delay = next_at - time.monotonic()
            if delay < 0:
                # Behind schedule. Skip the missed slots instead of firing them back
                # to back: a burst would exceed the declared rate the envelope was
                # written against, and the count makes the shortfall observable
                # rather than leaving it to be inferred from a thin series.
                missed = int(-delay // interval) + 1
                self._skipped_slots += missed
                next_at += missed * interval
                delay = max(0.0, next_at - time.monotonic())
            await self._sleep(delay)

    async def _maybe_trigger_comm_loss(self) -> None:
        """Evaluate DegradationPolicy.comm_loss_policy during a failure streak.

        Triggers at most once per contiguous loss streak; the flag is cleared on
        the next successful read.  No-op when no policy is declared or when the
        loss window has not yet exceeded ``max_comm_loss_s``.
        """
        policy = self._degradation_policy
        if policy is None or self._degradation_triggered:
            return
        if self._comm_loss_start is None:
            return
        elapsed = time.monotonic() - self._comm_loss_start
        if elapsed <= policy.max_comm_loss_s:
            return
        self._degradation_triggered = True
        await self._execute_degradation(policy.comm_loss_policy, reason="comm_loss")

    async def _track_sensor_quality(self, reading: Reading) -> None:
        """Evaluate DegradationPolicy.sensor_loss_policy on quality streaks.

        Non-OK readings (STALE/SATURATED/SUSPECT) accumulate a streak; a run of
        three that has also lasted ``sensor_loss_debounce_s`` triggers the declared
        ``sensor_loss_policy`` once.  An OK reading resets the streak and the
        triggered flag so a future loss can fire.
        """
        policy = self._degradation_policy
        if reading.quality == Quality.OK.value:
            self._sensor_loss_streak = 0
            self._sensor_loss_since = None
            if self._sensor_loss_triggered:
                self._registry.clear_degradation(
                    self._context.device_id, recovery="healthy_sensor_read"
                )
            self._sensor_loss_triggered = False
            return
        self._sensor_loss_streak += 1
        now = time.monotonic()
        if self._sensor_loss_since is None:
            self._sensor_loss_since = now
        if policy is None or self._sensor_loss_triggered:
            return
        # Three consecutive degraded samples mirrors the detector's own threshold
        # for QUALITY_DEGRADED: a lone suspect sample is noise, a run of them is
        # a fault worth acting on.  The duration floor keeps a fast channel from
        # reaching three samples inside one transient.
        if self._sensor_loss_streak < 3:
            return
        if now - self._sensor_loss_since < sensor_loss_debounce_s(self._registry):
            return
        self._sensor_loss_triggered = True
        await self._execute_degradation(policy.sensor_loss_policy, reason="sensor_loss")

    async def _execute_degradation(self, action: str, *, reason: str) -> None:
        """Delegate every transition to the shared fail-closed coordinator."""
        outcome = await self._degradation_coordinator.execute(
            action,
            reason=reason,
            failed_channel_id=self._channel.channel_id,
            sensor_switcher=self._switch_sensor,
        )
        self._last_degradation_outcome = outcome
        if outcome.completed and outcome.action == "switch_sensor":
            self._degradation_triggered = False
            self._sensor_loss_triggered = False

    async def _switch_sensor(self, target_channel_id: str) -> bool:
        """Verify and atomically switch this source to a declared fallback."""
        target = self._context.channel(target_channel_id)
        if target is None or not target.is_readable:
            return False
        try:
            async with self._registry.device_io(self._context.device_id):
                transport = await self._registry.transport(self._context.device_id)
                reading = await transport.read(target_channel_id)
        except Exception:  # noqa: BLE001 - coordinator records the fail-closed outcome
            return False
        if reading.quality != Quality.OK.value:
            return False
        self._channel = target
        self._detector = HardwareEventDetector(self._context, target,
                                               _observation_policy(self._registry, self._context, target))
        return True

    @property
    def degradation_triggered(self) -> bool:
        """Return True while a declared degradation action is in effect.

        Callers that gate commands (e.g. ``hold_position``) can read this to
        decide whether to issue new writes without blocking the sampling loop.
        A withheld (report-only) trigger is not in effect and blocks nothing.
        """
        if not degradation_actions_enabled(self._registry):
            return False
        return self._degradation_triggered or self._sensor_loss_triggered

    @property
    def last_degradation_outcome(self) -> Any | None:
        """Return the most recent governed degradation transition."""
        return self._last_degradation_outcome

    async def _persist(self, reading: Reading, *, lost: int) -> None:
        """Buffer one sample and, when a window closes, write it off the sampling path.

        Buffering stays on the loop because it only mutates dictionaries. The write
        does not: appending a file and opening DuckDB inside the sampling loop stalls
        it for the duration of that I/O, which on a 10 Hz channel is a whole period
        or more. Draining first and writing in a worker keeps buffer mutation
        single-threaded while the blocking part happens elsewhere.
        """
        store = self._store
        if store is None:
            return
        try:
            store.record(reading, dropped=lost)
            if not store.due_for_flush():
                return
            batches = store.drain()
            if batches:
                await asyncio.to_thread(store.write_batches, batches)
        except Exception as exc:  # noqa: BLE001 - persistence must not stop sampling
            logger.warning(
                "Hardware reading persistence failed for %s: %s",
                self.source_id,
                exc,
                exc_info=True,
            )

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    def _dispatch(self, events: Iterable[HardwareEvent], emit: Any) -> None:
        """Hand events to the sink, signal pipeline, and alert policy, paced per kind.

        ``emit`` receives the ``HardwareEvent`` itself rather than a pre-flattened
        signal, because the consumer decides the representation: the event family,
        the value and the unit are all needed downstream, and collapsing them to a
        detail string here would force every consumer to parse it back out.

        Sample-level safety evidence is evaluated before dispatch. Other hard events
        are evaluated before presentation pacing; suppression cannot suppress a guard.
        The policy dispatches asynchronous tasks and never blocks sampling.
        """
        for event in events:
            if self._alert_policy is not None and not event.sample_evidence:
                try:
                    self._alert_policy.evaluate(event)
                except Exception:  # noqa: BLE001 - policy must not stop sampling
                    logger.exception("Hardware alert policy raised")
            if not self._admit(event):
                continue
            if self._event_sink is not None:
                try:
                    self._event_sink(event)
                except Exception as exc:  # noqa: BLE001 - a sink must not stop sampling
                    logger.warning("Hardware event sink raised: %s", exc, exc_info=True)
            if emit is not None:
                try:
                    emit(event)
                except Exception as exc:  # noqa: BLE001 - as above
                    logger.warning("Hardware event emit raised: %s", exc, exc_info=True)

    def _admit(self, event: HardwareEvent) -> bool:
        """Return whether this event clears the per-kind-per-channel rate floor.

        Keyed by ``kind:device_id.channel_id`` so that:

        - A paced ``rate_exceeded`` on one channel can never hide a first-time
          ``threshold_exceeded`` on a *different* channel -- suppressing an
          observation from another channel would trade one flood for one blind
          spot.
        - The same kind on the *same* channel is still suppressed for the
          ``MIN_EVENT_INTERVAL_S`` floor, preventing level-triggered floods.
        """
        if event.observation.get("phase") in {"entry", "recovery", "reminder", "unknown"}:
            return True  # episode transitions must survive notification pacing
        now = time.monotonic()
        key = f"{event.kind}:{event.device_id}.{event.channel_id}"
        previous = self._last_emitted.get(key)
        if previous is not None and now - previous < MIN_EVENT_INTERVAL_S:
            self._paced_out += 1
            return False
        self._last_emitted[key] = now
        return True

    @property
    def health(self) -> dict[str, Any]:
        """Return sampling health for disclosure and for the board.

        ``observed_hz`` against ``declared_hz`` is the only way a cadence shortfall
        becomes visible: the stored series looks entirely normal when a channel runs
        at two thirds of its declared rate.
        """
        declared = float(self._channel.sample_rate_hz or 0.0)
        started = self._started_monotonic
        elapsed = (time.monotonic() - started) if started is not None else 0.0
        observed = (self._samples / elapsed) if elapsed > 0 else 0.0
        return {
            "source_id": self.source_id,
            "device_id": self._context.device_id,
            "channel_id": self._channel.channel_id,
            "declared_hz": declared,
            "observed_hz": observed,
            "rate_ratio": (observed / declared) if declared > 0 else 0.0,
            "samples": self._samples,
            "skipped_slots": self._skipped_slots,
            "dropped": self.ring.dropped,
            "events_paced_out": self._paced_out,
        }


class BatchStreamCoordinator:
    """Coordinates batch sampling for all streaming channels of one device.

    When the transport satisfies ``BatchTransport``, one coordinator replaces
    N individual ``HardwareStreamSource`` tasks with a single sampling loop
    that calls ``read_batch`` once per period.  The readings are then
    distributed to the per-channel rings and detectors, preserving the
    per-channel event model while eliminating per-read lock contention.

    Falls back to individual sources when the transport does not support
    batch -- the coordinator is an optimisation, not a requirement.
    """

    def __init__(
        self,
        registry: Any,
        context: HardwareContext,
        channels: tuple[Channel, ...],
        *,
        ring_capacity: int = DEFAULT_RING_CAPACITY,
        event_sink: EventSink | None = None,
        reading_store: Any = None,
        alert_policy: Any = None,
        degradation_policy: DegradationPolicy | None = None,
    ) -> None:
        self._registry = registry
        self._context = context
        self._channels = channels
        self._channel_ids = tuple(ch.channel_id for ch in channels)
        self._rings: dict[str, ReadingRing] = {
            ch.channel_id: ReadingRing(ring_capacity) for ch in channels
        }
        self._detectors: dict[str, HardwareEventDetector] = {
            ch.channel_id: HardwareEventDetector(context, ch, _observation_policy(registry, context, ch))
            for ch in channels
        }
        self._event_sink = event_sink
        self._store = reading_store
        self._alert_policy = alert_policy
        self._degradation_policy = degradation_policy
        self._degradation_coordinator = DegradationCoordinator(registry, context)
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._last_emitted: dict[str, float] = {}
        self._paced_out = 0
        self._samples = 0
        self._skipped_slots = 0
        self._started_monotonic: float | None = None
        self._comm_loss_start: float | None = None
        self._degradation_triggered: bool = False
        # Streaks are per channel: one batch carries one sample of every channel,
        # so a single summed counter would read one stale poll of three channels
        # as "three consecutive degraded samples".
        self._sensor_loss_streaks: dict[str, int] = {}
        self._sensor_loss_since: dict[str, float] = {}
        self._sensor_loss_triggered: bool = False
        self._failed_channel_id: str = ""
        self._last_degradation_outcome: Any = None

    @property
    def observation_detectors(self) -> tuple[HardwareEventDetector, ...]:
        return tuple(self._detectors.values())

    @property
    def source_id(self) -> str:
        return f"hw.batch:{self._context.device_id}"

    async def start(self, emit: Any) -> None:
        """Begin batch sampling. Returns promptly; the loop runs as an internal task."""
        if self._task is not None:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run(emit), name=self.source_id)

    async def stop(self) -> None:
        """Stop batch sampling. Idempotent."""
        self._stopping.set()
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception as exc:  # noqa: BLE001 - teardown must not propagate
            logger.warning("Batch stream %s stop raised: %s", self.source_id, exc)

    async def _run(self, emit: Any) -> None:
        """Core batch sampling loop.

        Uses deadline-based scheduling identical to HardwareStreamSource._run(),
        but calls read_batch() once to get all channels simultaneously.
        The batch readings are then split and fed to per-channel rings,
        detectors, persistence and event dispatch.

        The sampling rate is the maximum declared rate across all channels.
        """
        max_rate = max(
            (ch.sample_rate_hz for ch in self._channels if ch.sample_rate_hz > 0), default=0.0
        )
        interval = 1.0 / max_rate if max_rate > 0 else 1.0
        consecutive_failures = 0
        self._started_monotonic = time.monotonic()
        next_at = self._started_monotonic
        while not self._stopping.is_set():
            try:
                batch = await self._registry.read_batch(self._context.device_id, self._channel_ids)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one device must not stop the rest
                consecutive_failures += 1
                if consecutive_failures == 1:
                    for detector in self._detectors.values():
                        detector.invalidate_observation()
                    logger.warning(
                        "Batch stream %s read failed: %s", self.source_id, exc, exc_info=True
                    )
                    self._comm_loss_start = time.monotonic()
                # Check staleness on all channels before an action can await I/O.
                for detector in self._detectors.values():
                    self._dispatch(_stale_observation(detector, self._alert_policy), emit)
                # DegradationPolicy: comm_loss enforcement.
                await self._maybe_trigger_comm_loss()
                await self._sleep(min(interval * (2**consecutive_failures), 30.0))
                next_at = time.monotonic()
                continue

            # Recovery: clear comm-loss state.
            if consecutive_failures > 0 or self._comm_loss_start is not None:
                self._comm_loss_start = None
                self._degradation_triggered = False
                self._registry.clear_degradation(
                    self._context.device_id, recovery="healthy_batch_read"
                )
            consecutive_failures = 0
            self._samples += 1

            # Judge every channel before any persistence await can delay a hard guard.
            persisted = []
            for reading in batch.readings:
                ch_id = reading.channel_id
                ring = self._rings.get(ch_id)
                if ring is None:
                    continue
                lost = ring.record(reading)
                detector = self._detectors.get(ch_id)
                if detector is not None:
                    self._dispatch(_observe_sample(detector, reading, ring, self._registry,
                                                   self._alert_policy, lost=lost), emit)
                persisted.append((reading, lost))
            received = {reading.channel_id for reading in batch.readings}
            for ch_id, detector in self._detectors.items():
                if ch_id not in received:
                    self._dispatch(_stale_observation(detector, self._alert_policy), emit)
            for reading, lost in persisted:
                await self._persist(reading, lost=lost)
                await self._track_sensor_quality(reading)

            next_at += interval
            delay = next_at - time.monotonic()
            if delay < 0:
                missed = int(-delay // interval) + 1
                self._skipped_slots += missed
                next_at += missed * interval
                delay = max(0.0, next_at - time.monotonic())
            await self._sleep(delay)

    async def _maybe_trigger_comm_loss(self) -> None:
        """Evaluate DegradationPolicy.comm_loss_policy during a failure streak."""
        policy = self._degradation_policy
        if policy is None or self._degradation_triggered:
            return
        if self._comm_loss_start is None:
            return
        elapsed = time.monotonic() - self._comm_loss_start
        if elapsed <= policy.max_comm_loss_s:
            return
        self._degradation_triggered = True
        await self._execute_degradation(policy.comm_loss_policy, reason="comm_loss")

    async def _track_sensor_quality(self, reading: Reading) -> None:
        """Evaluate DegradationPolicy.sensor_loss_policy on per-channel quality streaks.

        A channel must be degraded in three consecutive batches spanning at least
        ``sensor_loss_debounce_s`` -- the same rule :class:`HardwareStreamSource`
        applies to its single channel. The latch clears only once every channel
        reads OK again, so a healthy sibling in the same batch cannot cancel a
        loss that is still in progress.
        """
        policy = self._degradation_policy
        channel_id = reading.channel_id
        if reading.quality == Quality.OK.value:
            self._sensor_loss_streaks[channel_id] = 0
            self._sensor_loss_since.pop(channel_id, None)
            if self._sensor_loss_triggered and not any(self._sensor_loss_streaks.values()):
                self._registry.clear_degradation(
                    self._context.device_id, recovery="healthy_batch_sensor_read"
                )
                self._sensor_loss_triggered = False
            return
        streak = self._sensor_loss_streaks.get(channel_id, 0) + 1
        self._sensor_loss_streaks[channel_id] = streak
        now = time.monotonic()
        since = self._sensor_loss_since.setdefault(channel_id, now)
        if policy is None or self._sensor_loss_triggered:
            return
        if streak < 3 or now - since < sensor_loss_debounce_s(self._registry):
            return
        self._sensor_loss_triggered = True
        self._failed_channel_id = channel_id
        await self._execute_degradation(policy.sensor_loss_policy, reason="sensor_loss")

    async def _execute_degradation(self, action: str, *, reason: str) -> None:
        """Delegate batch-source transitions to the shared coordinator."""
        failed_channel_id = self._failed_channel_id or (
            self._channel_ids[0] if self._channel_ids else ""
        )
        outcome = await self._degradation_coordinator.execute(
            action,
            reason=reason,
            failed_channel_id=failed_channel_id,
            sensor_switcher=self._switch_sensor,
        )
        self._last_degradation_outcome = outcome
        if outcome.completed and outcome.action == "switch_sensor":
            self._degradation_triggered = False
            self._sensor_loss_triggered = False
            self._sensor_loss_streaks.pop(failed_channel_id, None)
        self._failed_channel_id = ""

    async def _switch_sensor(self, target_channel_id: str) -> bool:
        """Replace one failed batch input only after healthy fallback readback."""
        target = self._context.channel(target_channel_id)
        if target is None or not target.is_readable:
            return False
        try:
            async with self._registry.device_io(self._context.device_id):
                transport = await self._registry.transport(self._context.device_id)
                reading = await transport.read(target_channel_id)
        except Exception:  # noqa: BLE001 - coordinator performs the fail-closed fallback
            return False
        if reading.quality != Quality.OK.value:
            return False
        failed = self._failed_channel_id or next(
            (channel_id for channel_id in self._channel_ids if channel_id != target_channel_id),
            "",
        )
        updated = tuple(
            target if channel.channel_id == failed else channel for channel in self._channels
        )
        self._channels = updated
        self._channel_ids = tuple(dict.fromkeys(channel.channel_id for channel in updated))
        self._rings.setdefault(target_channel_id, ReadingRing(DEFAULT_RING_CAPACITY))
        self._detectors[target_channel_id] = HardwareEventDetector(
            self._context, target, _observation_policy(self._registry, self._context, target))
        return True

    @property
    def degradation_triggered(self) -> bool:
        """Return True while a declared degradation action is in effect."""
        if not degradation_actions_enabled(self._registry):
            return False
        return self._degradation_triggered or self._sensor_loss_triggered

    @property
    def last_degradation_outcome(self) -> Any | None:
        """Return the most recent governed degradation transition."""
        return self._last_degradation_outcome

    async def _persist(self, reading: Reading, *, lost: int) -> None:
        """Buffer one sample for persistence, mirroring HardwareStreamSource._persist."""
        store = self._store
        if store is None:
            return
        try:
            store.record(reading, dropped=lost)
            if not store.due_for_flush():
                return
            batches = store.drain()
            if batches:
                await asyncio.to_thread(store.write_batches, batches)
        except Exception as exc:  # noqa: BLE001 - persistence must not stop sampling
            logger.warning(
                "Hardware reading persistence failed for %s: %s",
                self.source_id,
                exc,
                exc_info=True,
            )

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    def _dispatch(self, events: Iterable[HardwareEvent], emit: Any) -> None:
        """Hand events to the sink, signal pipeline, and alert policy, paced per kind."""
        for event in events:
            if self._alert_policy is not None and not event.sample_evidence:
                try:
                    self._alert_policy.evaluate(event)
                except Exception:  # noqa: BLE001 - policy must not stop sampling
                    logger.exception("Hardware alert policy raised")
            if not self._admit(event):
                continue
            if self._event_sink is not None:
                try:
                    self._event_sink(event)
                except Exception as exc:  # noqa: BLE001 - a sink must not stop sampling
                    logger.warning("Hardware event sink raised: %s", exc, exc_info=True)
            if emit is not None:
                try:
                    emit(event)
                except Exception as exc:  # noqa: BLE001 - as above
                    logger.warning("Hardware event emit raised: %s", exc, exc_info=True)

    def _admit(self, event: HardwareEvent) -> bool:
        """Return whether this event clears the per-kind-per-channel rate floor."""
        if event.observation.get("phase") in {"entry", "recovery", "reminder", "unknown"}:
            return True  # episode transitions must survive notification pacing
        now = time.monotonic()
        key = f"{event.kind}:{event.device_id}.{event.channel_id}"
        previous = self._last_emitted.get(key)
        if previous is not None and now - previous < MIN_EVENT_INTERVAL_S:
            self._paced_out += 1
            return False
        self._last_emitted[key] = now
        return True

    def ring(self, channel_id: str) -> ReadingRing | None:
        """Return the per-channel ring, or None for an unknown channel."""
        return self._rings.get(channel_id)

    @property
    def health(self) -> dict[str, Any]:
        """Return batch sampling health metrics."""
        max_rate = max(
            (ch.sample_rate_hz for ch in self._channels if ch.sample_rate_hz > 0), default=0.0
        )
        started = self._started_monotonic
        elapsed = (time.monotonic() - started) if started is not None else 0.0
        observed = (self._samples / elapsed) if elapsed > 0 else 0.0
        total_dropped = sum(r.dropped for r in self._rings.values())
        return {
            "source_id": self.source_id,
            "device_id": self._context.device_id,
            "channels": len(self._channels),
            "declared_hz": float(max_rate),
            "observed_hz": observed,
            "rate_ratio": (observed / max_rate) if max_rate > 0 else 0.0,
            "samples": self._samples,
            "skipped_slots": self._skipped_slots,
            "dropped": total_dropped,
            "events_paced_out": self._paced_out,
        }


def build_batch_coordinators(
    registry: Any,
    *,
    ring_capacity: int = DEFAULT_RING_CAPACITY,
    event_sink: EventSink | None = None,
    reading_store: Any = None,
    alert_policy: Any = None,
) -> tuple[BatchStreamCoordinator, ...]:
    """Return one coordinator per device whose transport supports BatchTransport.

    Devices whose transport does not satisfy ``BatchTransport`` are skipped;
    their channels are handled by individual ``HardwareStreamSource`` instances
    from ``build_stream_sources``.
    """
    from leapflow.hardware.transport import BatchTransport

    coordinators: list[BatchStreamCoordinator] = []
    for context in registry.contexts():
        channels = tuple(
            ch for ch in context.streaming_channels if ch.is_readable and not ch.is_media
        )
        if not channels:
            continue
        # Check whether the transport declaration supports batch.
        # We inspect the transport kind; the actual isinstance check against the
        # live transport is deferred to read_batch at runtime, but the coordinator
        # is only created for devices whose transport advertises BatchTransport.
        transport_kind = context.transport.kind
        try:
            from leapflow.hardware.transports import build_transport

            proto = build_transport(transport_kind, context.transport.config)
            if not isinstance(proto, BatchTransport):
                continue
        except Exception:  # noqa: BLE001 - skip devices we cannot probe
            continue
        coordinators.append(
            BatchStreamCoordinator(
                registry,
                context,
                channels,
                ring_capacity=ring_capacity,
                event_sink=event_sink,
                reading_store=reading_store,
                alert_policy=alert_policy,
                degradation_policy=context.degradation,
            )
        )
    return tuple(coordinators)


def build_stream_sources(
    registry: Any,
    *,
    ring_capacity: int = DEFAULT_RING_CAPACITY,
    event_sink: EventSink | None = None,
    reading_store: Any = None,
    alert_policy: Any = None,
    batch_device_ids: frozenset[str] = frozenset(),
) -> tuple[HardwareStreamSource, ...]:
    """Return one source per streaming channel across all admitted devices.

    ``sample_rate_hz > 0`` is the only thing consulted. No device type is enumerated
    anywhere, which is what lets a declaration add a sensor without touching code.

    Media channels are excluded even when they declare a rate. Admission already
    clears the rate on them; this is the second half of that guard, kept here
    because this is the loop that would otherwise push a few hundred kilobytes per
    sample into the raw NDJSON segment and try to average frames into a downsample
    window. Previews are pulled on demand by whoever is watching, never sampled.

    Channels belonging to devices in *batch_device_ids* are skipped: they are
    handled by a ``BatchStreamCoordinator`` instead, and building individual
    sources for them would double-sample the device.
    """
    sources: list[HardwareStreamSource] = []
    for context in registry.contexts():
        if context.device_id in batch_device_ids:
            continue
        for channel in context.streaming_channels:
            if not channel.is_readable or channel.is_media:
                continue
            sources.append(
                HardwareStreamSource(
                    registry,
                    context,
                    channel,
                    ring_capacity=ring_capacity,
                    event_sink=event_sink,
                    reading_store=reading_store,
                    alert_policy=alert_policy,
                    degradation_policy=context.degradation,
                )
            )
    return tuple(sources)


def _trend(values: list[float]) -> str:
    """Classify a series as rising, falling, or flat by comparing its halves."""
    if len(values) < 4:
        return "flat"
    midpoint = len(values) // 2
    first = sum(values[:midpoint]) / midpoint
    second = sum(values[midpoint:]) / (len(values) - midpoint)
    spread = max(values) - min(values)
    if spread == 0:
        return "flat"
    delta = (second - first) / spread
    if delta > 0.1:
        return "rising"
    if delta < -0.1:
        return "falling"
    return "flat"


def _bounds(envelope: Any) -> str:
    low = "-inf" if envelope.min_value is None else f"{envelope.min_value:g}"
    high = "+inf" if envelope.max_value is None else f"{envelope.max_value:g}"
    return f"{low}..{high}"


__all__ = [
    "BatchStreamCoordinator",
    "DEFAULT_RING_CAPACITY",
    "MIN_EVENT_INTERVAL_S",
    "EventKind",
    "HardwareDeliberativeObserver",
    "HardwareEvent",
    "HardwareEventDetector",
    "HardwareStreamSource",
    "ReadingRing",
    "build_batch_coordinators",
    "build_stream_sources",
]
