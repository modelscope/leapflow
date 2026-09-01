"""Continuous sampling: raw readings in, derived events out.

The layering here is the whole point, and it is a boundary decision rather than an
optimisation. ``SignalBuffer`` holds 50 interaction signals and every signal drives
causal fusion; a single 10 Hz channel would flush that buffer in five seconds and
run fusion at sampling rate, destroying both subsystems at once.

So raw readings stay inside this module, in a bounded per-channel ring. What crosses
into the interaction signal pipeline is *events* -- a threshold crossed, a rate
exceeded, samples lost -- which occur at the rate at which something worth noticing
actually happens.

Every detection rule is derived from the channel's own ``Envelope``. Nothing new is
declared: the limits a human already wrote down for approval are the same limits
that make an observation interesting.

Two clocks are in play and they are not interchangeable. Intervals, slew rates and
staleness read ``Reading.monotonic_at``, because wall-clock jumps would fabricate
rates no device produced. Anything that leaves this module -- an event's timestamp,
a persisted window -- carries ``Reading.observed_at``, because every consumer
outside it (findings, audit, the board's time axis) is wall-clock.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Iterable, Iterator

from leapflow.hardware.context import Channel, HardwareContext, Quality, as_numeric
from leapflow.hardware.transport import Reading

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
        rendered = "" if self.value is None else f" value={self.value}{f' {self.unit}' if self.unit else ''}"
        return f"[{self.kind}] {where}{rendered}: {self.detail}"

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
            "source": self.source,
            "device_id": self.device_id,
            "channel_id": self.channel_id,
            "quantity": self.quantity,
            "detail": self.detail,
            "value": self.value,
            "unit": self.unit,
            "ts": self.observed_at,
            "_mono_ts": time.monotonic(),
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

    def record(self, reading: Reading) -> int:
        """Append *reading*, returning how many samples appear to have been lost."""
        lost = 0
        if self._expected_sequence is not None and reading.sequence > self._expected_sequence:
            lost = reading.sequence - self._expected_sequence
            self._dropped_sequences += lost
        self._expected_sequence = reading.sequence + 1
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


class HardwareEventDetector:
    """Turns a stream of readings into events, using only declared envelope data.

    Stateful per channel: staleness and rate need to remember what came before. Kept
    separate from the ring so the detection rules can be tested against a synthetic
    series without a transport in the picture.
    """

    def __init__(self, context: HardwareContext, channel: Channel) -> None:
        self._context = context
        self._channel = channel
        self._last_numeric: float | None = None
        self._last_monotonic: float | None = None
        self._degraded_streak = 0
        self._breached = False
        self._stale = False

    def observe(self, reading: Reading, *, lost: int = 0) -> tuple[HardwareEvent, ...]:
        """Return the events this reading produces."""
        events: list[HardwareEvent] = []
        channel = self._channel
        envelope = channel.envelope

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
            # Three in a row rather than one: a lone suspect sample is noise, a run of
            # them is a fault. Reporting every one would reproduce the sampling-rate
            # flood this layer exists to prevent.
            if self._degraded_streak == 3:
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
            events.extend(self._numeric_events(reading, numeric))

        self._last_numeric = numeric if numeric is not None else self._last_numeric
        self._last_monotonic = reading.monotonic_at
        self._stale = False
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
        if channel.sample_rate_hz <= 0 or self._last_monotonic is None or self._stale:
            return ()
        deadline = 2.0 / channel.sample_rate_hz
        elapsed = (now if now is not None else time.monotonic()) - self._last_monotonic
        if elapsed <= deadline:
            return ()
        self._stale = True
        return (
            HardwareEvent(
                kind=EventKind.STALE,
                device_id=self._context.device_id,
                channel_id=channel.channel_id,
                quantity=channel.quantity,
                detail=(
                    f"no sample for {elapsed:.2f}s on a {channel.sample_rate_hz:g} Hz channel"
                ),
                unit=channel.unit,
                observed_at=time.time(),
            ),
        )

    def _numeric_events(self, reading: Reading, numeric: float) -> list[HardwareEvent]:
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
        elif envelope.contains(numeric, margin=envelope.settle_margin):
            # Re-entering is worth one event too, so a watcher can see recovery
            # instead of inferring it from silence.
            self._breached = False
            events.append(
                self._event(
                    EventKind.SETTLED, reading, "returned to the declared range"
                )
            )

        if (
            envelope.max_rate is not None
            and self._last_numeric is not None
            and self._last_monotonic is not None
        ):
            elapsed = reading.monotonic_at - self._last_monotonic
            if elapsed > 0 and envelope.rate_exceeded(
                delta=numeric - self._last_numeric, elapsed_s=elapsed
            ):
                observed = abs(numeric - self._last_numeric) / elapsed
                events.append(
                    self._event(
                        EventKind.RATE_EXCEEDED,
                        reading,
                        f"changing at {observed:g}/s, above the declared "
                        f"{envelope.max_rate:g}/s",
                    )
                )
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
        )


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
    ) -> None:
        self._registry = registry
        self._context = context
        self._channel = channel
        self.ring = ReadingRing(ring_capacity)
        self._detector = HardwareEventDetector(context, channel)
        self._event_sink = event_sink
        self._store = reading_store
        self._alert_policy = alert_policy
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._last_emitted: dict[str, float] = {}
        self._paced_out = 0
        self._samples = 0
        self._skipped_slots = 0
        self._started_monotonic: float | None = None

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
                    logger.warning(
                        "Hardware stream %s read failed: %s", self.source_id, exc, exc_info=True
                    )
                self._dispatch(self._detector.check_stale(), emit)
                await self._sleep(min(interval * (2**consecutive_failures), 30.0))
                # Backoff deliberately abandons the old cadence; resuming the prior
                # deadline would burst to "catch up" on a device that just recovered.
                next_at = time.monotonic()
                continue

            consecutive_failures = 0
            self._samples += 1
            lost = self.ring.record(reading)
            await self._persist(reading, lost=lost)
            self._dispatch(self._detector.observe(reading, lost=lost), emit)

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

        Alert policy evaluation runs after the emit so the event is recorded (and
        visible on the board) before a response is attempted.  The policy itself
        dispatches asynchronous tasks so it never blocks the sampling loop.
        """
        for event in events:
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
            # Alert policy evaluation: after emit so the event is visible first.
            if self._alert_policy is not None:
                try:
                    self._alert_policy.evaluate(event)
                    # Reset consecutive counters on recovery so a new breach cycle
                    # starts fresh rather than carrying stale counts.
                    if event.kind == EventKind.SETTLED:
                        self._alert_policy.reset_channel(
                            event.device_id, event.channel_id
                        )
                except Exception as exc:  # noqa: BLE001 - policy must not stop sampling
                    logger.warning("Hardware alert policy raised: %s", exc, exc_info=True)

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


def build_stream_sources(
    registry: Any,
    *,
    ring_capacity: int = DEFAULT_RING_CAPACITY,
    event_sink: EventSink | None = None,
    reading_store: Any = None,
    alert_policy: Any = None,
) -> tuple[HardwareStreamSource, ...]:
    """Return one source per streaming channel across all admitted devices.

    ``sample_rate_hz > 0`` is the only thing consulted. No device type is enumerated
    anywhere, which is what lets a declaration add a sensor without touching code.
    """
    sources: list[HardwareStreamSource] = []
    for context in registry.contexts():
        for channel in context.streaming_channels:
            if not channel.is_readable:
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
    "DEFAULT_RING_CAPACITY",
    "MIN_EVENT_INTERVAL_S",
    "EventKind",
    "HardwareEvent",
    "HardwareEventDetector",
    "HardwareStreamSource",
    "ReadingRing",
    "build_stream_sources",
]
