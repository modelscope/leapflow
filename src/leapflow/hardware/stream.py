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

EventSink = Callable[["HardwareEvent"], None]
"""Receives derived events. Must be thread-safe and non-blocking."""


class EventKind:
    """Derived observations, each traceable to a declared envelope field."""

    THRESHOLD_EXCEEDED = "threshold_exceeded"
    RATE_EXCEEDED = "rate_exceeded"
    STALE = "stale"
    SAMPLE_LOSS = "sample_loss"
    QUALITY_DEGRADED = "quality_degraded"
    SETTLED = "settled"


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
    timestamp: float = 0.0

    @property
    def signal_type(self) -> str:
        """Return the interaction-signal type used when this crosses the boundary."""
        return "hw_event"

    def to_detail(self) -> str:
        """Return a compact one-line description for the signal pipeline."""
        where = f"{self.device_id}.{self.channel_id}"
        rendered = "" if self.value is None else f" value={self.value}{f' {self.unit}' if self.unit else ''}"
        return f"[{self.kind}] {where}{rendered}: {self.detail}"


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
        self._last_timestamp: float | None = None
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
        self._last_timestamp = reading.timestamp
        self._stale = False
        return tuple(events)

    def check_stale(self, *, now: float | None = None) -> tuple[HardwareEvent, ...]:
        """Return a staleness event when sampling has stopped.

        Silence is itself an observation: a channel declared at 10 Hz that has said
        nothing for a second has failed, and the absence of readings is the only way
        that failure shows up.
        """
        channel = self._channel
        if channel.sample_rate_hz <= 0 or self._last_timestamp is None or self._stale:
            return ()
        deadline = 2.0 / channel.sample_rate_hz
        elapsed = (now if now is not None else time.monotonic()) - self._last_timestamp
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
                timestamp=time.monotonic(),
            ),
        )

    def _numeric_events(self, reading: Reading, numeric: float) -> list[HardwareEvent]:
        events: list[HardwareEvent] = []
        envelope = self._channel.envelope

        inside = envelope.contains(numeric)
        if not inside and not self._breached:
            self._breached = True
            events.append(
                self._event(
                    EventKind.THRESHOLD_EXCEEDED,
                    reading,
                    f"left the declared range ({_bounds(envelope)})",
                )
            )
        elif inside and self._breached:
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
            and self._last_timestamp is not None
        ):
            elapsed = reading.timestamp - self._last_timestamp
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
            timestamp=reading.timestamp,
        )


class HardwareStreamSource:
    """Samples one channel on a schedule, emitting derived events.

    Implements ``ActiveSignalSource`` structurally: ``source_id`` / ``channel_id`` /
    ``start(emit)`` / ``stop()``. It is registered with the session's
    ``ActiveSourceManager`` like any other lifecycle-bearing source, which is what puts
    device observations on the same path as every other environment signal instead of
    inventing a parallel one.
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
    ) -> None:
        self._registry = registry
        self._context = context
        self._channel = channel
        self.ring = ReadingRing(ring_capacity)
        self._detector = HardwareEventDetector(context, channel)
        self._event_sink = event_sink
        self._store = reading_store
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

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
        while not self._stopping.is_set():
            try:
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
                continue

            consecutive_failures = 0
            lost = self.ring.record(reading)
            if self._store is not None:
                # Buffered here rather than in the ring, because the ring is a bounded
                # window for the current decision while the store is the durable record
                # that later analysis reads. Contained so a full disk cannot stop sampling.
                try:
                    self._store.record(reading, dropped=lost)
                    self._store.flush()
                except Exception as exc:  # noqa: BLE001 - persistence must not stop sampling
                    logger.warning(
                        "Hardware reading persistence failed for %s: %s",
                        self.source_id,
                        exc,
                        exc_info=True,
                    )
            self._dispatch(self._detector.observe(reading, lost=lost), emit)
            await self._sleep(interval)

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    def _dispatch(self, events: Iterable[HardwareEvent], emit: Any) -> None:
        """Hand events to the sink and to the interaction signal pipeline."""
        for event in events:
            if self._event_sink is not None:
                try:
                    self._event_sink(event)
                except Exception as exc:  # noqa: BLE001 - a sink must not stop sampling
                    logger.warning("Hardware event sink raised: %s", exc, exc_info=True)
            if emit is None:
                continue
            try:
                emit(_as_interaction_signal(event))
            except Exception as exc:  # noqa: BLE001 - as above
                logger.warning("Hardware event emit raised: %s", exc, exc_info=True)


def _as_interaction_signal(event: HardwareEvent) -> Any:
    """Convert an event into the perception layer's signal type.

    Imported lazily so ``leapflow.hardware`` stays importable without the perception
    subsystem, and so the domain model keeps no compile-time dependency on it.
    """
    from leapflow.perception.types import InteractionSignal

    return InteractionSignal(
        timestamp=event.timestamp or time.monotonic(),
        signal_type=event.signal_type,
        app=event.device_id,
        detail=event.to_detail(),
    )


def build_stream_sources(
    registry: Any,
    *,
    ring_capacity: int = DEFAULT_RING_CAPACITY,
    event_sink: EventSink | None = None,
    reading_store: Any = None,
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
    "EventKind",
    "HardwareEvent",
    "HardwareEventDetector",
    "HardwareStreamSource",
    "ReadingRing",
    "build_stream_sources",
]
