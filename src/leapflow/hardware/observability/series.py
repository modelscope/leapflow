"""Versioned contract for the physical-signal payload the board renders.

Separate from the code that fills it, because the two change for different
reasons: this file changes when the *shape* the renderer reads changes, and
``digest`` changes when the analysis does. Every bound is explicit and every
payload carries its version, so a renderer meeting an older or newer producer can
refuse instead of drawing something plausible from a shape it does not understand.

The clock is stated in the payload rather than assumed. Physical readings are
timestamped with two different clocks inside ``leapflow.hardware``, and only one
of them can be put on a time axis; a chart drawn from the other looks entirely
normal while being wrong by decades.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)

SERIES_SCHEMA_VERSION = 2
"""Payload shape version, read by the renderer before anything else.

Bumped to 2 when the per-window ``conformance`` grid left the wire in favour of the
``conformance_mix`` distribution that is actually rendered. The grid had no renderer
and no other consumer, so it was pure weight -- but it was *declared* weight, and a
consumer that went looking for it deserves a version it can refuse rather than a key
that silently vanished.
"""

WALL_CLOCK = "wall"
"""The only clock a point's ``x`` may carry. See ``Reading.observed_at``."""

MAX_SERIES = 8
"""Most channels charted at once.

A bench declares up to ``hardware.max_devices`` (16) devices with a few channels
each, which is far more than a person can read on one screen. When the limit
bites, the channels kept are the ones whose quality is worst -- a misbehaving
channel is more worth seeing than a healthy one.
"""

MAX_POINTS = 480
"""Most points per series: eight hours at the default 60s downsample interval.

Chosen for the overnight-run case the whole subsystem exists to serve. Longer
history stays in DuckDB; the board is an operational view, not an archive.
"""

MAX_PAYLOAD_BYTES = 262_144
"""Hard ceiling on one finding's payload.

The payload is persisted as JSON, pushed over a WebSocket, and held in a bounded
ring, so it cannot be unbounded. On overflow the series are **decimated, never
truncated**: dropping the tail hides the present and dropping the head hides the
baseline, and either makes the chart lie about what happened.

This is a ceiling on what any *one* payload contributes, and it is load-bearing well
beyond this module: findings are returned in batches over a single newline-delimited
JSON-RPC frame, so a payload that overruns it does not degrade one panel -- it
overruns the frame and takes the whole Board down with it. A field that is exempt
from this ceiling is a defect in the field, not a limit to raise.
"""


@dataclass(frozen=True)
class SeriesPoint:
    """One downsampled interval of one channel.

    Carries the interval's shape rather than a single value: ``lo``/``hi`` are what
    make an excursion visible after averaging, which is the reason the storage tier
    keeps them.
    """

    x: float
    """Wall-clock epoch seconds. Never monotonic -- see ``WALL_CLOCK``."""
    y: float | None = None
    lo: float | None = None
    hi: float | None = None
    n: int = 0
    drop: int = 0
    q: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "lo": self.lo,
            "hi": self.hi,
            "n": self.n,
            "drop": self.drop,
            "q": self.q,
        }


@dataclass(frozen=True)
class EnvelopeBand:
    """The declared limits, carried alongside the trace they constrain.

    Sent with every series so the chart can draw the measured value against the
    limit a human wrote down. This is the one view in which ``Envelope`` stops
    being a number in a YAML file and becomes something an operator can see.
    """

    declared: bool = False
    min_value: float | None = None
    max_value: float | None = None
    quantization: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared": self.declared,
            "min": self.min_value,
            "max": self.max_value,
            "quantization": self.quantization,
        }


@dataclass(frozen=True)
class ChannelSeries:
    """One channel's charted history."""

    id: str
    label: str
    unit: str = ""
    quantity: str = ""
    kind: str = "line"
    quality_worst: str = "ok"
    envelope: EnvelopeBand = field(default_factory=EnvelopeBand)
    points: tuple[SeriesPoint, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "unit": self.unit,
            "quantity": self.quantity,
            "kind": self.kind,
            "quality_worst": self.quality_worst,
            "envelope": self.envelope.to_dict(),
            "points": [point.to_dict() for point in self.points],
        }


@dataclass(frozen=True)
class HardwareDigest:
    """Everything the board draws, in one versioned payload.

    Nothing here is declared twice: each field is derived from the device
    declaration, the reading store, or the event ring. A value the board shows that
    is not derivable from those is a second source of truth waiting to disagree
    with the first.
    """

    generated_at: float
    devices: tuple[dict[str, Any], ...] = ()
    series: tuple[ChannelSeries, ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    conformance: tuple[dict[str, Any], ...] = ()
    sampling: tuple[dict[str, Any], ...] = ()
    outcomes: tuple[dict[str, Any], ...] = ()
    calibration: tuple[dict[str, Any], ...] = ()
    storage: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to show, so callers can emit no finding."""
        return not self.devices and not self.series and not self.events

    def to_payload(self) -> dict[str, Any]:
        """Return the wire form, decimated if it exceeds the byte ceiling.

        ``counts`` is precomputed because the template path resolver walks mapping
        keys and list indices only -- there is no ``.length``, so a template asking
        for one silently renders an empty value.

        Conformance ships as its distribution only. The per-window grid is an
        intermediate of the analysis, not part of the wire contract: no renderer draws
        it (see ``_distribution``) and nothing else reads it, while it cost one row per
        charted window and grew to four times the entire rest of the payload.
        """
        payload = {
            "schema_version": SERIES_SCHEMA_VERSION,
            "clock": WALL_CLOCK,
            "generated_at": self.generated_at,
            "counts": {
                "devices": len(self.devices),
                "series": len(self.series),
                "events": len(self.events),
                "outcomes": len(self.outcomes),
                "calibration": len(self.calibration),
                "conformance_windows": len(self.conformance),
                "media_channels": sum(
                    int(row.get("media", 0) or 0) for row in self.devices
                ),
                "privacy_gated": sum(
                    int(row.get("privacy_gated", 0) or 0) for row in self.devices
                ),
            },
            "devices": list(self.devices),
            "device_classes": _distribution(self.devices, "device_class"),
            "series": [item.to_dict() for item in self.series],
            "events": list(self.events),
            "conformance_mix": _distribution(self.conformance, "state"),
            "sampling": list(self.sampling),
            "outcomes": list(self.outcomes),
            "calibration": list(self.calibration),
            "storage": dict(self.storage),
        }
        return _fit(payload)


def clamp_series(items: Sequence[ChannelSeries]) -> tuple[ChannelSeries, ...]:
    """Return at most ``MAX_SERIES`` channels, worst quality first.

    Ordering by quality rather than by name or arrival: when the limit bites, the
    channel that gets dropped should be a healthy one. Dropping by name would hide
    exactly the channel somebody opened the board to look at.
    """
    ranked = sorted(items, key=lambda item: (-_quality_rank(item.quality_worst), item.id))
    return tuple(ranked[:MAX_SERIES])


def decimate(points: Sequence[SeriesPoint], limit: int = MAX_POINTS) -> tuple[SeriesPoint, ...]:
    """Thin *points* to at most *limit*, keeping the first and last.

    Even sampling rather than a window: the shape of the whole interval survives,
    where taking a window would silently answer a different question than the one
    the axis labels claim.
    """
    total = len(points)
    if limit <= 0:
        return ()
    if total <= limit:
        return tuple(points)
    if limit == 1:
        return (points[-1],)
    step = (total - 1) / (limit - 1)
    picked = [points[min(total - 1, round(index * step))] for index in range(limit)]
    return tuple(picked)


def _fit(payload: dict[str, Any]) -> dict[str, Any]:
    """Decimate series until the encoded payload fits ``MAX_PAYLOAD_BYTES``.

    Series are the only field allowed to be large enough to need this, and they are
    the only field it can reduce -- so an overrun that survives the loop means some
    *other* field is unbounded, which is a producer defect this function cannot fix.
    It says so at ``warning`` rather than returning quietly: the previous silent
    return meant a payload four times the ceiling was persisted, pushed, and batched
    into an RPC frame with no record anywhere that the ceiling had been passed.
    """
    limit = MAX_POINTS
    while limit >= 2:
        if _encoded_size(payload) <= MAX_PAYLOAD_BYTES:
            return payload
        limit //= 2
        payload["series"] = [
            {**series, "points": [p.to_dict() for p in decimate(_as_points(series["points"]), limit)]}
            for series in payload["series"]
        ]
    size = _encoded_size(payload)
    if size > MAX_PAYLOAD_BYTES:
        logger.warning(
            "hardware digest: payload is %d bytes after decimating series to the floor, "
            "over the %d-byte ceiling; a non-series field is unbounded",
            size,
            MAX_PAYLOAD_BYTES,
        )
    return payload


def _as_points(rows: Sequence[dict[str, Any]]) -> tuple[SeriesPoint, ...]:
    return tuple(
        SeriesPoint(
            x=float(row.get("x") or 0.0),
            y=row.get("y"),
            lo=row.get("lo"),
            hi=row.get("hi"),
            n=int(row.get("n") or 0),
            drop=int(row.get("drop") or 0),
            q=str(row.get("q") or "ok"),
        )
        for row in rows
    )


def _encoded_size(payload: dict[str, Any]) -> int:
    try:
        return len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        # Unserialisable content is a producer defect, not a size problem; report it
        # as over-budget so the caller decimates rather than shipping a payload the
        # finding store cannot persist.
        return MAX_PAYLOAD_BYTES + 1


_QUALITY_ORDER = ("ok", "suspect", "stale", "saturated")


def _distribution(rows: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Count rows by one field, in the ``{label, value}`` shape a bar chart reads.

    Conformance is charted as a distribution rather than a per-window grid: a
    heatmap is in the component catalog but has no frontend renderer, so a template
    asking for one gets a fallback card printing its own type name.
    """
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(key, "") or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return [{"label": label, "value": value} for label, value in sorted(counts.items())]


def _quality_rank(value: str) -> int:
    return _QUALITY_ORDER.index(value) if value in _QUALITY_ORDER else len(_QUALITY_ORDER)


__all__ = [
    "MAX_PAYLOAD_BYTES",
    "MAX_POINTS",
    "MAX_SERIES",
    "SERIES_SCHEMA_VERSION",
    "WALL_CLOCK",
    "ChannelSeries",
    "EnvelopeBand",
    "HardwareDigest",
    "SeriesPoint",
    "clamp_series",
    "decimate",
]
