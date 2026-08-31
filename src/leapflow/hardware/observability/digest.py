"""Derive the board payload from what the registry already knows.

Pure with respect to the registry: it reads, it never samples, writes, or opens a
device. That keeps the whole analysis testable against a fake registry, and it
keeps a board refresh from being able to touch hardware -- an observation surface
that can actuate is not an observation surface.

Nothing here is a new declaration. Devices come from the admitted contexts, the
traces from the reading store, the limits from each channel's ``Envelope``, the
cadence from the stream sources, the experience from the outcome recorder. A value
the board shows that is not derivable from those would be a second source of truth,
free to disagree with the one the gate enforces.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from leapflow.hardware.observability.series import (
    ChannelSeries,
    EnvelopeBand,
    HardwareDigest,
    SeriesPoint,
    clamp_series,
    decimate,
)

logger = logging.getLogger(__name__)

EVENT_LIMIT = 60
"""Events kept on the timeline. The registry's own ring holds 200."""

OUTCOME_LIMIT = 3
"""Recalled outcomes per writable channel, matching the tool-side disclosure."""

NEAR_FRACTION = 0.05
"""Share of the declared span within which a value counts as *near* the limit.

Approaching a limit and sitting inside one are different operational facts, and a
two-colour in/out view cannot express the difference -- which is the whole reason
somebody watches a trace rather than a boolean.
"""


def build_digest(registry: Any, *, now: float | None = None) -> HardwareDigest:
    """Return the board payload for every admitted device.

    Contained end to end: a registry that cannot answer one question yields a digest
    missing that section rather than no digest at all. A monitor cycle that raises
    would stop the watch, and losing the board is a worse outcome than losing one
    panel of it.
    """
    moment = now if now is not None else time.time()
    if registry is None:
        return HardwareDigest(generated_at=moment)

    contexts = _safely(registry.contexts, default=())
    series: list[ChannelSeries] = []
    conformance: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []

    for context in contexts:
        for channel in getattr(context, "channels", ()):
            if not getattr(channel, "is_readable", False):
                continue
            windows = _history(registry, context.device_id, channel.channel_id)
            if not windows:
                continue
            built = _series_for(context, channel, windows)
            series.append(built)
            conformance.extend(_conformance_for(built))
        outcomes.extend(_outcomes_for(registry, context))

    return HardwareDigest(
        generated_at=moment,
        devices=tuple(_device_row(registry, context) for context in contexts),
        series=clamp_series(series),
        events=_events(registry),
        conformance=tuple(conformance),
        sampling=_sampling(registry),
        outcomes=tuple(outcomes),
        storage=_storage(registry),
    )


# ── Devices ──


def _device_row(registry: Any, context: Any) -> dict[str, Any]:
    """Summarise one device without touching its transport.

    ``probe()`` is deliberately not called: it is an I/O round trip per device, and
    a board refresh must not become a reason the bus is busy.
    """
    channels = tuple(getattr(context, "channels", ()))
    writable = tuple(c for c in channels if getattr(c, "is_writable", False))
    provenance = getattr(context, "provenance", None)
    return {
        "device_id": context.device_id,
        "label": getattr(context, "display_name", "") or context.device_id,
        "location": getattr(context, "location", ""),
        "transport_kind": str(getattr(getattr(context, "transport", None), "kind", "") or ""),
        "verified": bool(getattr(provenance, "verified_by", "")),
        "channels": len(channels),
        "writable": len(writable),
        "streaming": sum(1 for c in channels if float(getattr(c, "sample_rate_hz", 0) or 0) > 0),
        "halt_supported": bool(getattr(context, "halt_supported", False)),
        "opened": context.device_id in _safely(registry.opened_devices, default=()),
    }


# ── Series ──


def _history(registry: Any, device_id: str, channel_id: str) -> tuple[dict[str, Any], ...]:
    try:
        return tuple(registry.channel_history(device_id, channel_id, limit=_HISTORY_LIMIT))
    except Exception as exc:  # noqa: BLE001 - one unreadable channel must not lose the rest
        logger.debug("hardware digest: no history for %s.%s: %s", device_id, channel_id, exc)
        return ()


_HISTORY_LIMIT = 2000
"""Rows requested per channel before decimation, so the shape survives thinning."""


def _series_for(context: Any, channel: Any, windows: Sequence[dict[str, Any]]) -> ChannelSeries:
    envelope = getattr(channel, "envelope", None)
    points = tuple(
        SeriesPoint(
            x=float(row.get("ended_at") or 0.0),
            y=_as_float(row.get("mean_value")),
            lo=_as_float(row.get("min_value")),
            hi=_as_float(row.get("max_value")),
            n=int(row.get("samples") or 0),
            drop=int(row.get("dropped") or 0),
            q=str(row.get("quality_worst") or "ok"),
        )
        for row in windows
    )
    return ChannelSeries(
        id=f"{context.device_id}.{channel.channel_id}",
        label=f"{getattr(context, 'display_name', '') or context.device_id} · {channel.channel_id}",
        unit=str(getattr(channel, "unit", "") or ""),
        quantity=str(getattr(channel, "quantity", "") or ""),
        quality_worst=_worst(point.q for point in points),
        envelope=EnvelopeBand(
            declared=bool(getattr(envelope, "declared", False)),
            min_value=_as_float(getattr(envelope, "min_value", None)),
            max_value=_as_float(getattr(envelope, "max_value", None)),
            quantization=_as_float(getattr(envelope, "quantization", None)),
        ),
        points=decimate(points),
    )


# ── Envelope conformance ──


def _conformance_for(series: ChannelSeries) -> list[dict[str, Any]]:
    """Classify each window against the declared band.

    Judged on the window's ``lo``/``hi`` rather than its mean, because an excursion
    that averages back inside the band still left it -- and the mean is precisely
    what hides that.
    """
    band = series.envelope
    if not band.declared or (band.min_value is None and band.max_value is None):
        return [
            {"channel_id": series.id, "window_x": point.x, "state": "unknown"}
            for point in series.points
        ]
    span = _span(band)
    margin = span * NEAR_FRACTION if span > 0 else 0.0
    rows: list[dict[str, Any]] = []
    for point in series.points:
        rows.append({
            "channel_id": series.id,
            "window_x": point.x,
            "state": _conformance_state(point, band, margin),
        })
    return rows


def _conformance_state(point: SeriesPoint, band: EnvelopeBand, margin: float) -> str:
    low, high = point.lo, point.hi
    if low is None or high is None:
        return "unknown"
    if band.min_value is not None and low < band.min_value:
        return "outside"
    if band.max_value is not None and high > band.max_value:
        return "outside"
    if margin > 0:
        if band.min_value is not None and low < band.min_value + margin:
            return "near"
        if band.max_value is not None and high > band.max_value - margin:
            return "near"
    return "inside"


# ── Events, cadence, experience, storage ──


def _events(registry: Any) -> tuple[dict[str, Any], ...]:
    """Return the event ring, shaped for a timeline.

    ``title``/``summary``/``severity`` are the keys the timeline renderer reads; the
    structured fields are kept alongside them because a consumer that only got a
    rendered sentence would have to parse it back apart.
    """
    rows: list[dict[str, Any]] = []
    for event in _safely(lambda: registry.recent_events(limit=EVENT_LIMIT), default=()):
        kind = str(getattr(event, "kind", ""))
        device_id = str(getattr(event, "device_id", ""))
        channel_id = str(getattr(event, "channel_id", ""))
        rows.append({
            "title": f"{kind} · {device_id}.{channel_id}",
            "summary": str(getattr(event, "detail", "")),
            "severity": _event_severity(kind),
            "kind": kind,
            "device_id": device_id,
            "channel_id": channel_id,
            "detail": str(getattr(event, "detail", "")),
            "value": getattr(event, "value", None),
            "unit": str(getattr(event, "unit", "")),
            "x": float(getattr(event, "observed_at", 0.0) or 0.0),
        })
    rows.sort(key=lambda row: row["x"], reverse=True)
    return tuple(rows)


ALERT_KINDS = frozenset({"threshold_exceeded", "rate_exceeded", "stale", "unreachable"})
"""Event kinds that mean somebody has to look.

Exported rather than private because the producer decides push severity from the same
set, and two copies of one judgement drift: ``unreachable`` was added to the row
severity here while the producer kept its own three-item copy, so the board coloured
the row correctly and still declined to push it to anyone.

``settled`` and ``sample_loss`` are deliberately absent -- a recovery is good news and
a lost sample is already visible in the trace, so neither should interrupt anyone.
"""

_NOTABLE_KINDS = frozenset({"quality_degraded", "sample_loss"})


def _event_severity(kind: str) -> str:
    """Colour an event by what it means operationally.

    ``settled`` stays informational on purpose: a recovery is the one event nobody
    needs to be alarmed by, and colouring it like a breach would train a watcher to
    ignore the colour.
    """
    if kind in ALERT_KINDS:
        return "alert"
    if kind in _NOTABLE_KINDS:
        return "notable"
    return "info"


def _sampling(registry: Any) -> tuple[dict[str, Any], ...]:
    """Report observed cadence against declared cadence.

    The only place a shortfall becomes visible: a window records the samples it
    actually received, so a channel running at two thirds of its declared rate
    produces a series that looks entirely correct.
    """
    rows: list[dict[str, Any]] = []
    for source in _safely(registry.stream_sources, default=()):
        health = getattr(source, "health", None)
        if isinstance(health, dict):
            rows.append(dict(health))
    return tuple(rows)


def _outcomes_for(registry: Any, context: Any) -> list[dict[str, Any]]:
    """Surface recalled command outcomes, so learned parameters are visible.

    Until now this experience only reached the model, through ``hw_describe``. A
    human could not see what the agent had learned about their own bench.
    """
    recorder = getattr(registry, "outcome_recorder", None)
    if recorder is None:
        return []
    rows: list[dict[str, Any]] = []
    for channel in getattr(context, "writable_channels", ()):
        # The learned correction is reported per channel even when nothing was
        # recalled: "this valve runs 8% low" is the shortest true statement about a
        # bench, and it is the one thing the recalled text cannot express, because the
        # shared store keeps only the magnitude of an error and not its direction.
        calibration = _safely(
            lambda: recorder.calibration_for(context.device_id, channel.channel_id),
            default=None,
        )
        if calibration is not None:
            bias, samples = calibration
            rows.append({
                "device_id": context.device_id,
                "channel_id": channel.channel_id,
                "command": f"learned correction ({samples} observation(s))",
                "outcome": f"{bias:+g}{f' {channel.unit}' if channel.unit else ''}",
                "delta": abs(bias),
            })
        try:
            recalled = recorder.recall(
                device_id=context.device_id, channel=channel, limit=OUTCOME_LIMIT
            )
        except Exception as exc:  # noqa: BLE001 - recall is an optimisation, not a duty
            logger.debug("hardware digest: recall failed for %s: %s", context.device_id, exc)
            continue
        for row in recalled:
            rows.append({
                "device_id": context.device_id,
                "channel_id": channel.channel_id,
                "command": str(row.get("command", "")),
                "outcome": str(row.get("outcome", "")),
                "delta": _as_float(row.get("delta")),
                "unit": str(getattr(channel, "unit", "") or ""),
            })
    return rows


def _storage(registry: Any) -> dict[str, Any]:
    """Report persistence health, including the failure count.

    ``windows_written`` on its own is a numerator with no denominator: a database
    that cannot be opened looks exactly like an idle bench.
    """
    store = getattr(registry, "reading_store", None)
    if store is None:
        return {"persisting": False}
    return {
        "persisting": True,
        "raw_writes": int(getattr(store, "raw_writes", 0) or 0),
        "windows_written": int(getattr(store, "windows_written", 0) or 0),
        "write_failures": int(getattr(store, "write_failures", 0) or 0),
        "rows_pruned": int(getattr(store, "rows_pruned", 0) or 0),
        "pending_channels": int(getattr(store, "pending_channels", 0) or 0),
    }


# ── Helpers ──


def _safely(call: Any, *, default: Any) -> Any:
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - a missing section beats a missing board
        logger.debug("hardware digest: section unavailable: %s", exc)
        return default


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _span(band: EnvelopeBand) -> float:
    if band.min_value is None or band.max_value is None:
        return 0.0
    span = band.max_value - band.min_value
    return span if span > 0 else 0.0


_QUALITY_ORDER = ("ok", "suspect", "stale", "saturated")


def _worst(values: Any) -> str:
    worst, rank = "ok", 0
    for value in values:
        current = _QUALITY_ORDER.index(value) if value in _QUALITY_ORDER else len(_QUALITY_ORDER)
        if current > rank:
            worst, rank = value, current
    return worst


__all__ = ["EVENT_LIMIT", "NEAR_FRACTION", "OUTCOME_LIMIT", "build_digest"]
