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
    outcomes: list[dict[str, Any]] = []

    for context in contexts:
        for channel in getattr(context, "channels", ()):
            if not getattr(channel, "is_readable", False):
                continue
            windows = _history(registry, context.device_id, channel.channel_id)
            if not windows:
                continue
            series.append(_series_for(context, channel, windows))
        outcomes.extend(_outcomes_for(registry, context))

    # Clamp first, then classify. Conformance describes the windows of the channels
    # that are actually charted, so deriving it from the pre-clamp list made it
    # describe channels the board never draws -- and made it the one section no
    # ceiling applied to, at one row per window of every readable channel on the bench.
    charted = clamp_series(series)
    conformance = [row for item in charted for row in _conformance_for(item)]

    return HardwareDigest(
        generated_at=moment,
        devices=tuple(_device_row(registry, context) for context in contexts),
        series=charted,
        events=_events(registry),
        conformance=tuple(conformance),
        sampling=_sampling(registry),
        outcomes=tuple(outcomes),
        calibration=_calibration(registry, contexts, now=moment),
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
        # Presentation only, so the board can group a fleet instead of listing it.
        "device_class": str(getattr(context, "device_class", "") or "unclassified"),
        "transport_kind": str(getattr(getattr(context, "transport", None), "kind", "") or ""),
        "verified": bool(getattr(provenance, "verified_by", "")),
        "channels": len(channels),
        "writable": len(writable),
        "streaming": sum(1 for c in channels if getattr(c, "is_streaming", False)),
        "media": sum(1 for c in channels if getattr(c, "is_media", False)),
        "privacy_gated": sum(1 for c in channels if getattr(c, "is_privacy_gated", False)),
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

_NOTABLE_KINDS = frozenset({"quality_degraded", "sample_loss", "calibration_failed", "calibration_expired"})


def _event_severity(kind: str) -> str:
    """Colour an event by what it means operationally.

    ``settled`` stays informational on purpose: a recovery is the one event nobody
    needs to be alarmed by, and colouring it like a breach would train a watcher to
    ignore the colour. Calibration events are informational when started or completed,
    but notable when failed or expired -- an uncalibrated channel may produce
    misleading readings.
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


def _calibration(
    registry: Any, contexts: Sequence[Any], *, now: float
) -> tuple[dict[str, Any], ...]:
    """Report per-device calibration state, never-calibrated and expired first.

    This section is the one the board template already asked for and never received:
    ``hardware.yaml`` binds ``hardware.calibration`` behind a ``when:`` gate, the
    payload never carried the key, and the gate silently removed the whole panel. The
    producer ran, the template was valid, and nothing connected them -- exactly the
    failure the ``capability_plan`` binding hit before it.

    Ordered so the rows that need action come first, because the table is read from
    the top: a device that has never been calibrated cannot be trusted to measure,
    and one whose calibration has aged out is worse than one with none, since its
    readings look authoritative.
    """
    store = getattr(registry, "calibration_store", None)
    if store is None:
        return ()
    rows: list[dict[str, Any]] = []
    for context in contexts:
        record = _safely(lambda: store.latest(context.device_id), default=None)
        if record is None:
            rows.append({
                "device_id": context.device_id,
                "channel_id": "--",
                "state": "never",
                "calibrated_at": "",
                "days_since": "",
                "residual": "",
                "next_recal_due": "unknown",
                "_rank": 0,
            })
            continue
        age_days = max(0.0, (now - float(record.recorded_at)) / 86400.0)
        interval_days = _as_float(dict(record.parameters).get("recal_interval_days"))
        expired = interval_days is not None and age_days > interval_days
        rows.append({
            "device_id": context.device_id,
            "channel_id": str(record.procedure_id or "--"),
            "state": "expired" if expired else "valid",
            "calibrated_at": _stamp(record.recorded_at),
            "days_since": f"{age_days:.1f}",
            "residual": _residual(record),
            "next_recal_due": (
                _stamp(record.recorded_at + interval_days * 86400.0)
                if interval_days is not None
                else "not declared"
            ),
            "_rank": 1 if expired else 2,
        })
    rows.sort(key=lambda row: (row["_rank"], str(row["device_id"])))
    return tuple({key: value for key, value in row.items() if key != "_rank"} for row in rows)


def _residual(record: Any) -> str:
    """Render the fitted residual a procedure reported, when it reported one.

    Read from the procedure's own parameters rather than computed here: the residual
    of a hand-eye fit and of a two-point sensor calibration are different quantities,
    and inventing a common formula would put a number on the board that no procedure
    produced.
    """
    parameters = dict(getattr(record, "parameters", {}) or {})
    for key in ("residual", "rms_error", "error"):
        value = _as_float(parameters.get(key))
        if value is not None:
            return f"{value:g}"
    return "not reported"


def _stamp(value: Any) -> str:
    """Format a wall-clock instant for a table cell, tolerating a bad value."""
    moment = _as_float(value)
    if moment is None or moment <= 0:
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(moment))


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
