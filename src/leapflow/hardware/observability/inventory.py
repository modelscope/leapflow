"""Fleet inventory and per-device views: the board's on-demand read path.

A third data shape beside the digest. The digest is a *cycle* payload -- built on the
monitor cadence, byte-capped, pushed to whoever is listening, and deliberately
capped at eight charted channels because nobody reads more than that on one screen.
These two functions answer a different question: "what is attached" and "what is this
one device doing", asked by a person who just clicked on it.

Same discipline as the digest, for the same reasons:

- **Nothing is probed.** Values come from the sampling ring and the reading store,
  never from a transport round trip, so opening a device page cannot be the reason a
  bus is busy or a camera turns on.
- **Nothing is a second source of truth.** Channel limits come from the declaration,
  values from the store, capability from the admitted context. A number here that the
  gate could disagree with would be a defect by construction.
- **Media channels disclose metadata only.** A frame channel reports that it can be
  previewed and at what ceiling; the bytes are fetched separately, under consent.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from leapflow.hardware.context import PrivacyTier, Representation
from leapflow.hardware.observability.digest import ALERT_KINDS

logger = logging.getLogger(__name__)

DEVICE_CHANNEL_LIMIT = 64
"""Channels described per device view.

Higher than the digest's eight-series chart limit because this is a table for one
device rather than a chart for a fleet, and a discovered host legitimately has
twenty. Still bounded: the payload crosses a JSON-RPC boundary.
"""

TRACE_POINTS = 120
"""History windows per channel sparkline on a device page.

Enough to show a shape, small enough that a twenty-channel host stays a reasonably
sized reply.
"""


def build_inventory(registry: Any, *, now: float | None = None) -> dict[str, Any]:
    """Return every admitted device grouped by declared class.

    Grouping is presentational and comes from ``device_class``, a free-form label the
    declaration supplies. Nothing branches on it -- an unrecognised class simply
    becomes its own section, which is what lets a new kind of peripheral appear on the
    board without a code change.
    """
    if registry is None:
        return {"ok": False, "code": "hardware_disabled", "groups": [], "counts": {}}

    contexts = _safely(registry.contexts, default=())
    opened = set(_safely(registry.opened_devices, default=()))
    rows = [_inventory_row(context, opened) for context in contexts]

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["device_class"]), []).append(row)

    report = _safely(lambda: registry.report, default=None)
    return {
        "ok": True,
        "generated_at": now if now is not None else _now(),
        "groups": [
            {"device_class": name, "devices": sorted(items, key=lambda r: str(r["device_id"]))}
            for name, items in sorted(groups.items())
        ],
        "devices": rows,
        "counts": {
            "devices": len(rows),
            "classes": len(groups),
            "channels": sum(int(row["channels"]) for row in rows),
            "previewable": sum(int(row["media"]) for row in rows),
            "privacy_gated": sum(int(row["privacy_gated"]) for row in rows),
            "writable": sum(int(row["writable"]) for row in rows),
            "opened": len(opened),
        },
        # Surfaced rather than logged: a demotion is the difference between "this
        # device has no controls" and "this device was refused its controls, here is
        # why", and only the second is actionable.
        "notes": [note.to_dict() for note in getattr(report, "notes", ()) or ()],
        "rejected": list(getattr(report, "rejected", ()) or ()),
    }


def _inventory_row(context: Any, opened: set[str]) -> dict[str, Any]:
    channels = tuple(getattr(context, "channels", ()))
    previews = tuple(channel for channel in channels if _is_previewable(channel))
    provenance = getattr(context, "provenance", None)
    return {
        "device_id": context.device_id,
        "label": getattr(context, "display_name", "") or context.device_id,
        "device_class": str(getattr(context, "device_class", "") or "unclassified"),
        "vendor": str(getattr(context, "vendor", "") or ""),
        "model": str(getattr(context, "model", "") or ""),
        "location": str(getattr(context, "location", "") or ""),
        "transport_kind": str(getattr(getattr(context, "transport", None), "kind", "") or ""),
        "provenance_source": str(getattr(provenance, "source", "") or ""),
        "verified": bool(getattr(provenance, "verified_by", "")),
        "channels": len(channels),
        "writable": sum(1 for c in channels if c.is_writable),
        "streaming": sum(1 for c in channels if c.is_streaming),
        # "Preview" means a person can actually watch it: a frame, or the live scalar
        # meter a microphone exposes. Counting only ``is_media`` rendered 0 beside every
        # microphone and hid its Preview button even though the device page had a meter.
        "media": len(previews),
        "preview_channel": previews[0].channel_id if len(previews) == 1 else "",
        "privacy_gated": sum(1 for c in channels if c.is_privacy_gated),
        "halt_supported": bool(getattr(context, "halt_supported", False)),
        "opened": context.device_id in opened,
    }


def build_device_view(registry: Any, device_id: str, *, now: float | None = None) -> dict[str, Any]:
    """Return one device's identity, channels, live values and control surface.

    ``ok=False`` with a code rather than an exception for an unknown device: this is
    reached from a URL a person can edit and from a stale board link, and neither
    deserves a traceback.
    """
    if registry is None:
        return {"ok": False, "code": "hardware_disabled"}
    context = _safely(lambda: registry.context(str(device_id)), default=None)
    if context is None:
        return {"ok": False, "code": "unknown_device", "device_id": str(device_id)}

    opened = set(_safely(registry.opened_devices, default=()))
    channels = tuple(getattr(context, "channels", ()))[:DEVICE_CHANNEL_LIMIT]
    provenance = getattr(context, "provenance", None)

    return {
        "ok": True,
        "generated_at": now if now is not None else _now(),
        "device": _inventory_row(context, opened),
        "identity": {
            "device_id": context.device_id,
            "label": getattr(context, "display_name", "") or context.device_id,
            "notes": str(getattr(context, "notes", "") or ""),
            "provenance_source": str(getattr(provenance, "source", "") or ""),
            "verified_by": str(getattr(provenance, "verified_by", "") or ""),
            "provenance_notes": str(getattr(provenance, "notes", "") or ""),
            "lossy_fields": list(getattr(provenance, "lossy_fields", ()) or ()),
        },
        "channels": [_channel_row(registry, context, channel) for channel in channels],
        "traces": _traces(registry, context, channels),
        "controls": [_control_row(channel) for channel in channels if channel.is_writable],
        "previews": [
            _preview_row(registry.settings, context, channel)
            for channel in channels
            if _is_previewable(channel)
        ],
        "events": _device_events(registry, context.device_id),
        "channels_omitted": max(0, len(getattr(context, "channels", ())) - len(channels)),
    }


def _channel_row(registry: Any, context: Any, channel: Any) -> dict[str, Any]:
    """Describe one channel, with its latest sampled value when there is one.

    The value is read from the sampling ring, which is why a non-streaming channel
    shows "not sampled" rather than a stale number: a channel nobody samples has no
    current value that anyone measured, and printing the last on-demand read as if it
    were live is how a board starts lying about a device.
    """
    summary = _safely(
        lambda: registry.channel_summary(context.device_id, channel.channel_id), default=None
    ) or {}
    envelope = channel.envelope
    return {
        "channel_id": channel.channel_id,
        "quantity": channel.quantity,
        "unit": channel.unit,
        "direction": channel.direction,
        "effect": channel.effect,
        "representation": channel.representation,
        "privacy": channel.privacy,
        "description": channel.description,
        "value": summary.get("latest", None),
        "quality": str(summary.get("quality", "") or ("" if summary else "not sampled")),
        "samples": int(summary.get("samples", 0) or 0),
        "trend": str(summary.get("trend", "") or ""),
        "limits": _bounds_label(envelope),
        "declared_hz": channel.sample_rate_hz,
        "writable": channel.is_writable,
        "previewable": channel.is_media,
        "consent_required": channel.is_privacy_gated,
    }


def _traces(registry: Any, context: Any, channels: Sequence[Any]) -> list[dict[str, Any]]:
    """Return a compact history series per sampled numeric channel.

    Media and state channels are skipped: a sparkline of a frame reference or of a
    boolean is a line with no meaning, and it would cost the same bytes as a real one.
    """
    series: list[dict[str, Any]] = []
    for channel in channels:
        if channel.is_media or not channel.is_streaming:
            continue
        windows = _safely(
            lambda ch=channel: registry.channel_history(
                context.device_id, ch.channel_id, limit=TRACE_POINTS
            ),
            default=(),
        )
        points = [
            {"x": float(row.get("ended_at") or 0.0), "y": row.get("mean_value")}
            for row in windows
            if row.get("mean_value") is not None
        ]
        if not points:
            continue
        series.append({
            "id": f"{context.device_id}.{channel.channel_id}",
            "label": f"{channel.channel_id} ({channel.unit})" if channel.unit else channel.channel_id,
            "unit": channel.unit,
            "points": points,
        })
    return series


def _control_row(channel: Any) -> dict[str, Any]:
    """Describe the control surface of one writable channel.

    The declared envelope *is* the widget specification -- an enumerated domain is a
    select, a bounded numeric is a slider with a step, anything else is a plain field.
    Deriving it here means the board never hardcodes a control for a known device, and
    a new peripheral gets usable controls from its declaration alone.
    """
    envelope = channel.envelope
    if envelope.allowed_values:
        kind, options = "select", [str(value) for value in envelope.allowed_values]
    elif envelope.min_value is not None and envelope.max_value is not None:
        kind, options = "slider", []
    else:
        kind, options = "field", []
    return {
        "channel_id": channel.channel_id,
        "control": kind,
        "options": options,
        "unit": channel.unit,
        "effect": channel.effect,
        "min_value": envelope.min_value,
        "max_value": envelope.max_value,
        "step": envelope.quantization,
        "reversible": envelope.reversible,
        "settling_s": envelope.effective_settling_s,
        "requires_interlocks": list(envelope.requires_interlocks),
        "limits": _bounds_label(envelope),
        # Every write goes through approval per invocation, so the board's job is to
        # say so before somebody submits, not to predict the verdict.
        "approval": "required",
    }


def _is_previewable(channel: Any) -> bool:
    """Return whether this channel supports a *live* preview panel.

    Two shapes qualify, and the second is not obvious: a frame channel streams pictures,
    and a privacy-gated numeric channel is a live meter -- a microphone's input level being
    the case that matters. Both disclose the surroundings continuously, both need consent,
    and both are useless as a static table cell, which is what they were before this.

    A privacy-gated *state* channel is excluded: there is nothing to watch change.
    """
    if channel.is_media:
        return True
    return channel.is_privacy_gated and channel.representation == Representation.SCALAR.value


def _preview_row(settings: Any, context: Any, channel: Any) -> dict[str, Any]:
    """Describe a previewable channel without fetching a single byte or reading a value."""
    return {
        "device_id": context.device_id,
        "channel_id": channel.channel_id,
        "label": f"{getattr(context, 'display_name', '') or context.device_id} · {channel.channel_id}",
        # What the client renders with: a picture, or a meter. Derived from the declared
        # representation rather than the device class, so a non-camera frame source and a
        # non-microphone level source both work without a new case.
        "kind": "microphone" if not channel.is_media else "camera",
        "media_type": channel.media_type or ("image/jpeg" if channel.is_media else ""),
        "max_fps": channel.max_frame_rate_hz,
        # Runtime ceilings sent to the page so it can label each profile truthfully. They
        # are informational here; PreviewBroker is still authoritative and clamps a
        # hand-edited request before it reaches an encoder.
        "max_width": int(getattr(settings, "preview_max_width", 0) or 0),
        "max_quality": int(getattr(settings, "preview_quality", 0) or 0),
        "unit": channel.unit,
        # The meter's zero point. Taken from the declared envelope, so a channel whose
        # floor is -90 dBFS is not drawn against an invented -60.
        "floor": channel.envelope.min_value,
        "privacy": channel.privacy,
        "consent_required": channel.is_privacy_gated,
        "consent_reason": _consent_reason(channel),
    }


def _consent_reason(channel: Any) -> str:
    """Say what a preview would disclose, in the words the prompt will use.

    Written once, here, so the board's explanation and the approval prompt cannot
    describe the same act differently -- which is how a person learns to click through
    prompts they have stopped believing.
    """
    if channel.privacy == PrivacyTier.ENVIRONMENT.value:
        return "Observes the space around this machine."
    if channel.privacy == PrivacyTier.PERSONAL.value:
        return "Observes the person using this machine."
    return ""


def _device_events(registry: Any, device_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return this device's recent events, newest first.

    Severity uses the digest's ``ALERT_KINDS`` rather than a local copy: the same
    judgement held in two places has already drifted once in this subsystem, when the
    board coloured an event as an alert and the producer declined to push it.
    """
    rows: list[dict[str, Any]] = []
    for event in _safely(
        lambda: registry.recent_events(device_id, limit=limit), default=()
    ):
        kind = str(getattr(event, "kind", ""))
        rows.append({
            "title": f"{kind} · {getattr(event, 'channel_id', '')}",
            "summary": str(getattr(event, "detail", "")),
            "severity": "alert" if kind in ALERT_KINDS else "info",
            "x": float(getattr(event, "observed_at", 0.0) or 0.0),
        })
    rows.sort(key=lambda row: row["x"], reverse=True)
    return rows


def _bounds_label(envelope: Any) -> str:
    if not getattr(envelope, "declared", False):
        return "not declared"
    if envelope.allowed_values:
        return " | ".join(str(value) for value in envelope.allowed_values)
    low = "-inf" if envelope.min_value is None else f"{envelope.min_value:g}"
    high = "+inf" if envelope.max_value is None else f"{envelope.max_value:g}"
    return f"{low} .. {high}"


def _safely(call: Any, *, default: Any) -> Any:
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - a missing section beats a missing page
        logger.debug("hardware inventory: section unavailable: %s", exc)
        return default


def _now() -> float:
    return time.time()


__all__ = [
    "DEVICE_CHANNEL_LIMIT",
    "TRACE_POINTS",
    "build_device_view",
    "build_inventory",
]
