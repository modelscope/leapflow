"""Deterministic reference-document renderer.

Turns a ``HardwareContext`` into the text an agent reads before operating a
device. No model call and no heuristics are involved, which is what makes the
output testable and keeps it honest: every line traces to a declared field.

Two properties are deliberate.

Provenance appears in the header rather than in a footnote. A pseudo-implementation
of an unpublished standard must state that it is a guess, or it will be read as a
specification; an unverified context says so on the second line.

Machine verdicts and human prose come from the same source. "NOT reversible" is
rendered from ``Envelope.reversible``, not written by hand, so the sentence a model
reads and the rule a gate enforces cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from leapflow.hardware.context import (
    Channel,
    ContextSource,
    Envelope,
    HardwareContext,
    HardwareEffect,
)

_INDENT = "    "


def render_reference(context: HardwareContext) -> str:
    """Return the full reference document for one device."""
    lines: list[str] = []
    lines.extend(_render_header(context))
    lines.append("")
    lines.extend(_render_channels(context))
    if context.interlocks:
        lines.append("")
        lines.extend(_render_interlocks(context))
    if context.notes.strip():
        lines.append("")
        lines.append("NOTES")
        for note_line in _wrap_notes(context.notes):
            lines.append(f"{_INDENT}{note_line}")
    if context.provenance.is_lossy:
        lines.append("")
        lines.append("FIDELITY")
        lines.append(
            f"{_INDENT}Imported with loss. Unmapped upstream fields: "
            f"{', '.join(context.provenance.lossy_fields)}."
        )
        lines.append(
            f"{_INDENT}Treat limits below as incomplete and confirm against the device manual."
        )
    return "\n".join(lines)


def describe(context: HardwareContext) -> dict[str, Any]:
    """Return the structured payload behind the describe tool.

    Carries both the rendered text (what a model reads) and the machine fields
    (what a caller can act on) so neither consumer has to parse the other's form.
    """
    return {
        "device_id": context.device_id,
        "display_name": context.label,
        "location": context.location,
        "hc_version": context.hc_version,
        "halt_supported": context.halt_supported,
        "provenance": context.provenance.to_dict(),
        "writable_channels": [c.channel_id for c in context.writable_channels],
        "streaming_channels": [c.channel_id for c in context.streaming_channels],
        "channels": [c.to_dict() for c in context.channels],
        "interlocks": [i.to_dict() for i in context.interlocks],
        "reference": render_reference(context),
    }


def summarize(context: HardwareContext) -> dict[str, Any]:
    """Return the compact index entry behind the list tool.

    Envelopes are excluded on purpose. The index exists so a model can see what
    is available without paying for every limit of every channel; the limits are
    one describe call away when it actually intends to act.
    """
    return {
        "device_id": context.device_id,
        "display_name": context.label,
        "location": context.location,
        "channels": len(context.channels),
        "writable": len(context.writable_channels),
        "streaming": len(context.streaming_channels),
        "quantities": sorted({c.quantity for c in context.channels if c.quantity}),
        "verified": context.provenance.is_verified,
        "halt_supported": context.halt_supported,
    }


def _render_header(context: HardwareContext) -> list[str]:
    vendor_model = " ".join(part for part in (context.vendor, context.model) if part)
    where = f" ({context.location})" if context.location else ""
    title = f"DEVICE {context.device_id}"
    if vendor_model:
        title = f"{title} - {vendor_model}{where}"
    elif where:
        title = f"{title}{where}"

    provenance = context.provenance
    if provenance.is_verified:
        trust = f"{provenance.source}, VERIFIED by {provenance.verified_by}"
    elif provenance.source == ContextSource.IMPORTED.value:
        trust = f"{provenance.source} from {provenance.upstream_version or 'upstream'}, UNVERIFIED"
    else:
        trust = f"{provenance.source}, UNVERIFIED"

    writable = context.writable_channels
    if not writable:
        trust = f"{trust} - no writable channels (reads only)"

    halt = "supported" if context.halt_supported else "NOT SUPPORTED"
    return [
        title,
        f"context: {trust}",
        f"transport: {context.transport.kind or 'unset'}",
        f"emergency stop: {halt}",
    ]


def _render_channels(context: HardwareContext) -> list[str]:
    lines = ["CHANNELS"]
    if not context.channels:
        lines.append(f"{_INDENT}(none declared)")
        return lines
    for channel in context.channels:
        lines.append(f"{_INDENT}{_channel_headline(channel)}")
        lines.extend(f"{_INDENT * 2}{detail}" for detail in _channel_details(channel))
    return lines


def _channel_headline(channel: Channel) -> str:
    parts = [channel.channel_id, channel.direction]
    if channel.quantity:
        parts.append(channel.quantity)
    if channel.unit:
        parts.append(channel.unit)
    if channel.effect != HardwareEffect.READ.value:
        parts.append(f"effect={channel.effect}")
    if channel.is_streaming:
        parts.append(f"{channel.sample_rate_hz:g} Hz")
    return "  ".join(parts)


def _channel_details(channel: Channel) -> list[str]:
    details: list[str] = []
    envelope = channel.envelope
    if channel.description.strip():
        details.append(channel.description.strip())

    if not envelope.declared:
        if channel.is_writable:
            details.append("NO DECLARED ENVELOPE - writes are refused until limits are declared")
        return details

    bounds = _render_bounds(envelope)
    if bounds:
        details.append(bounds)
    if channel.is_writable and envelope.max_rate is not None:
        details.append(
            f"paced: consecutive commands may change this by at most "
            f"{envelope.max_rate:g} {channel.unit or 'units'}/s; a larger step is refused "
            "until enough time has passed"
        )
    if envelope.settling_time_s > 0:
        details.append(
            f"settling {envelope.settling_time_s:g}s - the value is not stable until it elapses"
        )
    if channel.is_writable and not envelope.reversible:
        details.append("NOT reversible - a repeated call applies the effect twice")
    if channel.verify_after_write:
        details.append("read back after every write")
    if envelope.requires_interlocks:
        details.append(f"interlocks: {', '.join(envelope.requires_interlocks)}")
    if envelope.notes.strip():
        details.extend(_wrap_notes(envelope.notes))
    return details


def _render_bounds(envelope: Envelope) -> str:
    parts: list[str] = []
    if envelope.min_value is not None or envelope.max_value is not None:
        low = "-inf" if envelope.min_value is None else f"{envelope.min_value:g}"
        high = "+inf" if envelope.max_value is None else f"{envelope.max_value:g}"
        parts.append(f"range {low}..{high}")
    if envelope.max_rate is not None:
        parts.append(f"max rate {envelope.max_rate:g}/s")
    if envelope.quantization is not None:
        parts.append(f"step {envelope.quantization:g}")
    return "   ".join(parts)


def _render_interlocks(context: HardwareContext) -> list[str]:
    lines = ["INTERLOCKS"]
    readable = {c.channel_id for c in context.channels if c.is_readable}
    for lock in context.interlocks:
        suffix = "" if lock.channel_id in readable else "   [UNEVALUABLE - guarded writes denied]"
        lines.append(
            f"{_INDENT}{lock.interlock_id}: {lock.channel_id} {lock.operator} {lock.value!r}{suffix}"
        )
        if lock.description.strip():
            lines.append(f"{_INDENT * 2}{lock.description.strip()}")
    return lines


def _wrap_notes(text: str, width: int = 84) -> list[str]:
    """Collapse whitespace and wrap prose to a readable width."""
    import textwrap

    collapsed = " ".join(text.split())
    return textwrap.wrap(collapsed, width=width) or [""]


__all__ = ["describe", "render_reference", "summarize"]
