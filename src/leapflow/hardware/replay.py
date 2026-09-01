"""Replay raw NDJSON segment files through the event detector.

Reads the segment files produced by ``ReadingStore._append_raw`` and feeds each
line — parsed back into a ``Reading`` — through ``HardwareEventDetector.observe()``
in strict file order. The result is a deterministic event sequence: replaying the
same file twice always yields identical output, because the detector is stateful per
channel and the inputs arrive in the same order.

Two entry points:

* ``replay_segment(path, detector)`` — library API returning the event list.
* ``run_replay(path)`` — CLI helper that builds a minimal detector from the first
  reading's metadata and prints each event.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from leapflow.hardware.context import Channel, Envelope, HardwareContext
from leapflow.hardware.stream import HardwareEvent, HardwareEventDetector
from leapflow.hardware.transport import Reading

logger = logging.getLogger(__name__)


def _reading_from_dict(data: dict[str, Any]) -> Reading:
    """Reconstruct a ``Reading`` from its ``to_dict()`` form.

    ``monotonic_at`` is absent in persisted data (see ``Reading.to_dict``).
    We synthesise it from ``observed_at`` so that elapsed-time calculations inside
    the detector produce the same wall-clock deltas the device originally did.
    Using wall-clock for monotonic is acceptable here because replay never competes
    with a live clock: there is no NTP step to worry about, and the only consumer
    of the monotonic field is rate-of-change arithmetic that needs a delta, not an
    absolute epoch.
    """
    return Reading(
        device_id=data.get("device_id", ""),
        channel_id=data.get("channel_id", ""),
        value=data.get("value"),
        quantity=data.get("quantity", ""),
        unit=data.get("unit", ""),
        observed_at=float(data.get("observed_at", 0.0)),
        monotonic_at=float(data.get("observed_at", 0.0)),
        sequence=int(data.get("sequence", 0)),
        quality=str(data.get("quality", "ok")),
    )


def replay_segment(
    path: Path,
    detector: HardwareEventDetector,
) -> list[HardwareEvent]:
    """Replay one NDJSON segment through *detector*, returning every event produced.

    Lines that cannot be parsed are logged and skipped — a partially written line at
    the tail of a segment is expected (the writer may have been interrupted) and must
    not abort the rest.

    The readings are fed in file order with gap detection: a jump in the ``sequence``
    field is reported as lost samples, exactly as the live sampling loop does.
    """
    events: list[HardwareEvent] = []
    last_seq: int | None = None

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.error("Cannot read segment %s: %s", path, exc)
        return events

    for lineno, raw_line in enumerate(lines, start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            data = json.loads(raw_line)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Skipping unparseable line %d in %s: %s", lineno, path, exc)
            continue

        reading = _reading_from_dict(data)
        lost = 0
        if last_seq is not None and reading.sequence > last_seq + 1:
            lost = reading.sequence - last_seq - 1
        last_seq = reading.sequence

        events.extend(detector.observe(reading, lost=lost))

    return events


def _build_replay_detector(
    device_id: str,
    channel_id: str,
    quantity: str = "",
    unit: str = "",
) -> HardwareEventDetector:
    """Build a minimal detector for a replay that has no live registry.

    Uses a permissive envelope — no limits, no rate constraints — so the replay
    faithfully surfaces quality, staleness, and sample-loss events without
    injecting threshold opinions that are absent from the segment file.
    """
    channel = Channel(
        channel_id=channel_id,
        direction="read",
        quantity=quantity,
        unit=unit,
        envelope=Envelope(),
    )
    context = HardwareContext(device_id=device_id, channels=(channel,))
    return HardwareEventDetector(context, channel)


def run_replay(path: Path) -> list[HardwareEvent]:
    """CLI convenience: replay a segment using metadata from the first reading.

    Returns the event list so the caller can render or serialise it.
    """
    try:
        first_line = ""
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    first_line = line
                    break
    except OSError as exc:
        logger.error("Cannot open segment %s: %s", path, exc)
        return []

    if not first_line:
        logger.warning("Segment %s is empty", path)
        return []

    try:
        first = json.loads(first_line)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("First line of %s is not valid JSON: %s", path, exc)
        return []

    detector = _build_replay_detector(
        device_id=first.get("device_id", "unknown"),
        channel_id=first.get("channel_id", "unknown"),
        quantity=first.get("quantity", ""),
        unit=first.get("unit", ""),
    )
    return replay_segment(path, detector)
