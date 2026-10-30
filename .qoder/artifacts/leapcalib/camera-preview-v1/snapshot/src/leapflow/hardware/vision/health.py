# Copyright (c) Alibaba, Inc. and its affiliates.
"""Frame-pump health as hardware signals.

A camera that stops delivering looks, from a skill's side, like a slow camera
until its capture times out mid-run. The pump already knows sooner: its
``FrameHub.stats()`` carry the age of the newest frame and a failure counter.
This module turns successive stats snapshots into *transitions* -- a stream that
went stale, a stream that recovered, frames lost to capture errors -- typed as
ordinary ``HardwareEvent`` kinds, so they enter the same signal path, alert
policy and board as every other device condition instead of a side channel.

Only transitions are reported: a camera stale for an hour is one ``stale`` event
and one ``settled`` when it comes back, not one per poll. A pump that is not
running is not judged -- a closed camera is idle, not broken.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from leapflow.hardware.stream import EventKind, HardwareEvent

FRAME_HUB_CHANNEL = "frame_hub"
"""The channel id frame-pump events are reported on (a device may have many streams)."""

DEFAULT_STALE_FLOOR_MS = 1000.0
"""A stream is never called stale sooner than this, whatever its rate."""
DEFAULT_STALE_FRAMES = 15
"""...nor sooner than this many of its own frame intervals (slow profiles wait longer)."""


@dataclass
class _PumpState:
    stale: bool = False
    failures: int = 0
    sequence: int = 0


class FrameHealthTracker:
    """Derive stale / recovered / sample-loss transitions from pump stats snapshots."""

    def __init__(
        self,
        *,
        stale_floor_ms: float = DEFAULT_STALE_FLOOR_MS,
        stale_frames: int = DEFAULT_STALE_FRAMES,
        clock: Any = time.time,
    ) -> None:
        self._floor_ms = max(0.0, float(stale_floor_ms))
        self._frames = max(1, int(stale_frames))
        self._clock = clock
        self._states: dict[str, _PumpState] = {}

    def stale_after_ms(self, fps: float) -> float:
        """The age past which a stream at *fps* counts as stale."""
        interval_ms = 1000.0 / fps if fps > 0 else 0.0
        return max(self._floor_ms, self._frames * interval_ms)

    def observe(self, device_id: str, stats: Mapping[str, Any] | None) -> list[HardwareEvent]:
        """Compare one snapshot with the previous one; return the transitions."""
        if not isinstance(stats, Mapping) or not stats.get("running"):
            self._states.pop(device_id, None)  # closed or never opened: start over next time
            return []
        state = self._states.get(device_id)
        failures = _count(stats.get("failures"))
        sequence = _count(stats.get("sequence"))
        first_sight = state is None or sequence < state.sequence  # new, or the pump restarted
        if first_sight:
            state = self._states[device_id] = _PumpState(failures=failures, sequence=sequence)
        events: list[HardwareEvent] = []
        lost = failures - state.failures
        if lost > 0:
            reason = str(stats.get("last_error") or "capture failed")
            events.append(self._event(EventKind.SAMPLE_LOSS, device_id, lost, "frames", reason))
        age = stats.get("frame_age_ms")
        limit = self.stale_after_ms(_number(stats.get("fps")))
        # No frame at all is stale only once a whole poll went by without the first one.
        is_stale = not first_sight if age is None else _number(age) > limit
        if is_stale and not state.stale:
            detail = f"no new frame for {_number(age):.0f} ms" if age is not None else "no frame yet"
            events.append(
                self._event(EventKind.STALE, device_id, age, "ms", f"{detail} (> {limit:.0f} ms)")
            )
        elif state.stale and not is_stale:
            events.append(
                self._event(EventKind.SETTLED, device_id, age, "ms", "frames are flowing again")
            )
        state.stale, state.failures, state.sequence = is_stale, failures, sequence
        return events

    def _event(
        self, kind: str, device_id: str, value: Any, unit: str, detail: str
    ) -> HardwareEvent:
        return HardwareEvent(
            kind=kind,
            device_id=device_id,
            channel_id=FRAME_HUB_CHANNEL,
            quantity="frame_age" if unit == "ms" else "frame_loss",
            detail=detail,
            value=value,
            unit=unit,
            observed_at=float(self._clock()),
        )


def fold_pump_stats(snapshots: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold several pumps of one device (e.g. one per ROS topic) into one snapshot.

    The device is as stale as its stalest running pump -- one that never delivered
    counts as stalest -- and that pump's rate sets the threshold. Sequences and
    failures add up, so new frames or losses on any pump register. No running pump
    folds to ``{}``, which the tracker reads as idle.
    """
    running = [stats for stats in snapshots if isinstance(stats, Mapping) and stats.get("running")]
    if not running:
        return {}
    stalest = max(
        running,
        key=lambda stats: (stats.get("frame_age_ms") is None, _number(stats.get("frame_age_ms"))),
    )
    return {
        "running": True,
        "sequence": sum(_count(stats.get("sequence")) for stats in running),
        "fps": stalest.get("fps", 0.0),
        "frame_age_ms": stalest.get("frame_age_ms"),
        "failures": sum(_count(stats.get("failures")) for stats in running),
        "last_error": next((str(s["last_error"]) for s in running if s.get("last_error")), ""),
        "pumps": len(running),
    }


def _count(value: Any) -> int:
    number = _number(value)
    return int(number) if number > 0 else 0


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


__all__ = [
    "DEFAULT_STALE_FLOOR_MS",
    "DEFAULT_STALE_FRAMES",
    "FRAME_HUB_CHANNEL",
    "FrameHealthTracker",
    "fold_pump_stats",
]
