"""Event signal timeline for video-mode recording.

Collects lightweight event markers during recording.  The compressed
timeline is later injected into VLM prompts as temporal anchors that
guide the model's attention within the video.
"""

from __future__ import annotations

import logging
import os.path
import threading
import time
from typing import Dict, FrozenSet, List, Optional, Protocol, TYPE_CHECKING

from leapflow.perception.types import TimelineMarker

if TYPE_CHECKING:
    from leapflow.domain.events import SystemEvent

logger = logging.getLogger(__name__)

_DEFAULT_MERGE_CHANNELS: FrozenSet[str] = frozenset({"keyboard", "scroll"})


class TimelineWriter(Protocol):
    """Timeline 写入接口 — 用于 EventBus 订阅。"""

    def set_start_time(self, t: float) -> None: ...
    def record_event(self, event: "SystemEvent") -> None: ...
    def clear(self) -> None: ...


class TimelineReader(Protocol):
    """Timeline 读取接口 — 用于 Analyzer 消费。"""

    def markers_in_range(self, start: float, end: float) -> List["TimelineMarker"]: ...
    def compress(self, *, max_markers: Optional[int] = None) -> List["TimelineMarker"]: ...
    def format_for_prompt(self, base_time: float = 0.0, *, max_markers: Optional[int] = None, end_time: Optional[float] = None) -> str: ...

_CHANNEL_MAP: Dict[str, str] = {
    "ui.action": "",
    "app.focus_change": "app_switch",
    "clipboard.change": "clipboard",
    "fs.change": "fs",
}


class SignalTimeline:
    """Thread-safe timeline of event markers aligned to video timestamps.

    Implements both TimelineWriter (for EventBus subscription) and
    TimelineReader (for Analyzer consumption) protocols.
    """

    # fs burst throttle window (seconds)
    _FS_BURST_WINDOW_S: float = 2.0
    # Fraction of fs events to evict when timeline is full
    _EVICT_RATIO: float = 0.2

    def __init__(
        self,
        *,
        max_markers: int = 5000,
        compress_max: int = 200,
        merge_channels: Optional[FrozenSet[str]] = None,
    ) -> None:
        self._markers: List[TimelineMarker] = []
        self._max_markers = max_markers
        self._compress_max = compress_max
        self._merge_channels = merge_channels or _DEFAULT_MERGE_CHANNELS
        self._lock = threading.Lock()
        self._start_time: float = 0.0
        self._overflow_count: int = 0
        # Throttle: maps directory path -> monotonic timestamp of last recorded fs event
        self._fs_burst_window: Dict[str, float] = {}

    def set_start_time(self, t: float) -> None:
        with self._lock:
            self._start_time = t

    def record_event(self, event: "SystemEvent") -> None:
        """Extract a TimelineMarker from a SystemEvent."""
        channel, action, coords, digest = _extract_marker_fields(event)
        if not channel:
            return

        # --- fs burst throttle: skip redundant fs events in the same directory ---
        if channel == "fs":
            dir_path = os.path.dirname(digest) if digest else ""
            now_mono = time.monotonic()
            last_ts = self._fs_burst_window.get(dir_path, 0.0)
            if now_mono - last_ts < self._FS_BURST_WINDOW_S:
                return
            self._fs_burst_window[dir_path] = now_mono
            # Periodically prune stale entries to avoid unbounded growth
            if len(self._fs_burst_window) > 200:
                cutoff = now_mono - self._FS_BURST_WINDOW_S
                self._fs_burst_window = {
                    k: v for k, v in self._fs_burst_window.items() if v > cutoff
                }

        marker = TimelineMarker(
            timestamp=event.timestamp,
            channel=channel,
            action=action,
            app=event.payload.get("app_name", "") or event.payload.get("bundle_id", ""),
            coordinates=coords,
            payload_digest=digest,
            priority=getattr(event, "priority", 5),
        )
        with self._lock:
            if len(self._markers) >= self._max_markers:
                self._evict_old_events()

            # After eviction attempt, check again
            if len(self._markers) >= self._max_markers:
                self._overflow_count += 1
                if self._overflow_count == 1 or self._overflow_count % 100 == 0:
                    logger.warning(
                        "SignalTimeline overflow: %d events dropped "
                        "(max_markers=%d, latest_ts=%.2fs, channel=%s)",
                        self._overflow_count,
                        self._max_markers,
                        marker.timestamp,
                        marker.channel,
                    )
                return
            self._markers.append(marker)

    def _evict_old_events(self) -> None:
        """Evict oldest low-priority (fs) events to make room for new ones.

        Must be called while holding self._lock.
        """
        # Collect indices of fs events (lowest priority channel)
        fs_indices = [i for i, m in enumerate(self._markers) if m.channel == "fs"]
        if not fs_indices:
            return

        # Evict the oldest 20% of fs events
        evict_count = max(1, int(len(fs_indices) * self._EVICT_RATIO))
        # fs_indices are already in insertion order (oldest first)
        indices_to_remove = set(fs_indices[:evict_count])

        self._markers = [
            m for i, m in enumerate(self._markers) if i not in indices_to_remove
        ]

        # Reset overflow counter since we freed space
        self._overflow_count = 0
        logger.info(
            "SignalTimeline compacted: evicted %d oldest fs events (remaining=%d/%d)",
            evict_count,
            len(self._markers),
            self._max_markers,
        )

    def markers_in_range(self, start: float, end: float) -> List[TimelineMarker]:
        with self._lock:
            return [m for m in self._markers if start <= m.timestamp <= end]

    def compress(self, *, max_markers: Optional[int] = None) -> List[TimelineMarker]:
        """Return a compressed timeline suitable for VLM prompt injection.

        Merging strategy:
        - Keep high-priority markers (app_switch, shortcut, clipboard, fs)
        - Merge consecutive keyboard events into one "typed: ..." marker
        - Merge consecutive scroll events into one "scrolled N times" marker
        - Trim to *max_markers* by dropping lowest-priority entries
        """
        cap = max_markers if max_markers is not None else self._compress_max

        with self._lock:
            raw = list(self._markers)
        if not raw:
            return []

        merged: List[TimelineMarker] = []
        run_channel: Optional[str] = None
        run_items: List[TimelineMarker] = []

        for m in raw:
            if m.channel in self._merge_channels:
                if m.channel == run_channel:
                    run_items.append(m)
                    continue
                # Channel switched within mergeable group — flush the old run
                if run_items:
                    merged.append(self._merge_run(run_channel, run_items))
                run_channel = m.channel
                run_items = [m]
            else:
                if run_items:
                    merged.append(self._merge_run(run_channel, run_items))
                    run_channel = None
                    run_items = []
                merged.append(m)

        if run_items:
            merged.append(self._merge_run(run_channel, run_items))

        if len(merged) > cap:
            # Drop lowest-priority entries first, then restore chronological order
            merged.sort(key=lambda m: m.priority)
            merged = merged[:cap]
            merged.sort(key=lambda m: m.timestamp)

        return merged

    @staticmethod
    def _merge_run(
        channel: Optional[str],
        run: List[TimelineMarker],
    ) -> TimelineMarker:
        """Collapse a run of same-channel markers into a single marker.

        Preserves the earliest timestamp and the highest priority found in the
        run (lower numeric priority value = higher importance, matching the
        existing convention in :func:`compress`).
        """
        first = run[0]
        # Highest priority => smallest priority value
        top_priority = min(m.priority for m in run)

        if channel == "keyboard":
            text = "".join(m.payload_digest for m in run)
            action = f'typed: "{text[:120]}" x{len(run)}' if len(run) > 1 else f'typed: "{text[:120]}"'
        elif channel == "scroll":
            action = f"scroll x{len(run)}"
        else:
            action = first.action

        return TimelineMarker(
            timestamp=first.timestamp,
            channel=channel or first.channel,
            action=action,
            app=first.app,
            coordinates=first.coordinates,
            payload_digest=first.payload_digest,
            priority=top_priority,
        )

    def format_for_prompt(
        self,
        base_time: float = 0.0,
        *,
        max_markers: Optional[int] = None,
        end_time: Optional[float] = None,
    ) -> str:
        """Format compressed timeline as human-readable text for VLM prompts."""
        markers = self.compress(max_markers=max_markers)
        if end_time is not None:
            markers = [m for m in markers if base_time <= m.timestamp <= end_time]
        if not markers:
            return "(no events recorded)"

        with self._lock:
            bt = base_time or self._start_time
        lines: List[str] = []
        for m in markers:
            rel = max(0.0, m.timestamp - bt)
            mins, secs = divmod(rel, 60)
            ts = f"[{int(mins):02d}:{secs:05.2f}]"
            coord_str = f" at ({m.coordinates[0]},{m.coordinates[1]})" if m.coordinates else ""
            app_str = f" in {m.app}" if m.app else ""
            lines.append(f"  {ts} {m.channel}: {m.action}{coord_str}{app_str}")
        return "\n".join(lines)

    def clear(self) -> None:
        with self._lock:
            self._markers.clear()
            self._overflow_count = 0


def _extract_marker_fields(event: "SystemEvent"):
    """Derive (channel, action, coords, digest) from a SystemEvent."""
    et = event.event_type
    p = event.payload

    if et == "ui.action":
        sub = p.get("sub_action", p.get("action", ""))
        channel = sub or "ui"
        action = sub
        coords = None
        if "x" in p and "y" in p:
            coords = (int(p["x"]), int(p["y"]))
        digest = ""
        if sub == "type":
            digest = p.get("text", p.get("chars", ""))[:50]
        elif sub == "shortcut":
            digest = p.get("shortcut", p.get("key", ""))
        return channel, action, coords, digest

    mapped = _CHANNEL_MAP.get(et)
    if mapped is not None:
        channel = mapped or et.split(".")[0]
        action = p.get("sub_action", p.get("action", et))
        digest = ""
        if et == "clipboard.change":
            digest = str(p.get("text", ""))[:50]
        elif et == "fs.change":
            digest = p.get("path", "")
        elif et == "app.focus_change":
            action = p.get("app_name", p.get("bundle_id", "app_switch"))
        return channel, action, None, digest

    return None, "", None, ""
