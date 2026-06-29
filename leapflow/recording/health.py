"""Recording Health Monitor — real-time degradation detection during learn recording.

Detects and warns about systemic issues that would silently corrupt the
trajectory (e.g., screen capture unavailable, UI event channel dead,
FS noise dominating). Designed to be polled periodically from the learn CLI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

if TYPE_CHECKING:
    from leapflow.perception.session import PerceptionSession
    from leapflow.recording.recorder import DemonstrationRecorder
    from leapflow.domain.trajectory import RecordingMode


@dataclass
class RecordingHealth:
    """Snapshot of recording health at a point in time."""

    warnings: List[str] = field(default_factory=list)
    screen_capture_ok: bool = True
    ui_events_ok: bool = True
    noise_ratio: float = 0.0
    step_count: int = 0
    timestamp: float = 0.0

    @property
    def healthy(self) -> bool:
        return len(self.warnings) == 0


class RecordingHealthMonitor:
    """Monitors recording channels for silent failures during learn sessions.

    Integrates with PerceptionSession (visual channel probe) and
    DemonstrationRecorder (event distribution analysis) to surface
    actionable warnings before a garbage trajectory is committed.
    """

    def __init__(
        self,
        perception: Optional["PerceptionSession"] = None,
        recorder: Optional["DemonstrationRecorder"] = None,
        *,
        visual_enabled: bool = False,
        recording_mode: Optional["RecordingMode"] = None,
        noise_ratio_threshold: float = 0.8,
        noise_min_steps: int = 5,
        ui_stale_threshold_s: float = 30.0,
    ) -> None:
        self._perception = perception
        self._recorder = recorder
        self._visual_enabled = visual_enabled
        self._recording_mode = recording_mode
        self._noise_ratio_threshold = noise_ratio_threshold
        self._noise_min_steps = noise_min_steps
        self._ui_stale_threshold = ui_stale_threshold_s

        self._emitted_warnings: Set[str] = set()
        self._last_check_time: float = 0.0

    def reset(self) -> None:
        """Reset state for a new recording session."""
        self._emitted_warnings.clear()
        self._last_check_time = 0.0

    async def check(self) -> RecordingHealth:
        """Run all health checks and return current health snapshot.

        Only surfaces each unique warning once (keyed by warning category).
        """
        now = time.time()
        self._last_check_time = now
        health = RecordingHealth(timestamp=now)

        await self._check_visual_channel(health)
        self._check_event_distribution(health)

        health.warnings = [
            w for w in health.warnings
            if self._is_new_warning(w)
        ]

        return health

    async def _check_visual_channel(self, health: RecordingHealth) -> None:
        """Probe screen capture availability when visual track is enabled."""
        if not self._visual_enabled or not self._perception:
            return

        channel_status = await self._perception.probe_channel_health()

        if not channel_status.screen_capture_available:
            health.screen_capture_ok = False
            health.warnings.append(
                "Visual track enabled but screen capture unavailable — "
                "frames will not be recorded"
            )

        skip_ui_check = (
            self._recording_mode is not None
            and self._recording_mode.skip_structural_events
        )
        if not skip_ui_check and not channel_status.ui_events_available:
            health.ui_events_ok = False
            health.warnings.append(
                "UI event channel appears inactive — "
                "no UI actions received in the last 30s"
            )

    def _check_event_distribution(self, health: RecordingHealth) -> None:
        """Analyze recorded event distribution for noise dominance."""
        if not self._recorder:
            return

        stats = self._recorder.event_stats
        if not stats:
            return

        total = sum(stats.values())
        health.step_count = total
        if total < self._noise_min_steps:
            return

        fs_count = stats.get("file_create", 0) + stats.get("file_modify", 0)
        if total > 0:
            health.noise_ratio = fs_count / total

        if health.noise_ratio > self._noise_ratio_threshold:
            pct = int(health.noise_ratio * 100)
            health.warnings.append(
                f"Recording appears noise-dominated — "
                f"{pct}% of {total} steps are file system events"
            )

    def _is_new_warning(self, warning: str) -> bool:
        """De-duplicate warnings by category prefix."""
        category = warning.split(" — ")[0] if " — " in warning else warning[:40]
        if category in self._emitted_warnings:
            return False
        self._emitted_warnings.add(category)
        return True
