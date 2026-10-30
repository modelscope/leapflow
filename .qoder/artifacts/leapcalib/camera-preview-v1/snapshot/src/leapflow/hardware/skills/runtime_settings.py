# Copyright (c) Alibaba, Inc. and its affiliates.
"""Node-side skill policy: which skills run, their caps, and the runtime's timeouts."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from leapflow.hardware.skills.episodes import EpisodeSettings

DEFAULT_PLAN_TTL_S = 300.0


@dataclass(frozen=True)
class SkillRuntimeSettings:
    """Node-side skill policy, read once when the node starts."""

    enabled: tuple[str, ...] = ("*",)
    speed_pct_max: float = 30.0
    max_duration_s: float = 120.0
    workcells_dir: Path | None = None
    plan_ttl_s: float = DEFAULT_PLAN_TTL_S
    halt_timeout_s: float = 2.0  # one actuator's halt call
    stop_grace_s: float = 1.0  # a stopped skill's unwind before it is cancelled
    dispatch_settle_s: float = 2.0  # in-flight writes reconciled before release
    episodes: EpisodeSettings = field(default_factory=EpisodeSettings)

    @classmethod
    def from_settings(cls, settings: Any) -> "SkillRuntimeSettings":
        raw = str(getattr(settings, "hardware_skills_enabled", "") or "")
        patterns = tuple(item.strip() for item in raw.split(",") if item.strip())
        layout = getattr(settings, "profile_layout", None)
        return cls(
            enabled=patterns or ("*",),
            speed_pct_max=float(getattr(settings, "hardware_skills_speed_pct_max", 30.0)),
            max_duration_s=float(getattr(settings, "hardware_skills_max_duration_s", 120.0)),
            workcells_dir=layout.hardware.workcells_dir if layout is not None else None,
            plan_ttl_s=float(getattr(settings, "hardware_skills_plan_ttl_s", DEFAULT_PLAN_TTL_S)),
            halt_timeout_s=float(getattr(settings, "hardware_skills_halt_timeout_s", 2.0)),
            stop_grace_s=float(getattr(settings, "hardware_skills_stop_grace_s", 1.0)),
            dispatch_settle_s=float(getattr(settings, "hardware_skills_dispatch_settle_s", 2.0)),
            episodes=EpisodeSettings(
                directory=layout.hardware.episodes_dir if layout is not None else None,
                fps=float(getattr(settings, "hardware_skills_episode_fps", 30.0)),
                min_free_mb=float(getattr(settings, "hardware_skills_episode_min_free_mb", 2048.0)),
            ),
        )

    def admits(self, skill_id: str) -> bool:
        return any(fnmatch.fnmatchcase(skill_id, pattern) for pattern in self.enabled)


__all__ = ["DEFAULT_PLAN_TTL_S", "SkillRuntimeSettings"]
