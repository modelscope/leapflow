"""Semantic relevance scoring with platform-aware weight profiles."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict

from leapflow.domain.events import SystemEvent
from leapflow.domain.platform import PlatformID, PlatformManifest


@dataclass(frozen=True)
class WeightProfile:
    """Platform-specific event weight configuration."""

    type_weights: Dict[str, float]
    recency_half_life: float = 60.0
    platform_boost: Dict[str, float] = field(default_factory=dict)


WEIGHT_PROFILES: Dict[PlatformID, WeightProfile] = {
    PlatformID.DARWIN_15: WeightProfile(
        type_weights={
            "fs.change": 0.7,
            "clipboard.change": 0.8,
            "app.focus_change": 0.6,
        },
        platform_boost={"app.focus_change": 0.15},
    ),
    PlatformID.DARWIN_26: WeightProfile(
        type_weights={
            "fs.change": 0.6,
            "clipboard.change": 0.7,
            "intent.signal": 0.95,
            "app.focus_change": 0.4,
        },
        platform_boost={"intent.signal": 0.3},
        recency_half_life=45.0,
    ),
    PlatformID.LINUX_GNOME: WeightProfile(
        type_weights={
            "fs.change": 0.75,
            "clipboard.change": 0.6,
            "app.focus_change": 0.5,
            "shell.output": 0.85,
        },
        platform_boost={"shell.output": 0.2},
    ),
}

_DEFAULT_PROFILE = WeightProfile(type_weights={})


def compute_relevance(
    event: SystemEvent,
    manifest: PlatformManifest,
    now: float | None = None,
) -> float:
    """Compute normalized relevance score in [0, 1] range.

    Formula: S = w_type + w_recency + w_platform_boost
    Clamped to [0.0, 1.0].
    """
    now = now or time.time()
    profile = WEIGHT_PROFILES.get(manifest.platform_id, _DEFAULT_PROFILE)

    w_type = profile.type_weights.get(event.event_type, 0.5)
    delta_t = max(now - event.timestamp, 0.01)
    w_recency = 0.5 ** (delta_t / profile.recency_half_life)
    w_boost = profile.platform_boost.get(event.event_type, 0.0)

    score = w_type * 0.5 + w_recency * 0.3 + w_boost * 0.2
    return max(0.0, min(1.0, score))
