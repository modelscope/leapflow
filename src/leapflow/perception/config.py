"""Perception subsystem configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from leapflow.config import Settings


@dataclass(frozen=True)
class ScorerConfig:
    """Weights and parameters for InformationGainScorer."""

    weights: Dict[str, float] = field(default_factory=lambda: {
        "novelty": 0.30,
        "event_density": 0.25,
        "hunger": 0.15,
        "structural": 0.15,
        "alignment": 0.15,
    })
    max_silent_s: float = 5.0


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration for the complete sampling engine."""

    min_interval_ms: int = 200
    max_fps: float = 5.0
    frame_budget: int = 60
    session_estimate_s: float = 120.0
    scorer: ScorerConfig = field(default_factory=ScorerConfig)

    # Event-driven sampling & decision cache
    event_driven_interval_ms: int = 50    # 事件驱动采样的最小间隔（ms）
    decision_cache_ttl_ms: int = 100      # 决策缓存有效期（ms）

    # Animation filter
    animation_window_size: int = 10
    animation_continuous_threshold: float = 0.7

    # Change detection thresholds
    global_diff_threshold: float = 0.35
    quadrant_diff_threshold: float = 0.20
    focus_diff_threshold: float = 0.15


@dataclass(frozen=True)
class PerceptionConfig:
    """Top-level perception module configuration."""

    enabled: bool = False
    frame_cache_dir: Path = field(default_factory=lambda: Path("~/.leapflow/profiles/default/cache/frames"))
    privacy_sensitive_apps: FrozenSet[str] = field(default_factory=frozenset)

    # Sampling
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # VLM
    vlm_model: str = ""
    vlm_api_key: str = ""
    vlm_base_url: str = ""

    # Encoding
    compression_max_resolution: int = 1024
    compression_quality: int = 75
    compression_adaptive: bool = True
    tiling_enabled: bool = True
    tiling_max_frames: int = 4
    tiling_tile_size: int = 384
    tiling_gap: int = 4

    # Cache
    cache_enabled: bool = True
    cache_ttl: float = 300.0
    cache_max_size: int = 1000

    # Semantic cache
    semantic_cache_ttl_days: int = 7

    # Signal Fusion
    signal_channels: FrozenSet[str] = field(default_factory=frozenset)
    signal_reactive_capture: bool = False
    signal_reactive_min_interval: float = 0.3
    signal_reactive_triggers: FrozenSet[str] = field(default_factory=frozenset)

    @classmethod
    def from_settings(cls, settings: "Settings") -> "PerceptionConfig":
        """Construct PerceptionConfig from the global Settings object."""
        return cls(
            enabled=settings.visual_track_enabled,
            frame_cache_dir=settings.visual_frame_cache_dir,
            privacy_sensitive_apps=frozenset(settings.privacy_sensitive_apps),
            vlm_model=settings.vlm_model or settings.llm_model,
            vlm_api_key=settings.vlm_api_key or settings.llm_api_key,
            vlm_base_url=settings.vlm_base_url or settings.llm_base_url,
            compression_max_resolution=settings.vlm_compression_max_resolution,
            compression_quality=settings.vlm_compression_quality,
            compression_adaptive=settings.vlm_compression_adaptive,
            tiling_enabled=settings.vlm_tiling_enabled,
            tiling_max_frames=settings.vlm_tiling_max_frames,
            tiling_tile_size=settings.vlm_tiling_tile_size,
            tiling_gap=settings.vlm_tiling_gap,
            cache_enabled=settings.vlm_cache_enabled,
            cache_ttl=settings.vlm_cache_ttl,
            cache_max_size=settings.vlm_cache_max_size,
            signal_channels=frozenset(settings.signal_channels),
            signal_reactive_capture=settings.signal_reactive_capture,
        )
