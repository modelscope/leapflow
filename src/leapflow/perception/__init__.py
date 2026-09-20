# Copyright (c) Alibaba, Inc. and its affiliates.
"""Visual-First Perception Subsystem.

Supports two modes:
  - Screenshot (legacy): adaptive keyframe capture + VLM extraction
  - Video (recommended): continuous recording + multi-scale VLM analysis

Core types are re-exported from perception.types; video components
from perception.video.
"""

from leapflow.perception.active_signal_source import (
    ActiveSignalSource,
    ActiveSourceManager,
    EmitCallback,
)
from leapflow.perception.config import PerceptionConfig, SamplingConfig, ScorerConfig
from leapflow.perception.environment_source import EnvironmentSource, EnvironmentSourceManager
from leapflow.perception.session import PerceptionSession
from leapflow.perception.signal_source import (
    SignalSource,
    SignalSourceRegistry,
    SignalTransformContext,
)
from leapflow.perception.signal_sources_builtin import build_default_signal_source_registry
from leapflow.perception.types import (
    ChannelStatus,
    InteractionSignal,
    Keyframe,
    MacroAnalysisResult,
    TimelineMarker,
    VideoAction,
    VideoSegment,
    VisualAction,
)

__all__ = [
    "ActiveSignalSource",
    "ActiveSourceManager",
    "DiscordBotSignalSource",
    "EmitCallback",
    "EnvironmentSource",
    "EnvironmentSourceManager",
    "FeishuIMSignalSource",
    "FileWatchSignalSource",
    "SlackBotSignalSource",
    "PerceptionConfig",
    "PerceptionSession",
    "SamplingConfig",
    "ScorerConfig",
    "SignalSource",
    "SignalSourceRegistry",
    "SignalTransformContext",
    "build_default_signal_source_registry",
    "ChannelStatus",
    "InteractionSignal",
    "Keyframe",
    "MacroAnalysisResult",
    "TelegramBotSignalSource",
    "TimelineMarker",
    "VideoAction",
    "VideoSegment",
    "VisualAction",
]


def __getattr__(name: str):  # noqa: N807
    """Lazy import for heavy ActiveSignalSource implementations."""
    if name == "FileWatchSignalSource":
        from leapflow.perception.active_sources_builtin import FileWatchSignalSource
        return FileWatchSignalSource
    if name == "FeishuIMSignalSource":
        from leapflow.perception.active_sources.feishu_im import FeishuIMSignalSource
        return FeishuIMSignalSource
    if name == "TelegramBotSignalSource":
        from leapflow.perception.active_sources.telegram_bot import TelegramBotSignalSource
        return TelegramBotSignalSource
    if name == "SlackBotSignalSource":
        from leapflow.perception.active_sources.slack_bot import SlackBotSignalSource
        return SlackBotSignalSource
    if name == "DiscordBotSignalSource":
        from leapflow.perception.active_sources.discord_bot import DiscordBotSignalSource
        return DiscordBotSignalSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
