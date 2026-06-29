"""Channel abstraction: table-driven behavior for all signal channels.

ChannelSpec defines a channel's causal role, aggregation policy, and
degradation behavior. ChannelRegistry is the single source of truth —
all downstream components (denoiser, chain builder, mapper) read from
the registry rather than hardcoding channel-specific logic.

Adding a new channel (voice, eye_tracking, touch) requires only:
    1. Define a ChannelSpec with semantic_prior, description_template, etc.
    2. Register it with ChannelRegistry (optionally with event_type_aliases)
No builder/denoiser/mapper code changes needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from leapflow.causal.types import CausalEvent, EventType

logger = logging.getLogger(__name__)


class AggregationPolicy(str, Enum):
    """How to aggregate bursts of same-channel events."""

    NONE = "none"
    TEMPORAL_WINDOW = "temporal_window"
    DIRECTIONAL = "directional"
    COMBO = "combo"


@dataclass(frozen=True)
class ChannelSpec:
    """Declarative specification for a signal channel.

    Adding a channel only requires creating a new :class:`ChannelSpec` and
    registering it; downstream components (denoiser, chain builder, mapper,
    adapters) read from the registry rather than hard-coding channel names.

    Attributes
    ----------
    description_template:
        Long-form template for verbose VLM prompts (used by
        :meth:`ChannelRegistry.describe_event`).
    short_template:
        Compact template used by causal adapters when summarising chains for
        token-budgeted prompts. Falls back to the channel name if empty or
        unresolvable.
    """

    channel: str
    default_role: EventType
    reactive_capture: bool = False
    privacy_level: int = 0
    aggregation: AggregationPolicy = AggregationPolicy.NONE
    aggregation_window_ms: float = 0.0
    fallback_inference: str = ""
    semantic_prior: Dict[str, float] = field(default_factory=dict)
    description_template: str = ""
    short_template: str = ""
    degraded_confidence_factor: float = 0.6
    detail_payload_key: str = "detail"


RoleResolver = Callable[[CausalEvent, Sequence[CausalEvent]], EventType]


class _SafeFormatDict(dict):
    """Dict that returns '?' for missing keys in str.format_map()."""

    def __missing__(self, key: str) -> str:
        return "?"


class ChannelRegistry:
    """Central registry for channel behavior. All access is table-driven."""

    __slots__ = ("_specs", "_resolvers", "_availability", "_event_type_map")

    def __init__(self) -> None:
        self._specs: Dict[str, ChannelSpec] = {}
        self._resolvers: Dict[str, RoleResolver] = {}
        self._availability: Dict[str, bool] = {}
        self._event_type_map: Dict[str, str] = {}

    def register(
        self,
        spec: ChannelSpec,
        role_resolver: Optional[RoleResolver] = None,
        event_type_aliases: Optional[List[str]] = None,
    ) -> None:
        self._specs[spec.channel] = spec
        self._availability[spec.channel] = True
        if role_resolver:
            self._resolvers[spec.channel] = role_resolver
        if event_type_aliases:
            for alias in event_type_aliases:
                self._event_type_map[alias] = spec.channel

    def resolve_role(self, event: CausalEvent, recent: Sequence[CausalEvent]) -> EventType:
        resolver = self._resolvers.get(event.channel)
        if resolver:
            return resolver(event, recent)
        spec = self._specs.get(event.channel)
        if spec:
            return spec.default_role
        return EventType.NOISE

    def get_spec(self, channel: str) -> Optional[ChannelSpec]:
        return self._specs.get(channel)

    def is_available(self, channel: str) -> bool:
        return self._availability.get(channel, False)

    def set_available(self, channel: str, available: bool) -> None:
        self._availability[channel] = available

    def fallback_inference(self, channel: str) -> str:
        spec = self._specs.get(channel)
        return spec.fallback_inference if spec else ""

    def get_semantic_prior(self, parent_channel: str, child_channel: str) -> float:
        spec = self._specs.get(parent_channel)
        if spec:
            return spec.semantic_prior.get(child_channel, 0.3)
        return 0.3

    def aggregation_policy(self, channel: str) -> AggregationPolicy:
        spec = self._specs.get(channel)
        return spec.aggregation if spec else AggregationPolicy.NONE

    def aggregation_window(self, channel: str) -> float:
        spec = self._specs.get(channel)
        return (spec.aggregation_window_ms / 1000.0) if spec else 0.0

    def requires_reactive_capture(self, channel: str) -> bool:
        spec = self._specs.get(channel)
        return spec.reactive_capture if spec else False

    def resolve_channel(self, event_type_str: str) -> str:
        """Map a system event type string to a channel name via registered aliases."""
        return self._event_type_map.get(event_type_str, event_type_str)

    def describe_event(self, channel: str, payload: Dict[str, Any]) -> str:
        """Generate description from channel's template and event payload."""
        spec = self._specs.get(channel)
        if not spec or not spec.description_template:
            return channel
        return self._render_template(spec.description_template, channel, payload)

    def describe_event_short(self, channel: str, payload: Dict[str, Any]) -> str:
        """Generate a compact description from the channel's ``short_template``.

        Falls back to the channel name when no short template is configured
        or template substitution fails.
        """
        spec = self._specs.get(channel)
        if not spec or not spec.short_template:
            return channel
        return self._render_template(spec.short_template, channel, payload)

    def _render_template(self, template: str, channel: str, payload: Dict[str, Any]) -> str:
        try:
            enriched = dict(payload)
            if "delta_y" in enriched:
                enriched.setdefault("direction", "down" if enriched["delta_y"] > 0 else "up")
            if "from_bundle" in enriched:
                enriched.setdefault("from", enriched["from_bundle"])
            if "to_bundle" in enriched:
                enriched.setdefault("to", enriched["to_bundle"])
            # Parse "A -> B" detail into from/to for structured templates
            if "detail" in enriched and "from" not in enriched:
                detail = str(enriched["detail"])
                if " -> " in detail:
                    parts = detail.split(" -> ", 1)
                    enriched.setdefault("from", parts[0])
                    enriched.setdefault("to", parts[1])
            if "text" in enriched:
                enriched["text"] = str(enriched["text"])[:50]
            return template.format_map(_SafeFormatDict(enriched))
        except (KeyError, ValueError):
            return channel

    @property
    def channels(self) -> List[str]:
        return list(self._specs.keys())

    def all_specs(self) -> List[ChannelSpec]:
        """Return all registered ChannelSpec instances."""
        return list(self._specs.values())

    @property
    def available_channels(self) -> List[str]:
        return [ch for ch, avail in self._availability.items() if avail]


# ── Default registry factory ──

def _resolve_app_switch_role(event: CausalEvent, recent: Sequence[CausalEvent]) -> EventType:
    """app_switch is RESPONSE if a trigger preceded within 500ms, else BOUNDARY."""
    for prev in reversed(recent):
        dt = event.timestamp - prev.timestamp
        if dt > 0.5:
            break
        if prev.event_type == EventType.TRIGGER:
            return EventType.RESPONSE
    return EventType.BOUNDARY


def _resolve_click_role(event: CausalEvent, recent: Sequence[CausalEvent]) -> EventType:
    """Click on a system dialog is RESPONSE; otherwise TRIGGER."""
    if recent and recent[-1].tags.get("dialog_open"):
        return EventType.RESPONSE
    return EventType.TRIGGER


def build_default_registry() -> ChannelRegistry:
    """Construct the standard 7-channel registry."""
    registry = ChannelRegistry()

    registry.register(ChannelSpec(
        channel="click",
        default_role=EventType.TRIGGER,
        reactive_capture=True,
        aggregation=AggregationPolicy.NONE,
        semantic_prior={"visual_change": 0.95, "app_switch": 0.4, "clipboard": 0.3},
        description_template="Click at ({x}, {y})",
        short_template="click({x},{y})",
    ), role_resolver=_resolve_click_role, event_type_aliases=["ui.click"])

    registry.register(ChannelSpec(
        channel="keyboard",
        default_role=EventType.TRIGGER,
        reactive_capture=False,
        aggregation=AggregationPolicy.COMBO,
        aggregation_window_ms=100.0,
        semantic_prior={"visual_change": 0.85, "clipboard": 0.9, "app_switch": 0.8},
        description_template="Key: {combo}",
        short_template="{combo}",
        detail_payload_key="combo",
    ), event_type_aliases=["ui.type", "ui.shortcut"])

    registry.register(ChannelSpec(
        channel="drag",
        default_role=EventType.TRIGGER,
        reactive_capture=True,
        aggregation=AggregationPolicy.NONE,
        semantic_prior={"visual_change": 0.9},
        description_template="Drag ({start_x},{start_y})→({end_x},{end_y})",
        short_template="drag",
    ), event_type_aliases=["ui.drag"])

    registry.register(ChannelSpec(
        channel="app_switch",
        default_role=EventType.BOUNDARY,
        reactive_capture=True,
        aggregation=AggregationPolicy.NONE,
        semantic_prior={"visual_change": 0.9},
        description_template="App: {from}→{to}",
        short_template="switch→{to}",
    ), role_resolver=_resolve_app_switch_role, event_type_aliases=["app.focus_change"])

    registry.register(ChannelSpec(
        channel="scroll",
        default_role=EventType.NAVIGATION,
        reactive_capture=False,
        aggregation=AggregationPolicy.DIRECTIONAL,
        aggregation_window_ms=200.0,
        semantic_prior={"visual_change": 0.7},
        description_template="Scroll {direction}",
        short_template="scroll",
    ), event_type_aliases=["ui.scroll"])

    registry.register(ChannelSpec(
        channel="clipboard",
        default_role=EventType.EFFECT,
        reactive_capture=False,
        privacy_level=1,
        aggregation=AggregationPolicy.NONE,
        semantic_prior={},
        description_template="Clipboard updated",
        short_template="clipboard",
    ), event_type_aliases=["clipboard.change"])

    registry.register(ChannelSpec(
        channel="clipboard_content",
        default_role=EventType.EFFECT,
        reactive_capture=False,
        privacy_level=2,
        aggregation=AggregationPolicy.NONE,
        fallback_inference="ocr",
        semantic_prior={},
        description_template="Clipboard: \"{text}\"",
        short_template="clipboard",
    ))

    registry.register(ChannelSpec(
        channel="visual_change",
        default_role=EventType.RESPONSE,
        reactive_capture=False,
        aggregation=AggregationPolicy.NONE,
        semantic_prior={},
        description_template="Visual change detected",
        short_template="visual_diff",
    ))

    return registry
