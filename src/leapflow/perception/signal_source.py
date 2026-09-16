# Copyright (c) Alibaba, Inc. and its affiliates.
"""SignalSource plugin protocol and registry for Perception signal extraction.

A SignalSource transforms a normalized SystemEvent (event_type + payload) into
an optional InteractionSignal. This replaces the hardcoded if-chain in
PerceptionSession._extract_signal() with a pluggable registry, enabling
community-contributed signal channels without modifying core perception code.

Design: transform-only (stateless). Sources are pure functions of
(event_type, payload, context). EventBus subscription and signal destinations
(SignalBuffer, CausalFusionPipeline) remain owned by PerceptionSession.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Protocol, runtime_checkable

from leapflow.perception.types import InteractionSignal


@dataclass(frozen=True)
class SignalTransformContext:
    """Immutable context passed to signal sources during transformation.

    Carries the gating state that sources need to reproduce the exact
    privacy/mode/channel behavior of the original _extract_signal().
    """
    now: float
    prev_app: str
    current_app: str
    enabled_channels: FrozenSet[str]
    privacy_sensitive_apps: FrozenSet[str]


@runtime_checkable
class SignalSource(Protocol):
    """Protocol for a signal source that transforms events into InteractionSignals.

    Design note:
        SignalSource is intentionally stateless and NOT managed by PluginFiber /
        EffectScope. It is a pure transform-only plugin category: EventBus
        subscription and resource lifecycle stay in PerceptionSession. This keeps
        the real-time signal path minimal (no per-event lifecycle overhead) and
        reserves Fiber lifecycle management for plugins that own external resources.

        Sources that need to subscribe to external event streams or hold resources
        (IM listeners, file watchers, IoT devices) should be implemented as a future
        ActiveSignalSource category that integrates with the EffectScope/PluginFiber
        lifecycle — NOT by adding lifecycle to this transform-only protocol.
    """

    @property
    def channel_id(self) -> str:
        """Primary channel identifier (e.g. 'click', 'app_switch')."""
        ...

    @property
    def event_types(self) -> FrozenSet[str]:
        """Set of raw event_types this source handles (e.g. {'ui.action'})."""
        ...

    @property
    def bypasses_privacy(self) -> bool:
        """Whether this source is allowed before privacy-sensitive suppression.

        Only app_switch bypasses privacy in current behavior.
        """
        ...

    def transform(
        self,
        event_type: str,
        payload: Dict[str, Any],
        context: SignalTransformContext,
    ) -> Optional[InteractionSignal]:
        """Transform an event into a signal, or return None if not applicable."""
        ...


class SignalSourceRegistry:
    """Registry of signal sources. Dispatches events to matching sources.

    Preserves the original single-signal-per-event behavior via 'first non-None
    wins' ordering.
    """

    def __init__(self) -> None:
        self._sources: list[SignalSource] = []

    def register(self, source: SignalSource) -> None:
        """Register a signal source."""
        self._sources.append(source)

    def sources_for(self, event_type: str) -> list[SignalSource]:
        """Return sources that declare interest in the given event_type."""
        return [s for s in self._sources if event_type in s.event_types]

    def transform_first(
        self,
        event_type: str,
        payload: Dict[str, Any],
        context: SignalTransformContext,
    ) -> Optional[InteractionSignal]:
        """Return the first non-None signal from matching sources.

        Reproduces the original _extract_signal() semantics (BEHAVIOR CONTRACT — do not
        change without updating downstream consumers and tests):
            - Sources with bypasses_privacy=True (app_switch) are evaluated BEFORE the
              privacy gate, so app switches are recorded even for privacy-sensitive apps
              (they carry no sensitive content).
            - For all other sources: if current_app is in privacy_sensitive_apps, NO
              signal is emitted (suppression).
            - Only ONE InteractionSignal is emitted per event (first non-None wins),
              following source registration order.
        """
        matching = self.sources_for(event_type)

        # Privacy-bypassing sources first (app_switch)
        for source in matching:
            if source.bypasses_privacy:
                sig = source.transform(event_type, payload, context)
                if sig is not None:
                    return sig

        # Privacy gate: skip remaining sources if current app is sensitive
        if context.current_app in context.privacy_sensitive_apps:
            return None

        # Non-bypassing sources
        for source in matching:
            if not source.bypasses_privacy:
                sig = source.transform(event_type, payload, context)
                if sig is not None:
                    return sig

        return None

    @property
    def sources(self) -> list[SignalSource]:
        """Read-only view of registered sources."""
        return list(self._sources)

    @property
    def channel_ids(self) -> FrozenSet[str]:
        """All channel_ids of registered sources."""
        return frozenset(s.channel_id for s in self._sources)
