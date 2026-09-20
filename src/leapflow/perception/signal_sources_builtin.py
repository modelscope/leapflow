# Copyright (c) Alibaba, Inc. and its affiliates.
"""Built-in signal sources reproducing the original _extract_signal() branches.

Each source encapsulates a single branch of the original hardcoded if-chain in
PerceptionSession._extract_signal(). Behavior is byte-for-byte identical:
field extraction, coercions, truncation, and detail formatting match the
original code exactly. The registry factory registers sources in the same
order the original chain used.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Optional

from leapflow.perception.signal_source import (
    SignalSourceRegistry,
    SignalTransformContext,
)
from leapflow.perception.types import InteractionSignal


class AppSwitchSignalSource:
    """Emits an ``app_switch`` signal on ``app.focus_change`` events.

    This is the only source that bypasses the privacy-sensitive-app gate,
    since a bundle-id transition carries no sensitive content.
    """

    @property
    def channel_id(self) -> str:
        return "app_switch"

    @property
    def event_types(self) -> FrozenSet[str]:
        return frozenset({"app.focus_change"})

    @property
    def bypasses_privacy(self) -> bool:
        return True

    def transform(
        self,
        event_type: str,
        payload: Dict[str, Any],
        context: SignalTransformContext,
    ) -> Optional[InteractionSignal]:
        if "app_switch" not in context.enabled_channels:
            return None
        new_app = payload.get("bundle_id", "")
        return InteractionSignal(
            timestamp=context.now,
            signal_type="app_switch",
            app=new_app,
            detail=f"{context.prev_app} -> {new_app}",
        )


class ClickSignalSource:
    """Emits a ``click`` signal on ``ui.action`` events with sub_type=='click'."""

    @property
    def channel_id(self) -> str:
        return "click"

    @property
    def event_types(self) -> FrozenSet[str]:
        return frozenset({"ui.action"})

    @property
    def bypasses_privacy(self) -> bool:
        return False

    def transform(
        self,
        event_type: str,
        payload: Dict[str, Any],
        context: SignalTransformContext,
    ) -> Optional[InteractionSignal]:
        if payload.get("sub_type", "") != "click":
            return None
        if "click" not in context.enabled_channels:
            return None
        return InteractionSignal(
            timestamp=context.now,
            signal_type="click",
            app=payload.get("app_bundle_id", "") or context.current_app,
            position=(
                int(payload.get("mouse_x", 0)),
                int(payload.get("mouse_y", 0)),
            ),
        )


class ScrollSignalSource:
    """Emits a ``scroll`` signal on ``ui.action`` events with sub_type=='scroll'."""

    @property
    def channel_id(self) -> str:
        return "scroll"

    @property
    def event_types(self) -> FrozenSet[str]:
        return frozenset({"ui.action"})

    @property
    def bypasses_privacy(self) -> bool:
        return False

    def transform(
        self,
        event_type: str,
        payload: Dict[str, Any],
        context: SignalTransformContext,
    ) -> Optional[InteractionSignal]:
        if payload.get("sub_type", "") != "scroll":
            return None
        if "scroll" not in context.enabled_channels:
            return None
        return InteractionSignal(
            timestamp=context.now,
            signal_type="scroll",
            app=payload.get("app_bundle_id", "") or context.current_app,
            position=(
                int(payload.get("mouse_x", 0)),
                int(payload.get("mouse_y", 0)),
            ),
            detail=f"dy={payload.get('delta_y', 0)}",
        )


class KeyboardShortcutSignalSource:
    """Emits a ``keyboard`` signal on ``ui.action`` events with sub_type=='shortcut'.

    Detail is the '+'-joined modifiers + optional char, matching the original.
    """

    @property
    def channel_id(self) -> str:
        return "keyboard"

    @property
    def event_types(self) -> FrozenSet[str]:
        return frozenset({"ui.action"})

    @property
    def bypasses_privacy(self) -> bool:
        return False

    def transform(
        self,
        event_type: str,
        payload: Dict[str, Any],
        context: SignalTransformContext,
    ) -> Optional[InteractionSignal]:
        if payload.get("sub_type", "") != "shortcut":
            return None
        if "keyboard" not in context.enabled_channels:
            return None
        modifiers = payload.get("modifiers", [])
        char = payload.get("char", "")
        combo = "+".join(modifiers + ([char] if char else []))
        return InteractionSignal(
            timestamp=context.now,
            signal_type="keyboard",
            app=payload.get("app_bundle_id", "") or context.current_app,
            detail=combo,
        )


class KeyboardTypeSignalSource:
    """Emits a ``keyboard`` signal on ``ui.action`` events with sub_type=='type'.

    Text is truncated to 50 characters, matching the original.
    """

    @property
    def channel_id(self) -> str:
        return "keyboard"

    @property
    def event_types(self) -> FrozenSet[str]:
        return frozenset({"ui.action"})

    @property
    def bypasses_privacy(self) -> bool:
        return False

    def transform(
        self,
        event_type: str,
        payload: Dict[str, Any],
        context: SignalTransformContext,
    ) -> Optional[InteractionSignal]:
        if payload.get("sub_type", "") != "type":
            return None
        if "keyboard" not in context.enabled_channels:
            return None
        text = str(payload.get("text", ""))[:50]
        return InteractionSignal(
            timestamp=context.now,
            signal_type="keyboard",
            app=payload.get("app_bundle_id", "") or context.current_app,
            detail=f"type:{text}",
        )


class DragSignalSource:
    """Emits a ``drag`` signal on ``ui.action`` events with sub_type=='drag'."""

    @property
    def channel_id(self) -> str:
        return "drag"

    @property
    def event_types(self) -> FrozenSet[str]:
        return frozenset({"ui.action"})

    @property
    def bypasses_privacy(self) -> bool:
        return False

    def transform(
        self,
        event_type: str,
        payload: Dict[str, Any],
        context: SignalTransformContext,
    ) -> Optional[InteractionSignal]:
        if payload.get("sub_type", "") != "drag":
            return None
        if "drag" not in context.enabled_channels:
            return None
        return InteractionSignal(
            timestamp=context.now,
            signal_type="drag",
            app=payload.get("app_bundle_id", "") or context.current_app,
            position=(
                int(payload.get("start_x", 0)),
                int(payload.get("start_y", 0)),
            ),
            end_position=(
                int(payload.get("end_x", 0)),
                int(payload.get("end_y", 0)),
            ),
        )


class ClipboardSignalSource:
    """Emits a ``clipboard`` signal on ``clipboard.change`` events.

    Two channels feed one signal_type ('clipboard'):
    - ``clipboard_content`` (preferred): detail='content:<text[:200]>'
    - ``clipboard`` (fallback): detail=payload['change_type']

    Reproduces the original elif exactly: if both channels are enabled, the
    content branch wins.

    Behavior contract (preserved from the original _extract_signal):
        - When BOTH "clipboard_content" and "clipboard" channels are enabled, the
          content branch ALWAYS wins (content-when-available precedence).
        - signal_type is ALWAYS "clipboard" (never "clipboard_content"), even when
          the clipboard_content channel triggered it — downstream consumers rely on
          this. Content text is truncated to 200 chars.
    """

    # Note: channel_id names the *primary* channel this source announces to
    # the registry catalog. Actual gating is performed inside transform() so
    # both 'clipboard_content' and 'clipboard' can activate this single source.
    @property
    def channel_id(self) -> str:
        return "clipboard"

    @property
    def event_types(self) -> FrozenSet[str]:
        return frozenset({"clipboard.change"})

    @property
    def bypasses_privacy(self) -> bool:
        return False

    def transform(
        self,
        event_type: str,
        payload: Dict[str, Any],
        context: SignalTransformContext,
    ) -> Optional[InteractionSignal]:
        if "clipboard_content" in context.enabled_channels:
            text = str(payload.get("text", ""))[:200]
            return InteractionSignal(
                timestamp=context.now,
                signal_type="clipboard",
                detail=f"content:{text}",
            )
        if "clipboard" in context.enabled_channels:
            return InteractionSignal(
                timestamp=context.now,
                signal_type="clipboard",
                detail=payload.get("change_type", "change"),
            )
        return None


def build_default_signal_source_registry() -> SignalSourceRegistry:
    """Register the built-in sources in the original if-chain order.

    Order matters for privacy semantics: the app_switch (bypass) source must
    precede any non-bypass source registered against the same event_type, so
    that ``transform_first`` short-circuits before the privacy gate.
    """
    registry = SignalSourceRegistry()
    # Privacy-bypassing first.
    registry.register(AppSwitchSignalSource())
    # ui.action variants, in the original if-chain order.
    registry.register(ClickSignalSource())
    registry.register(ScrollSignalSource())
    registry.register(KeyboardShortcutSignalSource())
    registry.register(KeyboardTypeSignalSource())
    registry.register(DragSignalSource())
    # clipboard.change.
    registry.register(ClipboardSignalSource())
    return registry
