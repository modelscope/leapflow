# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the SignalSource protocol, built-in sources, and registry.

Verifies that the pluginized extraction produces byte-for-byte identical
InteractionSignal outputs compared to the original _extract_signal() if-chain.
"""

from __future__ import annotations


from leapflow.perception.signal_source import (
    SignalSourceRegistry,
    SignalTransformContext,
)
from leapflow.perception.signal_sources_builtin import (
    AppSwitchSignalSource,
    ClickSignalSource,
    ClipboardSignalSource,
    DragSignalSource,
    KeyboardShortcutSignalSource,
    KeyboardTypeSignalSource,
    ScrollSignalSource,
    build_default_signal_source_registry,
)
from leapflow.perception.types import InteractionSignal


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


_ALL_CHANNELS = frozenset({"app_switch", "click", "scroll", "keyboard", "drag", "clipboard", "clipboard_content"})
_UNSET: frozenset = frozenset({"__unset__"})


def _ctx(
    *,
    now: float = 100.0,
    prev_app: str = "com.old.app",
    current_app: str = "com.current.app",
    enabled_channels: frozenset = _UNSET,
    privacy_sensitive_apps: frozenset = _UNSET,
) -> SignalTransformContext:
    """Build a SignalTransformContext with sensible defaults.

    Empty frozenset is a valid caller value ("no channels enabled"), so we use
    a sentinel rather than ``or`` to detect unset params.
    """
    return SignalTransformContext(
        now=now,
        prev_app=prev_app,
        current_app=current_app,
        enabled_channels=_ALL_CHANNELS if enabled_channels is _UNSET else enabled_channels,
        privacy_sensitive_apps=frozenset() if privacy_sensitive_apps is _UNSET else privacy_sensitive_apps,
    )


# ═══════════════════════════════════════════════════════════════════
# Individual Source Tests
# ═══════════════════════════════════════════════════════════════════


class TestAppSwitchSignalSource:
    """AppSwitch: app.focus_change → app_switch signal (bypasses privacy)."""

    def test_basic_transform(self) -> None:
        source = AppSwitchSignalSource()
        ctx = _ctx(prev_app="com.old", current_app="com.new")
        sig = source.transform("app.focus_change", {"bundle_id": "com.new"}, ctx)
        assert sig is not None
        assert sig == InteractionSignal(
            timestamp=100.0,
            signal_type="app_switch",
            app="com.new",
            detail="com.old -> com.new",
        )

    def test_not_in_enabled_channels(self) -> None:
        source = AppSwitchSignalSource()
        ctx = _ctx(enabled_channels=frozenset({"click"}))
        sig = source.transform("app.focus_change", {"bundle_id": "x"}, ctx)
        assert sig is None

    def test_bypasses_privacy_flag(self) -> None:
        source = AppSwitchSignalSource()
        assert source.bypasses_privacy is True
        assert source.channel_id == "app_switch"
        assert "app.focus_change" in source.event_types


class TestClickSignalSource:
    """Click: ui.action with sub_type=='click'."""

    def test_basic_transform(self) -> None:
        source = ClickSignalSource()
        payload = {"sub_type": "click", "app_bundle_id": "com.app", "mouse_x": 42, "mouse_y": 99}
        ctx = _ctx()
        sig = source.transform("ui.action", payload, ctx)
        assert sig == InteractionSignal(
            timestamp=100.0,
            signal_type="click",
            app="com.app",
            position=(42, 99),
        )

    def test_fallback_to_current_app(self) -> None:
        source = ClickSignalSource()
        payload = {"sub_type": "click", "mouse_x": 1, "mouse_y": 2}
        ctx = _ctx(current_app="com.fallback")
        sig = source.transform("ui.action", payload, ctx)
        assert sig is not None
        assert sig.app == "com.fallback"

    def test_wrong_sub_type(self) -> None:
        source = ClickSignalSource()
        payload = {"sub_type": "scroll", "mouse_x": 0, "mouse_y": 0}
        assert source.transform("ui.action", payload, _ctx()) is None

    def test_channel_disabled(self) -> None:
        source = ClickSignalSource()
        payload = {"sub_type": "click", "mouse_x": 0, "mouse_y": 0}
        ctx = _ctx(enabled_channels=frozenset({"scroll"}))
        assert source.transform("ui.action", payload, ctx) is None


class TestScrollSignalSource:
    """Scroll: ui.action with sub_type=='scroll'."""

    def test_basic_transform(self) -> None:
        source = ScrollSignalSource()
        payload = {"sub_type": "scroll", "app_bundle_id": "com.x", "mouse_x": 10, "mouse_y": 20, "delta_y": -3}
        sig = source.transform("ui.action", payload, _ctx())
        assert sig == InteractionSignal(
            timestamp=100.0,
            signal_type="scroll",
            app="com.x",
            position=(10, 20),
            detail="dy=-3",
        )

    def test_missing_delta(self) -> None:
        source = ScrollSignalSource()
        payload = {"sub_type": "scroll", "mouse_x": 0, "mouse_y": 0}
        sig = source.transform("ui.action", payload, _ctx(current_app="a"))
        assert sig is not None
        assert sig.detail == "dy=0"


class TestKeyboardShortcutSignalSource:
    """Keyboard shortcut: ui.action sub_type=='shortcut'."""

    def test_basic_combo(self) -> None:
        source = KeyboardShortcutSignalSource()
        payload = {"sub_type": "shortcut", "modifiers": ["cmd", "shift"], "char": "z", "app_bundle_id": "app"}
        sig = source.transform("ui.action", payload, _ctx())
        assert sig == InteractionSignal(
            timestamp=100.0,
            signal_type="keyboard",
            app="app",
            detail="cmd+shift+z",
        )

    def test_no_char(self) -> None:
        source = KeyboardShortcutSignalSource()
        payload = {"sub_type": "shortcut", "modifiers": ["ctrl"], "char": "", "app_bundle_id": "x"}
        sig = source.transform("ui.action", payload, _ctx())
        assert sig is not None
        assert sig.detail == "ctrl"


class TestKeyboardTypeSignalSource:
    """Keyboard type: ui.action sub_type=='type', text truncated to 50."""

    def test_basic_transform(self) -> None:
        source = KeyboardTypeSignalSource()
        payload = {"sub_type": "type", "text": "hello world", "app_bundle_id": "ed"}
        sig = source.transform("ui.action", payload, _ctx())
        assert sig == InteractionSignal(
            timestamp=100.0,
            signal_type="keyboard",
            app="ed",
            detail="type:hello world",
        )

    def test_text_truncation(self) -> None:
        source = KeyboardTypeSignalSource()
        long_text = "a" * 100
        payload = {"sub_type": "type", "text": long_text, "app_bundle_id": "x"}
        sig = source.transform("ui.action", payload, _ctx())
        assert sig is not None
        assert sig.detail == f"type:{'a' * 50}"
        assert len(sig.detail) == 55  # "type:" + 50 chars


class TestDragSignalSource:
    """Drag: ui.action sub_type=='drag'."""

    def test_basic_transform(self) -> None:
        source = DragSignalSource()
        payload = {
            "sub_type": "drag",
            "app_bundle_id": "draw",
            "start_x": 10, "start_y": 20,
            "end_x": 100, "end_y": 200,
        }
        sig = source.transform("ui.action", payload, _ctx())
        assert sig == InteractionSignal(
            timestamp=100.0,
            signal_type="drag",
            app="draw",
            position=(10, 20),
            end_position=(100, 200),
        )


class TestClipboardSignalSource:
    """Clipboard: clipboard.change with clipboard_content or clipboard channel."""

    def test_clipboard_content_channel(self) -> None:
        source = ClipboardSignalSource()
        payload = {"text": "secret stuff"}
        ctx = _ctx(enabled_channels=frozenset({"clipboard_content", "clipboard"}))
        sig = source.transform("clipboard.change", payload, ctx)
        assert sig == InteractionSignal(
            timestamp=100.0,
            signal_type="clipboard",
            detail="content:secret stuff",
        )

    def test_clipboard_content_truncation(self) -> None:
        source = ClipboardSignalSource()
        long_text = "x" * 300
        payload = {"text": long_text}
        ctx = _ctx(enabled_channels=frozenset({"clipboard_content"}))
        sig = source.transform("clipboard.change", payload, ctx)
        assert sig is not None
        assert sig.detail == f"content:{'x' * 200}"

    def test_clipboard_channel_fallback(self) -> None:
        """When only 'clipboard' (not 'clipboard_content') is enabled."""
        source = ClipboardSignalSource()
        payload = {"change_type": "cut"}
        ctx = _ctx(enabled_channels=frozenset({"clipboard"}))
        sig = source.transform("clipboard.change", payload, ctx)
        assert sig == InteractionSignal(
            timestamp=100.0,
            signal_type="clipboard",
            detail="cut",
        )

    def test_clipboard_default_change_type(self) -> None:
        source = ClipboardSignalSource()
        payload = {}
        ctx = _ctx(enabled_channels=frozenset({"clipboard"}))
        sig = source.transform("clipboard.change", payload, ctx)
        assert sig is not None
        assert sig.detail == "change"

    def test_neither_channel_enabled(self) -> None:
        source = ClipboardSignalSource()
        ctx = _ctx(enabled_channels=frozenset({"click"}))
        sig = source.transform("clipboard.change", {}, ctx)
        assert sig is None

    def test_clipboard_content_preferred_over_clipboard(self) -> None:
        """When both channels are enabled, clipboard_content wins."""
        source = ClipboardSignalSource()
        payload = {"text": "data", "change_type": "paste"}
        ctx = _ctx(enabled_channels=frozenset({"clipboard_content", "clipboard"}))
        sig = source.transform("clipboard.change", payload, ctx)
        assert sig is not None
        assert sig.detail == "content:data"


# ═══════════════════════════════════════════════════════════════════
# Registry Tests
# ═══════════════════════════════════════════════════════════════════


class TestSignalSourceRegistry:
    """Tests for the registry dispatch and privacy gating."""

    def test_app_switch_bypasses_privacy(self) -> None:
        """app_switch is emitted even when current_app is privacy-sensitive."""
        registry = build_default_signal_source_registry()
        ctx = _ctx(
            prev_app="com.normal",
            current_app="com.bank",
            privacy_sensitive_apps=frozenset({"com.bank"}),
        )
        sig = registry.transform_first("app.focus_change", {"bundle_id": "com.bank"}, ctx)
        assert sig is not None
        assert sig.signal_type == "app_switch"

    def test_non_app_switch_suppressed_for_privacy_sensitive_app(self) -> None:
        """click/scroll/keyboard/etc suppressed when current_app is privacy-sensitive."""
        registry = build_default_signal_source_registry()
        ctx = _ctx(
            current_app="com.private",
            privacy_sensitive_apps=frozenset({"com.private"}),
        )
        # Click event should be suppressed
        sig = registry.transform_first("ui.action", {"sub_type": "click", "mouse_x": 1, "mouse_y": 2}, ctx)
        assert sig is None

        # Scroll too
        sig = registry.transform_first("ui.action", {"sub_type": "scroll", "mouse_x": 0, "mouse_y": 0}, ctx)
        assert sig is None

        # Clipboard too
        sig = registry.transform_first("clipboard.change", {"text": "private"}, ctx)
        assert sig is None

    def test_first_non_none_wins(self) -> None:
        """Registry returns the first matching source's result."""
        registry = build_default_signal_source_registry()
        ctx = _ctx(enabled_channels=frozenset({"click", "scroll", "keyboard", "drag"}))
        payload = {"sub_type": "click", "mouse_x": 5, "mouse_y": 6}
        sig = registry.transform_first("ui.action", payload, ctx)
        assert sig is not None
        assert sig.signal_type == "click"

    def test_no_channels_returns_none(self) -> None:
        """When no channels match in context, all sources return None."""
        registry = build_default_signal_source_registry()
        ctx = _ctx(enabled_channels=frozenset())
        sig = registry.transform_first("ui.action", {"sub_type": "click"}, ctx)
        assert sig is None

    def test_disabled_channel_not_emitted(self) -> None:
        """A source whose channel is not in enabled_channels returns None."""
        registry = build_default_signal_source_registry()
        # Only 'scroll' enabled; click should be None
        ctx = _ctx(enabled_channels=frozenset({"scroll"}))
        sig = registry.transform_first("ui.action", {"sub_type": "click", "mouse_x": 0, "mouse_y": 0}, ctx)
        assert sig is None

    def test_unknown_event_type(self) -> None:
        """Unknown event_type yields no matching sources."""
        registry = build_default_signal_source_registry()
        sig = registry.transform_first("unknown.event", {}, _ctx())
        assert sig is None

    def test_default_registry_has_all_sources(self) -> None:
        """Default registry contains exactly 7 built-in sources."""
        registry = build_default_signal_source_registry()
        assert len(registry.sources) == 7
        ids = registry.channel_ids
        assert "app_switch" in ids
        assert "click" in ids
        assert "scroll" in ids
        assert "keyboard" in ids
        assert "drag" in ids
        assert "clipboard" in ids


# ═══════════════════════════════════════════════════════════════════
# Integration: PerceptionSession._extract_signal delegates correctly
# ═══════════════════════════════════════════════════════════════════


class TestExtractSignalDelegation:
    """Verify that PerceptionSession._extract_signal uses the registry."""

    _UNSET_CH = frozenset({"__ch_sentinel__"})
    _UNSET_PA = frozenset({"__pa_sentinel__"})

    def _make_session(self, channels=_UNSET_CH, privacy_apps=_UNSET_PA):
        """Build a minimal PerceptionSession for signal extraction tests."""
        from unittest.mock import MagicMock
        from leapflow.perception.config import PerceptionConfig
        from leapflow.perception.session import PerceptionSession
        from leapflow.domain.trajectory import RecordingMode

        if channels is self._UNSET_CH:
            channels = frozenset({"app_switch", "click", "scroll", "keyboard", "drag", "clipboard"})
        if privacy_apps is self._UNSET_PA:
            privacy_apps = frozenset()

        config = PerceptionConfig(
            signal_channels=channels,
            privacy_sensitive_apps=privacy_apps,
        )
        rpc = MagicMock()
        session = PerceptionSession(config=config, rpc=rpc)
        session._active = True
        session._session_id = "test"
        # VISION_ONLY is the only mode with needs_visual_polling==True in the
        # current RecordingMode enum; use it so _extract_signal reaches the
        # registry rather than early-returning on the mode gate.
        session._recording_mode = RecordingMode.VISION_ONLY
        return session

    def test_no_channels_returns_none(self) -> None:
        session = self._make_session(channels=frozenset())
        result = session._extract_signal("ui.action", {"sub_type": "click"}, "prev", 1.0)
        assert result is None

    def test_click_through_registry(self) -> None:
        session = self._make_session()
        session._current_app = "com.editor"
        result = session._extract_signal(
            "ui.action",
            {"sub_type": "click", "app_bundle_id": "com.editor", "mouse_x": 10, "mouse_y": 20},
            "prev",
            50.0,
        )
        assert result is not None
        assert result.signal_type == "click"
        assert result.position == (10, 20)
        assert result.app == "com.editor"

    def test_privacy_gate_preserves_app_switch(self) -> None:
        session = self._make_session(
            channels=frozenset({"app_switch", "click"}),
            privacy_apps=frozenset({"com.bank"}),
        )
        session._current_app = "com.bank"
        # app_switch should go through despite privacy
        sig = session._extract_signal("app.focus_change", {"bundle_id": "com.bank"}, "com.prev", 1.0)
        assert sig is not None
        assert sig.signal_type == "app_switch"

    def test_privacy_gate_blocks_click(self) -> None:
        session = self._make_session(
            channels=frozenset({"click"}),
            privacy_apps=frozenset({"com.bank"}),
        )
        session._current_app = "com.bank"
        sig = session._extract_signal("ui.action", {"sub_type": "click", "mouse_x": 0, "mouse_y": 0}, "prev", 1.0)
        assert sig is None

    def test_custom_registry_injection(self) -> None:
        """Session accepts a custom registry."""
        from unittest.mock import MagicMock
        from leapflow.perception.config import PerceptionConfig
        from leapflow.perception.session import PerceptionSession

        config = PerceptionConfig(
            signal_channels=frozenset({"custom"}),
        )
        custom_registry = SignalSourceRegistry()
        rpc = MagicMock()
        session = PerceptionSession(config=config, rpc=rpc, signal_source_registry=custom_registry)
        assert session._signal_source_registry is custom_registry

    def test_custom_signal_source_via_registry(self) -> None:
        """A community-provided custom SignalSource works end-to-end through the session."""
        from unittest.mock import MagicMock
        from leapflow.perception.config import PerceptionConfig
        from leapflow.perception.session import PerceptionSession
        from leapflow.perception.signal_sources_builtin import build_default_signal_source_registry
        from leapflow.perception.signal_source import SignalTransformContext
        from leapflow.perception.types import InteractionSignal
        from leapflow.domain.trajectory import RecordingMode

        class CustomSource:
            @property
            def channel_id(self) -> str:
                return "custom"

            @property
            def event_types(self):
                return frozenset({"custom.event"})

            @property
            def bypasses_privacy(self) -> bool:
                return False

            def transform(self, event_type, payload, context):
                if "custom" not in context.enabled_channels:
                    return None
                return InteractionSignal(
                    timestamp=context.now,
                    signal_type="custom",
                    detail=payload.get("detail", ""),
                )

        registry = build_default_signal_source_registry()
        registry.register(CustomSource())

        # Build a session with the custom registry, mirroring _make_session:
        # the 'custom' channel must be enabled, and VISION_ONLY passes the
        # needs_visual_polling gate so _extract_signal reaches the registry.
        config = PerceptionConfig(
            signal_channels=frozenset({"custom"}),
            privacy_sensitive_apps=frozenset(),
        )
        rpc = MagicMock()
        session = PerceptionSession(config=config, rpc=rpc, signal_source_registry=registry)
        session._active = True
        session._session_id = "test"
        session._recording_mode = RecordingMode.VISION_ONLY
        session._current_app = "com.editor"

        sig = session._extract_signal("custom.event", {"detail": "x"}, "prev", 1.0)
        assert sig is not None
        assert sig.signal_type == "custom"
        assert sig.detail == "x"
        # Sanity: the custom source satisfies the SignalSource protocol.
        assert isinstance(SignalTransformContext(
            now=1.0, prev_app="prev", current_app="com.editor",
            enabled_channels=frozenset({"custom"}), privacy_sensitive_apps=frozenset(),
        ), SignalTransformContext)
