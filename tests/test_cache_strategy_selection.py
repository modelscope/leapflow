# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for P0-OPT-1: capability-driven CacheStrategy selection.

Verifies that ``_select_cache_strategy`` / ``_resolve_cache_type`` in
``cli/context.py`` pick the correct strategy based on plugin capabilities:
- auto_prefix       → PrefixCacheOptimizer
- explicit_breakpoint → AnthropicCacheStrategy
- none              → NoCacheStrategy
- absent / unknown  → PrefixCacheOptimizer (safe default)
- DeepSeek (auto_prefix) → PrefixCacheOptimizer (regression zero-change)
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import patch

from leapflow.engine.prompt_cache import (
    AnthropicCacheStrategy,
    NoCacheStrategy,
    PrefixCacheOptimizer,
)
from leapflow.llm.provider_registry import LLMProviderRegistry


# ── Fake plugin for testing ───────────────────────────────────────────────

class _FakePlugin:
    """Minimal LLMProviderPlugin for capability testing."""

    def __init__(
        self,
        provider_id: str,
        cache_type: str = "auto_prefix",
        cache_usage_fields: List[str] | None = None,
    ) -> None:
        self._id = provider_id
        self._cache_type = cache_type
        self._cache_usage_fields = cache_usage_fields or []

    @property
    def provider_id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return f"Fake-{self._id}"

    @property
    def supported_models(self) -> List[str]:
        return ["*"]

    @property
    def capabilities(self) -> Dict[str, Any]:
        return {
            "supports_streaming": True,
            "cache_type": self._cache_type,
            "cache_usage_fields": self._cache_usage_fields,
        }

    def create_provider(self, config: Dict[str, Any]) -> Any:
        raise NotImplementedError("fake plugin — no real provider")


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_registry(*plugins: _FakePlugin) -> LLMProviderRegistry:
    """Create a registry populated with the given fake plugins."""
    reg = LLMProviderRegistry()
    for p in plugins:
        reg.register(p)
    return reg


def _select(
    base_url: str,
    registry: LLMProviderRegistry,
    *,
    provider_id: str | None = None,
) -> Any:
    """Import and call ``_select_cache_strategy`` with a mocked registry."""
    from leapflow.cli.context import _select_cache_strategy

    with patch(
        "leapflow.llm.provider_registry.get_default_registry",
        return_value=registry,
    ):
        return _select_cache_strategy(base_url, provider_id=provider_id)


# ── Tests ─────────────────────────────────────────────────────────────────

class TestCacheStrategySelection:
    """Capability-driven CacheStrategy selection."""

    def test_auto_prefix_returns_prefix_optimizer(self) -> None:
        """auto_prefix → PrefixCacheOptimizer."""
        reg = _make_registry(
            _FakePlugin("openai", cache_type="auto_prefix"),
        )
        strategy = _select("https://api.openai.com/v1", reg)
        assert isinstance(strategy, PrefixCacheOptimizer)

    def test_explicit_breakpoint_returns_anthropic_strategy(self) -> None:
        """explicit_breakpoint → AnthropicCacheStrategy."""
        reg = _make_registry(
            _FakePlugin("openai", cache_type="auto_prefix"),
            _FakePlugin("anthropic", cache_type="explicit_breakpoint"),
        )
        strategy = _select("https://api.anthropic.com/v1", reg)
        assert isinstance(strategy, AnthropicCacheStrategy)

    def test_none_returns_no_cache_strategy(self) -> None:
        """none → NoCacheStrategy."""
        reg = _make_registry(
            _FakePlugin("openai", cache_type="none"),
        )
        strategy = _select("https://api.openai.com/v1", reg)
        assert isinstance(strategy, NoCacheStrategy)

    def test_absent_capability_defaults_to_prefix_optimizer(self) -> None:
        """Missing plugin → safe default PrefixCacheOptimizer."""
        reg = LLMProviderRegistry()  # empty — no plugins
        strategy = _select("https://api.example.com/v1", reg)
        assert isinstance(strategy, PrefixCacheOptimizer)

    def test_unknown_cache_type_defaults_to_prefix_optimizer(self) -> None:
        """Unknown cache_type value → safe default PrefixCacheOptimizer."""
        reg = _make_registry(
            _FakePlugin("openai", cache_type="something_new"),
        )
        strategy = _select("https://api.openai.com/v1", reg)
        assert isinstance(strategy, PrefixCacheOptimizer)


class TestDeepSeekRegression:
    """DeepSeek existing paths must still select PrefixCacheOptimizer."""

    def test_deepseek_standard_url(self) -> None:
        reg = _make_registry(
            _FakePlugin("openai", cache_type="auto_prefix"),
        )
        strategy = _select("https://api.deepseek.com/v1", reg)
        assert isinstance(strategy, PrefixCacheOptimizer)

    def test_deepseek_anthropic_compat_endpoint(self) -> None:
        """DeepSeek /anthropic endpoint → AnthropicCacheStrategy (when plugin registered)."""
        reg = _make_registry(
            _FakePlugin("openai", cache_type="auto_prefix"),
            _FakePlugin("anthropic", cache_type="explicit_breakpoint"),
        )
        strategy = _select("https://api.deepseek.com/anthropic", reg)
        assert isinstance(strategy, AnthropicCacheStrategy)


class TestURLDetection:
    """URL-based provider plugin routing (best-effort fallback)."""

    def test_anthropic_com_host(self) -> None:
        reg = _make_registry(
            _FakePlugin("openai", cache_type="auto_prefix"),
            _FakePlugin("anthropic", cache_type="explicit_breakpoint"),
        )
        strategy = _select("https://api.anthropic.com/v1/messages", reg)
        assert isinstance(strategy, AnthropicCacheStrategy)

    def test_anthropic_path_suffix(self) -> None:
        """URL path ending in /anthropic routes to anthropic plugin."""
        reg = _make_registry(
            _FakePlugin("openai", cache_type="auto_prefix"),
            _FakePlugin("anthropic", cache_type="explicit_breakpoint"),
        )
        strategy = _select("https://api.deepseek.com/anthropic", reg)
        assert isinstance(strategy, AnthropicCacheStrategy)

    def test_anthropic_path_segment_with_trailing_path(self) -> None:
        """URL with /anthropic/ as a path segment followed by more path.

        This is the bug-fix scenario: custom gateway URL like
        ``https://proxy.internal/anthropic/v1/messages`` must route to
        the anthropic plugin, not openai.
        """
        reg = _make_registry(
            _FakePlugin("openai", cache_type="auto_prefix"),
            _FakePlugin("anthropic", cache_type="explicit_breakpoint"),
        )
        strategy = _select("https://proxy.internal/anthropic/v1/messages", reg)
        assert isinstance(strategy, AnthropicCacheStrategy)

    def test_anthropic_gateway_with_port(self) -> None:
        """Custom gateway with port and /anthropic segment."""
        reg = _make_registry(
            _FakePlugin("openai", cache_type="auto_prefix"),
            _FakePlugin("anthropic", cache_type="explicit_breakpoint"),
        )
        strategy = _select("https://gateway.corp:8443/anthropic/v1/messages", reg)
        assert isinstance(strategy, AnthropicCacheStrategy)

    def test_generic_url_routes_to_openai(self) -> None:
        reg = _make_registry(
            _FakePlugin("openai", cache_type="auto_prefix"),
        )
        strategy = _select("https://some-proxy.example.com/v1", reg)
        assert isinstance(strategy, PrefixCacheOptimizer)

    def test_empty_url_defaults_safely(self) -> None:
        reg = _make_registry(
            _FakePlugin("openai", cache_type="auto_prefix"),
        )
        strategy = _select("", reg)
        assert isinstance(strategy, PrefixCacheOptimizer)

    def test_anthropic_plugin_not_registered_falls_back(self) -> None:
        """When anthropic URL detected but plugin absent → safe default."""
        reg = _make_registry(
            _FakePlugin("openai", cache_type="auto_prefix"),
        )
        strategy = _select("https://api.anthropic.com/v1", reg)
        # Plugin not registered, falls back to auto_prefix default.
        assert isinstance(strategy, PrefixCacheOptimizer)

    def test_url_with_anthropic_in_query_param_does_not_match(self) -> None:
        """The word 'anthropic' in a query param must NOT route to anthropic."""
        reg = _make_registry(
            _FakePlugin("openai", cache_type="auto_prefix"),
            _FakePlugin("anthropic", cache_type="explicit_breakpoint"),
        )
        strategy = _select("https://proxy.example.com/v1?backend=anthropic", reg)
        # Query params are not path segments — should stay openai.
        assert isinstance(strategy, PrefixCacheOptimizer)


class TestExplicitProviderID:
    """When an explicit provider_id is supplied, URL is ignored."""

    def test_explicit_anthropic_id_overrides_openai_url(self) -> None:
        """provider_id='anthropic' wins even if URL looks like OpenAI."""
        reg = _make_registry(
            _FakePlugin("openai", cache_type="auto_prefix"),
            _FakePlugin("anthropic", cache_type="explicit_breakpoint"),
        )
        strategy = _select(
            "https://api.openai.com/v1", reg, provider_id="anthropic",
        )
        assert isinstance(strategy, AnthropicCacheStrategy)

    def test_explicit_openai_id_overrides_anthropic_url(self) -> None:
        """provider_id='openai' wins even if URL looks like Anthropic."""
        reg = _make_registry(
            _FakePlugin("openai", cache_type="auto_prefix"),
            _FakePlugin("anthropic", cache_type="explicit_breakpoint"),
        )
        strategy = _select(
            "https://api.anthropic.com/v1", reg, provider_id="openai",
        )
        assert isinstance(strategy, PrefixCacheOptimizer)

    def test_explicit_unknown_id_defaults_safely(self) -> None:
        """Unknown provider_id → safe default PrefixCacheOptimizer."""
        reg = _make_registry(
            _FakePlugin("openai", cache_type="auto_prefix"),
        )
        strategy = _select(
            "https://api.example.com/v1", reg, provider_id="unknown_provider",
        )
        assert isinstance(strategy, PrefixCacheOptimizer)


class TestOpenAICompatiblePluginCapabilities:
    """Verify the real OpenAICompatiblePlugin declares cache fields."""

    def test_capabilities_include_cache_type(self) -> None:
        from leapflow.llm._builtin_plugins import OpenAICompatiblePlugin

        plugin = OpenAICompatiblePlugin()
        caps = plugin.capabilities
        assert caps["cache_type"] == "auto_prefix"
        assert "cached_tokens" in caps["cache_usage_fields"]
        assert "prompt_cache_hit_tokens" in caps["cache_usage_fields"]
