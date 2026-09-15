# Copyright (c) Alibaba, Inc. and its affiliates.
"""Comprehensive tests for LLMProviderRegistry and ScopedLLMProviderRegistry.

Covers:
- Core registry operations (register, unregister, discover, create)
- Version bumping on mutations
- Instance cache invalidation
- Scoped lifecycle with reload semantics
- Error handling for unknown providers and malformed configs
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from leapflow.llm.provider_registry import LLMProviderRegistry, get_default_registry, reset_default_registry
from leapflow.llm.scoped_provider_registry import ScopedLLMProviderRegistry
from leapflow.llm._builtin_plugins import OpenAICompatiblePlugin


# ═══════════════════════════════════════════════════════════════
# Fake implementations
# ═══════════════════════════════════════════════════════════════


class FakeLLMProvider:
    """Minimal LLMProvider implementation for testing."""

    def __init__(self, provider_id: str, model: str = "test-model") -> None:
        self._provider_id = provider_id
        self._model = model

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def model(self) -> str:
        return self._model

    def chat(self, **kwargs: any) -> str:
        return "fake response"


class FakeLLMProviderPlugin:
    """Minimal LLMProviderPlugin implementation for testing."""

    def __init__(
        self,
        provider_id: str,
        display_name: str,
        supported_models: list | None = None,
        capabilities: dict | None = None,
    ) -> None:
        self._provider_id = provider_id
        self._display_name = display_name
        self._supported_models = supported_models or ["*"]
        self._capabilities = capabilities or {
            "supports_streaming": False,
            "supports_tools": False,
            "supports_vision": False,
        }

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def supported_models(self) -> list:
        return self._supported_models

    @property
    def capabilities(self) -> dict:
        return self._capabilities

    def create_provider(self, config: dict) -> FakeLLMProvider:
        return FakeLLMProvider(self._provider_id, config.get("model", "test-model"))


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def fresh_registry() -> LLMProviderRegistry:
    """Create a fresh LLMProviderRegistry without any built-in plugins."""
    return LLMProviderRegistry()


@pytest.fixture
def scoped_registry(fresh_registry: LLMProviderRegistry) -> ScopedLLMProviderRegistry:
    """Create a ScopedLLMProviderRegistry wrapping a fresh LLMProviderRegistry."""
    return ScopedLLMProviderRegistry(fresh_registry)


# ═══════════════════════════════════════════════════════════════
# LLMProviderRegistry core tests
# ═══════════════════════════════════════════════════════════════


class TestLLMProviderRegistryCore:
    """Tests for LLMProviderRegistry core functionality."""

    def test_register_provider_plugin(self, fresh_registry: LLMProviderRegistry) -> None:
        """Register OpenAICompatiblePlugin, verify in list_available."""
        plugin = OpenAICompatiblePlugin()
        fresh_registry.register(plugin)
        assert "openai" in fresh_registry.list_available()
        assert fresh_registry.get_plugin("openai") is not None

    def test_register_bumps_version(self, fresh_registry: LLMProviderRegistry) -> None:
        """Version increases on register."""
        initial_version = fresh_registry.version
        plugin = FakeLLMProviderPlugin(
            provider_id="version-test",
            display_name="Version Test",
        )
        fresh_registry.register(plugin)
        assert fresh_registry.version == initial_version + 1

    def test_unregister_returns_true_if_present(self, fresh_registry: LLMProviderRegistry) -> None:
        """Unregister known provider returns True + bumps version."""
        plugin = FakeLLMProviderPlugin(
            provider_id="unregister-test",
            display_name="Unregister Test",
        )
        fresh_registry.register(plugin)
        initial_version = fresh_registry.version
        result = fresh_registry.unregister("unregister-test")
        assert result is True
        assert fresh_registry.version == initial_version + 1
        assert fresh_registry.get_plugin("unregister-test") is None

    def test_unregister_returns_false_if_absent(self, fresh_registry: LLMProviderRegistry) -> None:
        """Unregister unknown returns False + no version bump."""
        initial_version = fresh_registry.version
        result = fresh_registry.unregister("nonexistent-provider-xyz")
        assert result is False
        assert fresh_registry.version == initial_version

    def test_discover_builtin_registers_openai(self, fresh_registry: LLMProviderRegistry) -> None:
        """After discover_builtin(), 'openai' is available."""
        fresh_registry.discover_builtin()
        assert "openai" in fresh_registry.list_available()
        assert fresh_registry.get_plugin("openai") is not None

    def test_bootstrap_combines_builtin_and_entry_points(self, fresh_registry: LLMProviderRegistry) -> None:
        """bootstrap() calls both discover methods."""
        with patch.object(fresh_registry, "discover_entry_points") as mock_ep:
            mock_ep.return_value = 0
            fresh_registry.bootstrap()
            assert "openai" in fresh_registry.list_available()
            mock_ep.assert_called_once()

    def test_notify_mutation_bumps_version_and_clears_instances(self, fresh_registry: LLMProviderRegistry) -> None:
        """Version bumps, _instances cache cleared."""
        plugin = OpenAICompatiblePlugin()
        fresh_registry.register(plugin)
        config = {
            "api_key": "test-key",
            "base_url": "https://api.test.com/v1",
            "model": "gpt-4o",
        }
        provider = fresh_registry.create_from_config(config)
        assert provider is not None
        assert "openai" in fresh_registry._instances

        initial_version = fresh_registry.version
        fresh_registry.notify_mutation()
        assert fresh_registry.version == initial_version + 1
        assert "openai" not in fresh_registry._instances

    def test_create_from_config_builds_provider(self, fresh_registry: LLMProviderRegistry) -> None:
        """Pass a valid OpenAI-compatible config and verify a provider is created."""
        fresh_registry.discover_builtin()
        config = {
            "provider": "openai",
            "api_key": "test-api-key-123",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
        }
        provider = fresh_registry.create_from_config(config)
        assert provider is not None
        assert provider.model == "gpt-4o"

    def test_get_plugin_returns_registered_plugin(self, fresh_registry: LLMProviderRegistry) -> None:
        """get_plugin("openai") returns the plugin instance."""
        plugin = OpenAICompatiblePlugin()
        fresh_registry.register(plugin)
        retrieved = fresh_registry.get_plugin("openai")
        assert retrieved is plugin

    def test_reset_default_registry_clears_singleton(self) -> None:
        """reset_default_registry() then get_default_registry() gives a fresh instance."""
        default_before = get_default_registry()
        # Mutate the before instance
        default_before.register(FakeLLMProviderPlugin("before-plugin", "Before"))
        assert "before-plugin" in default_before.list_available()

        reset_default_registry()
        default_after = get_default_registry()

        assert default_after is not default_before
        assert "before-plugin" not in default_after.list_available()


# ═══════════════════════════════════════════════════════════════
# ScopedLLMProviderRegistry reload tests
# ═══════════════════════════════════════════════════════════════


class TestScopedLLMProviderRegistryReload:
    """Tests for ScopedLLMProviderRegistry reload semantics."""

    def test_scoped_reload_bumps_version(
        self, fresh_registry: LLMProviderRegistry, scoped_registry: ScopedLLMProviderRegistry
    ) -> None:
        """reload increments the underlying registry version."""
        module_name = "tests._fake_llm_reload_test"
        plugin = OpenAICompatiblePlugin()
        fake_mod = types.ModuleType(module_name)
        fake_mod.plugin = plugin
        fake_mod.__spec__ = None
        sys.modules[module_name] = fake_mod
        plugin.__class__ = type(
            "OpenAICompatiblePlugin", (OpenAICompatiblePlugin,), {"__module__": module_name}
        )

        fiber = scoped_registry.create_fiber("openai")
        scoped_registry.scoped_register(plugin, fiber)
        fiber.activate()  # Must activate before we can dispose
        initial_version = fresh_registry.version
        initial_gen = fiber.generation

        new_plugin = OpenAICompatiblePlugin()
        fake_mod.plugin = new_plugin

        with patch("importlib.reload", return_value=fake_mod):
            new_fiber = scoped_registry.reload("openai")

        assert fresh_registry.version > initial_version
        assert new_fiber.generation > initial_gen

        sys.modules.pop(module_name, None)

    def test_scoped_reload_creates_new_fiber_with_higher_generation(
        self, fresh_registry: LLMProviderRegistry, scoped_registry: ScopedLLMProviderRegistry
    ) -> None:
        """New fiber has higher generation."""
        module_name = "tests._fake_llm_fiber_gen"
        plugin = OpenAICompatiblePlugin()
        fake_mod = types.ModuleType(module_name)
        fake_mod.plugin = plugin
        fake_mod.__spec__ = None
        sys.modules[module_name] = fake_mod
        plugin.__class__ = type(
            "OpenAICompatiblePlugin", (OpenAICompatiblePlugin,), {"__module__": module_name}
        )

        fiber = scoped_registry.create_fiber("openai")
        scoped_registry.scoped_register(plugin, fiber)
        fiber.activate()  # Must activate before we can dispose
        initial_gen = fiber.generation

        new_plugin = OpenAICompatiblePlugin()
        fake_mod.plugin = new_plugin

        with patch("importlib.reload", return_value=fake_mod):
            new_fiber = scoped_registry.reload("openai")

        assert new_fiber.generation > initial_gen
        assert new_fiber.state.name == "ACTIVE"

        sys.modules.pop(module_name, None)

    def test_scoped_reload_unknown_provider_raises_keyerror(
        self, scoped_registry: ScopedLLMProviderRegistry
    ) -> None:
        """reload("nonexistent") raises KeyError."""
        with pytest.raises(KeyError, match="not scoped-registered"):
            scoped_registry.reload("nonexistent-provider-xyz")

    def test_scoped_reload_module_without_plugin_raises_runtime_error(
        self, fresh_registry: LLMProviderRegistry, scoped_registry: ScopedLLMProviderRegistry
    ) -> None:
        """Monkeypatch to remove `plugin` attribute, reload raises RuntimeError."""
        module_name = "tests._fake_llm_no_plugin_attr"
        plugin = OpenAICompatiblePlugin()
        fake_mod = types.ModuleType(module_name)
        fake_mod.plugin = plugin
        fake_mod.__spec__ = None
        sys.modules[module_name] = fake_mod
        plugin.__class__ = type(
            "OpenAICompatiblePlugin", (OpenAICompatiblePlugin,), {"__module__": module_name}
        )

        fiber = scoped_registry.create_fiber("openai")
        scoped_registry.scoped_register(plugin, fiber)
        fiber.activate()  # Must activate before we can dispose

        del fake_mod.plugin

        with patch("importlib.reload", return_value=fake_mod):
            with pytest.raises(RuntimeError, match="no 'plugin' attribute"):
                scoped_registry.reload("openai")

        sys.modules.pop(module_name, None)
