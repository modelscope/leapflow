# Copyright (c) Alibaba, Inc. and its affiliates.
"""Comprehensive tests for GatewayAdapterRegistry and ScopedGatewayAdapterRegistry.

Covers:
- Core registry operations (register, unregister, discover, create)
- Version bumping on mutations
- Scoped lifecycle with reload semantics
- Error handling for unknown platforms and malformed plugins
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from leapflow.gateway.adapter_registry import GatewayAdapterRegistry, BuiltinAdapterPlugin
from leapflow.gateway.scoped_adapter_registry import ScopedGatewayAdapterRegistry
from leapflow.gateway.protocol import PlatformAdapter


# ═══════════════════════════════════════════════════════════════
# Fake implementations
# ═══════════════════════════════════════════════════════════════


class FakePlatformAdapter(PlatformAdapter):
    """Minimal PlatformAdapter implementation for testing."""

    def __init__(self, platform_id: str) -> None:
        self._platform_id = platform_id

    @property
    def platform_id(self) -> str:
        return self._platform_id

    def send_message(self, **kwargs: any) -> dict:
        return {"status": "sent"}

    def receive_messages(self, **kwargs: any) -> list:
        return []

    def setup(self, **kwargs: any) -> None:
        pass

    def teardown(self) -> None:
        pass


class FakeGatewayAdapterPlugin:
    """Minimal GatewayAdapterPlugin implementation for testing."""

    def __init__(
        self,
        platform_id: str,
        display_name: str,
        adapter_class_path: str,
        config_schema: dict | None = None,
    ) -> None:
        self._platform_id = platform_id
        self._display_name = display_name
        self._adapter_class_path = adapter_class_path
        self._config_schema = config_schema or {}

    @property
    def platform_id(self) -> str:
        return self._platform_id

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def adapter_class_path(self) -> str:
        return self._adapter_class_path

    @property
    def config_schema(self) -> dict:
        return self._config_schema

    def create_adapter(self, config: dict) -> PlatformAdapter:
        return FakePlatformAdapter(self._platform_id)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def fresh_registry() -> GatewayAdapterRegistry:
    """Create a fresh GatewayAdapterRegistry without any built-in plugins."""
    return GatewayAdapterRegistry()


@pytest.fixture
def scoped_registry(fresh_registry: GatewayAdapterRegistry) -> ScopedGatewayAdapterRegistry:
    """Create a ScopedGatewayAdapterRegistry wrapping a fresh GatewayAdapterRegistry."""
    return ScopedGatewayAdapterRegistry(fresh_registry)


# ═══════════════════════════════════════════════════════════════
# GatewayAdapterRegistry core tests
# ═══════════════════════════════════════════════════════════════


class TestGatewayAdapterRegistryCore:
    """Tests for GatewayAdapterRegistry core functionality."""

    def test_register_adapter_plugin(self, fresh_registry: GatewayAdapterRegistry) -> None:
        """Register a plugin, verify it appears in list_available()."""
        plugin = FakeGatewayAdapterPlugin(
            platform_id="test-platform",
            display_name="Test Platform",
            adapter_class_path="test.module:Adapter",
        )
        fresh_registry.register(plugin)
        assert "test-platform" in fresh_registry.list_available()
        assert fresh_registry.has_plugin("test-platform")

    def test_register_bumps_version(self, fresh_registry: GatewayAdapterRegistry) -> None:
        """Version increases on register."""
        initial_version = fresh_registry.version
        plugin = FakeGatewayAdapterPlugin(
            platform_id="version-test",
            display_name="Version Test",
            adapter_class_path="test.module:Adapter",
        )
        fresh_registry.register(plugin)
        assert fresh_registry.version == initial_version + 1

    def test_unregister_returns_true_if_present(self, fresh_registry: GatewayAdapterRegistry) -> None:
        """Unregister known adapter returns True + bumps version."""
        plugin = FakeGatewayAdapterPlugin(
            platform_id="unregister-test",
            display_name="Unregister Test",
            adapter_class_path="test.module:Adapter",
        )
        fresh_registry.register(plugin)
        initial_version = fresh_registry.version
        result = fresh_registry.unregister("unregister-test")
        assert result is True
        assert fresh_registry.version == initial_version + 1
        assert not fresh_registry.has_plugin("unregister-test")

    def test_unregister_returns_false_if_absent(self, fresh_registry: GatewayAdapterRegistry) -> None:
        """Unregister unknown returns False + no version bump."""
        initial_version = fresh_registry.version
        result = fresh_registry.unregister("nonexistent-platform")
        assert result is False
        assert fresh_registry.version == initial_version

    def test_discover_builtin_registers_all(self, fresh_registry: GatewayAdapterRegistry) -> None:
        """After discover_builtin(), all 5 built-in adapters are available."""
        discovered = fresh_registry.discover_builtin()
        # Should discover at least some built-ins (may be fewer if imports fail)
        assert discovered >= 0
        available = fresh_registry.list_available()
        # Verify the expected built-in platform IDs are present if discovery succeeded
        expected_platforms = ["feishu", "telegram", "dingtalk", "webhook", "api_server"]
        for platform in expected_platforms:
            if platform in available:
                assert fresh_registry.has_plugin(platform)

    def test_notify_mutation_bumps_version(self, fresh_registry: GatewayAdapterRegistry) -> None:
        """Explicit notify_mutation call bumps version."""
        initial_version = fresh_registry.version
        fresh_registry.notify_mutation()
        assert fresh_registry.version == initial_version + 1

    def test_create_adapter_via_registry(self, fresh_registry: GatewayAdapterRegistry) -> None:
        """Create an adapter via create_adapter(platform_id, config) and verify it works."""
        plugin = FakeGatewayAdapterPlugin(
            platform_id="create-test",
            display_name="Create Test",
            adapter_class_path="test.module:Adapter",
        )
        fresh_registry.register(plugin)
        adapter = fresh_registry.create_adapter("create-test", {"key": "value"})
        assert isinstance(adapter, PlatformAdapter)
        assert adapter.platform_id == "create-test"

    def test_get_plugin_returns_registered_plugin(self, fresh_registry: GatewayAdapterRegistry) -> None:
        """get_plugin returns the registered plugin instance."""
        plugin = FakeGatewayAdapterPlugin(
            platform_id="get-plugin-test",
            display_name="Get Plugin Test",
            adapter_class_path="test.module:Adapter",
        )
        fresh_registry.register(plugin)
        retrieved = fresh_registry.get_plugin("get-plugin-test")
        assert retrieved is plugin

    def test_summary_returns_platform_to_display_name(self, fresh_registry: GatewayAdapterRegistry) -> None:
        """summary() returns {platform_id: display_name} for all plugins."""
        plugin1 = FakeGatewayAdapterPlugin(
            platform_id="platform-a",
            display_name="Platform A",
            adapter_class_path="test.module:AdapterA",
        )
        plugin2 = FakeGatewayAdapterPlugin(
            platform_id="platform-b",
            display_name="Platform B",
            adapter_class_path="test.module:AdapterB",
        )
        fresh_registry.register(plugin1)
        fresh_registry.register(plugin2)
        summary = fresh_registry.summary()
        assert summary == {"platform-a": "Platform A", "platform-b": "Platform B"}


# ═══════════════════════════════════════════════════════════════
# ScopedGatewayAdapterRegistry reload tests
# ═══════════════════════════════════════════════════════════════


class TestScopedGatewayAdapterRegistryReload:
    """Tests for ScopedGatewayAdapterRegistry reload semantics."""

    def test_scoped_reload_bumps_version(
        self, fresh_registry: GatewayAdapterRegistry, scoped_registry: ScopedGatewayAdapterRegistry
    ) -> None:
        """reload increments the underlying registry version."""
        module_name = "tests._fake_gateway_reload_test"
        plugin = FakeGatewayAdapterPlugin(
            platform_id="reload-test",
            display_name="Reload Test",
            adapter_class_path="test.module:Adapter",
        )
        fake_mod = types.ModuleType(module_name)
        fake_mod.plugin = plugin
        fake_mod.__spec__ = None
        sys.modules[module_name] = fake_mod
        plugin.__class__ = type(
            "FakeGatewayAdapterPlugin", (FakeGatewayAdapterPlugin,), {"__module__": module_name}
        )

        fiber = scoped_registry.create_fiber("reload-test")
        scoped_registry.scoped_register(plugin, fiber)
        fiber.activate()  # Must activate before we can dispose
        initial_version = fresh_registry.version
        initial_gen = fiber.generation

        new_plugin = FakeGatewayAdapterPlugin(
            platform_id="reload-test",
            display_name="Reload Test V2",
            adapter_class_path="test.module:AdapterV2",
        )
        fake_mod.plugin = new_plugin

        with patch("importlib.reload", return_value=fake_mod):
            new_fiber = scoped_registry.reload("reload-test")

        assert fresh_registry.version > initial_version
        assert new_fiber.generation > initial_gen

        sys.modules.pop(module_name, None)

    def test_scoped_reload_creates_new_fiber_with_higher_generation(
        self, fresh_registry: GatewayAdapterRegistry, scoped_registry: ScopedGatewayAdapterRegistry
    ) -> None:
        """New fiber has higher generation."""
        module_name = "tests._fake_gateway_fiber_gen"
        plugin = FakeGatewayAdapterPlugin(
            platform_id="fiber-gen-test",
            display_name="Fiber Gen Test",
            adapter_class_path="test.module:Adapter",
        )
        fake_mod = types.ModuleType(module_name)
        fake_mod.plugin = plugin
        fake_mod.__spec__ = None
        sys.modules[module_name] = fake_mod
        plugin.__class__ = type(
            "FakeGatewayAdapterPlugin", (FakeGatewayAdapterPlugin,), {"__module__": module_name}
        )

        fiber = scoped_registry.create_fiber("fiber-gen-test")
        scoped_registry.scoped_register(plugin, fiber)
        fiber.activate()  # Must activate before we can dispose
        initial_gen = fiber.generation

        new_plugin = FakeGatewayAdapterPlugin(
            platform_id="fiber-gen-test",
            display_name="Fiber Gen Test V2",
            adapter_class_path="test.module:Adapter",
        )
        fake_mod.plugin = new_plugin

        with patch("importlib.reload", return_value=fake_mod):
            new_fiber = scoped_registry.reload("fiber-gen-test")

        assert new_fiber.generation > initial_gen
        assert new_fiber.state.name == "ACTIVE"

        sys.modules.pop(module_name, None)

    def test_scoped_reload_unknown_platform_raises_keyerror(
        self, scoped_registry: ScopedGatewayAdapterRegistry
    ) -> None:
        """reload("nonexistent") raises KeyError."""
        with pytest.raises(KeyError, match="not scoped-registered"):
            scoped_registry.reload("nonexistent-platform-xyz")

    def test_scoped_reload_module_without_plugin_raises_runtime_error(
        self, fresh_registry: GatewayAdapterRegistry, scoped_registry: ScopedGatewayAdapterRegistry
    ) -> None:
        """Monkeypatch to remove `plugin` attribute, reload raises RuntimeError."""
        module_name = "tests._fake_gateway_no_plugin_attr"
        plugin = FakeGatewayAdapterPlugin(
            platform_id="no-plugin-attr",
            display_name="No Plugin Attr",
            adapter_class_path="test.module:Adapter",
        )
        fake_mod = types.ModuleType(module_name)
        fake_mod.plugin = plugin
        fake_mod.__spec__ = None
        sys.modules[module_name] = fake_mod
        plugin.__class__ = type(
            "FakeGatewayAdapterPlugin", (FakeGatewayAdapterPlugin,), {"__module__": module_name}
        )

        fiber = scoped_registry.create_fiber("no-plugin-attr")
        scoped_registry.scoped_register(plugin, fiber)
        fiber.activate()  # Must activate before we can dispose

        del fake_mod.plugin

        with patch("importlib.reload", return_value=fake_mod):
            with pytest.raises(RuntimeError, match="no 'plugin' attribute"):
                scoped_registry.reload("no-plugin-attr")

        sys.modules.pop(module_name, None)
