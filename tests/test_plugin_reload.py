# Copyright (c) Alibaba, Inc. and its affiliates.
"""Comprehensive tests for plugin reload lifecycle.

Covers:
- Version bumping and fiber generation
- Error handling for unknown/missing/malformed plugins
- Preservation of sibling plugins
- Handler object replacement
- gp_ alias lifecycle
- Engine cache invalidation
- Config-driven plugin disabling
- Late-bound dependency re-injection
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from leapflow.domain.plugin_fiber import PluginFiber, FiberState
from leapflow.plugins.protocol import ToolMetadata
from leapflow.plugins.registry import ToolPluginRegistry
from leapflow.plugins.scoped_registry import ScopedToolRegistry


# ════════════════════════════════════════════════════════════════
# Fake implementations
# ════════════════════════════════════════════════════════════════


def _handler_v1(**kwargs: Any) -> str:
    return "v1"


def _handler_v2(**kwargs: Any) -> str:
    return "v2"


def _make_tool_metadata(name: str, handler: Any = None) -> ToolMetadata:
    """Create a minimal ToolMetadata for testing."""
    return ToolMetadata(
        name=name,
        description=f"Test tool: {name}",
        parameters_schema={"type": "object", "properties": {}},
        handler=handler or _handler_v1,
    )


@dataclass
class FakeToolPlugin:
    """Minimal ToolPlugin implementation for testing."""

    _plugin_id: str
    _tools: list[ToolMetadata] = field(default_factory=list)
    _category: str = "test"
    _dependencies: list[str] = field(default_factory=list)
    _bound_deps: dict[str, Any] = field(default_factory=dict)

    @property
    def plugin_id(self) -> str:
        return self._plugin_id

    @property
    def category(self) -> str:
        return self._category

    @property
    def tools(self) -> list[ToolMetadata]:
        return self._tools

    @property
    def dependencies(self) -> list[str]:
        return self._dependencies

    def bind_runtime(self, **deps: Any) -> None:
        self._bound_deps.update(deps)


# ════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════


@pytest.fixture
def fresh_registry() -> ToolPluginRegistry:
    """Create a fresh ToolPluginRegistry without any built-in plugins."""
    return ToolPluginRegistry()


@pytest.fixture
def scoped_registry(fresh_registry: ToolPluginRegistry) -> ScopedToolRegistry:
    """Create a ScopedToolRegistry wrapping a fresh ToolPluginRegistry."""
    return ScopedToolRegistry(fresh_registry)


def _register_and_assemble(
    scoped: ScopedToolRegistry,
    plugin: FakeToolPlugin,
    registry: ToolPluginRegistry,
) -> PluginFiber:
    """Helper: create fiber, scoped-register, assemble, and activate."""
    fiber = scoped.create_fiber(plugin.plugin_id)
    scoped.scoped_register(plugin, fiber)
    registry.assemble()
    fiber.activate()
    return fiber


def _make_fake_module(plugin_instance: Any, module_name: str) -> types.ModuleType:
    """Create a fake module with a `plugin` attribute and install in sys.modules."""
    mod = types.ModuleType(module_name)
    mod.plugin = plugin_instance
    mod.__spec__ = None  # Prevent importlib.reload from erroring on missing spec
    return mod


# ════════════════════════════════════════════════════════════════
# Test 1: reload replaces plugin and bumps version
# ════════════════════════════════════════════════════════════════


class TestReloadReplacesPluginAndBumpsVersion:
    """Register a plugin, capture initial version + fiber generation, reload, verify bumps."""

    def test_reload_replaces_plugin_and_bumps_version(
        self, fresh_registry: ToolPluginRegistry, scoped_registry: ScopedToolRegistry
    ) -> None:
        plugin = FakeToolPlugin(
            _plugin_id="reload-test",
            _tools=[_make_tool_metadata("reload_tool")],
        )

        # Create a fake module that importlib.reload will hit
        module_name = "tests._fake_reload_test_module"
        fake_mod = _make_fake_module(plugin, module_name)
        sys.modules[module_name] = fake_mod

        # Patch __class__.__module__ on the plugin so scoped_register records the right path
        plugin.__class__ = type(
            "FakeToolPlugin", (FakeToolPlugin,), {"__module__": module_name}
        )

        fiber = _register_and_assemble(scoped_registry, plugin, fresh_registry)
        initial_gen = fiber.generation
        initial_version = fresh_registry.version

        # Prepare a new plugin instance for the reload to discover
        new_plugin = FakeToolPlugin(
            _plugin_id="reload-test",
            _tools=[_make_tool_metadata("reload_tool")],
        )
        fake_mod.plugin = new_plugin

        # Perform reload
        with patch("importlib.reload", return_value=fake_mod):
            new_fiber = scoped_registry.reload("reload-test")

        assert new_fiber.generation > initial_gen
        assert fresh_registry.version > initial_version
        assert new_fiber.state == FiberState.ACTIVE

        # Cleanup
        sys.modules.pop(module_name, None)


# ════════════════════════════════════════════════════════════════
# Test 2: reload unknown plugin raises KeyError
# ════════════════════════════════════════════════════════════════


class TestReloadUnknownPluginRaisesKeyError:
    """Calling reload on a never-registered plugin must raise KeyError."""

    def test_reload_unknown_plugin_raises_keyerror(
        self, scoped_registry: ScopedToolRegistry
    ) -> None:
        with pytest.raises(KeyError, match="nonexistent_plugin_xyz"):
            scoped_registry.reload("nonexistent_plugin_xyz")


# ════════════════════════════════════════════════════════════════
# Test 3: reload preserves other plugins
# ════════════════════════════════════════════════════════════════


class TestReloadPreservesOtherPlugins:
    """Reloading plugin A does not affect plugin B."""

    def test_reload_preserves_other_plugins(
        self, fresh_registry: ToolPluginRegistry, scoped_registry: ScopedToolRegistry
    ) -> None:
        module_a = "tests._fake_plugin_a"
        module_b = "tests._fake_plugin_b"

        plugin_a = FakeToolPlugin(
            _plugin_id="plugin-a",
            _tools=[_make_tool_metadata("tool_a")],
        )
        plugin_b = FakeToolPlugin(
            _plugin_id="plugin-b",
            _tools=[_make_tool_metadata("tool_b")],
        )

        # Patch __module__ for A
        plugin_a.__class__ = type(
            "FakePluginA", (FakeToolPlugin,), {"__module__": module_a}
        )

        # Install fake modules
        fake_mod_a = _make_fake_module(plugin_a, module_a)
        fake_mod_b = _make_fake_module(plugin_b, module_b)
        sys.modules[module_a] = fake_mod_a
        sys.modules[module_b] = fake_mod_b

        # Register both
        fiber_a = scoped_registry.create_fiber("plugin-a")
        scoped_registry.scoped_register(plugin_a, fiber_a)
        fiber_b = scoped_registry.create_fiber("plugin-b")
        scoped_registry.scoped_register(plugin_b, fiber_b)

        fresh_registry.assemble()

        fiber_a.activate()
        fiber_b.activate()

        # Capture B's fiber gen
        b_gen_before = fiber_b.generation

        # Reload A only
        new_plugin_a = FakeToolPlugin(
            _plugin_id="plugin-a",
            _tools=[_make_tool_metadata("tool_a")],
        )
        fake_mod_a.plugin = new_plugin_a

        with patch("importlib.reload", return_value=fake_mod_a):
            scoped_registry.reload("plugin-a")

        # B is untouched
        assert "tool_b" in fresh_registry._tool_handlers
        assert "plugin-b" in fresh_registry._plugins
        fiber_b_current = scoped_registry.get_fiber("plugin-b")
        assert fiber_b_current is fiber_b
        assert fiber_b.generation == b_gen_before

        # Cleanup
        sys.modules.pop(module_a, None)
        sys.modules.pop(module_b, None)


# ════════════════════════════════════════════════════════════════
# Test 4: reload replaces tool handlers
# ════════════════════════════════════════════════════════════════


class TestReloadReplacesToolHandlers:
    """After reload, tool handlers point to new function objects."""

    def test_reload_replaces_tool_handlers(
        self, fresh_registry: ToolPluginRegistry, scoped_registry: ScopedToolRegistry
    ) -> None:
        module_name = "tests._fake_handler_replace"
        plugin = FakeToolPlugin(
            _plugin_id="handler-test",
            _tools=[_make_tool_metadata("htool", handler=_handler_v1)],
        )
        plugin.__class__ = type(
            "FakePlugin", (FakeToolPlugin,), {"__module__": module_name}
        )

        fake_mod = _make_fake_module(plugin, module_name)
        sys.modules[module_name] = fake_mod

        _register_and_assemble(scoped_registry, plugin, fresh_registry)

        # Capture original handler
        original_handler = fresh_registry.tool_handlers["htool"]
        assert original_handler is _handler_v1

        # Prepare new plugin with different handler
        new_plugin = FakeToolPlugin(
            _plugin_id="handler-test",
            _tools=[_make_tool_metadata("htool", handler=_handler_v2)],
        )
        fake_mod.plugin = new_plugin

        with patch("importlib.reload", return_value=fake_mod):
            scoped_registry.reload("handler-test")

        # New handler is NOT the original
        new_handler = fresh_registry.tool_handlers["htool"]
        assert new_handler is not original_handler
        assert new_handler is _handler_v2

        # Cleanup
        sys.modules.pop(module_name, None)


# ════════════════════════════════════════════════════════════════
# Test 5: reload does NOT produce gp_ aliases (Landing B)
# ════════════════════════════════════════════════════════════════


class TestReloadDoesNotProduceGpAliases:
    """After Landing B, reload() only adds the plain tool name — no gp_ alias."""

    def test_reload_only_adds_plain_name(self
        , fresh_registry: ToolPluginRegistry, scoped_registry: ScopedToolRegistry
    ) -> None:
        module_name = "tests._fake_gp_alias"
        plugin = FakeToolPlugin(
            _plugin_id="alias-plugin",
            _tools=[_make_tool_metadata("foo")],
        )
        plugin.__class__ = type(
            "FakePlugin", (FakeToolPlugin,), {"__module__": module_name}
        )

        fake_mod = _make_fake_module(plugin, module_name)
        sys.modules[module_name] = fake_mod

        _register_and_assemble(scoped_registry, plugin, fresh_registry)

        # Verify plain name present, no gp_ alias
        assert "foo" in fresh_registry._tool_handlers
        assert "gp_foo" not in fresh_registry._tool_handlers

        # Prepare new plugin for reload
        new_plugin = FakeToolPlugin(
            _plugin_id="alias-plugin",
            _tools=[_make_tool_metadata("foo")],
        )
        fake_mod.plugin = new_plugin

        with patch("importlib.reload", return_value=fake_mod):
            scoped_registry.reload("alias-plugin")

        # Only plain name re-added, no gp_ alias
        assert "foo" in fresh_registry._tool_handlers
        assert "gp_foo" not in fresh_registry._tool_handlers

        # Cleanup
        sys.modules.pop(module_name, None)


# ════════════════════════════════════════════════════════════════
# Test 6: engine cache invalidates on same-size reload
# ════════════════════════════════════════════════════════════════


class TestReloadEngineCacheInvalidatesOnSameSize:
    """Engine _registry_cache invalidates even when tool count is unchanged."""

    def test_reload_engine_cache_invalidates_on_same_size(
        self, fresh_registry: ToolPluginRegistry, scoped_registry: ScopedToolRegistry
    ) -> None:
        module_name = "tests._fake_cache_invalidation"
        plugin = FakeToolPlugin(
            _plugin_id="cache-test",
            _tools=[_make_tool_metadata("cache_tool")],
        )
        plugin.__class__ = type(
            "FakePlugin", (FakeToolPlugin,), {"__module__": module_name}
        )

        fake_mod = _make_fake_module(plugin, module_name)
        sys.modules[module_name] = fake_mod

        _register_and_assemble(scoped_registry, plugin, fresh_registry)

        # The cache key uses (len(td), len(th), version)
        # We simulate the engine cache by tracking version before/after
        version_before = fresh_registry.version

        # Prepare new plugin (same number of tools)
        new_plugin = FakeToolPlugin(
            _plugin_id="cache-test",
            _tools=[_make_tool_metadata("cache_tool")],
        )
        fake_mod.plugin = new_plugin

        with patch("importlib.reload", return_value=fake_mod):
            scoped_registry.reload("cache-test")

        version_after = fresh_registry.version

        # Version must have bumped (notify_mutation was called)
        assert version_after > version_before
        # Even though tool count is the same, version differs → cache would be invalidated

        # Cleanup
        sys.modules.pop(module_name, None)


# ════════════════════════════════════════════════════════════════
# Test 7: disabled plugin not registered
# ════════════════════════════════════════════════════════════════


class TestDisabledPluginNotRegistered:
    """When disabled_plugins includes a plugin_id, _discover_all() skips it."""

    def test_disabled_plugin_not_registered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Create a mock settings object with disabled_plugins set
        mock_settings = MagicMock()
        mock_settings.disabled_plugins = ("text_utils",)

        # Patch get_settings at the import source so _discover_all() picks it up
        monkeypatch.setattr(
            "leapflow.config.get_settings",
            lambda: mock_settings,
        )

        # We need to bypass the lazy singleton
        from leapflow.plugins.tool_plugins import _discover_all

        # Patch importlib.import_module to return controlled modules
        # that simulate just text_utils and system_info.
        # We need to keep leapflow.plugins.protocol importable for the ToolPlugin check.
        _real_import = __import__("importlib").import_module

        fake_text_utils = MagicMock()
        fake_text_utils.plugin.plugin_id = "text_utils"
        fake_system_info = MagicMock()
        fake_system_info.plugin.plugin_id = "system_info"

        def _mock_import(module_path: str) -> Any:
            if module_path == "leapflow.plugins.tool_plugins.text_utils":
                return fake_text_utils
            if module_path == "leapflow.plugins.tool_plugins.system_info":
                return fake_system_info
            if module_path.startswith("leapflow.plugins.tool_plugins."):
                raise ImportError(f"not testing: {module_path}")
            return _real_import(module_path)

        monkeypatch.setattr("importlib.import_module", _mock_import)

        plugins = _discover_all()
        plugin_ids = [p.plugin_id for p in plugins]

        # text_utils is disabled, should not appear
        assert "text_utils" not in plugin_ids
        # system_info should still be discovered
        assert "system_info" in plugin_ids


# ════════════════════════════════════════════════════════════════
# Test 8: reload missing module raises RuntimeError
# ════════════════════════════════════════════════════════════════


class TestReloadMissingModuleRaisesRuntimeError:
    """If the module is not in sys.modules, reload raises RuntimeError."""

    def test_reload_missing_module_raises_runtime_error(
        self, fresh_registry: ToolPluginRegistry, scoped_registry: ScopedToolRegistry
    ) -> None:
        # Manually set up a fiber and module_path pointing to a module NOT in sys.modules
        module_name = "tests._nonexistent_reload_module_xyz"

        plugin = FakeToolPlugin(
            _plugin_id="missing-mod",
            _tools=[_make_tool_metadata("missing_tool")],
        )
        plugin.__class__ = type(
            "FakePlugin", (FakeToolPlugin,), {"__module__": module_name}
        )

        # Install the module temporarily for registration
        fake_mod = _make_fake_module(plugin, module_name)
        sys.modules[module_name] = fake_mod

        _register_and_assemble(scoped_registry, plugin, fresh_registry)

        # Now remove the module from sys.modules BEFORE reload
        sys.modules.pop(module_name, None)

        with pytest.raises(RuntimeError, match="not in sys.modules"):
            scoped_registry.reload("missing-mod")


# ════════════════════════════════════════════════════════════════
# Test 9: reload module without plugin attribute raises
# ════════════════════════════════════════════════════════════════


class TestReloadModuleWithoutPluginAttrRaises:
    """If the reloaded module has no `plugin` attribute, reload raises RuntimeError."""

    def test_reload_module_without_plugin_attr_raises(
        self, fresh_registry: ToolPluginRegistry, scoped_registry: ScopedToolRegistry
    ) -> None:
        module_name = "tests._fake_no_plugin_attr"

        plugin = FakeToolPlugin(
            _plugin_id="no-attr",
            _tools=[_make_tool_metadata("noattr_tool")],
        )
        plugin.__class__ = type(
            "FakePlugin", (FakeToolPlugin,), {"__module__": module_name}
        )

        fake_mod = _make_fake_module(plugin, module_name)
        sys.modules[module_name] = fake_mod

        _register_and_assemble(scoped_registry, plugin, fresh_registry)

        # Remove the `plugin` attribute from the module BEFORE reload
        del fake_mod.plugin

        with patch("importlib.reload", return_value=fake_mod):
            with pytest.raises(RuntimeError, match="no 'plugin' attribute"):
                scoped_registry.reload("no-attr")

        # Cleanup
        sys.modules.pop(module_name, None)


# ════════════════════════════════════════════════════════════════
# Test 10: reload re-injects last-bound dependencies
# ════════════════════════════════════════════════════════════════


class TestReloadReinjectsLastBoundDeps:
    """After reload, the new plugin instance receives previously-bound runtime deps."""

    def test_reload_reinjects_last_bound_deps(
        self, fresh_registry: ToolPluginRegistry, scoped_registry: ScopedToolRegistry
    ) -> None:
        module_name = "tests._fake_deps_reinject"

        plugin = FakeToolPlugin(
            _plugin_id="deps-test",
            _tools=[_make_tool_metadata("deps_tool")],
            _dependencies=["memory_manager"],
        )
        plugin.__class__ = type(
            "FakePlugin", (FakeToolPlugin,), {"__module__": module_name}
        )

        fake_mod = _make_fake_module(plugin, module_name)
        sys.modules[module_name] = fake_mod

        _register_and_assemble(scoped_registry, plugin, fresh_registry)

        # Bind a runtime dependency
        mock_memory = MagicMock(name="MockMemoryManager")
        fresh_registry.bind_runtime(memory_manager=mock_memory)

        # Verify the original plugin received it
        assert plugin._bound_deps.get("memory_manager") is mock_memory

        # Prepare new plugin for reload (also declares memory_manager dep)
        new_plugin = FakeToolPlugin(
            _plugin_id="deps-test",
            _tools=[_make_tool_metadata("deps_tool")],
            _dependencies=["memory_manager"],
        )
        fake_mod.plugin = new_plugin

        with patch("importlib.reload", return_value=fake_mod):
            scoped_registry.reload("deps-test")

        # The new plugin instance should have received the dep via bind_runtime re-injection
        registered_plugin = fresh_registry._plugins.get("deps-test")
        assert registered_plugin is new_plugin
        assert new_plugin._bound_deps.get("memory_manager") is mock_memory

        # Cleanup
        sys.modules.pop(module_name, None)


# ════════════════════════════════════════════════════════════════
# Test 11: disabled_plugins E2E - tools excluded at runtime
# ════════════════════════════════════════════════════════════════


class TestDisabledPluginsEndToEnd:
    """End-to-end verification that disabled_plugins config actually excludes tools at runtime."""

    def test_disabled_plugins_removes_tools_from_fresh_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting disabled_plugins in settings causes those tools to not appear in a fresh registry."""
        
        # Force a settings object with disabled_plugins set
        from leapflow.config import get_settings as _real_get_settings
        real_settings = _real_get_settings()
        
        # Create a modified settings (frozen dataclass — use dataclasses.replace)
        import dataclasses
        modified = dataclasses.replace(real_settings, disabled_plugins=("text_utils",))
        
        # Patch get_settings to return our modified one
        import leapflow.config
        monkeypatch.setattr(leapflow.config, "get_settings", lambda: modified)
        # Also patch it in the plugins module in case it caches
        import leapflow.plugins.tool_plugins as plugins_mod
        # Reset the cached ALL_PLUGINS
        plugins_mod._all_plugins = None
        
        # Create a fresh registry and discover
        from leapflow.plugins.registry import ToolPluginRegistry
        reg = ToolPluginRegistry()
        reg.discover_builtin()
        reg.assemble()
        
        # text_utils should not be registered
        assert "text_utils" not in reg._plugins
        
        # Clean up: reset the cached list so other tests aren't affected
        plugins_mod._all_plugins = None


def _profile_plugin_source(version: str, *, extra_tool: bool = False) -> str:
    extra = ""
    if extra_tool:
        extra = ''',\n            ToolMetadata(\n                name="profile_version",\n                description="Return the active profile plugin version.",\n                parameters_schema={"type": "object", "properties": {}},\n                handler=self._version,\n                x_leapflow={"category": "profile", "risk_level": "read_only"},\n            )'''
    return f'''from __future__ import annotations
from typing import Any
from leapflow.plugins.protocol import ToolMetadata

VERSION = "{version}"

class ProfilePlugin:
    @property
    def plugin_id(self) -> str: return "profile_echo"
    @property
    def category(self) -> str: return "profile"
    @property
    def dependencies(self) -> list[str]: return []
    def bind_runtime(self, **deps: Any) -> None: return None
    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="profile_echo",
                description="Echo with profile plugin version.",
                parameters_schema={{"type": "object", "properties": {{"text": {{"type": "string"}}}}}},
                handler=self._echo,
                x_leapflow={{"category": "profile", "risk_level": "read_only"}},
            ){extra}
        ]
    async def _echo(self, text: str = "", **kwargs: Any) -> dict[str, Any]:
        return {{"ok": True, "version": VERSION, "text": text}}
    async def _version(self, **kwargs: Any) -> dict[str, Any]:
        return {{"ok": True, "version": VERSION}}

plugin = ProfilePlugin()
'''


def test_profile_scoped_plugin_discovery(monkeypatch, tmp_path) -> None:
    plugin_file = tmp_path / "profile_echo.py"
    plugin_file.write_text(_profile_plugin_source("v0"), encoding="utf-8")
    import leapflow.plugins.tool_plugins as discovery

    monkeypatch.setattr(discovery, "_profile_plugins_dir", lambda: tmp_path)
    discovered = discovery.discover_profile_plugins(disabled=set())

    assert [plugin.plugin_id for plugin in discovered] == ["profile_echo"]
    assert getattr(discovered[0], "__leapflow_plugin_path__") == str(plugin_file)


def test_file_backed_plugin_reload_without_syspath(monkeypatch, tmp_path) -> None:
    plugin_file = tmp_path / "profile_echo.py"
    plugin_file.write_text(_profile_plugin_source("v0"), encoding="utf-8")
    import leapflow.plugins.tool_plugins as discovery

    monkeypatch.setattr(discovery, "_profile_plugins_dir", lambda: tmp_path)
    plugin = discovery.discover_profile_plugins(disabled=set())[0]
    registry = ToolPluginRegistry()
    registry.register(plugin)
    registry.assemble()
    scoped = ScopedToolRegistry(registry)
    scoped.adopt_existing_plugins()

    assert registry.get_plugin("profile_echo") is not None
    plugin_file.write_text(_profile_plugin_source("v1", extra_tool=True), encoding="utf-8")
    fresh_fiber = scoped.reload("profile_echo")

    assert fresh_fiber.state == FiberState.ACTIVE
    assert scoped.get_plugin_file("profile_echo") == plugin_file
    assert "profile_echo" in registry.tool_handlers
    assert "profile_version" in registry.tool_handlers
    assert registry.get_plugin("profile_echo").tools[0].handler.__self__.__class__.__module__ == "profile_echo"
