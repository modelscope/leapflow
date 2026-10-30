# Copyright (c) Alibaba, Inc. and its affiliates.
"""Scoped lifecycle wrapper for GatewayAdapterRegistry.

Leverages the existing unregister() method for cleanup, and mirrors the
Tool subsystem's ScopedToolRegistry.reload() semantics: dispose the old
fiber, re-import the plugin module, register a fresh instance under a new
fiber, and bump the registry version for cache invalidation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from leapflow.domain.effect_scope import EffectScope
from leapflow.domain.plugin_fiber import FiberState, PluginFiber

logger = logging.getLogger(__name__)


class ScopedGatewayAdapterRegistry:
    """Composition wrapper adding lifecycle to GatewayAdapterRegistry."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._fibers: dict[str, PluginFiber] = {}
        # platform_id → dotted module path, used by reload() to re-import.
        self._plugin_modules: dict[str, str] = {}

    def create_fiber(self, platform_id: str) -> PluginFiber:
        """Create a fiber for a gateway adapter plugin."""
        scope = EffectScope(f"gateway-adapter:{platform_id}")
        fiber = PluginFiber(plugin_id=platform_id, scope=scope)
        self._fibers[platform_id] = fiber
        return fiber

    def get_fiber(self, platform_id: str) -> Optional[PluginFiber]:
        return self._fibers.get(platform_id)

    def scoped_register(self, plugin: Any, fiber: PluginFiber) -> None:
        """Register an adapter plugin with lifecycle tracking."""
        platform_id = plugin.platform_id
        # Resolve the module that exposes the module-level `plugin` attribute.
        # BuiltinAdapterPlugin stores the owning module in _adapter_module;
        # external plugins use their class's __module__ directly.
        module_path = getattr(plugin, "_adapter_module", None) or plugin.__class__.__module__
        self._plugin_modules[platform_id] = module_path
        self._registry.register(plugin)

        def _cleanup() -> None:
            self._registry.unregister(platform_id)
            logger.debug("Scoped-unregistered gateway adapter '%s'", platform_id)

        fiber.scope.effect(_cleanup)
        logger.debug("Scoped-registered gateway adapter '%s'", platform_id)

    def reload(self, platform_id: str) -> PluginFiber:
        """Reload a gateway adapter plugin: dispose old fiber, re-import module,
        register a fresh instance under a new fiber.

        Returns the new PluginFiber in ACTIVE state.

        Raises:
            KeyError: if platform_id was never scoped-registered.
            RuntimeError: if the module cannot be reloaded or has no ``plugin`` attribute.
        """
        if platform_id not in self._fibers:
            raise KeyError(
                f"Gateway adapter '{platform_id}' not scoped-registered"
            )

        module_path = self._plugin_modules.get(platform_id)
        if module_path is None:
            raise RuntimeError(
                f"Module path unknown for gateway adapter '{platform_id}'"
            )

        old_fiber = self._fibers[platform_id]

        # 1. Dispose old fiber — EffectScope cleanup runs unregister().
        if old_fiber.state == FiberState.ACTIVE:
            old_fiber.begin_unload()
        if old_fiber.state != FiberState.DISPOSED:
            old_fiber.dispose()

        # 2. Re-import the plugin module to get a fresh instance.
        import importlib
        import sys
        if module_path not in sys.modules:
            raise RuntimeError(
                f"Gateway module '{module_path}' not in sys.modules; cannot reload"
            )
        fresh_module = importlib.reload(sys.modules[module_path])
        fresh_plugin = getattr(fresh_module, "plugin", None)
        if fresh_plugin is None:
            raise RuntimeError(
                f"Reloaded module '{module_path}' has no 'plugin' attribute"
            )

        # 3. Create new fiber and register the fresh plugin.
        new_fiber = self.create_fiber(platform_id)
        self.scoped_register(fresh_plugin, new_fiber)
        new_fiber.activate()

        # 4. Bump the registry version so consumers invalidate any caches.
        self._registry.notify_mutation()

        return new_fiber

    def adopt_existing_plugins(self) -> None:
        """Create fibers for adapters already registered directly on the underlying registry.

        Used during boot to bring all built-in gateway adapters under fiber lifecycle
        management WITHOUT re-registering them (which would overwrite existing entries).
        """
        for platform_id in self._registry.list_available():
            if platform_id in self._fibers:
                continue  # already adopted
            plugin = self._registry.get_plugin(platform_id)
            if plugin is None:
                continue
            fiber = self.create_fiber(platform_id)
            module_path = getattr(plugin, "_adapter_module", None) or plugin.__class__.__module__
            self._plugin_modules[platform_id] = module_path

            def _cleanup(pid: str = platform_id) -> None:
                self._registry.unregister(pid)
                logger.debug("Scoped-unregistered gateway adapter '%s'", pid)

            fiber.scope.effect(_cleanup)
            fiber.activate()

    @property
    def fibers(self) -> dict[str, PluginFiber]:
        return dict(self._fibers)
