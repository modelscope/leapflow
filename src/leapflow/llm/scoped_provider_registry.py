# Copyright (c) Alibaba, Inc. and its affiliates.
"""Scoped lifecycle wrapper for LLMProviderRegistry.

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


class ScopedLLMProviderRegistry:
    """Composition wrapper adding lifecycle to LLMProviderRegistry."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._fibers: dict[str, PluginFiber] = {}
        # provider_id → dotted module path, used by reload() to re-import.
        self._plugin_modules: dict[str, str] = {}

    def create_fiber(self, provider_id: str) -> PluginFiber:
        """Create a fiber for an LLM provider plugin."""
        scope = EffectScope(f"llm-provider:{provider_id}")
        fiber = PluginFiber(plugin_id=provider_id, scope=scope)
        self._fibers[provider_id] = fiber
        return fiber

    def get_fiber(self, provider_id: str) -> Optional[PluginFiber]:
        return self._fibers.get(provider_id)

    def scoped_register(self, plugin: Any, fiber: PluginFiber) -> None:
        """Register a provider plugin with lifecycle tracking."""
        provider_id = plugin.provider_id
        # Remember the module so reload() can re-import a fresh instance.
        self._plugin_modules[provider_id] = plugin.__class__.__module__
        self._registry.register(plugin)

        def _cleanup() -> None:
            self._registry.unregister(provider_id)
            logger.debug("Scoped-unregistered LLM provider '%s'", provider_id)

        fiber.scope.effect(_cleanup)
        logger.debug("Scoped-registered LLM provider '%s'", provider_id)

    def reload(self, provider_id: str) -> PluginFiber:
        """Reload an LLM provider plugin: dispose old fiber, re-import module,
        register a fresh instance under a new fiber.

        Returns the new PluginFiber in ACTIVE state.

        Raises:
            KeyError: if provider_id was never scoped-registered.
            RuntimeError: if the module cannot be reloaded or has no ``plugin`` attribute.
        """
        if provider_id not in self._fibers:
            raise KeyError(
                f"LLM provider '{provider_id}' not scoped-registered"
            )

        module_path = self._plugin_modules.get(provider_id)
        if module_path is None:
            raise RuntimeError(
                f"Module path unknown for LLM provider '{provider_id}'"
            )

        old_fiber = self._fibers[provider_id]

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
                f"LLM module '{module_path}' not in sys.modules; cannot reload"
            )
        fresh_module = importlib.reload(sys.modules[module_path])
        fresh_plugin = getattr(fresh_module, "plugin", None)
        if fresh_plugin is None:
            raise RuntimeError(
                f"Reloaded module '{module_path}' has no 'plugin' attribute"
            )

        # 3. Create new fiber and register the fresh plugin.
        new_fiber = self.create_fiber(provider_id)
        self.scoped_register(fresh_plugin, new_fiber)
        new_fiber.activate()

        # 4. Bump the registry version so consumers invalidate any caches.
        self._registry.notify_mutation()

        return new_fiber

    def adopt_existing_plugins(self) -> None:
        """Create fibers for providers already registered directly on the underlying registry.

        Used during boot to bring all built-in LLM providers under fiber lifecycle
        management WITHOUT re-registering them (which would replace existing entries).
        """
        for provider_id in self._registry.list_available():
            if provider_id in self._fibers:
                continue  # already adopted
            plugin = self._registry.get_plugin(provider_id)
            if plugin is None:
                continue
            fiber = self.create_fiber(provider_id)
            self._plugin_modules[provider_id] = plugin.__class__.__module__

            def _cleanup(pid: str = provider_id) -> None:
                self._registry.unregister(pid)
                logger.debug("Scoped-unregistered LLM provider '%s'", pid)

            fiber.scope.effect(_cleanup)
            fiber.activate()

    @property
    def fibers(self) -> dict[str, PluginFiber]:
        return dict(self._fibers)
