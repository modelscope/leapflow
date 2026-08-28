"""Scoped lifecycle wrapper for ToolPluginRegistry.

Provides reversible plugin registration: registering through this wrapper
automatically tracks cleanup effects on a PluginFiber's scope. When the
fiber is disposed, the plugin's tools are removed from the underlying registry.

This is a composition wrapper — the underlying ToolPluginRegistry is NOT modified.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path
from typing import Any, Optional

from leapflow.domain.effect_scope import EffectScope
from leapflow.domain.plugin_fiber import PluginFiber, FiberState
from leapflow.plugins.protocol import ToolPlugin

logger = logging.getLogger(__name__)


class ScopedToolRegistry:
    """Composition wrapper adding lifecycle management to ToolPluginRegistry.

    Usage:
        from leapflow.plugins import get_registry
        registry = get_registry()
        scoped = ScopedToolRegistry(registry)

        fiber = scoped.create_fiber("my-plugin")
        scoped.scoped_register(my_plugin, fiber)
        fiber.activate()
        # ... plugin tools are now available ...
        fiber.begin_unload()
        fiber.dispose()  # tools automatically removed
    """

    def __init__(self, registry: Any) -> None:
        """Wrap an existing ToolPluginRegistry instance."""
        self._registry = registry
        self._fibers: dict[str, PluginFiber] = {}
        self._plugin_modules: dict[str, str] = {}  # plugin_id → module path
        self._plugin_files: dict[str, Path] = {}  # plugin_id → installed source file

    def create_fiber(self, plugin_id: str) -> PluginFiber:
        """Create a new PluginFiber for managing a plugin's lifecycle."""
        scope = EffectScope(f"tool-plugin:{plugin_id}")
        fiber = PluginFiber(plugin_id=plugin_id, scope=scope)
        self._fibers[plugin_id] = fiber
        return fiber

    def get_fiber(self, plugin_id: str) -> Optional[PluginFiber]:
        """Get an existing fiber by plugin ID."""
        return self._fibers.get(plugin_id)

    def scoped_register(self, plugin: ToolPlugin, fiber: PluginFiber) -> None:
        """Register a plugin with lifecycle tracking.

        The plugin is registered on the underlying registry, and a cleanup
        effect is added to the fiber's scope that will remove all the plugin's
        tools when the fiber is disposed.
        """
        plugin_id = plugin.plugin_id
        # Track reload metadata so reload() can re-import the plugin later. A
        # generated wrapper may instantiate a class defined in a core module
        # (e.g. DshBridgePlugin), so prefer the wrapper module explicitly attached
        # by the file loader over plugin.__class__.__module__. Reloading the core
        # module from the wrapper path would overwrite sys.modules and make the
        # wrapper import itself recursively.
        self._plugin_modules[plugin_id] = str(
            getattr(plugin, "__leapflow_plugin_module__", plugin.__class__.__module__)
        )
        plugin_path = getattr(plugin, "__leapflow_plugin_path__", None)
        if plugin_path:
            self._plugin_files[plugin_id] = Path(str(plugin_path))
        # Register on underlying registry
        self._registry.register(plugin)

        # Capture tool names for cleanup
        tool_names = [t.name for t in plugin.tools]

        # Register cleanup effect on the fiber's scope
        def _cleanup() -> None:
            self._unregister_tools(plugin_id, tool_names)

        fiber.scope.effect(_cleanup)
        logger.debug("Scoped-registered plugin '%s' with %d tools", plugin_id, len(tool_names))

        # A newly registered plugin may satisfy a dependency that other fibers
        # are waiting on (LOADING). Re-run the activation check so those fibers
        # can transition LOADING → ACTIVE now that their provider is present.
        self._check_pending_activations()

    def scoped_register_late_tool(
        self,
        definition: dict[str, Any],
        handler: Any,
        name: str,
        fiber: PluginFiber,
    ) -> None:
        """Register a late tool with lifecycle tracking."""
        self._registry.register_late_tool(definition, handler, name)

        def _cleanup() -> None:
            self._remove_late_tool(name)

        fiber.scope.effect(_cleanup)

    def _unregister_tools(self, plugin_id: str, tool_names: list[str]) -> None:
        """Cleanup callback: remove plugin+tools from the underlying registry.

        Delegates to ToolPluginRegistry public API to preserve encapsulation.
        """
        # Try full plugin removal first (also removes from _plugins dict)
        if not self._registry.unregister_plugin(plugin_id):
            # Fallback: plugin not in registry (may have been removed already);
            # ensure tool names are cleaned up anyway.
            self._registry.unregister_tools(tool_names)

    def _remove_late_tool(self, name: str) -> None:
        """Remove a single late-registered tool via public API."""
        self._registry.unregister_tools([name])

    def adopt_existing_plugins(self) -> None:
        """Create fibers for plugins already registered directly on the underlying registry.

        Used during boot to bring all built-in plugins under fiber lifecycle management
        WITHOUT re-registering them (which would raise Duplicate plugin_id).

        Activation is dependency-driven: a plugin declaring no dependencies is
        activated immediately (PENDING → ACTIVE), while a plugin with declared
        dependencies enters LOADING (PENDING → LOADING) and is only promoted to
        ACTIVE once its dependencies are satisfiable. After every fiber has been
        seeded, ``_check_pending_activations()`` resolves the LOADING set to a
        fixpoint, and any straggler is force-activated for graceful degradation.
        """
        for plugin_id, plugin in self._registry.plugins.items():
            if plugin_id in self._fibers:
                continue  # already adopted
            fiber = self.create_fiber(plugin_id)
            self._plugin_modules[plugin_id] = str(
                getattr(plugin, "__leapflow_plugin_module__", plugin.__class__.__module__)
            )
            plugin_path = getattr(plugin, "__leapflow_plugin_path__", None)
            if plugin_path:
                self._plugin_files[plugin_id] = Path(str(plugin_path))
            tool_names = [t.name for t in plugin.tools]

            def _cleanup(pid: str = plugin_id, names: list = tool_names) -> None:
                self._unregister_tools(pid, names)

            fiber.scope.effect(_cleanup)
            # Backward-compatible fast path: no declared dependencies means the
            # plugin can activate right away, exactly as before P1.
            if plugin.dependencies:
                fiber.begin_loading()
            else:
                fiber.activate()

        # Promote every LOADING fiber whose dependencies are now satisfiable.
        self._check_pending_activations()
        # Any fiber still LOADING has unsatisfiable or late-bound dependencies;
        # force-activate it so a missing runtime dep never blocks boot.
        self._force_activate_stragglers()

    # ── Dependency-driven activation ──

    def _dependencies_satisfied(self, plugin: ToolPlugin) -> bool:
        """Return True when every declared dependency of *plugin* is available.

        A dependency name is satisfied when it is either present in the
        registry's last-bound runtime deps (injected via ``bind_runtime``) or
        provided by another plugin whose fiber is already ACTIVE (the provider's
        ``plugin_id`` equals the dependency name). Requiring the provider to be
        ACTIVE — not merely registered — is what lets a genuine dependency cycle
        deadlock cleanly instead of activating members out of order.
        """
        bound = self._registry.last_bound_deps
        for dep in plugin.dependencies:
            if dep in bound:
                continue
            provider = self._fibers.get(dep)
            if provider is not None and provider.state == FiberState.ACTIVE:
                continue
            return False
        return True

    def _check_pending_activations(self) -> None:
        """Activate LOADING fibers whose dependencies have become satisfiable.

        Runs to a fixpoint: activating one provider may satisfy a consumer that
        depends on it, so the scan repeats until no further fiber transitions.
        This makes activation order-independent — an arbitrary registration or
        discovery order still resolves a full provider → consumer chain.
        """
        progressed = True
        while progressed:
            progressed = False
            for plugin_id, fiber in self._fibers.items():
                if fiber.state != FiberState.LOADING:
                    continue
                plugin = self._registry.get_plugin(plugin_id)
                if plugin is None:
                    continue
                if self._dependencies_satisfied(plugin):
                    self._activate_fiber(plugin_id, fiber)
                    progressed = True

    def _activate_fiber(self, plugin_id: str, fiber: PluginFiber) -> None:
        """Transition a fiber to ACTIVE and bind its satisfied runtime deps.

        The transition tolerates both PENDING → ACTIVE and LOADING → ACTIVE.
        After activation the plugin receives any last-bound deps it declared, so
        a fiber promoted after ``bind_runtime`` still gets its injections.
        """
        if fiber.state == FiberState.ACTIVE:
            return
        fiber.activate()
        plugin = self._registry.get_plugin(plugin_id)
        if plugin is None:
            return
        relevant = {
            k: v
            for k, v in self._registry.last_bound_deps.items()
            if k in plugin.dependencies
        }
        if relevant:
            plugin.bind_runtime(**relevant)

    def _has_unsatisfied_plugin_dep(self, plugin_id: str) -> bool:
        """Return True if the plugin waits on another *plugin* that is not ACTIVE.

        Distinguishes a genuine inter-plugin dependency problem (a cycle or a
        provider that never activates) from an ordinary late-bound runtime dep
        (e.g. ``file_read_gate``) that is injected after boot via
        ``bind_runtime`` and legitimately absent at adoption time.
        """
        plugin = self._registry.get_plugin(plugin_id)
        if plugin is None:
            return False
        for dep in plugin.dependencies:
            provider = self._fibers.get(dep)
            if provider is not None and provider.state != FiberState.ACTIVE:
                return True
        return False

    def _force_activate_stragglers(self) -> None:
        """Force-activate any fiber still LOADING (graceful degradation).

        A straggler blocked on another plugin (cycle / never-activating provider)
        is force-activated with a warning; one merely awaiting a late-bound
        runtime dependency is force-activated quietly, since that dep arrives
        later through ``bind_runtime`` and the plugin already tolerates its
        temporary absence.
        """
        stragglers = [
            pid for pid, fiber in self._fibers.items()
            if fiber.state == FiberState.LOADING
        ]
        if not stragglers:
            return
        blocked = [pid for pid in stragglers if self._has_unsatisfied_plugin_dep(pid)]
        for plugin_id in stragglers:
            self._activate_fiber(plugin_id, self._fibers[plugin_id])
        if blocked:
            logger.warning(
                "Force-activated plugin fiber(s) with unresolved inter-plugin "
                "dependencies (possible cycle): %s",
                blocked,
            )
        else:
            logger.debug(
                "Activated plugin fiber(s) awaiting late-bound runtime deps: %s",
                stragglers,
            )

    @property
    def fibers(self) -> dict[str, PluginFiber]:
        """Read-only view of managed fibers."""
        return dict(self._fibers)

    def get_plugin_module(self, plugin_id: str) -> str | None:
        """Return the module path used to reload a plugin, if known."""
        return self._plugin_modules.get(plugin_id)

    def get_plugin_file(self, plugin_id: str) -> Path | None:
        """Return the file backing a profile-scoped plugin, if known."""
        return self._plugin_files.get(plugin_id)

    def dispose_plugin(self, plugin_id: str, *, prune_metadata: bool = False) -> PluginFiber:
        """Dispose a plugin fiber and remove its tools from the live registry.

        ``prune_metadata`` is reserved for terminal removal: disable keeps module
        metadata so plugin_enable/plugin_reload can bring the plugin back, while
        remove drops the reload metadata and sys.modules entry.
        """
        fiber = self._fibers.get(plugin_id)
        if fiber is None:
            raise KeyError(f"Plugin '{plugin_id}' has no fiber")
        if fiber.state == FiberState.ACTIVE:
            fiber.begin_unload()
        if fiber.state != FiberState.DISPOSED:
            fiber.dispose()
        if prune_metadata:
            module_path = self._plugin_modules.pop(plugin_id, None)
            self._plugin_files.pop(plugin_id, None)
            if module_path:
                sys.modules.pop(module_path, None)
        return fiber

    def _load_fresh_plugin(self, plugin_id: str, module_path: str) -> ToolPlugin:
        """Load a fresh plugin instance using file-backed reload when available."""
        file_path = self._plugin_files.get(plugin_id)
        if file_path is not None:
            module = self._load_module_from_file(module_path, file_path)
        else:
            if module_path not in sys.modules:
                raise RuntimeError(
                    f"Plugin module '{module_path}' not in sys.modules; cannot reload"
                )
            module = importlib.reload(sys.modules[module_path])
        plugin = getattr(module, "plugin", None)
        if plugin is None:
            raise RuntimeError(
                f"Reloaded module '{module_path}' has no 'plugin' attribute"
            )
        if file_path is not None:
            try:
                setattr(plugin, "__leapflow_plugin_path__", str(file_path))
                setattr(plugin, "__leapflow_plugin_module__", module_path)
            except Exception:
                logger.debug("Cannot attach plugin file path metadata for %s", plugin_id, exc_info=True)
        return plugin

    @staticmethod
    def _load_module_from_file(module_path: str, file_path: Path) -> Any:
        """Load ``module_path`` from current source text, bypassing pyc caches."""
        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Cannot read plugin file '{file_path}': {exc}") from exc
        module = types.ModuleType(module_path)
        module.__file__ = str(file_path)
        module.__package__ = ""
        module.__loader__ = None
        module.__spec__ = None
        sys.modules[module_path] = module
        try:
            exec(compile(source, str(file_path), "exec"), module.__dict__)
        except Exception:
            sys.modules.pop(module_path, None)
            raise
        return module

    def reload(self, plugin_id: str) -> PluginFiber:
        """Reload a plugin: dispose old fiber, re-import module, register new instance.

        Returns the new PluginFiber in ACTIVE state.

        Raises:
            KeyError: if plugin_id was never scoped-registered.
            RuntimeError: if the plugin module cannot be reloaded or has no `plugin` attribute.

        Concurrency safety:
            LeapFlow's engine snapshots handlers per-turn via `dict(_plugin_registry.tool_handlers)`.
            Existing turns keep their snapshot and finish with old handlers. New turns starting
            after this call pick up the new handlers. Single-threaded asyncio ensures no
            mid-turn tool table swap.

        Late-bound dependency re-injection:
            After the new fiber is activated, the registry's last_bound_deps are re-applied
            via bind_runtime(). This ensures gates, managers, and other runtime deps that
            were previously injected are also available to the new plugin instance.
        """
        if plugin_id not in self._fibers:
            raise KeyError(f"Plugin '{plugin_id}' not scoped-registered")

        module_path = self._plugin_modules.get(plugin_id)
        if module_path is None:
            raise RuntimeError(f"Module path unknown for plugin '{plugin_id}'")

        old_fiber = self._fibers[plugin_id]
        old_tool_names: list[str] = []
        # Capture current tool names BEFORE disposing so we know what to remove.
        old_plugin = self._registry.get_plugin(plugin_id)
        if old_plugin is not None:
            old_tool_names = [t.name for t in old_plugin.tools]

        # 1. Dispose old fiber (EffectScope cleanup runs unregister)
        if old_fiber.state == FiberState.ACTIVE:
            old_fiber.begin_unload()
        if old_fiber.state != FiberState.DISPOSED:
            old_fiber.dispose()

        # Belt-and-suspenders: fiber.dispose() already triggered scope cleanup which
        # should have called unregister_plugin. This is defensive in case the effect
        # callback didn't run (e.g., disposed via a different path). It's idempotent.
        if old_tool_names:
            self._unregister_tools(plugin_id, old_tool_names)

        # 2. Re-import the plugin module to get a fresh instance.
        fresh_plugin = self._load_fresh_plugin(plugin_id, module_path)

        # 3. Create new fiber and register the fresh plugin
        new_fiber = self.create_fiber(plugin_id)
        self.scoped_register(fresh_plugin, new_fiber)
        new_fiber.activate()

        # 4. Publish the fresh plugin's tools into the already-assembled catalog
        #    and bump the registry version so consumer caches (e.g. the engine
        #    tool registry) rebuild on the next turn.
        self._registry.publish_plugin_tools(fresh_plugin)

        # 5. Re-inject last-bound runtime dependencies onto the new plugin instance
        if self._registry.last_bound_deps:
            self._registry.bind_runtime(**self._registry.last_bound_deps)

        return new_fiber
