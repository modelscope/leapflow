# Copyright (c) Alibaba, Inc. and its affiliates.
"""Gateway Adapter Plugin Registry.

Provides discovery, registration, and lifecycle management for platform adapters.
Adapters can be:
- Built-in (discovered from gateway/adapters/ package)
- External (registered via entry_points or explicit registration)
- Config-driven (enabled/disabled via gateway config)

The registry complements the existing ManifestLoader; manifests declare
*what* a platform needs (credentials, setup guide, options), while plugins
declare *how* to instantiate the adapter and expose metadata for tooling.
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from leapflow.gateway.protocol import PlatformAdapter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Plugin Protocol
# ═══════════════════════════════════════════════════════════════


@runtime_checkable
class GatewayAdapterPlugin(Protocol):
    """Protocol for gateway adapter plugins.

    Each plugin knows how to create one type of platform adapter.
    Plugins are stateless factories — they hold metadata and produce
    configured adapter instances on demand.
    """

    @property
    def platform_id(self) -> str:
        """Unique identifier for this platform (e.g. 'feishu', 'telegram')."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable platform name for UI display."""
        ...

    @property
    def adapter_class_path(self) -> str:
        """Dotted import path to the adapter class (module:ClassName)."""
        ...

    @property
    def config_schema(self) -> Dict[str, Any]:
        """JSON-schema-like dict describing accepted configuration keys.

        Used for validation and documentation. Empty dict means any config
        is accepted without validation.
        """
        ...

    def create_adapter(self, config: Dict[str, Any]) -> PlatformAdapter:
        """Instantiate an adapter with the given configuration.

        *config* typically merges credentials + options from the manifest/config
        store. The plugin is responsible for passing the correct kwargs.
        """
        ...


# ═══════════════════════════════════════════════════════════════
# Built-in plugin descriptor (concrete implementation)
# ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BuiltinAdapterPlugin:
    """Concrete plugin descriptor for adapters shipped with LeapFlow.

    Each built-in adapter module exposes a module-level ``plugin`` instance
    of this class for auto-discovery.
    """

    _platform_id: str
    _display_name: str
    _adapter_module: str
    _adapter_class: str
    _config_schema: Dict[str, Any] = field(default_factory=dict)

    @property
    def platform_id(self) -> str:
        return self._platform_id

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def adapter_class_path(self) -> str:
        return f"{self._adapter_module}:{self._adapter_class}"

    @property
    def config_schema(self) -> Dict[str, Any]:
        return self._config_schema

    def create_adapter(self, config: Dict[str, Any]) -> PlatformAdapter:
        """Import the adapter class and instantiate with config kwargs."""
        module = importlib.import_module(self._adapter_module)
        cls = getattr(module, self._adapter_class)
        return cls(**config)


# ═══════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════


class GatewayAdapterRegistry:
    """Central registry for gateway platform adapter plugins.

    Responsibilities:
    - Discover built-in adapter plugins from the adapters package
    - Accept external plugin registrations (entry_points, explicit)
    - Instantiate adapters from config via the registered plugin
    - List available platforms for tooling/UI

    Thread-safety: not thread-safe; intended for single-threaded async use
    within the gateway server lifecycle.
    """

    def __init__(self) -> None:
        self._plugins: Dict[str, GatewayAdapterPlugin] = {}
        self._version: int = 0

    @property
    def version(self) -> int:
        """Monotonic counter incremented on every mutation. Used for cache invalidation."""
        return self._version

    def notify_mutation(self) -> None:
        """Public API to signal a mutation happened (increments version)."""
        self._version += 1

    # ── Registration ──────────────────────────────────────────

    def register(self, plugin: GatewayAdapterPlugin) -> None:
        """Register a single adapter plugin.

        Overwrites any existing plugin for the same platform_id.
        """
        pid = plugin.platform_id
        if pid in self._plugins:
            logger.info(
                "Overwriting adapter plugin for platform '%s' "
                "(previous: %s, new: %s)",
                pid,
                self._plugins[pid].adapter_class_path,
                plugin.adapter_class_path,
            )
        self._plugins[pid] = plugin
        self._version += 1
        logger.debug("Registered adapter plugin: %s (%s)", pid, plugin.display_name)

    def unregister(self, platform_id: str) -> bool:
        """Remove a registered plugin. Returns True if it was present."""
        removed = self._plugins.pop(platform_id, None) is not None
        if removed:
            self._version += 1
        return removed

    # ── Discovery ─────────────────────────────────────────────

    def discover_builtin(self) -> int:
        """Scan the built-in adapters package for plugin instances.

        Each adapter module is expected to expose a module-level ``plugin``
        attribute satisfying the ``GatewayAdapterPlugin`` protocol.

        Returns the number of plugins discovered.
        """
        adapter_modules = [
            "leapflow.gateway.adapters.feishu",
            "leapflow.gateway.adapters.telegram",
            "leapflow.gateway.adapters.dingtalk",
            "leapflow.gateway.adapters.webhook",
            "leapflow.gateway.adapters.api_server",
        ]
        discovered = 0
        for module_path in adapter_modules:
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                logger.debug("Skipping adapter module %s (import failed)", module_path)
                continue
            plugin = getattr(module, "plugin", None)
            if plugin is not None and isinstance(plugin, GatewayAdapterPlugin):
                self.register(plugin)
                discovered += 1
            else:
                logger.debug(
                    "Adapter module %s has no 'plugin' attribute or it "
                    "does not satisfy GatewayAdapterPlugin",
                    module_path,
                )
        return discovered

    def discover_entry_points(self, group: str = "leapflow.gateway.adapters") -> int:
        """Discover plugins registered via setuptools entry_points.

        Entry points should point to a module-level ``plugin`` instance.
        Returns the number of plugins discovered.
        """
        discovered = 0
        try:
            from importlib.metadata import entry_points

            eps = entry_points()
            # Python 3.12+ returns a SelectableGroups; fallback for 3.9+
            if hasattr(eps, "select"):
                group_eps = eps.select(group=group)
            else:
                group_eps = eps.get(group, [])

            for ep in group_eps:
                try:
                    plugin = ep.load()
                    if isinstance(plugin, GatewayAdapterPlugin):
                        self.register(plugin)
                        discovered += 1
                    else:
                        logger.warning(
                            "Entry point '%s' does not satisfy GatewayAdapterPlugin",
                            ep.name,
                        )
                except Exception:
                    logger.warning(
                        "Failed to load entry point '%s'", ep.name, exc_info=True,
                    )
        except ImportError:
            logger.debug("importlib.metadata not available; skipping entry_points discovery")
        return discovered

    # ── Queries ───────────────────────────────────────────────

    def get_plugin(self, platform_id: str) -> Optional[GatewayAdapterPlugin]:
        """Return the registered plugin for a platform, or None."""
        return self._plugins.get(platform_id)

    def list_available(self) -> List[str]:
        """Return sorted list of all registered platform IDs."""
        return sorted(self._plugins.keys())

    def list_plugins(self) -> List[GatewayAdapterPlugin]:
        """Return all registered plugins (ordered by platform_id)."""
        return [self._plugins[k] for k in sorted(self._plugins)]

    def has_plugin(self, platform_id: str) -> bool:
        """Check if a plugin is registered for the given platform."""
        return platform_id in self._plugins

    # ── Adapter creation ──────────────────────────────────────

    def create_adapter(
        self,
        platform_id: str,
        config: Dict[str, Any],
    ) -> PlatformAdapter:
        """Create an adapter instance using the registered plugin.

        Raises KeyError if no plugin is registered for the platform.
        Raises any exception from the plugin's create_adapter on failure.
        """
        plugin = self._plugins.get(platform_id)
        if plugin is None:
            raise KeyError(
                f"No adapter plugin registered for platform '{platform_id}'. "
                f"Available: {', '.join(self.list_available()) or '(none)'}"
            )
        return plugin.create_adapter(config)

    def create_adapter_safe(
        self,
        platform_id: str,
        config: Dict[str, Any],
    ) -> Optional[PlatformAdapter]:
        """Create an adapter, returning None on any failure (logged)."""
        try:
            return self.create_adapter(platform_id, config)
        except KeyError:
            logger.debug("No plugin for platform '%s'", platform_id)
            return None
        except Exception:
            logger.warning(
                "Failed to create adapter for '%s'", platform_id, exc_info=True,
            )
            return None

    # ── Info / debugging ──────────────────────────────────────

    def summary(self) -> Dict[str, str]:
        """Return {platform_id: display_name} for all plugins."""
        return {p.platform_id: p.display_name for p in self.list_plugins()}

    def __len__(self) -> int:
        return len(self._plugins)

    def __contains__(self, platform_id: str) -> bool:
        return platform_id in self._plugins

    def __repr__(self) -> str:
        return (
            f"GatewayAdapterRegistry(plugins={self.list_available()})"
        )
