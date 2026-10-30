# Copyright (c) Alibaba, Inc. and its affiliates.
"""LLM Provider Plugin Registry.

Provides discovery, registration, and lifecycle management for LLM providers.
Providers can be:
- Built-in (OpenAIChat compatible format)
- External (registered via entry_points or explicit registration)
- Config-driven (selected via llm config section)

The registry is the single entry point for provider instantiation. It decouples
the engine from concrete provider implementations and enables third-party
providers to be added without modifying core code.
"""
from __future__ import annotations

import importlib.metadata
import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from leapflow.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Entry point group name for external provider plugins.
ENTRY_POINT_GROUP = "leapflow.llm_providers"


@runtime_checkable
class LLMProviderPlugin(Protocol):
    """Protocol for LLM provider plugins.

    A plugin declares its identity, supported models, and provides a factory
    method to create configured LLMProvider instances. Implementations may be
    built-in or discovered via setuptools entry_points.
    """

    @property
    def provider_id(self) -> str:
        """Unique provider identifier, e.g. 'openai', 'anthropic', 'local-llama'."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable provider name for UI/logging."""
        ...

    @property
    def supported_models(self) -> List[str]:
        """Model ID patterns this provider can serve.

        May include exact model names or glob-style hints (e.g. 'gpt-4*').
        Used for informational purposes and routing suggestions.
        """
        ...

    @property
    def capabilities(self) -> Dict[str, Any]:
        """Provider-level capability declarations.

        Keys may include:
        - 'supports_streaming': bool
        - 'supports_tools': bool
        - 'supports_vision': bool
        - 'supports_thinking': bool
        - 'max_context_length': int
        - 'credential_rotation': bool
        - 'cache_type': str — prompt cache mechanism used by the provider.
              'auto_prefix'          — automatic prefix caching (OpenAI, DeepSeek).
              'explicit_breakpoint'  — explicit cache_control breakpoints (Anthropic).
              'none'                 — no prompt caching support.
        - 'cache_usage_fields': List[str] — names of usage-dict keys that
              report cache hit / creation token counts for this provider.
        """
        ...

    def create_provider(self, config: Dict[str, Any]) -> LLMProvider:
        """Factory method to create a configured LLMProvider instance.

        Args:
            config: Provider-specific configuration dict. Expected keys vary
                    by provider but typically include 'api_key', 'base_url',
                    'model', 'max_retries', 'timeout_s'.

        Returns:
            A ready-to-use LLMProvider instance.
        """
        ...


class LLMProviderRegistry:
    """Central registry for LLM provider plugins.

    Responsibilities:
    - Registration of built-in and external provider plugins
    - Discovery of plugins via setuptools entry_points
    - Config-driven provider instantiation
    - Listing available providers for UI/diagnostics

    Thread-safety: Not thread-safe. Expected to be populated at startup
    and read concurrently thereafter (no mutation after init).
    """

    def __init__(self) -> None:
        self._plugins: Dict[str, LLMProviderPlugin] = {}
        self._instances: Dict[str, LLMProvider] = {}
        self._version: int = 0

    @property
    def version(self) -> int:
        """Monotonic counter incremented on every mutation. Used for cache invalidation."""
        return self._version

    def notify_mutation(self) -> None:
        """Public API to signal a mutation happened (increments version)."""
        self._version += 1
        # Cached instances become stale on any mutation; drop them so callers
        # re-create providers against the current plugin set.
        self._instances.clear()

    def register(self, plugin: LLMProviderPlugin) -> None:
        """Register a provider plugin.

        If a plugin with the same provider_id already exists, the new one
        replaces it (allows overriding built-ins with custom implementations).
        """
        pid = plugin.provider_id
        if pid in self._plugins:
            logger.info(
                "llm_registry: replacing provider plugin '%s' (%s -> %s)",
                pid,
                self._plugins[pid].display_name,
                plugin.display_name,
            )
        self._plugins[pid] = plugin
        # Invalidate cached instance on re-registration.
        self._instances.pop(pid, None)
        self._version += 1
        logger.debug("llm_registry: registered provider '%s'", pid)

    def unregister(self, provider_id: str) -> bool:
        """Remove a provider plugin. Returns True if it existed."""
        removed = self._plugins.pop(provider_id, None) is not None
        self._instances.pop(provider_id, None)
        if removed:
            self._version += 1
            logger.debug("llm_registry: unregistered provider '%s'", provider_id)
        return removed

    def get_plugin(self, provider_id: str) -> Optional[LLMProviderPlugin]:
        """Retrieve a registered plugin by ID."""
        return self._plugins.get(provider_id)

    def get_provider(self, provider_id: str, config: Optional[Dict[str, Any]] = None) -> Optional[LLMProvider]:
        """Get or create an LLMProvider instance by provider_id.

        Uses a cached instance if available; creates one from config if not.
        Pass config=None to retrieve a previously-created instance only.
        """
        if provider_id in self._instances:
            return self._instances[provider_id]

        plugin = self._plugins.get(provider_id)
        if plugin is None:
            return None

        if config is None:
            return None

        try:
            instance = plugin.create_provider(config)
            self._instances[provider_id] = instance
            logger.info(
                "llm_registry: created provider '%s' (model: %s)",
                provider_id,
                config.get("model", "unknown"),
            )
            return instance
        except Exception as exc:
            logger.error(
                "llm_registry: failed to create provider '%s': %s",
                provider_id, exc,
            )
            return None

    def create_from_config(self, config: Dict[str, Any]) -> Optional[LLMProvider]:
        """Create a provider instance from a config dict.

        The config must include a 'provider' key identifying which plugin to use.
        Falls back to 'openai' if not specified (backward compatibility).

        Args:
            config: Dict with at minimum 'provider' (or defaults to 'openai'),
                    plus provider-specific keys (api_key, base_url, model, etc.).

        Returns:
            LLMProvider instance, or None if the provider is not registered.
        """
        provider_id = config.get("provider", "openai")
        return self.get_provider(provider_id, config)

    def list_available(self) -> List[str]:
        """Return sorted list of registered provider IDs."""
        return sorted(self._plugins.keys())

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Return detailed info about all registered plugins (for diagnostics)."""
        result = []
        for pid, plugin in sorted(self._plugins.items()):
            result.append({
                "provider_id": pid,
                "display_name": plugin.display_name,
                "supported_models": plugin.supported_models,
                "capabilities": plugin.capabilities,
                "active": pid in self._instances,
            })
        return result

    def discover_entry_points(self) -> int:
        """Discover and register plugins from setuptools entry_points.

        Looks for entry points in the 'leapflow.llm_providers' group.
        Each entry point should resolve to a class implementing LLMProviderPlugin.

        Returns:
            Number of plugins successfully loaded.
        """
        loaded = 0
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.12+ returns a SelectableGroups; 3.9-3.11 returns a dict.
            if hasattr(eps, "select"):
                group_eps = eps.select(group=ENTRY_POINT_GROUP)
            else:
                group_eps = eps.get(ENTRY_POINT_GROUP, [])

            for ep in group_eps:
                try:
                    plugin_cls = ep.load()
                    # Instantiate if it's a class, use directly if already an instance.
                    plugin = plugin_cls() if isinstance(plugin_cls, type) else plugin_cls
                    if isinstance(plugin, LLMProviderPlugin):
                        self.register(plugin)
                        loaded += 1
                        logger.info(
                            "llm_registry: loaded entry_point plugin '%s' from %s",
                            plugin.provider_id, ep.value,
                        )
                    else:
                        logger.warning(
                            "llm_registry: entry_point '%s' does not satisfy "
                            "LLMProviderPlugin protocol, skipped",
                            ep.name,
                        )
                except Exception as exc:
                    logger.warning(
                        "llm_registry: failed to load entry_point '%s': %s",
                        ep.name, exc,
                    )
        except Exception as exc:
            logger.debug("llm_registry: entry_point discovery failed: %s", exc)

        if loaded:
            logger.info("llm_registry: discovered %d external plugin(s)", loaded)
        return loaded

    def discover_builtin(self) -> None:
        """Register all built-in provider plugins.

        Currently registers:
        - OpenAICompatiblePlugin (covers OpenAI, Azure, DeepSeek, Dashscope, etc.)
        - AnthropicPlugin (native Anthropic Messages API; skipped when the
          ``anthropic`` SDK is not installed)
        """
        from leapflow.llm._builtin_plugins import OpenAICompatiblePlugin

        self.register(OpenAICompatiblePlugin())

        # Anthropic: optional — gracefully degrade when SDK is absent.
        try:
            from leapflow.llm._anthropic_plugin import AnthropicPlugin

            self.register(AnthropicPlugin())
        except ImportError:
            logger.debug(
                "llm_registry: anthropic SDK not installed, "
                "AnthropicPlugin skipped (install with 'pip install anthropic')"
            )
        except Exception as exc:
            logger.warning(
                "llm_registry: failed to load AnthropicPlugin: %s", exc,
            )

    def bootstrap(self) -> None:
        """Full initialization: register built-ins, then discover external plugins.

        Call once at application startup.
        """
        self.discover_builtin()
        self.discover_entry_points()
        logger.info(
            "llm_registry: bootstrap complete — %d provider(s) available: %s",
            len(self._plugins), ", ".join(self.list_available()),
        )

    def clear(self) -> None:
        """Remove all plugins and instances (useful for testing)."""
        had_state = bool(self._plugins) or bool(self._instances)
        self._plugins.clear()
        self._instances.clear()
        if had_state:
            self._version += 1


# Module-level singleton for convenient access.
_default_registry: Optional[LLMProviderRegistry] = None
_scoped_default_registry: Optional[Any] = None


def get_default_registry() -> LLMProviderRegistry:
    """Return the module-level default registry, creating if needed."""
    global _default_registry
    if _default_registry is None:
        _default_registry = LLMProviderRegistry()
    return _default_registry


def get_scoped_default_registry() -> "Any":
    """Return a ScopedLLMProviderRegistry wrapping the default registry.

    Ensures the underlying registry is bootstrapped (built-ins + entry points),
    then adopts every registered provider under a PluginFiber so the LLM
    subsystem is uniformly under fiber lifecycle management. Adoption is
    additive tracking only and does not re-register providers.
    """
    global _scoped_default_registry
    if _scoped_default_registry is None:
        from leapflow.llm.scoped_provider_registry import ScopedLLMProviderRegistry
        registry = get_default_registry()
        if not registry.list_available():
            registry.bootstrap()
        _scoped_default_registry = ScopedLLMProviderRegistry(registry)
    _scoped_default_registry.adopt_existing_plugins()
    return _scoped_default_registry


def reset_default_registry() -> None:
    """Reset the default registry (for testing)."""
    global _default_registry, _scoped_default_registry
    if _default_registry is not None:
        _default_registry.clear()
    _default_registry = None
    _scoped_default_registry = None
