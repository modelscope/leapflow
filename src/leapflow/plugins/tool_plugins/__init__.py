"""Built-in tool plugin discovery.

Each module in this package exposes a module-level ``plugin`` instance
satisfying the ToolPlugin Protocol. ``get_all_plugins()`` aggregates them for
``ToolPluginRegistry.discover_builtin()``.

To add a built-in plugin: create the module, expose ``plugin``, and list it in
``_BUILTIN_PLUGIN_MODULES`` below.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leapflow.plugins.protocol import ToolPlugin

logger = logging.getLogger(__name__)

# Discovery order is a product contract: it fixes the order tools appear in the
# LLM tool index, which is part of the system prompt the journey cassettes are
# fingerprinted against. Do not reorder without reseeding cassettes.
_BUILTIN_PLUGIN_MODULES = (
    "leapflow.plugins.tool_plugins.text_utils",
    "leapflow.plugins.tool_plugins.system_info",
    "leapflow.plugins.tool_plugins.skill_discovery",
    "leapflow.plugins.tool_plugins.code_intel",
    "leapflow.plugins.tool_plugins.scm_git",
    "leapflow.plugins.tool_plugins.dev_tools",
    "leapflow.plugins.tool_plugins.file_ops",
    "leapflow.plugins.tool_plugins.shell_terminal",
    "leapflow.plugins.tool_plugins.config_tools",
    "leapflow.plugins.tool_plugins.web_access",
    "leapflow.plugins.tool_plugins.memory_research",
    "leapflow.plugins.tool_plugins.orchestration",
    "leapflow.plugins.tool_plugins.hub",
    "leapflow.plugins.tool_plugins.gateway",
    "leapflow.plugins.tool_plugins.self_management",
    # Desktop semantics — tools activate only once perception is bound.
    "leapflow.plugins.tool_plugins.desktop_semantic",
    # Hardware Context Protocol — appended last, and contributes no tools until a
    # hardware registry is bound. With hardware disabled the tool index is
    # byte-identical to a build without it, which is what keeps the journey
    # cassette fingerprints valid.
    "leapflow.hardware.plugin",
)


def _disabled_plugin_ids() -> set[str]:
    """Read ``disabled_plugins`` from settings, tolerating early bootstrap."""
    try:
        from leapflow.config import get_settings

        return set(getattr(get_settings(), "disabled_plugins", ()) or ())
    except (ImportError, AttributeError, RuntimeError):
        # Config not available during early init; treat as no filter.
        return set()


def _profile_plugins_dir() -> Path | None:
    """Return the active profile's installed plugin directory, if configured."""
    try:
        from leapflow.config import get_settings

        settings = get_settings()
        profile_layout = getattr(settings, "profile_layout", None)
        if profile_layout is None:
            return None
        return Path(profile_layout.plugins_dir)
    except (ImportError, AttributeError, RuntimeError, TypeError, OSError):
        return None


def _load_plugin_from_file(path: Path) -> "ToolPlugin | None":
    """Load one profile-scoped plugin file and attach its source path metadata."""
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        logger.warning("Cannot create import spec for profile plugin %s", path)
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        logger.warning("Failed to import profile plugin %s", path, exc_info=True)
        return None
    plugin = getattr(module, "plugin", None)
    if plugin is None:
        sys.modules.pop(module_name, None)
        logger.warning("Profile plugin %s has no module-level 'plugin'", path)
        return None
    try:
        setattr(plugin, "__leapflow_plugin_path__", str(path))
        setattr(plugin, "__leapflow_plugin_module__", module_name)
    except Exception:
        logger.debug("Cannot attach plugin source path metadata for %s", path, exc_info=True)
    return plugin


def discover_profile_plugins(disabled: set[str] | None = None) -> "list[ToolPlugin]":
    """Discover profile-scoped plugins installed under ProfileLayout.plugins_dir."""
    plugins_dir = _profile_plugins_dir()
    if plugins_dir is None or not plugins_dir.exists():
        return []
    disabled_ids = disabled if disabled is not None else _disabled_plugin_ids()
    discovered: list[ToolPlugin] = []
    for path in sorted(plugins_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        plugin = _load_plugin_from_file(path)
        if plugin is None:
            continue
        if plugin.plugin_id in disabled_ids:
            logger.info("Skipping disabled profile plugin: %s", plugin.plugin_id)
            continue
        discovered.append(plugin)
    return discovered


def _discover_all() -> "list[ToolPlugin]":
    """Import all built-in plugin modules and collect their plugin instances."""
    disabled = _disabled_plugin_ids()
    plugins: list[ToolPlugin] = []

    for module_path in _BUILTIN_PLUGIN_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            logger.error("Failed to import plugin module %s: %s", module_path, exc)
            continue

        plugin = getattr(module, "plugin", None)
        if plugin is None:
            logger.warning("Plugin module %s does not define a 'plugin' variable", module_path)
            continue
        if plugin.plugin_id in disabled:
            logger.info("Skipping disabled plugin: %s", plugin.plugin_id)
            continue
        plugins.append(plugin)

    for plugin in discover_profile_plugins(disabled):
        if plugin.plugin_id in {p.plugin_id for p in plugins}:
            logger.warning("Skipping duplicate profile plugin id: %s", plugin.plugin_id)
            continue
        plugins.append(plugin)

    return plugins


# Lazy singleton — no side effects at import time.
_all_plugins: "list[ToolPlugin] | None" = None


def get_all_plugins() -> "list[ToolPlugin]":
    """Return all built-in plugin instances, discovering lazily on first access."""
    global _all_plugins
    if _all_plugins is None:
        _all_plugins = _discover_all()
    return _all_plugins


__all__ = ["discover_profile_plugins", "get_all_plugins"]
