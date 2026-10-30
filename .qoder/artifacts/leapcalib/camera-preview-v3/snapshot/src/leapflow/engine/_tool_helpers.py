# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tool registry helpers and skill-registry builder.

Pure functions and module-level state for the runtime tool registry and the
built-in skill registry.  Extracted from ``engine.py`` to reduce file size.
"""

from __future__ import annotations

from typing import Any, Dict

from leapflow.platform.protocol import HostRpc
from leapflow.llm.base import LLMProvider
from leapflow.memory.providers.semantic import SemanticMemoryProvider
from leapflow.memory.providers.working import WorkingMemoryProvider
from leapflow.skills.builtin import app_launcher, clipboard_manager, file_organizer
from leapflow.skills.registry import Skill, SkillRegistry
from leapflow.tools.name_resolver import ToolRegistry, ToolResolution

# ---------------------------------------------------------------------------
# Module-level registry cache
# ---------------------------------------------------------------------------

_registry_cache: tuple[int, int, int, ToolRegistry] | None = None


def _default_tool_registry() -> ToolRegistry:
    """Return the runtime tool registry, rebuilding when late-registered tools arrive."""
    global _registry_cache
    from leapflow.plugins import get_registry

    _plugin_registry = get_registry()
    from leapflow.tools.name_resolver import TOOL_NAME_ALIASES

    _plugin_registry.assemble()  # idempotent: no-op once assembled

    td = _plugin_registry.tool_definitions
    th = _plugin_registry.tool_handlers

    size_key = (len(td), len(th), _plugin_registry.version)
    if _registry_cache is not None and _registry_cache[:3] == size_key:
        return _registry_cache[3]
    # Rebuild
    registry = ToolRegistry.from_definitions(
        td,
        th,
        aliases=TOOL_NAME_ALIASES,
    )
    _registry_cache = (*size_key, registry)
    return registry


def _resolve_tool_name(tool_name: str, arguments: Dict[str, Any] | None = None) -> ToolResolution:
    """Resolve a tool name through the runtime registry."""
    return _default_tool_registry().resolve(tool_name, arguments or {})


def _normalize_tool_name(tool_name: str) -> str:
    """Return the canonical executable tool name when resolution is safe."""
    return _default_tool_registry().normalize_name(tool_name)


def _concurrency_spec_lookup(tool_name: str) -> Any:
    """Return the registry ToolSpec for a (possibly gp_-prefixed) tool name.

    Injected into the tool concurrency policy so parallel-safety is classified
    from the same registry metadata that drives idempotency and the batch-stop
    gate (one source of truth). Returns None for an unregistered tool, which the
    policy treats as sequential.
    """
    specs = _default_tool_registry().specs
    return specs.get(tool_name) or specs.get(tool_name.removeprefix("gp_"))


def _normalize_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Return a resolved tool call while preserving the original tool name."""
    original_name = str(tool_call.get("name", ""))
    arguments = tool_call.get("arguments") or {}
    resolution = _resolve_tool_name(original_name, arguments)
    if not resolution.auto_executable or resolution.normalized_name is None:
        return {**tool_call, **resolution.to_metadata()}
    return {
        **tool_call,
        "name": resolution.normalized_name,
        **resolution.to_metadata(),
    }


# ---------------------------------------------------------------------------
# Built-in skill registry builder
# ---------------------------------------------------------------------------


def build_default_registry(
    rpc: HostRpc, llm: LLMProvider, wm: WorkingMemoryProvider, lt: SemanticMemoryProvider
) -> SkillRegistry:
    """Register built-in skills with closures (dependency injection)."""

    reg = SkillRegistry()

    async def _file_organizer(goal: str, **_kwargs: Any) -> str:
        return await file_organizer.run(rpc, llm, wm, lt, user_goal=goal)

    async def _clipboard(goal: str, **_kwargs: Any) -> str:
        return await clipboard_manager.run(rpc, llm, wm, lt, user_goal=goal)

    async def _app_launch(goal: str, **_kwargs: Any) -> str:
        return await app_launcher.run(rpc, user_goal=goal)

    reg.register(
        Skill(
            name="file_organizer",
            description="Organize PDFs/files using LLM plan + RPC file moves.",
            run=_file_organizer,
        )
    )
    reg.register(
        Skill(
            name="clipboard_manager",
            description="Summarize clipboard and store durable memory.",
            run=_clipboard,
        )
    )
    reg.register(
        Skill(
            name="app_launcher",
            description="Launch/activate apps and request simple automation actions.",
            run=_app_launch,
        )
    )
    return reg
