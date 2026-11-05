# Copyright (c) Alibaba, Inc. and its affiliates.
"""Plugin subsystem — contracts, discovery, lifecycle, and the live registry.

This package owns everything about *extending* LeapFlow: the ``ToolPlugin``
contract, built-in plugin discovery, the process-global tool registry and its
fiber lifecycle, sandbox isolation, and marketplace distribution.

Tool *implementations* live in ``leapflow.tools``; plugins wrap them as
``ToolMetadata`` and this package publishes them to the runtime. The dependency
direction is one-way: plugin core never imports a concrete tool module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from leapflow.plugins.adaptive_loop import (
    AdaptiveLoopMutation,
    AdaptiveLoopRequest,
    AdaptiveLoopResult,
    AdaptivePluginLoop,
    CapabilityDecisionRecorder,
    SelfManagementLifecycleActor,
)
from leapflow.plugins.adaptive_policy import AdaptiveEvolutionPolicy, AdaptivePolicyDecision
from leapflow.plugins.capability_plan import CapabilityPlan, CapabilityPlanStep
from leapflow.plugins.capability_resolver import (
    CapabilityCandidate,
    CapabilityResolution,
    CapabilityResolver,
    ResolverContext,
)
from leapflow.plugins.protocol import ToolMetadata, ToolPlugin
from leapflow.plugins.registry import ToolPluginRegistry
from leapflow.plugins.scoped_registry import ScopedToolRegistry

if TYPE_CHECKING:
    from leapflow.domain.plugin_fiber import PluginFiber

# Lazy singletons — importing this package has no side effects.
_registry: ToolPluginRegistry | None = None
_scoped_registry: ScopedToolRegistry | None = None


def get_registry() -> ToolPluginRegistry:
    """Return the process-global tool registry, discovering built-ins once.

    This registry is the single authority for tool definitions, handlers, and
    the cross-cutting runtime gates the tools dispatch through.
    """
    global _registry
    if _registry is None:
        registry = ToolPluginRegistry()
        registry.discover_builtin()
        _registry = registry
    return _registry


def get_scoped_registry() -> ScopedToolRegistry:
    """Return the process-global lifecycle wrapper around the tool registry.

    On first access every plugin already registered (all built-ins discovered
    at boot) is adopted under a ``PluginFiber``, so the whole tool subsystem is
    uniformly under fiber lifecycle management. Adoption is additive tracking
    only — it does not re-register plugins or change how tools are dispatched.
    """
    global _scoped_registry
    if _scoped_registry is None:
        scoped = ScopedToolRegistry(get_registry())
        scoped.adopt_existing_plugins()
        _scoped_registry = scoped
    return _scoped_registry


def reload_plugin(plugin_id: str) -> "PluginFiber":
    """Hot-reload one plugin, returning its fresh fiber in ACTIVE state."""
    return get_scoped_registry().reload(plugin_id)


__all__ = [
    "AdaptiveLoopMutation",
    "AdaptiveLoopRequest",
    "AdaptiveLoopResult",
    "AdaptivePluginLoop",
    "AdaptiveEvolutionPolicy",
    "AdaptivePolicyDecision",
    "CapabilityDecisionRecorder",
    "CapabilityCandidate",
    "CapabilityPlan",
    "CapabilityPlanStep",
    "CapabilityResolution",
    "CapabilityResolver",
    "ResolverContext",
    "ScopedToolRegistry",
    "SelfManagementLifecycleActor",
    "ToolMetadata",
    "ToolPlugin",
    "ToolPluginRegistry",
    "get_registry",
    "get_scoped_registry",
    "reload_plugin",
]
