"""ToolPlugin Protocol — the unified contract for tool plugin modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class ToolPlugin(Protocol):
    """Tool plugin protocol.

    Each plugin module implements this Protocol to register its tool set
    with the ToolPluginRegistry. The Registry assembles all tools' OpenAI
    schemas and handler mappings in a single assemble() pass.
    """

    @property
    def plugin_id(self) -> str:
        """Unique plugin identifier, e.g. 'file_operations', 'shell_terminal'.

        Used for debug logging, runtime dependency resolution, and conflict detection.
        """
        ...

    @property
    def category(self) -> str:
        """Tool category label for PCD capability_expand and grouped display.

        Must be consistent with x_leapflow.category.
        """
        ...

    @property
    def tools(self) -> list[ToolMetadata]:
        """List of all tool metadata registered by this plugin.

        ToolMetadata is the single source of truth (SSOT).
        """
        ...

    @property
    def dependencies(self) -> list[str]:
        """List of runtime dependency names required by this plugin.

        The Registry distributes deps matching this list during bind_runtime().
        Example: ['memory_manager', 'research_ledger', 'file_read_gate']
        """
        ...

    def bind_runtime(self, **deps: Any) -> None:
        """Receive runtime-injected dependencies.

        Called by ToolPluginRegistry.bind_runtime() uniformly.
        Plugins should ignore kwargs not declared in their dependencies.
        """
        ...


@dataclass(frozen=True)
class ToolMetadata:
    """Unified tool metadata — Single Source of Truth for each tool.

    A tool only needs to define ToolMetadata once to:
    - Generate OpenAI function-calling schema (consumed by LLM)
    - Provide handler mapping (dispatched by engine)
    - Carry PCD / capability metadata (consumed by context_disclosure)
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]  # OpenAI JSON Schema format
    handler: Callable[..., Any]
    # Runtime dispatch goes through leapflow.plugins.handler_invocation.invoke_tool_handler,
    # which supports both generated-plugin **kwargs handlers and older params-dict handlers.
    x_leapflow: dict[str, Any] = field(default_factory=dict)
    mutates_state: bool = False
    # Declarative capability metadata consumed by the capability resolver and
    # environment-fit scoring. ``provides_capabilities`` are abstract capability
    # tags this tool offers (matched against a requirement). ``requires_capabilities``
    # are abstract capability tags another selected tool must provide earlier in
    # an orchestration plan. Their tag vocabulary is owned by the resolver, not
    # this type. ``requires_platform_capabilities`` are
    # ``leapflow.domain.platform.Capability`` values (e.g. "shell.exec") the
    # host must support for the tool to run. All default empty so a tool that
    # neither offers nor depends on named capabilities declares nothing -- the
    # common case, kept noise-free.
    provides_capabilities: tuple[str, ...] = ()
    requires_capabilities: tuple[str, ...] = ()
    requires_platform_capabilities: tuple[str, ...] = ()
    # ``requires_environment_affordances`` are task-environment (app-level)
    # preconditions the tool drives -- e.g. ``ui.chat.send.v2`` -- as opposed to
    # host ``requires_platform_capabilities`` (e.g. ``shell.exec``). Separating the
    # two lets the resolver score a candidate against the *task environment* an
    # app presents without conflating it with the host the agent runs on, and lets
    # the same tool be compared across app adapters. Defaults empty: a tool that
    # depends on no named app affordance declares nothing.
    requires_environment_affordances: tuple[str, ...] = ()

    def to_openai_schema(self) -> dict[str, Any]:
        """Generate OpenAI function-calling schema dict.

        ``mutates_state`` is folded into ``x_leapflow`` so schema-only
        consumers (e.g. ToolRegistry.from_definitions) can classify
        side-effecting tools without access to the metadata object.
        """
        x_leapflow = dict(self.x_leapflow)
        if self.mutates_state:
            x_leapflow.setdefault("mutates_state", True)
        if self.provides_capabilities:
            x_leapflow.setdefault("provides_capabilities", list(self.provides_capabilities))
        if self.requires_capabilities:
            x_leapflow.setdefault("requires_capabilities", list(self.requires_capabilities))
        if self.requires_platform_capabilities:
            x_leapflow.setdefault(
                "requires_platform_capabilities", list(self.requires_platform_capabilities)
            )
        if self.requires_environment_affordances:
            x_leapflow.setdefault(
                "requires_environment_affordances",
                list(self.requires_environment_affordances),
            )
        entry: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
        if x_leapflow:
            entry["function"]["x_leapflow"] = x_leapflow
        return entry
