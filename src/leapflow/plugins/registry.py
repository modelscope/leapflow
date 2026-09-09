"""ToolPluginRegistry — the single entry point for tool system initialization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from leapflow.domain.tool_pipeline import ToolExecutionPipeline
from leapflow.plugins.protocol import ToolMetadata, ToolPlugin

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CapabilityConflict:
    """A rejected duplicate tool-name claim.

    Tool names form a single global namespace consumed by the provider, so two
    plugins cannot both expose the same name. The registry keeps the incumbent
    (first indexed) and rejects the challenger, surfacing the rejected claim
    here -- and through ``plugin_list`` -- instead of silently overwriting the
    live handler or emitting a duplicate schema.
    """

    tool_name: str
    kept_plugin: str
    rejected_plugin: str
    kept_description: str = ""
    rejected_description: str = ""


class ToolPluginRegistry:
    """Central tool plugin registry.

    Lifecycle:
    1. discover_builtin() — auto-discover built-in plugins under tool_plugins/
    2. register() — register additional plugins (installed, marketplace, external)
    3. bind_runtime(**deps) — inject runtime dependencies into all plugins
    4. assemble() — one-shot assembly of final tool_definitions and tool_handlers

    Plugins registered *after* assembly (install, hot-reload) publish their
    tools through publish_plugin_tools() instead of a full reassemble.

    Consumer API:
    - registry.tool_definitions → List[dict]  (OpenAI schemas)
    - registry.tool_handlers → Dict[str, Callable]  (name → handler)
    - registry.get_tools_by_category(cat) → List[ToolMetadata]

    Also serves as the central holder for the cross-cutting runtime gates
    (file_read_gate, file_write_gate, desktop_gate, ...) that tool handlers
    dispatch through, so no tool module needs a mutable module-level global.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, ToolPlugin] = {}
        self._tool_definitions: list[dict[str, Any]] = []
        self._tool_handlers: dict[str, Any] = {}
        self._all_metadata: list[ToolMetadata] = []
        # tool name -> owning plugin_id, the authority for first-wins name
        # arbitration and for tearing down only the names a plugin owns live.
        self._tool_owner: dict[str, str] = {}
        self._conflicts: list[CapabilityConflict] = []
        self._assembled = False
        self._version: int = 0
        self._last_bound_deps: dict[str, Any] = {}  # Track last-injected deps for re-injection on reload
        self._tool_pipeline = ToolExecutionPipeline()

        # ── Cross-cutting runtime gates ──
        self._file_read_gate: Any = None
        self._file_write_gate: Any = None
        self._desktop_gate: Any = None
        self._capability_catalog_provider: Optional[Callable[[], List[Dict[str, Any]]]] = None
        self._subagent_manager: Any = None
        self._memory_manager: Any = None
        self._gateway_server: Any = None
        self._research_ledger: Any = None
        self._reentry_scheduler: Any = None

    # ── Gate Accessors (replacements for registry_bootstrap module-globals) ──

    def set_file_read_gate(self, gate: Any) -> None:
        """Install a file-read approval gate."""
        self._file_read_gate = gate

    def get_file_read_gate(self) -> Any:
        return self._file_read_gate

    def set_file_write_gate(self, gate: Any) -> None:
        """Install a file-write approval gate."""
        self._file_write_gate = gate

    def get_file_write_gate(self) -> Any:
        return self._file_write_gate

    def set_desktop_gate(self, gate: Any) -> None:
        """Install an approval gate for mutating semantic desktop tools."""
        self._desktop_gate = gate

    def get_desktop_gate(self) -> Any:
        return self._desktop_gate

    def set_capability_catalog_provider(self, provider: Optional[Callable[[], List[Dict[str, Any]]]]) -> None:
        """Install a late-bound provider for the live tool catalog."""
        self._capability_catalog_provider = provider
        # Propagate via standard DI path — plugins declare 'capability_catalog_provider' in dependencies
        self.bind_runtime(capability_catalog_provider=provider)

    def capability_catalog(self) -> List[Dict[str, Any]]:
        """Resolve the live tool catalog for capability discovery."""
        if self._capability_catalog_provider is not None:
            try:
                catalog = self._capability_catalog_provider()
            except (RuntimeError, ValueError, TypeError) as exc:
                logger.debug("capability_catalog provider failed: %s", exc, exc_info=True)
                catalog = None
            if catalog:
                return list(catalog)
        if not self._assembled:
            self.assemble()
        return self._tool_definitions

    def set_memory_manager(self, mgr: Any) -> None:
        """Install memory manager reference for memory tools."""
        self._memory_manager = mgr
        # Also propagate to plugins that need it
        self.bind_runtime(memory_manager=mgr)

    def set_gateway_server(self, server: Any) -> None:
        """Install gateway server reference for gateway tools.

        Propagation stops at the standard DI path: the gateway plugin declares
        'gateway_server' and forwards it to its handler module itself, so the
        registry stays free of any concrete tool import.
        """
        self._gateway_server = server
        self.bind_runtime(gateway_server=server)

    def set_research_ledger(self, ledger: Any) -> None:
        """Install research ledger reference."""
        self._research_ledger = ledger
        self.bind_runtime(research_ledger=ledger)

    def set_reentry_scheduler(self, scheduler: Any) -> None:
        """Install re-entry scheduler callable."""
        self._reentry_scheduler = scheduler
        self.bind_runtime(reentry_scheduler=scheduler)

    def set_subagent_manager(self, manager: Any) -> None:
        """Install SubagentManager reference for delegate_task dispatch."""
        self._subagent_manager = manager
        self.bind_runtime(subagent_manager=manager)

    # ── Registration ──

    def register(self, plugin: ToolPlugin) -> None:
        """Register a tool plugin. Raises on duplicate plugin_id."""
        if plugin.plugin_id in self._plugins:
            raise ValueError(
                f"Duplicate plugin_id: {plugin.plugin_id!r} "
                f"(existing: {self._plugins[plugin.plugin_id].__class__.__name__})"
            )
        if not isinstance(plugin, ToolPlugin):
            raise TypeError(f"Plugin must satisfy ToolPlugin Protocol: {type(plugin)}")
        self._plugins[plugin.plugin_id] = plugin
        self._bump_version("plugin_registered", plugin_id=plugin.plugin_id, tools=len(plugin.tools))
        logger.debug("Registered tool plugin: %s (%d tools)", plugin.plugin_id, len(plugin.tools))

    # ── Built-in Discovery ──

    def discover_builtin(self) -> None:
        """Discover and load all built-in tool plugins from leapflow.plugins.tool_plugins."""
        from leapflow.plugins.tool_plugins import get_all_plugins

        for plugin in get_all_plugins():
            if plugin.plugin_id not in self._plugins:
                self.register(plugin)

    # ── Dependency Injection ──

    def bind_runtime(self, **deps: Any) -> None:
        """Inject runtime dependencies into all registered plugins.

        Only distributes deps that a plugin has declared in its dependencies.
        Can be called multiple times (incremental binding).

        Plugins are visited in provider → consumer (topological) order derived
        from their declared inter-plugin dependencies, so a provider is always
        bound before any plugin that depends on it. Ordering is independent of
        registration/discovery order; see ``_topological_plugin_order``.
        """
        for plugin_id in self._topological_plugin_order():
            plugin = self._plugins.get(plugin_id)
            if plugin is None:
                continue
            relevant = {k: v for k, v in deps.items() if k in plugin.dependencies}
            if relevant:
                plugin.bind_runtime(**relevant)
        # Track last-bound deps for potential re-injection on plugin reload
        self._last_bound_deps.update(deps)

    def _topological_plugin_order(self) -> List[str]:
        """Return plugin ids ordered so providers precede their consumers.

        The dependency graph is built from ``{plugin_id: plugin.dependencies}``,
        keeping only dependency names that name another registered plugin —
        external runtime deps (e.g. ``file_read_gate``) are not plugins and do
        not constrain ordering. ``graphlib.TopologicalSorter`` yields providers
        before dependents while preserving registration order among independent
        plugins. A dependency cycle cannot be ordered, so the original
        registration order is used for all plugins as a safe fallback.
        """
        from graphlib import CycleError, TopologicalSorter

        plugin_ids = list(self._plugins.keys())
        registered = set(plugin_ids)
        graph: dict[str, set[str]] = {
            pid: {dep for dep in self._plugins[pid].dependencies if dep in registered}
            for pid in plugin_ids
        }
        try:
            return list(TopologicalSorter(graph).static_order())
        except CycleError:
            logger.warning(
                "Circular inter-plugin dependency detected; falling back to "
                "registration order for bind_runtime distribution"
            )
            return plugin_ids

    # ── Assembly ──

    def assemble(self) -> None:
        """Assemble final outputs. After this, properties become available.

        Should only be called once. Registry enters read-only state after assembly.
        Can be called again via reassemble() if late tools are registered.
        """
        if self._assembled:
            return

        for plugin in self._plugins.values():
            for tool in plugin.tools:
                self._index_tool(tool, plugin.plugin_id)

        # Bumped before the flag flips, so this trace is classified as boot
        # composition rather than a runtime change. Assembly *is* the composition
        # event; labelling it ``runtime`` made every daemon start publish an
        # ``evolution.*`` event and put a boot row in the live activity feed.
        self._bump_version("assembled", plugins=len(self._plugins), tools=len(self._tool_handlers))
        self._assembled = True

        logger.info(
            "Tool registry assembled: %d plugins, %d tools",
            len(self._plugins),
            len(self._tool_handlers),
        )

    def publish_plugin_tools(self, plugin: ToolPlugin) -> list[str]:
        """Publish an already-registered plugin's tools into the live catalog.

        assemble() runs once at boot; a plugin that arrives later (install,
        hot-reload) makes its tools dispatchable through this method, keeping
        the definitions, metadata, and handler table in one place instead of
        letting callers write to the registry's internals.

        Returns the published tool names and bumps the version counter so
        downstream caches (engine tool registry, PCD catalog) invalidate.
        """
        tool_names = [tool.name for tool in plugin.tools]
        # Before the first assemble() the pending pass will pick these tools up
        # from the plugin itself; publishing now would duplicate every schema.
        if self._assembled:
            for tool in plugin.tools:
                self._index_tool(tool, plugin.plugin_id)
        self._bump_version(
            "tools_published", plugin_id=plugin.plugin_id, tool_names=list(tool_names)
        )
        return tool_names

    def register_late_tool(
        self, definition: dict[str, Any], handler: Any, name: str, owner: str = "late_tool"
    ) -> None:
        """Register a standalone tool after assembly (session_search, MCP, ...).

        Used for tools that have no owning plugin; plugin-owned tools go
        through publish_plugin_tools(). Subject to the same first-wins name
        arbitration: a late tool cannot shadow a name a plugin already claimed.
        """
        if name in self._tool_owner:
            self._record_conflict(
                name, owner, definition.get("function", {}).get("description", "")
            )
            return
        self._tool_definitions.append(definition)
        self._tool_handlers[name] = handler
        self._tool_owner[name] = owner
        self._bump_version("late_tool_registered", tool_name=name, owner=owner)

    def _index_tool(self, tool: ToolMetadata, owner: str) -> None:
        """Add one tool to the metadata, schema, and handler indexes.

        Tool names are a single global namespace: the first plugin to claim a
        name keeps it, and a later plugin declaring the same name is rejected
        and recorded in ``conflicts`` rather than silently overwriting the live
        handler or emitting a duplicate schema. Rejection is non-fatal so one
        colliding plugin cannot break assembly for every other plugin.
        """
        if tool.name in self._tool_owner:
            self._record_conflict(tool.name, owner, tool.description)
            return
        self._all_metadata.append(tool)
        self._tool_definitions.append(tool.to_openai_schema())
        self._tool_handlers[tool.name] = tool.handler
        self._tool_owner[tool.name] = owner

    def _record_conflict(self, name: str, challenger: str, challenger_description: str) -> None:
        """Record a rejected duplicate tool-name claim; never raises."""
        incumbent = self._tool_owner.get(name, "")
        incumbent_meta = next((m for m in self._all_metadata if m.name == name), None)
        self._conflicts.append(
            CapabilityConflict(
                tool_name=name,
                kept_plugin=incumbent,
                rejected_plugin=challenger,
                kept_description=incumbent_meta.description if incumbent_meta else "",
                rejected_description=challenger_description,
            )
        )
        logger.warning(
            "Tool-name conflict on %r: kept %r, rejected duplicate from %r",
            name,
            incumbent,
            challenger,
        )

    # ── Unregistration ──

    def unregister_plugin(self, plugin_id: str) -> bool:
        """Remove a plugin and all its tools from the registry.

        Removes:
        - The plugin from _plugins
        - All handlers contributed by the plugin
        - All tool_definitions matching the plugin's tool names
        - All all_metadata entries matching the plugin's tool names

        Returns True if the plugin was present, False otherwise.
        Bumps the version counter to invalidate downstream caches.
        """
        plugin = self._plugins.pop(plugin_id, None)
        if plugin is None:
            return False

        # Only tear down names this plugin owns live. A plugin whose duplicate
        # claim was rejected owns none of the colliding names, so disposing it
        # must not remove the incumbent's live handler.
        owned = {t.name for t in plugin.tools if self._tool_owner.get(t.name) == plugin_id}
        self._remove_tools_by_name(owned)
        self._conflicts = [
            c for c in self._conflicts if plugin_id not in (c.kept_plugin, c.rejected_plugin)
        ]
        self._bump_version("plugin_unregistered", plugin_id=plugin_id, tools=sorted(owned))
        return True

    def unregister_tools(self, tool_names: Iterable[str]) -> int:
        """Remove specific tools by name (independent of plugin association).

        Used for late-tool cleanup where the tool was not registered via a plugin.
        Returns the number of tools actually removed.
        Bumps the version counter.
        """
        names_set = set(tool_names)
        removed = self._remove_tools_by_name(names_set)
        if removed > 0:
            self._bump_version("tools_unregistered", tool_names=sorted(names_set), removed=removed)
        return removed

    def _remove_tools_by_name(self, names: set[str]) -> int:
        """Internal: remove tools from handlers/definitions/metadata by name set.

        Does NOT bump version.
        Returns the number of handler entries removed.
        """
        initial_count = len(self._tool_handlers)

        # Remove handlers and their ownership claims
        for name in names:
            self._tool_handlers.pop(name, None)
            self._tool_owner.pop(name, None)

        # Remove from _tool_definitions
        self._tool_definitions = [
            d for d in self._tool_definitions
            if d.get("function", {}).get("name") not in names
        ]

        # Remove from _all_metadata
        self._all_metadata = [
            m for m in self._all_metadata if m.name not in names
        ]

        return initial_count - len(self._tool_handlers)

    # ── Public API (for consumers) ──

    @property
    def tool_definitions(self) -> list[dict[str, Any]]:
        """OpenAI function-calling schemas for all registered tools."""
        if not self._assembled:
            self.assemble()
        return self._tool_definitions

    @property
    def tool_handlers(self) -> dict[str, Any]:
        """Tool name → handler callable mapping."""
        if not self._assembled:
            self.assemble()
        return self._tool_handlers

    @property
    def all_metadata(self) -> list[ToolMetadata]:
        """All ToolMetadata entries (for PCD, capability manifests, etc.)."""
        if not self._assembled:
            self.assemble()
        return self._all_metadata

    @property
    def conflicts(self) -> list[CapabilityConflict]:
        """Rejected duplicate tool-name claims recorded during indexing."""
        if not self._assembled:
            self.assemble()
        return list(self._conflicts)

    @property
    def tool_owners(self) -> dict[str, str]:
        """Live tool name → owning plugin id mapping.

        This exposes the registry's first-wins arbitration result to adaptive
        capability selection without letting consumers mutate ownership state.
        """
        if not self._assembled:
            self.assemble()
        return dict(self._tool_owner)

    def get_tools_by_category(self, category: str) -> list[ToolMetadata]:
        """Query tools by x_leapflow.category."""
        return [t for t in self._all_metadata if t.x_leapflow.get("category") == category]

    @property
    def version(self) -> int:
        """Monotonic counter incremented on every mutation. Used by cache invalidation."""
        return self._version

    def notify_mutation(self) -> None:
        """Public API to signal a mutation happened (increments version)."""
        self._bump_version("scope_disposed")

    def _bump_version(self, kind: str, **detail: Any) -> None:
        """Single point where the registry's version changes, and is observed.

        Convergence here is not cosmetic. The version was previously incremented at
        seven separate statements across five methods, so a probe placed at any one
        of them -- including ``notify_mutation``, which only two scope-disposal
        callers reach -- would silently miss the rest. Routing every increment
        through one method makes "the capability set changed" a single fact, which
        is the only way an observer of it can be trusted.

        ``_assembled`` separates boot-time composition from a later runtime change.
        Both bump the version, but only the second is an *evolution*: without the
        distinction, every daemon start would bury the rare real mutation under a
        replay of the initial plugin load.
        """
        self._version += 1
        try:
            from leapflow.domain.evolution_trace import EvolutionStage
            from leapflow.telemetry.evolution_tap import emit_trace, is_enabled

            if not is_enabled():
                return
            emit_trace(
                EvolutionStage.ACT,
                f"registry_{kind}",
                correlation={"registry_version": str(self._version)},
                summary=f"registry {kind} -> v{self._version}",
                detail={
                    "version": self._version,
                    "phase": "runtime" if self._assembled else "composition",
                    "conflicts": [c.to_dict() for c in self._conflicts],
                    **detail,
                },
            )
        except Exception:  # noqa: BLE001 - the registry must never fail on telemetry
            logger.debug("registry: evolution trace failed", exc_info=True)

    @property
    def last_bound_deps(self) -> dict[str, Any]:
        """Read-only view of the most-recently bound runtime dependencies."""
        return dict(self._last_bound_deps)

    @property
    def plugins(self) -> dict[str, ToolPlugin]:
        """Read-only view of registered plugins."""
        return dict(self._plugins)

    def get_plugin(self, plugin_id: str) -> Optional[ToolPlugin]:
        """Get a specific plugin by ID, or None if not registered."""
        return self._plugins.get(plugin_id)

    def get_desktop_semantic_plugin(self) -> Optional[Any]:
        """Return the DesktopSemanticPlugin instance, or None if not registered.

        The engine uses this to query dynamic semantic schemas and handlers
        (the desktop_semantic plugin is the semantic tool registration site).
        """
        return self._plugins.get("desktop_semantic")

    @property
    def tool_pipeline(self) -> ToolExecutionPipeline:
        """The composable tool execution pipeline for interceptor registration.

        Interceptors registered here wrap engine-dispatched tool executions through
        ``invoke_tool_handler`` while preserving the existing approval and
        resolution gates around the handler call.
        """
        return self._tool_pipeline

    @property
    def categories(self) -> set[str]:
        """Set of all registered plugin categories."""
        return {p.category for p in self._plugins.values()}
