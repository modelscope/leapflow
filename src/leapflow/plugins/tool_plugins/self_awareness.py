# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unified self-awareness plugin — faceted agent self-cognition surface.

Aggregates registry, daemon, engine, and build_info into two read-only tools:

* ``self_describe(facet=...)`` — structured introspection by facet
* ``runtime_snapshot()`` — lightweight ~150-token flat dict for quick orientation

All data sources are injected via ``bind_runtime``; missing dependencies degrade
gracefully per facet rather than raising.
"""

from __future__ import annotations

import logging
import time
import weakref
from collections import Counter
from typing import TYPE_CHECKING, Any

from leapflow.plugins.protocol import ToolMetadata

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Cache configuration ──────────────────────────────────────────────────────

_DAEMON_CACHE_TTL_S = 30.0

# ── Facet enum values ────────────────────────────────────────────────────────

_FACETS = ("identity", "capabilities", "runtime", "evolution", "platform", "all")


def _format_uptime(seconds: float) -> str:
    """Format seconds into a human-friendly string like '2h 13m'."""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds) // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    if hours < 24:
        return f"{hours}h {remaining_minutes}m" if remaining_minutes else f"{hours}h"
    days = hours // 24
    remaining_hours = hours % 24
    return f"{days}d {remaining_hours}h" if remaining_hours else f"{days}d"


def _format_context(used: int, total: int) -> str:
    """Format context usage as '112K/1M (11%)'."""
    def _human(n: int) -> str:
        if n >= 1_000_000:
            val = n / 1_000_000
            return f"{val:.1f}M" if val != int(val) else f"{int(val)}M"
        if n >= 1_000:
            val = n / 1_000
            return f"{val:.0f}K" if val >= 10 else f"{val:.1f}K"
        return str(n)

    pct = round(used / total * 100) if total > 0 else 0
    return f"{_human(used)}/{_human(total)} ({pct}%)"


# ── Plugin class ─────────────────────────────────────────────────────────────


class SelfAwarenessPlugin:
    """Unified self-cognition surface for the agent.

    Facade plugin that reads live runtime state from four injected sources
    (registry, daemon_client, engine, build_info) and exposes it through
    two read-only tools. Registry version gating and daemon TTL caching keep
    data current without hot-path cost.
    """

    def __init__(self) -> None:
        # Injected dependencies — all optional, degrade per-facet
        self._daemon_client: Any = None
        self._registry: Any = None
        self._engine_ref: weakref.ref | None = None
        self._build_info: Any = None

        # ── Caches ──
        self._registry_version: int = -1
        self._capabilities_cache: dict[str, Any] = {}

        self._daemon_cache: dict[str, Any] = {}
        self._daemon_cache_ts: float = 0.0

    # ── Protocol properties ──────────────────────────────────────────────

    @property
    def plugin_id(self) -> str:
        return "self_awareness"

    @property
    def category(self) -> str:
        return "system"

    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="self_describe",
                description=(
                    "Introspect LeapFlow's own identity, capabilities, runtime state, "
                    "evolution metrics, or platform connections. Use facet='all' only "
                    "when a comprehensive self-check is explicitly requested."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "facet": {
                            "type": "string",
                            "enum": list(_FACETS),
                            "description": (
                                "Which aspect to inspect: identity (version/model/uptime), "
                                "capabilities (tools/plugins/trust), runtime (context/posture/"
                                "disclosure), evolution (performance/proposals), platform "
                                "(gateway/hardware/env), or all."
                            ),
                        },
                    },
                },
                handler=self._handle_self_describe,
                x_leapflow={
                    "category": "system",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("system.self_describe",),
            ),
            ToolMetadata(
                name="runtime_snapshot",
                description=(
                    "Lightweight (~150 token) flat snapshot of current runtime state: "
                    "model, context budget, posture, disclosure level, turn count, "
                    "cache hit rate, uptime, tool count, and pending approvals."
                ),
                parameters_schema={"type": "object", "properties": {}},
                handler=self._handle_runtime_snapshot,
                x_leapflow={
                    "category": "system",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("system.runtime_snapshot",),
            ),
        ]

    @property
    def dependencies(self) -> list[str]:
        return ["daemon_client"]

    def bind_runtime(self, **deps: Any) -> None:
        """Receive runtime-injected dependencies.

        Accepts: daemon_client, registry, engine (stored as weak ref), build_info.
        """
        if "daemon_client" in deps:
            self._daemon_client = deps["daemon_client"]
        if "registry" in deps:
            self._registry = deps["registry"]
        if "engine" in deps:
            engine = deps["engine"]
            if engine is not None:
                try:
                    self._engine_ref = weakref.ref(engine)
                except TypeError:
                    # Some stub objects cannot be weak-referenced
                    self._engine_ref = lambda: engine  # type: ignore[assignment]
            else:
                self._engine_ref = None
        if "build_info" in deps:
            self._build_info = deps["build_info"]

    # ── Tool handlers ────────────────────────────────────────────────────

    def _handle_self_describe(self, facet: str = "identity", **_: Any) -> dict[str, Any]:
        """Dispatch to facet builders, merging all when facet='all'."""
        if facet not in _FACETS:
            return {"error": f"Unknown facet: {facet!r}. Valid: {', '.join(_FACETS)}"}

        if facet == "all":
            result: dict[str, Any] = {}
            for f in _FACETS:
                if f == "all":
                    continue
                result[f] = self._build_facet(f)
            return result

        return self._build_facet(facet)

    def _handle_runtime_snapshot(self, **_: Any) -> dict[str, Any]:
        """Return a lightweight flat dict for quick agent orientation."""
        engine = self._resolve_engine()
        daemon = self._get_daemon_cache()

        model = ""
        context_str = "unknown"
        posture = "unknown"
        disclosure = "unknown"
        turn = 0
        cache_hit_rate = "unknown"

        if engine is not None:
            snapshot = getattr(engine, "context_budget_snapshot", None)
            if callable(snapshot):
                snapshot = snapshot()
            if isinstance(snapshot, dict):
                used = int(snapshot.get("total_tokens", 0) or 0)
                total = int(snapshot.get("context_length", 0) or 0)
                context_str = _format_context(used, total)
                posture = str(snapshot.get("context_posture", "baseline") or "baseline")
                disclosure = str(snapshot.get("disclosure_level", "unknown") or "unknown")
            turn = int(getattr(engine, "_session_turn_count", 0) or 0)
            # Cache hit rate from usage tracker
            tracker = getattr(engine, "_usage_tracker", None)
            if tracker is not None:
                summary = getattr(tracker, "summary", None)
                if callable(summary):
                    s = summary()
                    rate = getattr(s, "cache_hit_rate", None)
                    if rate is not None and rate >= 0:
                        cache_hit_rate = f"{rate:.1f}%"

        if daemon:
            model = str(daemon.get("model", "") or "")
            if not model:
                model = "unknown"
        elif engine is not None:
            settings = getattr(engine, "_settings", None)
            model = str(getattr(settings, "llm_model", "") or "") if settings else "unknown"

        uptime = "unknown"
        if daemon:
            uptime_s = daemon.get("uptime_s")
            if isinstance(uptime_s, (int, float)) and uptime_s >= 0:
                uptime = _format_uptime(uptime_s)
        elif self._build_info is not None:
            started = getattr(self._build_info, "started_at", 0.0) or 0.0
            if started > 0:
                uptime = _format_uptime(time.time() - started)

        tools_available = 0
        if self._registry is not None:
            try:
                tools_available = len(self._registry.tool_handlers)
            except Exception:
                pass

        pending = int(daemon.get("pending_approvals", 0) or 0) if daemon else 0

        return {
            "model": model,
            "context": context_str,
            "posture": posture,
            "disclosure": disclosure,
            "turn": turn,
            "cache_hit_rate": cache_hit_rate,
            "uptime": uptime,
            "tools_available": tools_available,
            "pending_approvals": pending,
        }

    # ── Facet builders ───────────────────────────────────────────────────

    def _build_facet(self, facet: str) -> dict[str, Any]:
        builder = {
            "identity": self._facet_identity,
            "capabilities": self._facet_capabilities,
            "runtime": self._facet_runtime,
            "evolution": self._facet_evolution,
            "platform": self._facet_platform,
        }.get(facet)
        if builder is None:
            return {"error": f"No builder for facet: {facet!r}"}
        try:
            return builder()
        except Exception as exc:
            logger.debug("self_describe facet %s failed: %s", facet, exc, exc_info=True)
            return {"available": False, "reason": f"Facet {facet!r} raised: {exc}"}

    def _facet_identity(self) -> dict[str, Any]:
        """Version, build, model, provider, context limit, uptime."""
        result: dict[str, Any] = {"available": True}

        if self._build_info is not None:
            result["version"] = getattr(self._build_info, "version", "unknown")
            result["commit"] = getattr(self._build_info, "commit", None) or "unknown"
            dirty = getattr(self._build_info, "dirty_digest", None)
            result["dirty"] = dirty is not None and dirty != ""
        else:
            result["version"] = "unknown"
            result["commit"] = "unknown"
            result["dirty"] = None

        daemon = self._get_daemon_cache()
        if daemon:
            result["model"] = daemon.get("model", "unknown")
            result["context_limit"] = daemon.get("llm_context_length", 0)
            uptime_s = daemon.get("uptime_s", 0)
            result["uptime"] = _format_uptime(uptime_s) if isinstance(uptime_s, (int, float)) else "unknown"
            result["pid"] = daemon.get("pid")
        else:
            engine = self._resolve_engine()
            if engine is not None:
                settings = getattr(engine, "_settings", None)
                result["model"] = str(getattr(settings, "llm_model", "unknown") or "unknown") if settings else "unknown"
                result["context_limit"] = int(getattr(settings, "llm_context_length", 0) or 0) if settings else 0
            else:
                result["model"] = "unknown"
                result["context_limit"] = 0
            if self._build_info is not None:
                started = getattr(self._build_info, "started_at", 0.0) or 0.0
                result["uptime"] = _format_uptime(time.time() - started) if started > 0 else "unknown"
            else:
                result["uptime"] = "unknown"

        return result

    def _facet_capabilities(self) -> dict[str, Any]:
        """Tool count by category, plugin count, trust summary, evolution readiness."""
        if self._registry is None:
            return {"available": False, "reason": "registry not bound"}

        # Registry version gate: rebuild only on mismatch
        current_version = self._registry.version
        if current_version != self._registry_version:
            self._capabilities_cache = self._build_capabilities_report()
            self._registry_version = current_version

        return {**self._capabilities_cache, "available": True}

    def _build_capabilities_report(self) -> dict[str, Any]:
        """Construct capability report from live registry state."""
        reg = self._registry
        all_meta = reg.all_metadata
        plugins = reg.plugins

        # Tools by category
        category_counts: Counter[str] = Counter()
        for meta in all_meta:
            cat = meta.x_leapflow.get("category", "uncategorized") if meta.x_leapflow else "uncategorized"
            category_counts[cat] += 1

        # Trust summary from plugin metadata (if available via scoped registry)
        trust_summary: dict[str, int] = {}
        try:
            from leapflow.plugins.scoped_registry import get_fiber_registry

            fiber_reg = get_fiber_registry()
            if fiber_reg is not None:
                for pid, fiber in fiber_reg.fibers.items():
                    level = str(getattr(fiber, "trust_level", "UNKNOWN"))
                    trust_summary[level] = trust_summary.get(level, 0) + 1
        except (ImportError, AttributeError, RuntimeError):
            pass

        # Evolution readiness
        evolution_ready = False
        try:
            from leapflow.config import get_settings
            settings = get_settings()
            evolution_ready = bool(getattr(settings, "evolution_enabled", False))
        except (ImportError, AttributeError, RuntimeError):
            pass

        return {
            "tool_count": len(all_meta),
            "tools_by_category": dict(category_counts.most_common()),
            "plugin_count": len(plugins),
            "plugin_ids": sorted(plugins.keys()),
            "trust_summary": trust_summary if trust_summary else {"note": "fiber registry unavailable"},
            "evolution_ready": evolution_ready,
            "conflicts": len(reg.conflicts),
        }

    def _facet_runtime(self) -> dict[str, Any]:
        """Context budget, disclosure level, posture, session turns, cache hit rate."""
        engine = self._resolve_engine()
        if engine is None:
            return {"available": False, "reason": "engine not bound"}

        result: dict[str, Any] = {"available": True}

        # Context budget
        snapshot = getattr(engine, "context_budget_snapshot", None)
        if callable(snapshot):
            snapshot = snapshot()
        if isinstance(snapshot, dict):
            used = int(snapshot.get("total_tokens", 0) or 0)
            total = int(snapshot.get("context_length", 0) or 0)
            result["context_used"] = used
            result["context_total"] = total
            result["context_percentage"] = round(used / total * 100, 1) if total > 0 else 0
            result["context_formatted"] = _format_context(used, total)
            result["posture"] = str(snapshot.get("context_posture", "baseline") or "baseline")
            result["disclosure_level"] = str(snapshot.get("disclosure_level", "unknown") or "unknown")
        else:
            result["context_used"] = 0
            result["context_total"] = 0
            result["context_formatted"] = "unknown"
            result["posture"] = "unknown"
            result["disclosure_level"] = "unknown"

        result["session_turn_count"] = int(getattr(engine, "_session_turn_count", 0) or 0)

        # Cache hit rate
        tracker = getattr(engine, "_usage_tracker", None)
        if tracker is not None:
            summary_fn = getattr(tracker, "summary", None)
            if callable(summary_fn):
                s = summary_fn()
                rate = getattr(s, "cache_hit_rate", None)
                result["cache_hit_rate"] = f"{rate:.1f}%" if rate is not None and rate >= 0 else "unknown"
            else:
                result["cache_hit_rate"] = "unknown"
        else:
            result["cache_hit_rate"] = "unknown"

        return result

    def _facet_evolution(self) -> dict[str, Any]:
        """Evolution performance metrics and active proposals count."""
        daemon = self._get_daemon_cache()
        if not daemon:
            if self._daemon_client is None:
                return {"available": False, "reason": "daemon_client not bound"}
            return {"available": False, "reason": "daemon status unavailable"}

        result: dict[str, Any] = {"available": True}
        result["performance"] = daemon.get("evolution_performance", {})

        # Active proposals from watch summary or pending approvals
        result["pending_approvals"] = int(daemon.get("pending_approvals", 0) or 0)

        return result

    def _facet_platform(self) -> dict[str, Any]:
        """Gateway connections, hardware backend, environment sources, active clients."""
        daemon = self._get_daemon_cache()
        if not daemon:
            if self._daemon_client is None:
                return {"available": False, "reason": "daemon_client not bound"}
            return {"available": False, "reason": "daemon status unavailable"}

        result: dict[str, Any] = {"available": True}
        result["active_clients"] = daemon.get("active_clients", 0)
        result["connected_clients"] = daemon.get("connected_clients", 0)
        result["host_backend"] = daemon.get("host_backend", {})
        result["environment_sources"] = daemon.get("environment_sources", {})

        return result

    # ── Internal helpers ─────────────────────────────────────────────────

    def _resolve_engine(self) -> Any:
        """Resolve engine from weak ref, returning None if unavailable."""
        if self._engine_ref is None:
            return None
        engine = self._engine_ref()
        return engine

    def _get_daemon_cache(self) -> dict[str, Any]:
        """Return cached daemon status, refreshing when TTL expires.

        Uses synchronous access only — the daemon_client.status() is async,
        so we store the last-fetched result and expose a sync refresh method
        that callers in an async context can await.
        """
        now = time.monotonic()
        if self._daemon_cache and (now - self._daemon_cache_ts) < _DAEMON_CACHE_TTL_S:
            return self._daemon_cache
        return self._refresh_daemon_cache_sync()

    def _refresh_daemon_cache_sync(self) -> dict[str, Any]:
        """Try to refresh daemon cache synchronously via an existing event loop."""
        if self._daemon_client is None:
            return {}
        try:
            import asyncio

            asyncio.get_running_loop()  # Verify we are in an async context
            # We are inside an async context (tool handler runs within the agent loop).
            # Schedule the coroutine and use a shim to get the result.
            future = asyncio.ensure_future(self._daemon_client.status())
            # Cannot await here in a sync handler; return stale cache and
            # schedule background refresh.
            future.add_done_callback(self._on_daemon_status_fetched)
            return self._daemon_cache
        except RuntimeError:
            # No running event loop — likely in tests or sync CLI
            pass
        return self._daemon_cache

    def _on_daemon_status_fetched(self, future: Any) -> None:
        """Callback when async daemon status completes."""
        try:
            result = future.result()
            if isinstance(result, dict):
                self._daemon_cache = result
                self._daemon_cache_ts = time.monotonic()
        except Exception:
            logger.debug("self_awareness: daemon status refresh failed", exc_info=True)

    def inject_daemon_cache(self, status: dict[str, Any]) -> None:
        """Inject daemon status directly (used by tests and daemon service)."""
        self._daemon_cache = status
        self._daemon_cache_ts = time.monotonic()


# Module-level instance for plugin discovery
plugin = SelfAwarenessPlugin()
