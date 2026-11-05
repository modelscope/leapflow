# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unified self-awareness plugin — the agent's single self-cognition surface.

This is the one place the agent answers questions about *itself* — its identity,
capabilities, runtime state, evolution readiness, or platform connections — from
live runtime evidence rather than from source code, documentation, or memory.

Two read-only tools:

* ``self_describe(facet=...)`` — structured introspection by facet
* ``runtime_snapshot()`` — lightweight ~150-token flat dict for quick orientation

Data sources are resolved by *pull*, not by push-DI, so the surface cannot go
silently dark when a bootstrap wiring step is forgotten:

* capabilities  → the process-global tool/scoped registries (``get_registry`` /
  ``get_scoped_registry``) — the very evidence source ``plugin_list`` uses.
* identity      → this process's captured ``BuildInfo`` + live ``Settings``.
* runtime / evolution / platform → the *session-scoped* daemon ``status`` snapshot,
  obtained through an injected ``runtime_status_provider`` keyed by the active
  turn's ``session_id``. This honours the "session engine is the only reporting
  source" rule: the provider resolves the caller's own session, never a template
  engine (whose figures read zero). Absent that provider (in-process CLI), these
  facets degrade gracefully instead of reporting stale zeros.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections import Counter
from typing import Any

from leapflow.plugins.protocol import ToolMetadata

logger = logging.getLogger(__name__)

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


def _settings_or_none() -> Any:
    """Return live Settings, or None during early bootstrap."""
    try:
        from leapflow.config import get_settings

        return get_settings()
    except (ImportError, AttributeError, RuntimeError):
        return None


def _cache_hit_rate_str(value: Any) -> str:
    """Render a numeric cache hit rate as a percentage string, else 'unknown'."""
    if isinstance(value, (int, float)) and value >= 0:
        return f"{float(value):.1f}%"
    return "unknown"


# ── Plugin class ─────────────────────────────────────────────────────────────


class SelfAwarenessPlugin:
    """Unified self-cognition surface for the agent.

    Facade plugin that resolves live runtime state by pull (global registries,
    captured build info, live settings) plus an injected session-scoped status
    provider, and exposes it through two read-only tools. Registry version
    gating keeps the capabilities report current without hot-path cost.
    """

    def __init__(self) -> None:
        # Injected: a callable ``(session_id: str) -> dict | Awaitable[dict]``
        # returning the session-scoped daemon status snapshot. Optional — the
        # runtime/evolution/platform facets degrade when it is unbound.
        self._status_provider: Any = None

        # Captured once per process, mirroring how the daemon captures its own
        # build fingerprint at startup, so identity works without any wiring.
        try:
            from leapflow.utils.build_info import capture_build_info

            self._build_info: Any = capture_build_info()
        except Exception:  # noqa: BLE001 - identity must never fail to load
            logger.debug("self_awareness: build info capture failed", exc_info=True)
            self._build_info = None

        # ── capabilities cache (rebuilt on registry version change) ──
        self._registry_version: int = -1
        self._capabilities_cache: dict[str, Any] = {}

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
                    "Answer questions about LeapFlow ITSELF from live runtime evidence — its "
                    "identity/version/model, what tools and plugins it has, whether it supports "
                    "plugins/self-evolution/hot-reload, its runtime state, or whether it has any "
                    "form of self-awareness. Prefer this over reading LeapFlow's own source code. "
                    "Pick facet=capabilities for 'what can you do / do you support X', "
                    "identity for version/model, runtime for context/posture, or all for a full "
                    "self-check when explicitly requested."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "facet": {
                            "type": "string",
                            "enum": list(_FACETS),
                            "description": (
                                "Which aspect to inspect: identity (version/model/uptime), "
                                "capabilities (tools/plugins/trust/evolution readiness), runtime "
                                "(context/posture/disclosure), evolution (performance/proposals), "
                                "platform (gateway/hardware/env), or all."
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
                    "summary": "introspect LeapFlow's own identity/capabilities/runtime state",
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
                    "summary": "flat snapshot of live runtime state",
                },
                provides_capabilities=("system.runtime_snapshot",),
            ),
        ]

    @property
    def dependencies(self) -> list[str]:
        return ["runtime_status_provider"]

    def bind_runtime(self, **deps: Any) -> None:
        """Receive runtime-injected dependencies.

        Accepts ``runtime_status_provider`` — a ``(session_id) -> dict`` callable
        (sync or async) resolving the session-scoped daemon status snapshot.
        """
        if "runtime_status_provider" in deps:
            self._status_provider = deps["runtime_status_provider"]

    # ── Tool handlers ────────────────────────────────────────────────────

    async def _handle_self_describe(self, facet: str = "identity", **_: Any) -> dict[str, Any]:
        """Dispatch to facet builders, merging all when facet='all'."""
        if facet not in _FACETS:
            return {"error": f"Unknown facet: {facet!r}. Valid: {', '.join(_FACETS)}"}

        status = await self._resolve_status()

        if facet == "all":
            result: dict[str, Any] = {}
            for f in _FACETS:
                if f == "all":
                    continue
                result[f] = self._build_facet(f, status)
            return result

        return self._build_facet(facet, status)

    async def _handle_runtime_snapshot(self, **_: Any) -> dict[str, Any]:
        """Return a lightweight flat dict for quick agent orientation."""
        status = await self._resolve_status()
        snapshot = status.get("context_budget_snapshot") if isinstance(status, dict) else None
        snapshot = snapshot if isinstance(snapshot, dict) else {}

        used = int(snapshot.get("total_tokens", status.get("context_used", 0)) or 0)
        total = int(snapshot.get("context_length", status.get("llm_context_length", 0)) or 0)
        context_str = _format_context(used, total) if total > 0 else "unknown"
        posture = str(snapshot.get("context_posture", status.get("context_posture", "")) or "unknown")
        disclosure = str(snapshot.get("disclosure_level", "") or "unknown")
        turn = int(status.get("session_turn_count", 0) or 0)
        cache_hit_rate = _cache_hit_rate_str(status.get("cache_hit_rate"))

        model = str(status.get("model", "") or "")
        if not model:
            settings = _settings_or_none()
            model = str(getattr(settings, "llm_model", "") or "unknown") if settings else "unknown"

        uptime = self._uptime(status)

        tools_available = 0
        reg = self._registry_or_none()
        if reg is not None:
            try:
                tools_available = len(reg.tool_handlers)
            except Exception:  # noqa: BLE001 - snapshot is best-effort
                tools_available = 0

        pending = int(status.get("pending_approvals", 0) or 0)

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

    def _build_facet(self, facet: str, status: dict[str, Any]) -> dict[str, Any]:
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
            return builder(status)
        except Exception as exc:  # noqa: BLE001 - a facet fault degrades, never raises
            logger.debug("self_describe facet %s failed: %s", facet, exc, exc_info=True)
            return {"available": False, "reason": f"Facet {facet!r} raised: {exc}"}

    def _facet_identity(self, status: dict[str, Any]) -> dict[str, Any]:
        """Version, build, model, provider, context limit, uptime."""
        result: dict[str, Any] = {"available": True}

        # Prefer the daemon's authoritative build fingerprint; fall back to the
        # fingerprint this process captured at import time.
        build = status.get("build") if isinstance(status.get("build"), dict) else {}
        bi = self._build_info
        result["version"] = (
            build.get("version") or (getattr(bi, "version", None) if bi else None) or "unknown"
        )
        commit = build.get("commit") or (getattr(bi, "commit", None) if bi else None)
        result["commit"] = commit or "unknown"
        dirty = build.get("dirty_digest")
        if dirty is None and bi is not None:
            dirty = getattr(bi, "dirty_digest", None)
        result["dirty"] = bool(dirty)

        if status:
            result["model"] = status.get("model") or "unknown"
            result["context_limit"] = status.get("llm_context_length", 0)
            result["pid"] = status.get("pid") or build.get("pid")
        else:
            settings = _settings_or_none()
            result["model"] = (
                str(getattr(settings, "llm_model", "") or "unknown") if settings else "unknown"
            )
            result["context_limit"] = (
                int(getattr(settings, "llm_context_length", 0) or 0) if settings else 0
            )
            result["pid"] = getattr(bi, "pid", None) if bi else None

        result["uptime"] = self._uptime(status)
        return result

    def _facet_capabilities(self, _status: dict[str, Any]) -> dict[str, Any]:
        """Tool/plugin inventory, trust summary, evolution readiness — from the
        live registry, the same evidence source ``plugin_list`` uses."""
        reg = self._registry_or_none()
        if reg is None:
            return {"available": False, "reason": "tool registry unavailable"}

        # Registry version gate: rebuild only on mismatch.
        current_version = reg.version
        if current_version != self._registry_version:
            self._capabilities_cache = self._build_capabilities_report(reg)
            self._registry_version = current_version

        # Skills are enumerated live (they carry their own index cache) rather
        # than folded into the version-gated report, so a skill added without a
        # tool-registry change still shows up.
        return {**self._capabilities_cache, "skills": self._skill_inventory(), "available": True}

    def _build_capabilities_report(self, reg: Any) -> dict[str, Any]:
        """Construct the capability report from live registry state."""
        all_meta = reg.all_metadata
        plugins = reg.plugins

        category_counts: Counter[str] = Counter()
        mutating = 0
        approval_required = 0
        for meta in all_meta:
            meta_x = meta.x_leapflow or {}
            cat = str(meta_x.get("category", "uncategorized") or "uncategorized")
            category_counts[cat] += 1
            if bool(getattr(meta, "mutates_state", False)):
                mutating += 1
            if meta_x.get("requires_approval") is True:
                approval_required += 1

        # Trust summary from the scoped registry's fibers, when available.
        trust_summary: dict[str, int] = {}
        try:
            from leapflow.plugins.scoped_registry import get_fiber_registry

            fiber_reg = get_fiber_registry()
            if fiber_reg is not None:
                for _pid, fiber in fiber_reg.fibers.items():
                    level = str(getattr(fiber, "trust_level", "UNKNOWN"))
                    trust_summary[level] = trust_summary.get(level, 0) + 1
        except (ImportError, AttributeError, RuntimeError):
            pass

        # Evolution readiness from live settings.
        evolution_ready = False
        try:
            from leapflow.config import get_settings

            evolution_ready = bool(getattr(get_settings(), "evolution_enabled", False))
        except (ImportError, AttributeError, RuntimeError):
            pass

        return {
            "tool_count": len(all_meta),
            "tools_by_category": dict(category_counts.most_common()),
            "mutating_tool_count": mutating,
            "approval_required_tool_count": approval_required,
            "plugin_count": len(plugins),
            "plugin_ids": sorted(plugins.keys()),
            "supports_plugins": "self_management" in plugins,
            "trust_summary": trust_summary if trust_summary else {"note": "fiber registry unavailable"},
            "evolution_ready": evolution_ready,
            "conflicts": len(reg.conflicts),
        }

    def _facet_runtime(self, status: dict[str, Any]) -> dict[str, Any]:
        """Context budget, disclosure level, posture, session turns, cache hit rate."""
        if not status:
            return {"available": False, "reason": "runtime status unavailable (no daemon bound)"}

        result: dict[str, Any] = {"available": True}
        snapshot = status.get("context_budget_snapshot")
        snapshot = snapshot if isinstance(snapshot, dict) else {}

        used = int(snapshot.get("total_tokens", status.get("context_used", 0)) or 0)
        total = int(snapshot.get("context_length", status.get("llm_context_length", 0)) or 0)
        result["context_used"] = used
        result["context_total"] = total
        result["context_percentage"] = round(used / total * 100, 1) if total > 0 else 0
        result["context_formatted"] = _format_context(used, total)
        result["posture"] = str(
            snapshot.get("context_posture", status.get("context_posture", "baseline")) or "baseline"
        )
        result["disclosure_level"] = str(snapshot.get("disclosure_level", "") or "unknown")
        result["session_turn_count"] = int(status.get("session_turn_count", 0) or 0)
        result["cache_hit_rate"] = _cache_hit_rate_str(status.get("cache_hit_rate"))
        return result

    def _facet_evolution(self, status: dict[str, Any]) -> dict[str, Any]:
        """Evolution performance metrics and active proposals count."""
        if not status:
            return {"available": False, "reason": "evolution status unavailable (no daemon bound)"}
        return {
            "available": True,
            "performance": status.get("evolution_performance", {}),
            "pending_approvals": int(status.get("pending_approvals", 0) or 0),
        }

    def _facet_platform(self, status: dict[str, Any]) -> dict[str, Any]:
        """Gateway connections, hardware backend, environment sources, active clients."""
        if not status:
            return {"available": False, "reason": "platform status unavailable (no daemon bound)"}
        return {
            "available": True,
            "active_clients": status.get("active_clients", 0),
            "connected_clients": status.get("connected_clients", 0),
            "host_backend": status.get("host_backend", {}),
            "environment_sources": status.get("environment_sources", {}),
        }

    # ── Internal helpers ─────────────────────────────────────────────────

    def _uptime(self, status: dict[str, Any]) -> str:
        """Resolve uptime from the daemon status, else the captured build info."""
        uptime_s = status.get("uptime_s") if status else None
        if isinstance(uptime_s, (int, float)) and uptime_s >= 0:
            return _format_uptime(uptime_s)
        if self._build_info is not None:
            started = getattr(self._build_info, "started_at", 0.0) or 0.0
            if started > 0:
                return _format_uptime(time.time() - started)
        return "unknown"

    def _registry_or_none(self) -> Any:
        """Return the process-global tool registry, or None if unavailable."""
        try:
            from leapflow.plugins import get_registry

            return get_registry()
        except (ImportError, AttributeError, RuntimeError):
            logger.debug("self_awareness: tool registry unavailable", exc_info=True)
            return None

    def _skill_inventory(self) -> dict[str, Any]:
        """Live skills summary from the skill discovery subsystem (pull).

        Skills are a distinct capability surface from tools/plugins, so the
        canonical self-cognition facet reports them too — otherwise the agent
        must reach for skills_list separately and self_describe under-reports
        what LeapFlow can do.
        """
        try:
            from leapflow.skills.discovery import skill_inventory_summary

            return skill_inventory_summary()
        except Exception:  # noqa: BLE001 - skills are optional; degrade, never raise
            logger.debug("self_awareness: skill inventory unavailable", exc_info=True)
            return {"available": False, "reason": "skill discovery unavailable"}

    async def _resolve_status(self) -> dict[str, Any]:
        """Fetch the session-scoped status snapshot via the injected provider.

        The active turn's ``session_id`` is read from the per-turn tool execution
        context so the daemon resolves the caller's own session engine — never a
        shared/most-recent one. Returns an empty dict when no provider is bound
        (in-process CLI) or on any failure; facets degrade accordingly.
        """
        if self._status_provider is None:
            return {}

        session_id = ""
        try:
            from leapflow.tools.execution_context import current_tool_context

            ctx = current_tool_context()
            if ctx is not None:
                session_id = str(getattr(ctx, "session_id", "") or "")
        except Exception:  # noqa: BLE001 - context lookup must never break the tool
            session_id = ""

        try:
            result = self._status_provider(session_id)
            if inspect.isawaitable(result):
                result = await result
            return result if isinstance(result, dict) else {}
        except Exception:  # noqa: BLE001 - a status fault degrades, never raises
            logger.debug("self_awareness: status provider failed", exc_info=True)
            return {}


# Module-level instance for plugin discovery
plugin = SelfAwarenessPlugin()
