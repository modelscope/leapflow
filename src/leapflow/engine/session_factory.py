"""Session-scoped engine factory for concurrent, isolated turn execution (Stage 3).

Builds a per-session ``AgentEngine`` that SHARES the base engine's stateless /
already-wired services (LLM client, DuckDB stores, skill registry, tool bridge,
compressor config, guardrail, subagent manager, ...) by reference, but owns a
FRESH per-session working memory and idempotency ledger and starts from a clean
per-turn state slate.

This isolates exactly the substrate that concurrent turns corrupt — the working
memory (a single unkeyed deque) and the engine's per-turn state — without
duplicating the engine wiring (which is scattered across the context setup) or
changing the engine's single-turn internals.

Phase P3-1: additive only. The factory is not yet used by the daemon (that is
P3-2's SessionRegistry); this phase adds the mechanism and proves isolation.

See ``temp/plan/concurrent_turns_stage3.md`` (Approach D, §4.1–4.3).
"""
from __future__ import annotations

import atexit
import copy
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from leapflow.engine.prefix_commitment import PrefixCommitmentController
from leapflow.engine.recovery_coordinator import RecoveryCoordinator
from leapflow.engine.research_ledger import ResearchLedger
from leapflow.engine.tool_execution import ToolExecutionLedger
from leapflow.engine.turn_usage import TurnUsageTracker
from leapflow.learning.plugin_trust import PluginTrustLedger

logger = logging.getLogger(__name__)

# Process-global DuckDB store backing the plugin trust ledger. Set once during
# first-time sink wiring; used by ``persist_plugin_trust_state`` (atexit / daemon
# shutdown) so trust earned in one process survives a restart.
_DEFAULT_STATS_STORE: Any = None


class _PersistingTrustLedger(PluginTrustLedger):
    """Trust ledger that flushes to a DuckDB store on trust-level transitions.

    Trust levels change rarely (only on a promotion/demotion after a streak, or
    a hard-failure freeze), so persisting on a *level change* keeps DuckDB writes
    off the per-tool hot path while guaranteeing the durable state tracks the
    in-memory ledger. The final counter state is additionally flushed on process
    exit via ``persist_plugin_trust_state`` (registered with ``atexit``).
    """

    def __init__(self, *, store: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._store = store

    def set_store(self, store: Any) -> None:
        """Attach (or replace) the durable store used for flushing."""
        self._store = store

    def record_success(self, plugin_id: str) -> None:
        before = self.level(plugin_id)
        super().record_success(plugin_id)
        after = self.level(plugin_id)
        if after != before:
            self._flush()
            self._trace_transition(plugin_id, before, after, hard=False)

    def record_failure(self, plugin_id: str, *, hard: bool = False) -> None:
        before = self.level(plugin_id)
        super().record_failure(plugin_id, hard=hard)
        after = self.level(plugin_id)
        # ``hard`` freezes the plugin even when the reported level is unchanged
        # (already DRAFT), so persist it explicitly to record the frozen set.
        if hard or after != before:
            self._flush()
            self._trace_transition(plugin_id, before, after, hard=hard)

    def _trace_transition(
        self, plugin_id: str, before: Any, after: Any, *, hard: bool
    ) -> None:
        """Emit the trust transition, which nothing else records durably.

        Placed here rather than in ``PluginTrustLedger`` for two reasons. The base
        ledger is a pure domain object with no dependencies, and an observability
        import does not belong in it; and this subclass has *already* computed the
        before/after pair for the flush, so the transition is a fact in hand rather
        than one that has to be detected a second time.

        Only the level a plugin currently holds is persisted. The moment it moved,
        and the direction, exist nowhere else -- which is exactly why a promotion to
        PRODUCTION or a freeze on an internal defect cannot be reconstructed after
        the fact from the trust state alone.
        """
        try:
            from leapflow.domain.evolution_trace import EvolutionStage
            from leapflow.telemetry.evolution_tap import emit_trace, is_enabled

            if not is_enabled():
                return
            from_name = getattr(before, "name", str(before))
            to_name = getattr(after, "name", str(after))
            frozen = bool(self.is_frozen(plugin_id))
            emit_trace(
                EvolutionStage.LEARN,
                "trust_frozen" if hard else "trust_transition",
                correlation={"plugin_id": plugin_id},
                summary=(
                    f"{plugin_id}: frozen at {to_name} by an internal defect"
                    if hard
                    else f"{plugin_id}: {from_name} -> {to_name}"
                ),
                detail={
                    "plugin_id": plugin_id,
                    "from": from_name,
                    "to": to_name,
                    # A frozen plugin reports DRAFT, so the level alone cannot say
                    # whether it is new or permanently disqualified.
                    "frozen": frozen,
                    "hard_failure": hard,
                },
            )
        except Exception:  # noqa: BLE001 - trust accounting must not fail on telemetry
            logger.debug("plugin trust: evolution trace failed", exc_info=True)

    def _flush(self) -> None:
        """Persist current ledger state; failures degrade to memory-only."""
        store = self._store
        if store is None:
            return
        try:
            store.save_trust_state(self.to_state())
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            logger.warning("Plugin trust flush failed (memory-only): %s", exc)


def _default_stats_db_path() -> Optional[Path]:
    """Derive ``plugin_stats.duckdb`` from the active profile layout's DB dir.

    Placed alongside the other profile DuckDB stores (e.g. the memory store),
    using only existing ``ProfileLayout`` APIs — no new config field or layout
    descriptor. Returns ``None`` when no profile layout is reachable.
    """
    try:
        from leapflow.config import get_settings

        layout = getattr(get_settings(), "profile_layout", None)
        db_dir = getattr(layout, "db_dir", None)
        if db_dir is None:
            return None
        return Path(db_dir) / "plugin_stats.duckdb"
    except (ImportError, RuntimeError, AttributeError, OSError) as exc:
        logger.warning("Cannot derive plugin stats DB path: %s", exc)
        return None


def _resolve_stats_store(db_path: str | Path | None) -> Any:
    """Build a ``PluginStatsStore`` for ``db_path`` (or the profile default).

    Degrades to ``None`` (memory-only) if the store cannot be constructed, so a
    missing DuckDB backend never crashes engine wiring.
    """
    try:
        from leapflow.learning.plugin_stats_store import PluginStatsStore

        path = Path(db_path) if db_path is not None else _default_stats_db_path()
        if path is None:
            return None
        return PluginStatsStore(path)
    except (ImportError, RuntimeError, OSError) as exc:
        logger.warning("Plugin stats persistence unavailable: %s", exc)
        return None


def _load_or_new_trust_ledger(store: Any) -> _PersistingTrustLedger:
    """Restore the persisted trust ledger, or start fresh. Never raises.

    A missing or corrupt store yields a fresh (DRAFT) ledger rather than an
    error, honoring graceful degradation.
    """
    if store is not None:
        try:
            state = store.load_trust_state()
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning("Failed to load plugin trust state: %s", exc)
            state = None
        if state:
            try:
                ledger = _PersistingTrustLedger.load_state(state)
                ledger.set_store(store)
                return ledger
            except (ValueError, TypeError, KeyError) as exc:
                logger.warning("Corrupt plugin trust state ignored: %s", exc)
    return _PersistingTrustLedger(store=store)


def _load_or_new_usage_tracker(store: Any) -> Any:
    """Restore the persisted usage tracker, or start fresh. Never raises.

    Reliability scoring needs history that outlives the process; a missing or
    corrupt blob yields an empty tracker rather than an error, so a bad write
    costs recent signal instead of the session.
    """
    from leapflow.learning.plugin_stats import PluginUsageTracker as _PUTracker

    if store is not None:
        try:
            state = store.load_usage_state()
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning("Failed to load plugin usage state: %s", exc)
            state = None
        if state:
            try:
                return _PUTracker.load_state(state)
            except (ValueError, TypeError, KeyError) as exc:
                logger.warning("Corrupt plugin usage state ignored: %s", exc)
    return _PUTracker()


def persist_plugin_trust_state() -> bool:
    """Flush the process-global plugin trust ledger to its DuckDB store.

    Safe to call at any time (registered with ``atexit`` and callable on daemon
    shutdown). Returns ``True`` when state was written, ``False`` when no store /
    ledger is wired or the write failed. Never raises.
    """
    store = _DEFAULT_STATS_STORE
    if store is None:
        return False
    try:
        from leapflow.learning.plugin_advisor import get_default_advisor

        advisor = get_default_advisor()
        ledger = getattr(advisor, "_trust_ledger", None) if advisor is not None else None
        if ledger is None:
            return False
        saved = bool(store.save_trust_state(ledger.to_state()))
        # Usage samples are flushed alongside trust: the two are read back as a
        # pair by reliability scoring, and persisting only one would restore a
        # trust level with no evidence behind it.
        usage_tracker = getattr(advisor, "_usage_tracker", None)
        if usage_tracker is not None:
            store.save_usage_state(usage_tracker.to_state())
        return saved
    except (ImportError, RuntimeError, AttributeError, OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to persist plugin trust state: %s", exc)
        return False


def _wire_plugin_stats_sink(
    tracker: TurnUsageTracker, db_path: str | Path | None = None
) -> None:
    """Attach process-global plugin learning sink to a TurnUsageTracker.

    Lazily initializes the PluginUsageTracker, PluginTrustLedger, and
    PluginAdvisor singletons on first call. Safe to call multiple times;
    subsequent calls simply set the sink reference.

    On first-time initialization the trust ledger and the rolling usage samples
    are restored from a profile-scoped DuckDB store (``plugin_stats.duckdb``
    beside the other profile DBs) so trust and reliability history survive
    process restarts. ``db_path`` overrides the derived path (used by tests);
    when omitted the profile layout supplies it. If the store is unavailable both
    stay in memory only.
    """
    try:
        from leapflow.learning.plugin_advisor import (
            PluginAdvisor,
            get_default_advisor,
            set_default_advisor,
        )

        advisor = get_default_advisor()
        if advisor is not None:
            # Already initialized — just set the sink
            tracker.set_plugin_stats_sink(advisor._usage_tracker)
            return

        # First-time initialization: restore durable trust + usage state if present.
        store = _resolve_stats_store(db_path)
        trust_ledger = _load_or_new_trust_ledger(store)
        usage_tracker = _load_or_new_usage_tracker(store)
        usage_tracker.set_trust_ledger(trust_ledger)
        # Quarantine had no feed at all: trust demotion was immediate but a failing
        # plugin was never disabled. The tracker is process-level so the tool-outcome
        # sink that increments it and the cold-path sweep that drains it share one
        # instance.
        try:
            from leapflow.evolution.observations import current_quarantine_tracker

            usage_tracker.set_quarantine_tracker(current_quarantine_tracker())
        except Exception:  # noqa: BLE001 - governance wiring must not fail composition
            logger.debug("quarantine tracker not attached", exc_info=True)
        advisor = PluginAdvisor(trust_ledger, usage_tracker)
        set_default_advisor(advisor)
        tracker.set_plugin_stats_sink(usage_tracker)

        global _DEFAULT_STATS_STORE
        if store is not None and _DEFAULT_STATS_STORE is None:
            _DEFAULT_STATS_STORE = store
            atexit.register(persist_plugin_trust_state)
    except (ImportError, RuntimeError, AttributeError):
        pass  # Learning module not available — degrade gracefully


def _settings_for_workspace(settings: Any, workspace_root: str | Path | None) -> Any:
    if settings is None or not workspace_root:
        return settings
    root = Path(str(workspace_root)).expanduser().resolve()
    try:
        return replace(settings, workspace_root=root)
    except TypeError:
        cloned = copy.copy(settings)
        setattr(cloned, "workspace_root", root)
        return cloned


def build_session_engine(
    base_engine: Any,
    *,
    session_id: str,
    working_memory: Any,
    workspace_root: str | Path | None = None,
) -> Any:
    """Return a per-session engine sharing ``base_engine``'s wired services.

    The returned engine has its own working memory, idempotency ledger, and
    FRESH per-turn subsystems (governance / research ledger / commitment / usage
    / recovery), plus a clean per-turn state slate. This is required because some
    of those subsystems accumulate state across a turn/session (e.g. context
    governance tracks exploration rounds) and must not be shared with the base or
    other sessions, or concurrent turns would trigger each other's nudges.

    Stateless / session-keyed shared services (LLM, DuckDB stores, registry, tool
    bridge, and the context compressor — which operates on passed messages and
    keeps its archive_fn wiring) are shared by reference. The engine's single-turn
    internals are unchanged.
    """
    engine = copy.copy(base_engine)  # shallow copy: own __dict__, shared attr refs
    engine._settings = _settings_for_workspace(
        getattr(base_engine, "_settings", None),
        workspace_root,
    )
    # Fresh per-session substrate (the concurrency-corrupting parts).
    engine._wm = working_memory
    engine._tool_execution_ledger = ToolExecutionLedger()
    # Fresh per-turn subsystems (stateful accumulators): a session engine must not
    # share governance/ledger/usage/recovery with the base or other sessions.
    engine._context_governance_controller = engine._new_governance()
    engine._research_ledger = ResearchLedger()
    engine._prefix_commitment = PrefixCommitmentController()
    engine._usage_tracker = TurnUsageTracker()
    _wire_plugin_stats_sink(engine._usage_tracker)
    engine._recovery_coordinator = RecoveryCoordinator()
    engine._last_context_snapshot = {}
    engine._last_turn_tool_categories = frozenset()
    # Clean per-turn state slate (each turn also reassigns these, but a fresh
    # session engine must not inherit the base engine's in-flight state).
    engine._current_session_id = session_id
    engine._current_turn_id = ""
    engine._current_command_id = ""
    engine._active_frame = None
    engine._cancel_requested = False
    engine._active_task = None
    engine._session_turn_count = 0
    return engine
