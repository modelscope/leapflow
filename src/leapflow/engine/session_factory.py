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

import copy
from typing import Any

from leapflow.engine.prefix_commitment import PrefixCommitmentController
from leapflow.engine.recovery_coordinator import RecoveryCoordinator
from leapflow.engine.research_ledger import ResearchLedger
from leapflow.engine.tool_execution import ToolExecutionLedger
from leapflow.engine.turn_usage import TurnUsageTracker


def build_session_engine(base_engine: Any, *, session_id: str, working_memory: Any) -> Any:
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
    # Fresh per-session substrate (the concurrency-corrupting parts).
    engine._wm = working_memory
    engine._tool_execution_ledger = ToolExecutionLedger()
    # Fresh per-turn subsystems (stateful accumulators): a session engine must not
    # share governance/ledger/usage/recovery with the base or other sessions.
    engine._context_governance_controller = engine._new_governance()
    engine._research_ledger = ResearchLedger()
    engine._prefix_commitment = PrefixCommitmentController()
    engine._usage_tracker = TurnUsageTracker()
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
