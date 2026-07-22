"""Scenario-based integration tests for the agent execution pipeline."""

from __future__ import annotations

import asyncio
import tempfile
from typing import List
from unittest.mock import AsyncMock

import pytest

from conftest import StubLLM, make_settings
from leapflow.engine.engine import (
    AgentEngine,
    _normalize_tool_name,
    _resolve_tool_name,
    _tool_args_metadata,
    build_default_registry,
)
from leapflow.engine.intent_classifier import Intent
from leapflow.engine.task_graph import (
    GraphValidationError,
    RetryPolicy,
    TaskGraph,
    TaskNode,
    TaskStatus,
)
from leapflow.memory import (
    EpisodicMemoryProvider, SemanticMemoryProvider, WorkingMemoryProvider,
)


class _FixedClassifier:
    """Deterministic intent classifier for routing tests."""

    def __init__(self, label: str) -> None:
        self._intent = Intent(label=label, reason="test")

    async def classify(self, user_text: str) -> Intent:
        return self._intent


def _node(
    id: str,
    *,
    action: str = "test_skill",
    depends_on: List[str] | None = None,
    **kwargs,
) -> TaskNode:
    return TaskNode(
        id=id,
        name=f"Node {id}",
        action=action,
        depends_on=depends_on or [],
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════
# Engine scenarios
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_simple_query_returns_answer() -> None:
    """memory_recent intent: LLM synthesizes an answer from recent events."""
    answer = (
        '{"thought":"done","action":{"type":"answer","name":"final",'
        '"payload":{"text":"You edited README.md"}}}'
    )
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM([answer])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        imm.ingest("event.fs_change", "File modified: /tmp/README.md", path="/tmp/README.md")
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("memory_recent")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            out = await engine.run("What did I change recently?")
            assert "README" in out
            assert llm.call_count == 1
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_react_loop_tool_then_answer() -> None:
    """Unified tool loop: executes a tool call, then returns final answer."""
    # Provide a parseable tool call (time_get), then a plain text final answer
    tool_reply = '<tool_call>{"name": "time_get", "arguments": {}}</tool_call>'
    final_reply = "UI observed successfully"
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM(
            [
                tool_reply,
                final_reply,
            ]
        )
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            out = await engine.run("Observe the desktop and tell me what you see.")
            assert out == "UI observed successfully"
            assert llm.call_count >= 2
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_child_frame_runs_isolated_from_parent() -> None:
    """RB4: a recursive child frame runs the full loop without contaminating the
    parent's per-turn subsystems.

    This is the completeness check for per-turn state isolation: if any per-turn
    subsystem were not swapped around the child run, the parent's reference would
    change (or its state mutate) and an assertion below would fail.
    """
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM(["child final answer"])  # single round: direct final answer, no tools
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            # Capture the parent (engine baseline) per-turn subsystems.
            parent_governance = engine._context_governance_controller
            parent_ledger = engine._research_ledger
            parent_usage = engine._usage_tracker
            parent_commitment = engine._prefix_commitment
            parent_compressor = engine._compressor
            assert engine._active_frame is None

            # A child frame must have fresh, distinct subsystems.
            child = engine._build_child_frame("do a subtask", depth=1)
            assert child.governance is not parent_governance
            assert child.ledger is not parent_ledger
            assert child.usage_tracker is not parent_usage
            assert child.depth == 1

            out = await engine._run_child_frame(child)
            assert out == "child final answer"

            # Parent's per-turn subsystems must be restored (references unchanged).
            assert engine._context_governance_controller is parent_governance
            assert engine._research_ledger is parent_ledger
            assert engine._usage_tracker is parent_usage
            assert engine._prefix_commitment is parent_commitment
            assert engine._compressor is parent_compressor
            assert engine._active_frame is None
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_full_loop_subagent_executor_runs_isolated() -> None:
    """RB4 wiring: EngineFrameSubagentExecutor runs a delegated subagent through
    the engine's full loop on an isolated child frame, restricted to permitted
    tools and leaving the parent's per-turn state untouched.
    """
    from leapflow.engine.subagent import EngineFrameSubagentExecutor, SubagentConfig

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM(["subagent completed the task"])  # single-round final answer
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            parent_ledger = engine._research_ledger
            parent_governance = engine._context_governance_controller

            executor = EngineFrameSubagentExecutor(
                run_child=engine._run_subagent_goal,
                tool_names=["time_get", "shell_run", "delegate_task"],
                settings=settings,
            )
            # depth=1 with max_depth=2 gates delegate_task out of the child.
            config = SubagentConfig(goal="do a focused subtask", depth=1)
            result = await executor.execute_subagent(config)

            assert result.status == "completed"
            assert "subagent" in result.summary
            # The child ran on an isolated frame: parent references are intact.
            assert engine._research_ledger is parent_ledger
            assert engine._context_governance_controller is parent_governance
            assert engine._active_frame is None
        finally:
            lt.close()


def test_engine_recalibrate_difficulty_applies_and_resets(tmp_path) -> None:
    """S3-L3: enabled calibration installs a bounded scale_k derived from the
    baseline from stored turn signals; reset reverts it exactly.
    """
    import dataclasses

    from leapflow.storage.evolution_store import DuckDBEvolutionStore

    settings = dataclasses.replace(make_settings(str(tmp_path)), agent_calibration_enabled=True)
    from leapflow.platform.mock import MockBridge

    rpc = MockBridge()
    llm = StubLLM(["x"])
    wm = WorkingMemoryProvider(max_tokens=1024)
    lt = SemanticMemoryProvider(source=settings.duckdb_path)
    imm = EpisodicMemoryProvider()
    store = DuckDBEvolutionStore(str(tmp_path / "evo.duckdb"))
    try:
        reg = build_default_registry(rpc, llm, wm, lt)
        classifier = _FixedClassifier("complex")
        engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
        baseline = engine._budget_config.scale_k

        # 20 over-predicted turns: high difficulty costs *less* effort than low.
        for i in range(20):
            difficulty = 0.9 if i % 2 == 0 else 0.1
            steps = 2 if difficulty > 0.5 else 4
            store.save_episode(
                episode_id=f"e{i}", skill_name="turn_x", actions=[],
                outcome="completed", reward=1.0,
                context={"final_difficulty": difficulty, "steps": steps},
            )

        result = engine.recalibrate_difficulty(store)
        assert result.applied is True
        assert engine._budget_config.scale_k < baseline          # over-predicted -> reduce
        assert 0.25 <= engine._budget_config.scale_k <= 3.0       # clamped

        engine.reset_calibration()
        assert engine._budget_config.scale_k == baseline          # exact revert
    finally:
        store.close()
        lt.close()


def test_engine_recalibrate_difficulty_disabled_is_noop(tmp_path) -> None:
    """S3-L3: with calibration disabled (default), recalibration never changes the weight."""
    from leapflow.storage.evolution_store import DuckDBEvolutionStore

    settings = make_settings(str(tmp_path))  # agent_calibration_enabled defaults False
    from leapflow.platform.mock import MockBridge

    rpc = MockBridge()
    llm = StubLLM(["x"])
    wm = WorkingMemoryProvider(max_tokens=1024)
    lt = SemanticMemoryProvider(source=settings.duckdb_path)
    imm = EpisodicMemoryProvider()
    store = DuckDBEvolutionStore(str(tmp_path / "evo.duckdb"))
    try:
        reg = build_default_registry(rpc, llm, wm, lt)
        engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, _FixedClassifier("complex"))
        baseline = engine._budget_config.scale_k
        for i in range(20):
            store.save_episode(
                episode_id=f"e{i}", skill_name="turn_x", actions=[],
                outcome="completed", reward=1.0,
                context={"final_difficulty": 0.9, "steps": 1},
            )
        result = engine.recalibrate_difficulty(store)
        assert result.applied is False
        assert "disabled" in result.reason
        assert engine._budget_config.scale_k == baseline
    finally:
        store.close()
        lt.close()


def test_engine_recalibrate_thresholds_applies_and_resets(tmp_path) -> None:
    """S3-L4: premature-finalization signal raises the finalize threshold
    (bounded, from baseline), rebuilding governance; reset reverts to None.
    """
    import dataclasses

    from leapflow.storage.evolution_store import DuckDBEvolutionStore

    settings = dataclasses.replace(make_settings(str(tmp_path)), agent_calibration_enabled=True)
    from leapflow.platform.mock import MockBridge

    rpc = MockBridge()
    llm = StubLLM(["x"])
    wm = WorkingMemoryProvider(max_tokens=1024)
    lt = SemanticMemoryProvider(source=settings.duckdb_path)
    imm = EpisodicMemoryProvider()
    store = DuckDBEvolutionStore(str(tmp_path / "evo.duckdb"))
    try:
        reg = build_default_registry(rpc, llm, wm, lt)
        engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, _FixedClassifier("complex"))
        baseline = settings.context_finalizing_ratio
        # 20 finalizing-posture turns that failed -> premature -> raise threshold.
        for i in range(20):
            store.save_episode(
                episode_id=f"e{i}", skill_name="turn_x", actions=[],
                outcome="failed", reward=-0.5,
                context={"final_posture": "finalizing", "final_difficulty": 0.5},
            )
        result = engine.recalibrate_thresholds(store)
        assert result.applied is True
        assert engine._calibrated_finalizing_ratio >= baseline    # premature -> raise (or clamp)
        assert 0.6 <= engine._calibrated_finalizing_ratio <= 0.98  # clamped band

        engine.reset_threshold_calibration()
        assert engine._calibrated_finalizing_ratio is None
    finally:
        store.close()
        lt.close()


def test_engine_orientation_view_aggregates_ledger(tmp_path) -> None:
    """S4-D1: orientation_view is a read-only unified query over existing state;
    the current research ledger forms the working layer. Changes nothing.
    """
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM(["x"])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, _FixedClassifier("complex"))
            engine._research_ledger.note("finding", "A uses DuckDB")
            engine._research_ledger.note("open_question", "does B cache?")

            orientation = engine.orientation_view()
            working_texts = [it.text for it in orientation.by_layer("working")]
            assert "A uses DuckDB" in working_texts
            assert any("does B cache?" in t for t in working_texts)
            # Read-only: the ledger is unchanged (open question still tracked).
            assert engine._research_ledger.open_question_count == 1
        finally:
            lt.close()


def test_child_frame_gets_isolated_session(tmp_path) -> None:
    """S4-E: a recursive child frame persists under its own isolated ``sub_``
    session, never the parent turn's session.
    """
    import dataclasses

    class _FakeConvStore:
        def __init__(self) -> None:
            self.created: list = []

        def create_session(self, session_id, **kwargs) -> None:
            self.created.append(session_id)

    with tempfile.TemporaryDirectory() as td:
        settings = dataclasses.replace(make_settings(td), session_persistence_enabled=True)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM(["x"])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, _FixedClassifier("complex"))
            engine._conversation_store = _FakeConvStore()

            root_sid = engine._ensure_session_for_frame(engine._build_root_frame("hello"), "hello")
            child = engine._build_child_frame("sub goal", depth=1)
            child_sid = engine._ensure_session_for_frame(child, "sub goal")

            assert child_sid is not None and child_sid.startswith("sub_")
            assert child_sid != root_sid                     # isolated from the root session
            assert child_sid != engine._current_session_id   # never the parent's session
        finally:
            lt.close()


def test_periodic_recalibration_runs_every_interval(tmp_path) -> None:
    """E-2: with a positive interval, S3 calibration re-runs every N root turns
    (0 = one-shot only). Bounded/gated like the underlying recalibration.
    """
    import dataclasses

    from leapflow.storage.evolution_store import DuckDBEvolutionStore

    settings = dataclasses.replace(
        make_settings(str(tmp_path)),
        agent_calibration_enabled=True,
        agent_calibration_interval_turns=2,
    )
    from leapflow.platform.mock import MockBridge

    rpc = MockBridge()
    llm = StubLLM(["x"])
    wm = WorkingMemoryProvider(max_tokens=1024)
    lt = SemanticMemoryProvider(source=settings.duckdb_path)
    imm = EpisodicMemoryProvider()
    store = DuckDBEvolutionStore(str(tmp_path / "evo.duckdb"))
    try:
        reg = build_default_registry(rpc, llm, wm, lt)
        engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, _FixedClassifier("complex"))
        # Over-predicted episodes so recalibration reduces scale_k when it fires.
        for i in range(20):
            difficulty = 0.9 if i % 2 == 0 else 0.1
            steps = 2 if difficulty > 0.5 else 4
            store.save_episode(
                episode_id=f"e{i}", skill_name="turn_x", actions=[],
                outcome="completed", reward=1.0,
                context={"final_difficulty": difficulty, "steps": steps},
            )
        engine.set_calibration_store(store)
        baseline = engine._budget_config.scale_k

        engine._maybe_periodic_recalibration()          # turn 1: counter 1 < 2 -> no change
        assert engine._budget_config.scale_k == baseline
        engine._maybe_periodic_recalibration()          # turn 2: interval hit -> recalibrate
        assert engine._budget_config.scale_k < baseline
    finally:
        store.close()
        lt.close()


def test_periodic_recalibration_off_by_default(tmp_path) -> None:
    """E-2: interval 0 (default) never re-calibrates even when enabled."""
    import dataclasses

    from leapflow.storage.evolution_store import DuckDBEvolutionStore

    settings = dataclasses.replace(make_settings(str(tmp_path)), agent_calibration_enabled=True)
    from leapflow.platform.mock import MockBridge

    rpc = MockBridge()
    llm = StubLLM(["x"])
    wm = WorkingMemoryProvider(max_tokens=1024)
    lt = SemanticMemoryProvider(source=settings.duckdb_path)
    imm = EpisodicMemoryProvider()
    store = DuckDBEvolutionStore(str(tmp_path / "evo.duckdb"))
    try:
        reg = build_default_registry(rpc, llm, wm, lt)
        engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, _FixedClassifier("complex"))
        for i in range(20):
            store.save_episode(
                episode_id=f"e{i}", skill_name="turn_x", actions=[],
                outcome="completed", reward=1.0,
                context={"final_difficulty": 0.9, "steps": 1},
            )
        engine.set_calibration_store(store)
        baseline = engine._budget_config.scale_k
        for _ in range(5):
            engine._maybe_periodic_recalibration()
        assert engine._budget_config.scale_k == baseline    # interval 0 -> never fires
    finally:
        store.close()
        lt.close()


def _long_messages(pairs: int = 30) -> list:
    msgs = [{"role": "system", "content": "You are a helpful agent."}]
    for i in range(pairs):
        msgs.append({"role": "user", "content": ("context token " * 40) + f" #{i}"})
        msgs.append({"role": "assistant", "content": ("response detail " * 40) + f" #{i}"})
    return msgs


def _writeback_engine(td, *, enabled: bool):
    import dataclasses

    from conftest import make_settings
    settings = dataclasses.replace(
        make_settings(td), llm_context_length=400, agent_compression_writeback=enabled,
    )
    from leapflow.platform.mock import MockBridge
    rpc = MockBridge()
    llm = StubLLM(["x"])
    wm = WorkingMemoryProvider(max_tokens=1024)
    lt = SemanticMemoryProvider(source=settings.duckdb_path)
    imm = EpisodicMemoryProvider()
    reg = build_default_registry(rpc, llm, wm, lt)
    engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, _FixedClassifier("complex"))
    return engine, lt


def test_compression_writeback_persists_when_enabled() -> None:
    """E-3 (CL-8): with writeback on, structural compression is persisted back into
    the loop's message history (so the frozen prefix stays byte-stable). Off = intact.
    """
    with tempfile.TemporaryDirectory() as td:
        engine, lt = _writeback_engine(td, enabled=True)
        try:
            messages = _long_messages()
            before = len(messages)
            engine._prepare_llm_messages(messages)
            assert len(messages) < before        # write-back shrank the history
            assert messages[0]["role"] == "system"  # cacheable prefix preserved
        finally:
            lt.close()


def test_compression_writeback_off_leaves_history_intact() -> None:
    with tempfile.TemporaryDirectory() as td:
        engine, lt = _writeback_engine(td, enabled=False)
        try:
            messages = _long_messages()
            before = len(messages)
            engine._prepare_llm_messages(messages)
            assert len(messages) == before       # default off: history unchanged
        finally:
            lt.close()


def _adaptive_engine(td):
    from conftest import make_settings
    from leapflow.platform.mock import MockBridge
    settings = make_settings(td)
    rpc = MockBridge()
    llm = StubLLM(["x"])
    wm = WorkingMemoryProvider(max_tokens=1024)
    lt = SemanticMemoryProvider(source=settings.duckdb_path)
    imm = EpisodicMemoryProvider()
    reg = build_default_registry(rpc, llm, wm, lt)
    engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, _FixedClassifier("complex"))
    return engine, lt, settings


def test_progress_gated_budget_extension_decision() -> None:
    """P0: extend the budget only when productively unfinished (open ledger work,
    not stalled, within resources); stop when stalled or the ledger is complete.
    """
    from leapflow.engine.agent_loop import AgentLoopFrame

    with tempfile.TemporaryDirectory() as td:
        engine, lt, settings = _adaptive_engine(td)
        try:
            frame = AgentLoopFrame(user_text="x")
            # Open question + not stalled + resources ok -> extend.
            engine._research_ledger.note("open_question", "How does X cache?")
            frame.stalled_rounds = 0
            assert engine._should_extend_budget(frame) is True
            # Stalled beyond threshold -> stop (let it converge).
            frame.stalled_rounds = settings.agent_stall_rounds
            assert engine._should_extend_budget(frame) is False
            # Ledger reports completion (0 open questions) -> stop.
            frame.stalled_rounds = 0
            engine._research_ledger.note("resolved", "How does X cache")
            assert engine._research_ledger.open_question_count == 0
            assert engine._should_extend_budget(frame) is False
        finally:
            lt.close()


def test_update_progress_and_stall_tracks_progress() -> None:
    """P0: stall counter increments when the progress marker is unchanged and
    resets when new evidence/ledger progress appears."""
    from leapflow.engine.agent_loop import AgentLoopFrame

    with tempfile.TemporaryDirectory() as td:
        engine, lt, _ = _adaptive_engine(td)
        try:
            frame = AgentLoopFrame(user_text="x")
            engine._last_context_snapshot = {"context_governance": {"evidence_count": 1, "sources_seen": 1}}
            engine._update_progress_and_stall(frame)     # first marker recorded
            assert frame.stalled_rounds == 0
            engine._update_progress_and_stall(frame)     # unchanged -> stall++
            assert frame.stalled_rounds == 1
            engine._update_progress_and_stall(frame)
            assert frame.stalled_rounds == 2
            # New evidence surfaces -> progress -> reset.
            engine._last_context_snapshot = {"context_governance": {"evidence_count": 9, "sources_seen": 3}}
            engine._update_progress_and_stall(frame)
            assert frame.stalled_rounds == 0
        finally:
            lt.close()


def test_stagnation_guard_ignores_injected_context() -> None:
    """Guardrail fix: StagnationGuard counts only genuine tool results, not
    injected user context (ledger/live signals/memory), so a context-heavy long
    task with successful tools is not falsely flagged as stagnating."""
    from leapflow.engine.tool_guardrails import StagnationGuard

    guard = StagnationGuard(window=5, min_success_rate=0.5)
    history: list = []
    for i in range(5):
        history.append({"role": "tool", "content": '{"ok": true, "n": %d}' % i})
        history.append({"role": "user", "content": "MEMORY_CONTEXT: prior experience ..."})
        history.append({"role": "user", "content": "## Research Ledger (task state) ..."})

    assert guard.check(history).violated is False   # noise excluded; all tools succeeded


def test_stagnation_guard_flags_genuine_tool_failures() -> None:
    from leapflow.engine.tool_guardrails import StagnationGuard

    guard = StagnationGuard(window=5, min_success_rate=0.5)
    history = [{"role": "tool", "content": '{"ok": false, "error": "boom"}'} for _ in range(6)]
    assert guard.check(history).violated is True    # genuine failures -> flagged


def test_guardrail_halt_suppressed_while_progressing() -> None:
    """Guardrail is progress-aware: a halt/nudge is suppressed while the task is
    advancing (stall counter 0) and only escalates once the task is stalled."""
    from leapflow.engine.agent_loop import AgentLoopFrame
    from leapflow.engine.tool_guardrails import GuardrailViolation

    class _HaltGuard:
        def check(self, history):
            return GuardrailViolation(violated=True, reason="loop", severity="halt", suggestion="stop")

        def reset(self):
            pass

    with tempfile.TemporaryDirectory() as td:
        engine, lt, _ = _adaptive_engine(td)
        try:
            engine._guardrail = _HaltGuard()
            frame = AgentLoopFrame(user_text="x")
            engine._active_frame = frame
            msgs = [{"role": "user", "content": "x"}]

            frame.stalled_rounds = 0
            assert engine._check_guardrail(msgs) is None      # progressing -> halt suppressed
            frame.stalled_rounds = 2
            assert engine._check_guardrail(msgs) == "halt"     # stalled -> halt fires
        finally:
            lt.close()


def _with_coordinator(engine):
    from leapflow.engine.recovery_budget import RecoveryBudget
    from leapflow.engine.recovery_coordinator import RecoveryCoordinator
    from leapflow.engine.recovery_strategies import default_strategies
    engine._recovery_coordinator = RecoveryCoordinator(
        strategies=default_strategies(), budget=RecoveryBudget(total_recovery_actions=12),
    )
    return engine._recovery_coordinator


def test_recoverable_tool_failures_feed_back_never_break() -> None:
    """TF: many consecutive recoverable tool failures are fed back for autonomous
    diagnosis (no halt) — no blanket count-based break."""
    with tempfile.TemporaryDirectory() as td:
        engine, lt, _ = _adaptive_engine(td)
        try:
            _with_coordinator(engine)
            failed = [("shell_run", {"ok": False, "error": "boom", "retryable": True})] * 10
            assert engine._evaluate_tool_failures(failed, turn_id=1) is None   # never halts
        finally:
            lt.close()


def test_recoverable_tool_failures_do_not_spend_recovery_budget() -> None:
    """TF: tool failures are agent-level observations, not infrastructure recovery,
    so feeding them back must not deplete the shared system recovery budget."""
    with tempfile.TemporaryDirectory() as td:
        engine, lt, _ = _adaptive_engine(td)
        try:
            coord = _with_coordinator(engine)
            before = coord.budget.remaining()
            engine._evaluate_tool_failures(
                [("shell_run", {"ok": False, "error": "x", "retryable": True})] * 8, turn_id=1,
            )
            assert coord.budget.remaining() == before   # zero-cost feedback
        finally:
            lt.close()


def test_non_recoverable_tool_failure_halts_via_coordinator() -> None:
    """TF: only a genuinely non-recoverable failure (e.g. permission denied) halts
    the turn — routed through the single recovery decision point."""
    with tempfile.TemporaryDirectory() as td:
        engine, lt, _ = _adaptive_engine(td)
        try:
            _with_coordinator(engine)
            perm = [("gateway_send", {
                "ok": False,
                "failure_class": "authorization",
                "error": "permission denied",
                "execution_policy": "external_side_effect",
            })]
            reason = engine._evaluate_tool_failures(perm, turn_id=1)
            assert reason is not None and reason != ""
        finally:
            lt.close()


def test_turn_recovery_rearm_after_progress_content_only() -> None:
    """P1-A: progress re-arms content-level one-shots (so a long task can recover
    again) but keeps storm-prone infrastructure one-shots strict for the turn."""
    from leapflow.engine.turn_recovery import TurnRecoveryState

    rec = TurnRecoveryState()
    assert rec.try_length_continuation() is True    # content one-shot fires
    assert rec.try_length_continuation() is False   # exhausted within the streak
    assert rec.try_provider_failover() is True      # infra one-shot fires

    assert rec.rearm_after_progress() is True        # progress re-arms content guards
    assert rec.try_length_continuation() is True     # available again
    assert rec.try_compress() is True                # content guard re-armed too
    assert rec.try_provider_failover() is False      # infra stays strict

    # No-op (returns False) when no content guard had been used.
    assert TurnRecoveryState().rearm_after_progress() is False


def test_should_stop_after_tool_result_is_policy_driven() -> None:
    """P1-B: the side-effect batch-stop gate is driven by the declared
    execution_policy, not a hardcoded tool-name list."""
    from leapflow.engine.engine import _should_stop_after_tool_result

    # Any mutating/side-effect policy failure stops the batch…
    assert _should_stop_after_tool_result("any_tool", {"ok": False, "execution_policy": "external_side_effect"}) is True
    assert _should_stop_after_tool_result("any_tool", {"ok": False, "execution_policy": "mutating_once"}) is True
    # …while a read-only failure does not — even for a tool that used to be in the
    # hardcoded side-effect name list (proves it is now policy-driven).
    assert _should_stop_after_tool_result("shell_run", {"ok": False, "execution_policy": "read_only"}) is False
    # A non-failure never stops the batch.
    assert _should_stop_after_tool_result("gateway_send", {"ok": True}) is False


@pytest.mark.asyncio
async def test_exact_canonical_tool_names_execute_without_guessing() -> None:
    """Only exact canonical tool names (plus case/separator formatting) execute."""
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM([])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        captured: dict[str, object] = {}

        async def file_list_handler(args):
            captured["args"] = args
            return {"ok": True, "path": args.get("path", ""), "entries": []}

        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            engine._tool_bridge = None

            result = await engine._execute_general_tool(
                {"name": "file_list", "arguments": {"path": "."}},
                {"file_list": file_list_handler},
            )
            metadata = _tool_args_metadata(
                "file_list",
                {"path": "."},
                original_tool_name="File-List",
            )

            assert result["ok"] is True
            assert captured["args"] == {"path": "."}
            # Case/separator formatting of the *same* canonical name still resolves.
            assert _normalize_tool_name("File_List") == "file_list"
            assert _normalize_tool_name("file-list") == "file_list"
            # Known LLM drift patterns resolve via static alias table.
            assert _normalize_tool_name("list_directory") == "file_list"
            assert _normalize_tool_name("execute_command") == "shell_run"
            assert _normalize_tool_name("run_terminal") == "shell_run"
            alias_resolution = _resolve_tool_name("list_directory", {"path": "."})
            assert alias_resolution.normalized_name == "file_list"
            assert alias_resolution.status == "aliased"
            assert alias_resolution.auto_executable is True
            # Names NOT in alias table remain unknown.
            directory_resolution = _resolve_tool_name("directory_scan", {"path": "."})
            risky_resolution = _resolve_tool_name("please_do", {"command": "ls -la"})
            assert directory_resolution.normalized_name is None
            assert directory_resolution.status == "unknown"
            assert directory_resolution.auto_executable is False
            assert risky_resolution.normalized_name is None
            assert risky_resolution.status == "unknown"
            assert risky_resolution.auto_executable is False
            assert metadata["original_tool_name"] == "File-List"
            assert metadata["normalized_tool_name"] == "file_list"
            assert metadata["resolved_from"] == "File-List"
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_tool_execution_ledger_skips_duplicate_external_tool() -> None:
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM([])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        calls: list[dict[str, object]] = []

        async def shell_handler(args):
            calls.append(dict(args))
            return {"ok": True, "stdout": "pushed"}

        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            engine._tool_bridge = None
            engine._current_session_id = "session-1"
            engine._session_turn_count = 1
            engine._begin_turn_context("push once")
            call = {"name": "shell_run", "arguments": {"command": "git push"}}

            first = await engine._execute_tool_with_ledger(call, {"shell_run": shell_handler}, tool_call_id="a")
            second = await engine._execute_tool_with_ledger(call, {"shell_run": shell_handler}, tool_call_id="b")

            assert len(calls) == 1
            assert first["ok"] is True
            assert first["execution_policy"] == "external_side_effect"
            assert second["already_executed"] is True
            assert second["original_result"]["stdout"] == "pushed"
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_tool_execution_ledger_waits_for_inflight_duplicate_external_tool() -> None:
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM([])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        started = asyncio.Event()
        release = asyncio.Event()
        calls: list[dict[str, object]] = []

        async def shell_handler(args):
            calls.append(dict(args))
            started.set()
            await release.wait()
            return {"ok": True, "stdout": "pushed"}

        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            engine._tool_bridge = None
            engine._current_session_id = "session-1"
            engine._session_turn_count = 1
            engine._begin_turn_context("push once")
            call = {"name": "shell_run", "arguments": {"command": "git push"}}

            first_task = asyncio.create_task(
                engine._execute_tool_with_ledger(call, {"shell_run": shell_handler}, tool_call_id="a")
            )
            await started.wait()
            second_task = asyncio.create_task(
                engine._execute_tool_with_ledger(call, {"shell_run": shell_handler}, tool_call_id="b")
            )
            await asyncio.sleep(0)

            assert len(calls) == 1
            assert not second_task.done()

            release.set()
            first, second = await asyncio.gather(first_task, second_task)

            assert first["ok"] is True
            assert second["already_executed"] is True
            assert second["ok"] is True
            assert second["original_result"]["stdout"] == "pushed"
            assert len(calls) == 1
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_tool_execution_ledger_allows_repeated_read_only_tool() -> None:
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM([])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        calls: list[dict[str, object]] = []

        async def file_list_handler(args):
            calls.append(dict(args))
            return {"ok": True, "entries": []}

        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            engine._tool_bridge = None
            engine._current_session_id = "session-1"
            engine._session_turn_count = 1
            engine._begin_turn_context("list twice")
            call = {"name": "file_list", "arguments": {"path": "."}}

            first = await engine._execute_tool_with_ledger(call, {"file_list": file_list_handler}, tool_call_id="a")
            second = await engine._execute_tool_with_ledger(call, {"file_list": file_list_handler}, tool_call_id="b")

            assert len(calls) == 2
            assert first["execution_policy"] == "read_only"
            assert second["execution_policy"] == "read_only"
            assert "already_executed" not in second
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_side_effect_failure_stops_remaining_native_tool_batch() -> None:
    from leapflow.engine.execution_trace import ExecutionTrace
    from leapflow.llm.base import ToolCallInfo
    from leapflow.platform.mock import MockBridge

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        rpc = MockBridge()
        llm = StubLLM([])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        calls: list[str] = []

        async def execute_tool(tool_call, _handlers):
            calls.append(str(tool_call.get("arguments", {}).get("command")))
            return {"ok": False, "returncode": 1, "stderr": "cd: no such file or directory"}

        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            engine._tool_bridge = None
            engine._execute_general_tool = AsyncMock(side_effect=execute_tool)  # type: ignore[method-assign]
            engine._current_session_id = "session-1"
            engine._session_turn_count = 1
            engine._begin_turn_context("run git commands")
            messages: list[dict[str, object]] = []

            results = await engine._execute_tools_concurrent(
                [
                    ToolCallInfo(id="tc1", name="shell_run", arguments={"command": "cd missing"}),
                    ToolCallInfo(id="tc2", name="shell_run", arguments={"command": "git status"}),
                ],
                {"shell_run": execute_tool},
                trace=ExecutionTrace(),
                messages=messages,
            )

            assert calls == ["cd missing"]
            assert len(results) == 2
            assert results[0]["result"]["ok"] is False
            assert results[1]["result"]["execution_skipped"] is True
            assert results[1]["result"]["counts_as_failure"] is False
            assert AgentEngine._count_consecutive_tool_failures(messages) == 1
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_unknown_tool_returns_structured_retry_feedback() -> None:
    """Unknown tools should produce structured feedback instead of a bare string."""
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM([])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            result = await engine._execute_general_tool(
                {"name": "missing_magic_tool", "arguments": {"foo": "bar"}},
                {},
            )

            assert result["ok"] is False
            assert result["error_type"] == "unknown_tool"
            assert result["original_tool_name"] == "missing_magic_tool"
            assert result["retryable"] is True
            assert "available_tools" in result
            assert "suggestions" in result
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_unknown_tool_triggers_single_self_healing_retry() -> None:
    """The loop should give the LLM one structured chance to retry an unknown tool."""
    class CaptureLLM(StubLLM):
        def __init__(self) -> None:
            super().__init__([
                '<tool_call>{"name": "missing_magic_tool", "arguments": {"foo": "bar"}}</tool_call>',
                "recovered answer",
            ])
            self.seen_messages: list[list[dict[str, object]]] = []

        async def achat(self, messages, *, stream=True, enable_thinking=False, **kwargs):
            self.seen_messages.append(list(messages))
            return await super().achat(messages, stream=stream, enable_thinking=enable_thinking, **kwargs)

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = CaptureLLM()
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            out = await engine.run("Use a missing tool then recover")

            assert out == "recovered answer"
            assert llm.call_count == 2
            second_call_messages = "\n".join(str(message.get("content", "")) for message in llm.seen_messages[1])
            assert "unavailable tool name" in second_call_messages
            assert "missing_magic_tool" in second_call_messages
            assert "Available tools include" in second_call_messages
        finally:
            lt.close()

@pytest.mark.asyncio
async def test_app_connector_context_is_injected_without_extra_llm_call() -> None:
    class CaptureLLM(StubLLM):
        def __init__(self) -> None:
            super().__init__(["Use platform_connect for supported app onboarding."])
            self.seen_messages: list[list[dict[str, object]]] = []

        async def achat(self, messages, *, stream=True, enable_thinking=False, **kwargs):
            self.seen_messages.append(list(messages))
            return await super().achat(messages, stream=stream, enable_thinking=enable_thinking, **kwargs)

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.gateway.server import GatewayServer
        from leapflow.platform.mock import MockBridge
        from leapflow.tools.gateway_tool import set_gateway_approval_gate, set_gateway_server

        rpc = MockBridge()
        llm = CaptureLLM()
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        server = GatewayServer(settings.profile_dir)
        server.discover_manifests()
        set_gateway_server(server)
        set_gateway_approval_gate(None)
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            out = await engine.run("接入飞书")

            system_prompt = str(llm.seen_messages[0][0].get("content", ""))
            assert out == "Use platform_connect for supported app onboarding."
            assert "App Connector Capability Index" in system_prompt
            assert "`feishu`" in system_prompt
            assert "`telegram`" in system_prompt
            assert "platform_connect" in system_prompt
            assert llm.call_count == 1
        finally:
            await server.stop()
            set_gateway_approval_gate(None)
            set_gateway_server(None)
            lt.close()


@pytest.mark.asyncio
async def test_app_connector_llm_tool_call_uses_same_unified_loop() -> None:
    tool_reply = '<tool_call>{"name": "platform_connect", "arguments": {"action": "guide", "platform": "telegram"}}</tool_call>'
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.gateway.server import GatewayServer
        from leapflow.platform.mock import MockBridge
        from leapflow.tools.gateway_tool import set_gateway_approval_gate, set_gateway_server

        rpc = MockBridge()
        llm = StubLLM([tool_reply, "Telegram guide ready"])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        server = GatewayServer(settings.profile_dir)
        server.discover_manifests()
        set_gateway_server(server)
        set_gateway_approval_gate(None)
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            events = [event async for event in engine.run_stream("配置 Telegram")]

            tool_events = [event for event in events if event.type in {"tool_start", "tool_complete"}]
            assert [event.content for event in tool_events] == ["platform_connect", "platform_connect"]
            assert tool_events[1].metadata["ok"] is True
            assert events[-1].type == "final"
            assert "Telegram guide ready" in events[-1].content
            assert llm.call_count == 2
        finally:
            await server.stop()
            set_gateway_approval_gate(None)
            set_gateway_server(None)
            lt.close()


@pytest.mark.asyncio
async def test_app_connector_empty_final_uses_onboarding_recovery_state() -> None:
    tool_reply = (
        '<tool_call>{"name": "platform_connect", "arguments": '
        '{"action": "guide", "platform": "feishu", '
        '"options": {"binary": "definitely-missing-cli-for-onboarding-test"}}}</tool_call>'
    )
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.gateway.server import GatewayServer
        from leapflow.platform.mock import MockBridge
        from leapflow.tools.gateway_tool import set_gateway_approval_gate, set_gateway_server

        rpc = MockBridge()
        llm = StubLLM([tool_reply, ""])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        server = GatewayServer(settings.profile_dir)
        server.discover_manifests()
        set_gateway_server(server)
        set_gateway_approval_gate(None)
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            final = await engine.run("继续接入")
        finally:
            await server.stop()
            set_gateway_approval_gate(None)
            set_gateway_server(None)
            lt.close()

    assert llm.call_count == 2
    assert "App onboarding is paused" in final
    assert "cli_missing" in final
    assert "definitely-missing-cli-for-onboarding-test" in final


@pytest.mark.asyncio
async def test_aliased_tool_in_stream_resolves_and_executes() -> None:
    """Text-mode tool calls with a known drifted name resolve via alias and execute normally."""
    tool_reply = '<tool_call>{"name": "list_directory", "arguments": {"path": "."}}</tool_call>'
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM([tool_reply, "directory checked"])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            events = [event async for event in engine.run_stream("List current directory")]

            tool_events = [event for event in events if event.type in {"tool_start", "tool_complete"}]
            assert tool_events[0].metadata["original_tool_name"] == "list_directory"
            assert tool_events[0].metadata["tool_resolution_status"] == "aliased"
            assert tool_events[0].metadata["normalized_tool_name"] == "file_list"
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_unknown_tool_in_stream_triggers_structured_retry() -> None:
    """Text-mode tool calls with a truly unknown name surface a structured unknown with suggestions."""
    tool_reply = '<tool_call>{"name": "directory_scan", "arguments": {"path": "."}}</tool_call>'
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM([tool_reply, "directory checked"])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            events = [event async for event in engine.run_stream("List current directory")]

            tool_events = [event for event in events if event.type in {"tool_start", "tool_complete"}]
            assert [event.content for event in tool_events] == ["directory_scan", "directory_scan"]
            assert tool_events[0].metadata["original_tool_name"] == "directory_scan"
            assert tool_events[0].metadata["tool_resolution_status"] == "unknown"
            assert tool_events[1].metadata["ok"] is False
            assert tool_events[1].metadata["error_type"] == "unknown_tool"
            assert "resolved_from" not in tool_events[1].metadata
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_engine_remembers_context() -> None:
    """Engine run stores user query and assistant reply in working memory."""
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        # Unified loop: plain text response is returned as final answer
        llm = StubLLM(
            [
                "Hello from the assistant.",
            ]
        )
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            query = "What can you do?"
            out = await engine.run(query)
            assert "Hello from the assistant." in out

            messages = wm.as_chat_messages()
            roles = [m["role"] for m in messages]
            contents = [str(m["content"]) for m in messages]
            assert "user" in roles
            assert "assistant" in roles
            assert query in contents
            assert any("Hello from the assistant." in c for c in contents)
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_streaming_engine_estimates_context_tokens_without_provider_usage() -> None:
    """Streaming providers often omit usage; status still needs prompt utilization."""
    from leapflow.llm.base import LLMChatResponse, LLMProvider
    from leapflow.platform.mock import MockBridge

    class StreamingOnlyLLM(LLMProvider):
        async def achat(self, messages, *, stream=True, enable_thinking=False, on_chunk=None, **kwargs):
            return LLMChatResponse(content="fallback")

        async def achat_stream(self, messages, *, enable_thinking=False, **kwargs):
            yield "streamed answer"

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        settings = settings.__class__(
            **{
                **settings.__dict__,
                "stream_output": True,
                "native_tool_calling_enabled": False,
            }
        )
        rpc = MockBridge()
        llm = StreamingOnlyLLM()
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            events = [event async for event in engine.run_stream("Summarize a long conversation")]

            assert any(event.type == "final" for event in events)
            assert engine.context_token_count > 0
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_progressive_disclosure_light_query_omits_tools_and_thinking() -> None:
    """Plain chat should stay on the light path even when thinking is requested."""
    from leapflow.llm.base import LLMChatResponse, LLMProvider
    from leapflow.platform.mock import MockBridge

    class CaptureLLM(LLMProvider):
        def __init__(self) -> None:
            self.messages: list[dict] = []
            self.kwargs: dict = {}
            self.enable_thinking = True
            self.call_count = 0

        async def achat(self, messages, *, stream=True, enable_thinking=False, on_chunk=None, **kwargs):
            self.call_count += 1
            self.messages = list(messages)
            self.kwargs = dict(kwargs)
            self.enable_thinking = enable_thinking
            return LLMChatResponse(content="I am LeapFlow.")

        async def achat_stream(self, messages, *, enable_thinking=False, **kwargs):
            if False:
                yield ""

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        settings = settings.__class__(
            **{
                **settings.__dict__,
                "native_tool_calling_enabled": True,
            }
        )
        rpc = MockBridge()
        llm = CaptureLLM()
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("chat")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            out = await engine.run("hello", enable_thinking=True)

            assert out == "I am LeapFlow."
            assert llm.call_count == 1
            # CORE disclosure keeps a static low-risk tool whitelist always callable
            # (never an empty/contradictory tool contract), but excludes heavy/mutating tools.
            core_names = {
                tool.get("function", {}).get("name", "")
                for tool in llm.kwargs.get("tools", [])
            }
            assert "shell_run" not in core_names
            assert "hub_push" not in core_names
            assert llm.enable_thinking is False
            system_prompt = str(llm.messages[0].get("content", ""))
            assert "## Presentation Style" in system_prompt
            assert "Avoid redundant tool calls" in system_prompt
            assert "same tool with the same arguments" in system_prompt
            assert "existing tool result already answers" in system_prompt
            assert "No leaked tool protocol" in system_prompt
            assert "Theme-safe colors" in system_prompt
            assert "## Task Contract" in system_prompt
            assert "Original user request: hello" in system_prompt
            assert "Workspace root:" in system_prompt
            assert "never infer `.` as the project root" in system_prompt
            assert "LeapFlow workspace config is optional" in system_prompt
            assert "~/.leapflow/config/user.yaml" in system_prompt
            assert "~/.leapflow/profiles/<profile>/config/*.yaml" in system_prompt
            assert "<workspace>/.leapflow/config.yaml" in system_prompt
            snapshot = engine.context_budget_snapshot
            assert snapshot["disclosure_level"] == "core"
            assert snapshot["disclosure"]["native_tools"] is True
        finally:
            lt.close()


def test_task_contract_replaces_stale_contract_block() -> None:
    """Compression recovery should keep exactly one current task contract."""
    from leapflow.platform.mock import MockBridge

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        rpc = MockBridge()
        llm = StubLLM(["ok"])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("chat")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            engine._session_turn_count = 1
            engine._begin_turn_context("first request")
            stale_contract = engine._task_contract_block()
            engine._session_turn_count = 2
            engine._begin_turn_context("second request")

            prepared = engine._ensure_task_contract_message([
                {"role": "system", "content": f"base system\n\n{stale_contract}\n"},
                {"role": "system", "content": stale_contract},
                {"role": "user", "content": "second request"},
            ])
            system_text = "\n".join(
                str(message.get("content", ""))
                for message in prepared
                if message.get("role") == "system"
            )

            assert system_text.count("## Task Contract") == 1
            assert "Original user request: second request" in system_text
            assert "Original user request: first request" not in system_text
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_progressive_disclosure_file_query_selects_file_schemas() -> None:
    """File-oriented requests should disclose file schemas without the full catalog."""
    from leapflow.llm.base import LLMChatResponse, LLMProvider
    from leapflow.platform.mock import MockBridge

    class CaptureLLM(LLMProvider):
        def __init__(self) -> None:
            self.kwargs: dict = {}

        async def achat(self, messages, *, stream=True, enable_thinking=False, on_chunk=None, **kwargs):
            self.kwargs = dict(kwargs)
            return LLMChatResponse(content="Done")

        async def achat_stream(self, messages, *, enable_thinking=False, **kwargs):
            if False:
                yield ""

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        settings = settings.__class__(
            **{
                **settings.__dict__,
                "native_tool_calling_enabled": True,
            }
        )
        rpc = MockBridge()
        llm = CaptureLLM()
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("file")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            await engine.run("Read src/leapflow/engine/engine.py")

            tools = llm.kwargs.get("tools", [])
            names = {tool.get("function", {}).get("name", "") for tool in tools}
            assert "file_read" in names
            assert "file_list" in names
            assert "shell_run" not in names
            # file_read/file_list are part of the static Tier 0.5 core whitelist, so a
            # plain file-oriented turn (no prior-turn tool-category continuity, no
            # slash command / escalation signal) stays at the CORE floor level.
            assert engine.context_budget_snapshot["disclosure_level"] == "core"
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_progressive_disclosure_expands_write_category_after_prior_turn_tool_use() -> None:
    """Tier 1 continuity: a native tool_call executed in turn N structurally
    opens its capability category for turn N+1 — a purely structural signal,
    never a re-reading of user text. Regression guard for the dedicated
    ``AgentEngine._last_turn_tool_categories`` state: working memory only
    stores a synthetic "[Called: ...]" summary with no structured tool_calls,
    so continuity must not be derived from ``wm.as_chat_messages()``.
    """
    from leapflow.llm.base import LLMChatResponse, LLMProvider, ToolCallInfo
    from leapflow.platform.mock import MockBridge

    class CaptureLLM(LLMProvider):
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def achat(self, messages, *, stream=True, enable_thinking=False, on_chunk=None, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return LLMChatResponse(
                    content="",
                    tool_calls=[
                        ToolCallInfo(
                            id="tc1",
                            name="text_replace",
                            arguments={"text": "a", "old": "a", "new": "b"},
                        )
                    ],
                )
            return LLMChatResponse(content="Turn done")

        async def achat_stream(self, messages, *, enable_thinking=False, **kwargs):
            if False:
                yield ""

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        settings = settings.__class__(
            **{**settings.__dict__, "native_tool_calling_enabled": True}
        )
        rpc = MockBridge()
        llm = CaptureLLM()
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("chat")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            await engine.run("Replace a with b in some text")
            first_turn_names = {
                t.get("function", {}).get("name") for t in llm.calls[0].get("tools", [])
            }
            # text_replace is not in the static core whitelist and nothing opened
            # its category yet, so the model's own tools schema does not include it
            # (the mock LLM here bypasses that constraint only to exercise the
            # engine's post-execution bookkeeping, not provider-side enforcement).
            assert "text_replace" not in first_turn_names

            await engine.run("hi again")
            second_turn_names = {
                t.get("function", {}).get("name") for t in llm.calls[-1].get("tools", [])
            }
            assert "text_replace" in second_turn_names
            assert "file_write" in second_turn_names  # same "write" category opened
            assert "memory_add" in second_turn_names
            assert engine.context_budget_snapshot["disclosure_level"] == "expanded"
            assert "write" in engine.context_budget_snapshot["disclosure"]["expanded_categories"]

            # A third turn with no tool use must not carry the category forever —
            # continuity is exactly one turn, not a sticky escalation.
            await engine.run("just chatting, no tools needed")
            third_turn_names = {
                t.get("function", {}).get("name") for t in llm.calls[-1].get("tools", [])
            }
            assert "text_replace" not in third_turn_names
            assert engine.context_budget_snapshot["disclosure_level"] == "core"
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_immediate_memory_integration() -> None:
    """EpisodicMemoryProvider fragments surface in memory_recent responses."""
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM(["You recently modified /tmp/README.md"])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        imm.ingest("event.fs_change", "File modified: /tmp/README.md", path="/tmp/README.md")
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("memory_recent")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            out = await engine.run("What did I change just now?")
            assert "README" in out
        finally:
            lt.close()


# ═══════════════════════════════════════════════════════════════════
# TaskGraph scenarios
# ═══════════════════════════════════════════════════════════════════


def test_task_graph_linear_chain() -> None:
    """A → B → C: topological order and ready_nodes advance step by step."""
    g = TaskGraph(goal="linear")
    g.add_node(_node("a"))
    g.add_node(_node("b", depends_on=["a"]))
    g.add_node(_node("c", depends_on=["b"]))

    order = g.topological_order()
    assert order.index("a") < order.index("b") < order.index("c")

    ready = g.ready_nodes()
    assert [n.id for n in ready] == ["a"]

    g.mark_completed("a", "a-out")
    ready = g.ready_nodes()
    assert [n.id for n in ready] == ["b"]

    g.mark_completed("b", "b-out")
    ready = g.ready_nodes()
    assert [n.id for n in ready] == ["c"]

    g.mark_completed("c", "c-out")
    assert g.ready_nodes() == []
    assert g.is_complete


def test_task_graph_diamond_dependency() -> None:
    """A → {B, C} → D: B and C become ready in parallel after A completes."""
    g = TaskGraph(goal="diamond")
    g.add_node(_node("a"))
    g.add_node(_node("b", depends_on=["a"]))
    g.add_node(_node("c", depends_on=["a"]))
    g.add_node(_node("d", depends_on=["b", "c"]))

    assert [n.id for n in g.ready_nodes()] == ["a"]

    g.mark_completed("a", "root")
    ready_ids = {n.id for n in g.ready_nodes()}
    assert ready_ids == {"b", "c"}

    g.mark_completed("b", "left")
    assert [n.id for n in g.ready_nodes()] == ["c"]

    g.mark_completed("c", "right")
    assert [n.id for n in g.ready_nodes()] == ["d"]


def test_task_graph_cycle_detection() -> None:
    """A → B → A cycle is rejected by validate() and from_dict()."""
    g = TaskGraph(goal="cyclic")
    g.nodes["a"] = _node("a", depends_on=["b"])
    g.nodes["b"] = _node("b", depends_on=["a"])

    errors = g.validate()
    assert any("cycle" in e.lower() for e in errors)

    with pytest.raises(GraphValidationError):
        TaskGraph.from_dict(
            {
                "goal": "cyclic",
                "nodes": [
                    {"id": "a", "action": "skill_a", "depends_on": ["b"]},
                    {"id": "b", "action": "skill_b", "depends_on": ["a"]},
                ],
            }
        )


def test_task_graph_param_resolution() -> None:
    """${a.output} and ${graph.goal} substitute upstream results and goal text."""
    g = TaskGraph(goal="Ship release")
    g.add_node(_node("a"))
    g.add_node(
        _node(
            "b",
            depends_on=["a"],
            params={
                "upstream": "${a.output}",
                "goal": "${graph.goal}",
                "nested": "${a.result.name}",
            },
        )
    )
    g.mark_completed("a", {"name": "artifact", "version": "1.0"})

    resolved = g.resolve_params(g.nodes["b"])
    assert resolved["upstream"] == {"name": "artifact", "version": "1.0"}
    assert resolved["goal"] == "Ship release"
    assert resolved["nested"] == "artifact"


def test_task_graph_retry_policy() -> None:
    """Failed nodes can be reset while retries remain; exhausted retries stay failed."""
    g = TaskGraph(goal="retry")
    policy = RetryPolicy(max_retries=2)
    g.add_node(_node("a", retry_policy=policy))

    node = g.nodes["a"]

    g.mark_running("a")
    assert node.attempt_count == 1
    g.mark_failed("a", "transient error")
    assert node.status == TaskStatus.FAILED

    g.reset_node("a")
    assert node.status == TaskStatus.PENDING
    assert node.error is None

    g.mark_running("a")
    g.mark_failed("a", "transient error")
    g.reset_node("a")

    g.mark_running("a")
    g.mark_failed("a", "permanent error")
    assert node.status == TaskStatus.FAILED
    assert node.attempt_count == 3
    assert node.error == "permanent error"


# ═══════════════════════════════════════════════════════════════════
# Idempotency guard and failure recovery tests
# ═══════════════════════════════════════════════════════════════════


def test_platform_action_idempotency_key_deduplicates_identical_calls() -> None:
    """Unified idempotency keys replace the old platform_action fingerprint."""
    from leapflow.engine.tool_execution import build_idempotency_key

    args = {"platform": "feishu", "action": "im.send_message", "payload": {"chat_id": "oc_1", "text": "hi"}}
    key1 = build_idempotency_key(
        session_id="session-1",
        turn_id="turn-1",
        tool_name="platform_action",
        arguments=args,
        policy="external_side_effect",
    )
    key2 = build_idempotency_key(
        session_id="session-1",
        turn_id="turn-2",
        tool_name="platform_action",
        arguments=args,
        policy="external_side_effect",
    )
    different_payload = build_idempotency_key(
        session_id="session-1",
        turn_id="turn-1",
        tool_name="platform_action",
        arguments={"platform": "feishu", "action": "im.send_message", "payload": {"chat_id": "oc_2", "text": "hi"}},
        policy="external_side_effect",
    )
    other_tool = build_idempotency_key(
        session_id="session-1",
        turn_id="turn-1",
        tool_name="file_list",
        arguments={"path": "."},
        policy="read_only",
    )

    assert key1 == key2, "External side effects deduplicate across turns in the same session"
    assert key1 != different_payload, "Different payload must produce a different idempotency key"
    assert key1 != other_tool, "Tool name and policy participate in the key"


def test_last_tool_failures_recovery_message_from_unknown_action() -> None:
    """_last_tool_failures_recovery_message extracts context from unknown_platform_action results."""
    import json
    from leapflow.engine.engine import _last_tool_failures_recovery_message

    failure_payload = {
        "ok": False,
        "failure_code": "unknown_platform_action",
        "error": "Unknown platform action: feishu.im.chat.list",
        "platform": "feishu",
        "requested_action": "im.chat.list",
        "available_action_names": ["im.send_message", "im.list_chats", "im.search_chats"],
        "recovery_hint": "Use exactly one registered action name from available_action_names.",
        "retryable": True,
    }
    messages = [
        {"role": "tool", "content": json.dumps(failure_payload)},
    ]
    result = _last_tool_failures_recovery_message(messages)

    assert result, "Should produce non-empty recovery message"
    assert "feishu.im.chat.list" in result or "im.chat.list" in result
    assert "im.list_chats" in result
    assert "im.send_message" in result


def test_last_tool_failures_recovery_message_missing_fields() -> None:
    """_last_tool_failures_recovery_message handles Missing required fields errors."""
    import json
    from leapflow.engine.engine import _last_tool_failures_recovery_message

    failure_payload = {
        "ok": False,
        "error": "Missing required fields: text",
    }
    messages = [{"role": "tool", "content": json.dumps(failure_payload)}]
    result = _last_tool_failures_recovery_message(messages)

    assert result
    assert "text" in result


def test_duplicate_suppression_is_not_counted_as_consecutive_tool_failure() -> None:
    """Suppressed duplicate side effects are control signals, not failed executions."""
    import json
    from leapflow.engine.engine import _last_tool_failures_recovery_message

    root_failure = {
        "ok": False,
        "error": "git push rejected",
        "stderr": "non-fast-forward",
        "execution_policy": "external_side_effect",
    }
    duplicate_suppressed = {
        "ok": False,
        "already_executed": True,
        "duplicate_suppressed": True,
        "counts_as_failure": False,
        "error": "An identical side-effect attempt is already recorded. Review the original result before retrying.",
    }
    messages = [
        {"role": "tool", "content": json.dumps(root_failure)},
        {"role": "tool", "content": json.dumps(duplicate_suppressed)},
    ]

    assert AgentEngine._count_consecutive_tool_failures(messages) == 1
    recovery = _last_tool_failures_recovery_message(messages)
    assert "git push rejected" in recovery
    assert "consecutive tool failures" not in recovery
    assert "duplicate execution was not replayed" not in recovery



    import json
    from leapflow.engine.engine import _last_tool_failures_recovery_message

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "sure"},
        {"role": "tool", "content": json.dumps({"ok": True, "data": {}})},
    ]
    assert _last_tool_failures_recovery_message(messages) == ""


@pytest.mark.asyncio
async def test_permission_failure_hard_stops_text_tool_loop() -> None:
    """Authorization failures are terminal business blockers, not retry prompts."""
    tool_reply = '<tool_call>{"name": "platform_action", "arguments": {"platform": "feishu", "action": "im.list_chats", "payload": {}}}</tool_call>'
    failure_payload = {
        "ok": False,
        "platform": "feishu",
        "action": "im.list_chats",
        "capability": "im.chat.read",
        "failure_class": "authorization",
        "failure_code": "access_denied",
        "missing_scopes": ["im:chat:read"],
        "scope_relation": "all_required",
        "scope_source": "authoritative",
        "recoverability": "admin_required",
        "retryable": False,
        "blocks_approval": True,
    }

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM([tool_reply, "SHOULD NOT BE CALLED"])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            engine._execute_general_tool = AsyncMock(return_value=failure_payload)  # type: ignore[method-assign]

            out = await engine.run("列出飞书群聊")

            assert llm.call_count == 1
            engine._execute_general_tool.assert_awaited_once()  # type: ignore[attr-defined]
            assert "Authorization failed" in out
            assert "im:chat:read" in out
            assert "Do NOT retry" in out
            assert "SHOULD NOT BE CALLED" not in out
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_permission_failure_hard_stops_native_tool_loop() -> None:
    """Native tool-calling must also stop immediately on authorization blockers."""
    from leapflow.llm.base import LLMChatResponse, LLMProvider, ToolCallInfo
    from leapflow.platform.mock import MockBridge

    failure_payload = {
        "ok": False,
        "platform": "feishu",
        "action": "im.list_chats",
        "capability": "im.chat.read",
        "failure_class": "authorization",
        "failure_code": "access_denied",
        "missing_scopes": ["im:chat:read"],
        "scope_relation": "all_required",
        "scope_source": "authoritative",
        "recoverability": "admin_required",
        "retryable": False,
        "blocks_approval": True,
    }

    class PermissionLLM(LLMProvider):
        def __init__(self) -> None:
            self.call_count = 0

        async def achat(self, messages, *, stream=True, enable_thinking=False, on_chunk=None, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                return LLMChatResponse(
                    content="",
                    tool_calls=[
                        ToolCallInfo(
                            id="tc1",
                            name="platform_action",
                            arguments={"platform": "feishu", "action": "im.list_chats", "payload": {}},
                        ),
                        ToolCallInfo(
                            id="tc2",
                            name="platform_action",
                            arguments={
                                "platform": "feishu",
                                "action": "im.send_message",
                                "payload": {"chat_id": "chat-1", "text": "should-not-send"},
                            },
                        ),
                    ],
                )
            return LLMChatResponse(content="SHOULD NOT BE CALLED")

        async def achat_stream(self, messages, *, enable_thinking=False, **kwargs):
            if False:
                yield ""

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        settings = settings.__class__(**{**settings.__dict__, "native_tool_calling_enabled": True})
        rpc = MockBridge()
        llm = PermissionLLM()
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            engine._execute_general_tool = AsyncMock(return_value=failure_payload)  # type: ignore[method-assign]

            out = await engine.run("列出飞书群聊")

            assert llm.call_count == 1
            engine._execute_general_tool.assert_awaited_once()  # type: ignore[attr-defined]
            assert "Authorization failed" in out
            assert "im:chat:read" in out
            assert "SHOULD NOT BE CALLED" not in out
        finally:
            lt.close()


def test_permission_recovery_text_quotes_only_listed_scopes() -> None:
    """The deterministic renderer must never invent or expand scope names."""
    from leapflow.engine.engine import _build_permission_recovery_text

    text = _build_permission_recovery_text({
        "platform": "feishu",
        "capability": "im.chat.read",
        "failure_class": "authorization",
        "failure_code": "missing_scope",
        "missing_scopes": ["im:chat:read"],
        "scope_relation": "all_required",
        "recoverability": "admin_required",
        "console_url": "https://open.feishu.cn/app/cli_xxx/auth",
    })

    assert "im:chat:read" in text
    assert "one of" not in text.lower()
    assert "https://open.feishu.cn/app/cli_xxx/auth" in text
    assert "Do NOT retry" in text


def test_permission_recovery_text_uses_one_of_only_when_declared() -> None:
    """"one of" phrasing only appears when scope_relation explicitly says so."""
    from leapflow.engine.engine import _build_permission_recovery_text

    text = _build_permission_recovery_text({
        "platform": "feishu",
        "capability": "im.message.send",
        "failure_class": "authorization",
        "failure_code": "missing_scope",
        "required_scopes": ["im:message.send_as_user", "im:message:send_as_bot"],
        "scope_relation": "one_of",
        "recoverability": "admin_required",
    })

    assert "ANY ONE" in text
    assert "im:message.send_as_user" in text
    assert "im:message:send_as_bot" in text


def test_permission_override_message_replaces_free_text_after_unresolved_failure() -> None:
    """An unresolved permission failure as the turn's last tool signal must
    override any free-text LLM answer, preventing scope hallucination."""
    import json
    from leapflow.engine.engine import _permission_override_message

    failure_payload = {
        "ok": False,
        "platform": "feishu",
        "capability": "im.chat.read",
        "failure_class": "authorization",
        "failure_code": "missing_scope",
        "missing_scopes": ["im:chat:read"],
        "scope_relation": "all_required",
        "console_url": "https://open.feishu.cn/app/cli_xxx/auth",
    }
    messages = [
        {"role": "user", "content": "list my groups"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "platform_action", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": json.dumps(failure_payload)},
    ]

    override = _permission_override_message(messages)

    assert override
    assert "im:chat:read" in override
    assert "im:chat.group_info" not in override


def test_permission_override_message_empty_after_successful_followup() -> None:
    """No override once a later tool call in the same turn succeeded."""
    import json
    from leapflow.engine.engine import _permission_override_message

    messages = [
        {"role": "user", "content": "list my groups"},
        {"role": "tool", "content": json.dumps({"ok": False, "failure_class": "authorization", "failure_code": "missing_scope"})},
        {"role": "tool", "content": json.dumps({"ok": True, "data": {}})},
    ]

    assert _permission_override_message(messages) == ""


def test_record_tool_call_categories_caches_capability_manifests(monkeypatch) -> None:
    """Capability manifests are cached instead of rebuilt on every tool-call round."""
    from types import SimpleNamespace

    import leapflow.engine.engine as engine_module

    calls = 0

    def fake_build_capability_manifests(tool_definitions):
        nonlocal calls
        calls += 1
        return [SimpleNamespace(name="text_replace", category="write")]

    monkeypatch.setattr(engine_module, "build_capability_manifests", fake_build_capability_manifests)
    engine = object.__new__(AgentEngine)
    engine._last_turn_tool_categories = frozenset()
    engine._manifests_by_name = None

    engine._record_tool_call_categories([SimpleNamespace(name="text_replace")])
    engine._record_tool_call_categories([SimpleNamespace(name="text_replace")])

    assert calls == 1
    assert engine._last_turn_tool_categories == frozenset({"write"})
