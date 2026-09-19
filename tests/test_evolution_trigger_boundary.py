# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regressions for the evolution trigger and the segments it makes observable.

Three defects measured against a live daemon, each of which made the board report an
absence that was really a skipped step:

* the governance sweep was nested inside the trajectory branch, so an empty
  trajectory skipped it entirely — producing exactly the ambiguity the sweep's own
  comment said its no-op traces existed to remove;
* a proposal queue full of never-advanced records was reported ``wired``, the healthy
  class, beside a genuinely healthy segment;
* the learning boundary was reachable only from context cleanup, which in daemon mode
  means process shutdown.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from leapflow.monitor.evolution_producer import (
    NO_EVIDENCE,
    UNVERIFIABLE,
    WIRED,
    EvolutionProducer,
)

_CONTEXT_PY = Path(__file__).parents[1] / "src" / "leapflow" / "cli" / "context.py"


# ════════════════════════════════════════════════════════════════
# The sweep runs on a quiet boundary too
# ════════════════════════════════════════════════════════════════


def _learning_boundary_body() -> ast.AST:
    """Return the AST of the durable semantic session boundary."""
    tree = ast.parse(_CONTEXT_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_learning_boundary":
            return node
    raise AssertionError("run_learning_boundary not found")


def test_the_governance_sweep_is_not_nested_in_the_teacher_job_branch() -> None:
    """A boundary with no teacher job must still leave governance evidence."""
    boundary = _learning_boundary_body()
    sweep_calls = [
        node
        for node in ast.walk(boundary)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_run_coevolution_sweep"
    ]
    assert len(sweep_calls) == 1, "the sweep must be driven from exactly one place"
    sweep_line = sweep_calls[0].lineno

    for node in ast.walk(boundary):
        if not isinstance(node, ast.If):
            continue
        test_names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if not ({"jobs", "finalizations"} & test_names):
            continue
        guarded = [
            child.lineno
            for stmt in node.body
            for child in ast.walk(stmt)
            if hasattr(child, "lineno")
        ]
        assert sweep_line not in guarded, (
            "the sweep is nested inside teacher-work availability; a quiet boundary "
            "would be indistinguishable from a sweep that never ran"
        )


def test_the_legacy_shutdown_only_learning_path_is_absent() -> None:
    """Teacher work must be driven only by durable session finalization."""
    source = _CONTEXT_PY.read_text(encoding="utf-8")
    assert "_on_session_end_learning" not in source
    assert "_drive_world_model_evolution" not in source


# ════════════════════════════════════════════════════════════════
# The learning boundary is callable, not only reachable at shutdown
# ════════════════════════════════════════════════════════════════


def test_the_learning_boundary_has_a_public_entry_point() -> None:
    """Bound to ``cleanup`` alone, "when does it evolve" was unanswerable.

    A daemon that ran for a week never evolved and one killed with SIGKILL never
    evolved at all, because the only caller was process teardown.
    """
    from leapflow.cli.context import Context

    assert hasattr(Context, "run_learning_boundary")
    signature = inspect.signature(Context.run_learning_boundary)
    assert "reason" in signature.parameters, (
        "the caller must be recorded, so a shutdown flush is distinguishable from a "
        "session boundary and from a hand-run one"
    )


def test_cleanup_drives_the_boundary_through_the_public_entry_point() -> None:
    """One path, so the shutdown flush and an explicit run cannot diverge."""
    source = _CONTEXT_PY.read_text(encoding="utf-8")
    assert 'run_learning_boundary(reason="shutdown")' in source


def test_the_daemon_exposes_the_boundary_as_an_rpc() -> None:
    """It has to run in the daemon: that process owns the durable event stream."""
    from leapflow.daemon.protocol import LeapService, METHOD_REGISTRY

    assert METHOD_REGISTRY.get("evolution.run") == "evolution_run"
    signature = inspect.signature(LeapService.evolution_run)
    assert signature.parameters["session_id"].default is inspect.Parameter.empty


def test_the_cli_can_run_the_boundary_without_stopping_the_daemon() -> None:
    from leapflow.cli.commands.evolve import cmd_evolve

    assert callable(cmd_evolve)
    source = (Path(__file__).parents[1] / "src" / "leapflow" / "cli" / "cli.py").read_text()
    assert '"--session"' in source
    assert "required=True" in source


def test_the_trajectory_buffer_is_bounded() -> None:
    """Unbounded, it accumulated every turn of every session for a daemon lifetime."""
    from leapflow.world_model.prediction import _MAX_TRAJECTORY_STEPS, PredictionLoop

    source = inspect.getsource(PredictionLoop.__init__)
    assert "deque(maxlen=_MAX_TRAJECTORY_STEPS)" in source
    assert _MAX_TRAJECTORY_STEPS > 0


# ════════════════════════════════════════════════════════════════
# A write-only queue is not a governed one
# ════════════════════════════════════════════════════════════════


class _Item:
    def __init__(self, status: str) -> None:
        self.status = status


class _Queue:
    def __init__(self, *statuses: str) -> None:
        self._items = [_Item(s) for s in statuses]

    def list_items(self, limit: int = 0) -> list[_Item]:
        return list(self._items)


class _RaisingQueue:
    def list_items(self, limit: int = 0) -> list[_Item]:
        raise OSError("queue unreadable")


@pytest.fixture()
def producer() -> EvolutionProducer:
    return EvolutionProducer()


def _run(producer: EvolutionProducer, queue: object) -> dict:
    """Call the segment with a projection shaped from the supplied lifecycle rows."""
    try:
        proposals = [
            {"status": item.status}
            for item in queue.list_items(limit=0)  # type: ignore[attr-defined]
        ]
    except (AttributeError, OSError):
        return producer._segment_lifecycle(None)
    return producer._segment_lifecycle({"proposals": proposals})


def test_a_queue_that_never_advances_is_not_wired(producer: EvolutionProducer) -> None:
    """The defect: 212 records, every one PENDING, reported as the healthy class.

    Nothing drained the queue and nothing ever would, so it grew monotonically while
    the board showed it green beside a genuinely healthy ``Trust accrual``.
    """
    row = _run(producer, _Queue(*["PENDING"] * 212))
    assert row["status"] == NO_EVIDENCE
    assert "PENDING=212" in row["evidence"], "the spread stays visible"
    assert "212 record(s)" in row["next_step"]
    assert "only grows" in row["next_step"], "the row must name the consequence"


def test_one_advanced_record_is_evidence_the_queue_is_read_back(
    producer: EvolutionProducer,
) -> None:
    """Evidence is a transition, not a row count."""
    row = _run(producer, _Queue("PENDING", "PENDING", "INSTALLED"))
    assert row["status"] == WIRED


@pytest.mark.parametrize("status", ["GENERATED", "APPROVED", "PROBATION", "VERIFIED",
                                    "REJECTED", "FAILED", "QUARANTINED"])
def test_every_post_entry_state_counts_as_advanced(
    producer: EvolutionProducer, status: str
) -> None:
    """Entry states are enumerated, so a new terminal state cannot read as ungoverned."""
    assert _run(producer, _Queue("PENDING", status))["status"] == WIRED


def test_an_empty_queue_is_distinguished_from_a_stuck_one(
    producer: EvolutionProducer,
) -> None:
    row = _run(producer, _Queue())
    assert row["status"] == NO_EVIDENCE
    assert "queue is empty" in row["evidence"]
    assert "plugin_propose" in row["next_step"]


def test_an_unreadable_queue_is_unverifiable_not_no_evidence(
    producer: EvolutionProducer,
) -> None:
    """A source that could not be read is a fault, not an absence of activity."""
    assert _run(producer, _RaisingQueue())["status"] == UNVERIFIABLE
    assert _run(producer, None)["status"] == UNVERIFIABLE


# ════════════════════════════════════════════════════════════════
# A manual refresh must leave a trace that it ran
# ════════════════════════════════════════════════════════════════


def test_a_manual_refresh_records_that_a_cycle_completed() -> None:
    """The defect: refresh looked broken because nothing it touched moved.

    Findings dedup on content, so a cycle confirming "nothing changed" writes nothing.
    The scheduler normally does the run bookkeeping and this path skips the scheduler,
    so without it the board could not tell a refresh that ran from one that never
    happened — which is the one thing a reader presses refresh to find out.
    """
    from leapflow.monitor.manager import MonitorManager

    source = inspect.getsource(MonitorManager.run_watch_once)
    assert "increment_run_count" in source
    # After the cycle, not before: last_run_at means "a cycle completed".
    assert source.index("self._executor.execute") < source.index("increment_run_count")
    assert 'result.get("ok")' in source, "a producer that raised did not complete a cycle"


def test_a_watch_view_publishes_its_declared_cadence() -> None:
    """The board judges its own freshness against it; only the watch knows it."""
    from leapflow.monitor.types import WatchView

    view = WatchView(
        watch_id="w", name="n", domain="d", trigger="every 120s", state="armed",
        muted=False, run_count=1, next_due_at=2.0, last_run_at=1.0,
        interval_seconds=120.0,
    )
    assert view.to_dict()["interval_seconds"] == 120.0


def test_a_watch_with_no_interval_trigger_declares_no_cadence() -> None:
    """Zero means "no fixed cadence", which the board reads as a withheld verdict."""
    from leapflow.monitor.types import WatchView

    assert WatchView(
        watch_id="w", name="n", domain="d", trigger="event:x", state="armed",
        muted=False, run_count=0, next_due_at=0.0, last_run_at=0.0,
    ).to_dict()["interval_seconds"] == 0.0
