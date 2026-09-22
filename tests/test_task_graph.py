# Copyright (c) Alibaba, Inc. and its affiliates.
"""Task graph data-structure unit tests.

Extracted from test_agent_execution.py — pure TaskGraph scenarios with no
engine or LLM dependency.
"""

from __future__ import annotations

from typing import List

import pytest

from leapflow.engine.task_planning.task_graph import (
    GraphValidationError,
    RetryPolicy,
    TaskGraph,
    TaskNode,
    TaskStatus,
)


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
