"""S4-D1: multi-layer orientation aggregation (observe-only).

Hermetic unit tests for the pure orientation aggregator and the research-ledger
adapter — layer salience + time decay, ranking, caps, and the working-layer
mapping. No I/O, no engine.
"""
from __future__ import annotations

from leapflow.world_model.orientation import (
    aggregate_orientation,
    build_orientation_from_ledger,
)


def test_aggregate_orientation_layers_and_decay() -> None:
    now = 1000.0
    orientation = aggregate_orientation(
        immediate=[("fresh signal", now), ("old signal", now - 300.0)],  # 1.0, 0.5 (halflife 300)
        working=[("task finding", now)],                                  # 0.7
        long_term=[("durable fact", now)],                                # 0.4
        now=now,
    )
    texts = [it.text for it in orientation.items]
    # Weight order: fresh immediate (1.0) > working (0.7) > decayed immediate (0.5) > long_term (0.4).
    assert texts == ["fresh signal", "task finding", "old signal", "durable fact"]
    assert orientation.items[0].weight == 1.0
    assert orientation.by_layer("immediate")[0].text == "fresh signal"


def test_aggregate_orientation_caps_and_empty() -> None:
    capped = aggregate_orientation(working=[f"f{i}" for i in range(30)], now=0.0, max_items=5)
    assert len(capped.items) == 5

    empty = aggregate_orientation(now=0.0)
    assert empty.items == ()
    assert empty.summary()["total"] == 0


def test_build_orientation_from_ledger_maps_working_layer() -> None:
    now = 100.0
    state = {
        "findings": ["A uses DuckDB"],
        "open_questions": ["does B cache?"],
        "next_step": "check B",
        "decisions": [],
    }
    orientation = build_orientation_from_ledger(
        state, now=now,
        immediate=[("live event", now)],
        long_term=[("old preference", now - 86400.0)],
    )
    working_texts = [it.text for it in orientation.by_layer("working")]
    assert "A uses DuckDB" in working_texts
    assert "[open] does B cache?" in working_texts
    assert "[next] check B" in working_texts
    # Fresh immediate outranks working; long-term (aged one half-life) sinks.
    assert orientation.items[0].layer == "immediate"
    assert orientation.by_layer("long_term")[0].text == "old preference"


def test_build_orientation_from_ledger_empty_is_safe() -> None:
    assert build_orientation_from_ledger(None, now=0.0).items == ()
    assert build_orientation_from_ledger({}, now=0.0).items == ()


def test_orientation_render_and_summary() -> None:
    now = 0.0
    orientation = aggregate_orientation(immediate=["sig"], working=["task"], now=now)
    rendered = orientation.render()
    assert "(immediate) sig" in rendered
    assert "(working) task" in rendered
    summary = orientation.summary()
    assert summary["total"] == 2
    assert summary["by_layer"]["immediate"] == 1
    assert summary["by_layer"]["working"] == 1
