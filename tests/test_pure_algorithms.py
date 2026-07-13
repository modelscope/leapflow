"""Cherry-picked pure algorithm tests — deterministic, stateless primitives."""

from __future__ import annotations

import pytest

from leapflow.causal.types import (
    CausalEvent,
    CausalGraph,
    EventSource,
    EventType,
)
from leapflow.learning.active_learning import _lcs, _merge_steps_lcs, _union_dedup
from leapflow.learning.similarity import (
    _action_sequence_similarity,
    _jaccard,
    _levenshtein,
    _token_jaccard,
)
from leapflow.memory import decay_weight
from leapflow.prompts.templates import UNIFIED_SYSTEM_TEMPLATE
from leapflow.world_model._json_utils import extract_json_object


def _make_event(channel: str, ts: float) -> CausalEvent:
    return CausalEvent(
        id=CausalEvent.make_id(),
        timestamp=ts,
        event_type=EventType.TRIGGER,
        source=EventSource.SIGNAL,
        channel=channel,
        payload={},
        confidence=1.0,
    )


def test_jaccard_similarity() -> None:
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard({"a"}, {"b"}) == 0.0
    assert _jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)


def test_levenshtein_distance() -> None:
    assert _levenshtein([], []) == 0
    assert _levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert _levenshtein(["a", "c"], ["a", "b", "c"]) == 1


def test_token_jaccard() -> None:
    assert _token_jaccard("organize files in downloads", "organize files in downloads") == 1.0
    assert _token_jaccard("hello world", "goodbye moon") == 0.0
    score = _token_jaccard("organize files downloads", "organize downloads")
    assert 0.0 < score < 1.0


def test_action_sequence_similarity() -> None:
    assert _action_sequence_similarity(["click", "type", "submit"], ["click", "type", "submit"]) == 1.0
    assert _action_sequence_similarity([], []) == 1.0
    divergent = _action_sequence_similarity(["click", "type", "submit"], ["click", "scroll", "close"])
    assert 0.0 < divergent < 1.0


def test_decay_weight_aging() -> None:
    fresh = decay_weight(1.0, age_seconds=0.0, frequency=1.0, decay_lambda=1e-3)
    aged = decay_weight(1.0, age_seconds=3600.0, frequency=1.0, decay_lambda=1e-3)
    assert aged < fresh


def test_decay_weight_frequency_boost() -> None:
    low_freq = decay_weight(1.0, age_seconds=100.0, frequency=1.0, decay_lambda=1e-3)
    high_freq = decay_weight(1.0, age_seconds=100.0, frequency=10.0, decay_lambda=1e-3)
    assert high_freq > low_freq


def test_lcs_merge_steps() -> None:
    existing = ["List", "Classify", "Move"]
    candidate = ["List", "Rename", "Move"]
    merged = _merge_steps_lcs(existing, candidate)
    assert _lcs(existing, candidate) == ["List", "Move"]
    assert merged.index("List") < merged.index("Move")
    assert "Classify" in merged
    assert "Rename" in merged


def test_union_dedup_preserves_order() -> None:
    assert _union_dedup(["a", "b", "c"], ["b", "c", "d"]) == ["a", "b", "c", "d"]
    assert _union_dedup(["x", "y"], ["x", "z"]) == ["x", "y", "z"]
    assert _union_dedup([], ["m", "n"]) == ["m", "n"]


def test_causal_graph_topology() -> None:
    graph = CausalGraph()
    parent = _make_event("click", 1.0)
    middle = _make_event("visual_change", 1.2)
    child = _make_event("clipboard", 1.4)
    for ev in (parent, middle, child):
        graph.add_event(ev)
    graph.add_edge(parent.id, middle.id)
    graph.add_edge(middle.id, child.id)

    order = graph.topological_order()
    ids = [e.id for e in order]
    assert ids.index(parent.id) < ids.index(middle.id) < ids.index(child.id)


def test_causal_graph_serialization() -> None:
    graph = CausalGraph()
    parent = _make_event("click", 1.0)
    child = _make_event("visual_change", 1.2)
    graph.add_event(parent)
    graph.add_event(child)
    graph.add_edge(parent.id, child.id)
    graph.metadata["session"] = "test"

    restored = CausalGraph.from_dict(graph.to_dict())
    assert restored.event_count == 2
    assert restored.metadata["session"] == "test"
    assert restored.events[child.id].caused_by == parent.id
    assert child.id in restored.events[parent.id].causes


def test_causal_graph_connected_components() -> None:
    graph = CausalGraph()
    a, b = _make_event("click", 1.0), _make_event("visual_change", 1.2)
    c, d = _make_event("click", 5.0), _make_event("visual_change", 5.3)
    for ev in (a, b, c, d):
        graph.add_event(ev)
    graph.add_edge(a.id, b.id)
    graph.add_edge(c.id, d.id)

    components = graph.connected_components()
    assert len(components) == 2
    comp_sets = [frozenset(c) for c in components]
    assert frozenset({a.id, b.id}) in comp_sets
    assert frozenset({c.id, d.id}) in comp_sets


def test_unified_system_template_escapes_literal_tool_protocol_json() -> None:
    rendered = UNIFIED_SYSTEM_TEMPLATE.format(
        tool_catalog="- **skills_list**(): List installed skills",
        skill_section="",
        memory_context="",
    )

    assert '{"name": "tool_name", "arguments": {"key": "value"}}' in rendered
    assert '{"name": ..., "arguments": ...}' in rendered
    assert "Avoid redundant tool calls" in rendered
    assert "same tool with the same arguments" in rendered
    assert "existing tool result already answers" in rendered


def test_json_extraction_variants() -> None:
    assert extract_json_object('{"key": "value", "n": 42}') == {"key": "value", "n": 42}

    fenced = 'Analysis:\n```json\n{"insights": [{"type": "pattern"}]}\n```\nDone.'
    assert extract_json_object(fenced) == {"insights": [{"type": "pattern"}]}

    wrapped = 'prefix {"outer": {"inner": 1}, "items": [1, 2]} suffix'
    assert extract_json_object(wrapped) == {"outer": {"inner": 1}, "items": [1, 2]}

    assert extract_json_object("") == {}
    assert extract_json_object("no json here") == {}
    assert extract_json_object('{"broken": [1,],}') == {}
