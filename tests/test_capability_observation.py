# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for structured capability observations."""

from __future__ import annotations

from leapflow.learning.capability_observation import CapabilityObservationBuffer


def test_observation_buffer_accepts_unknown_tool_results() -> None:
    buffer = CapabilityObservationBuffer()

    accepted = buffer.add_result(
        {
            "ok": False,
            "error_type": "unknown_tool",
            "original_tool_name": "json_pretty",
            "suggestions": ["json_pretty_loop"],
        }
    )

    assert accepted is True
    requirements = buffer.requirements(min_count=1)
    assert len(requirements) == 1
    assert requirements[0].origin == "unknown_tool"
    assert requirements[0].capability == "json_pretty"
    assert dict(requirements[0].metadata)["original_tool_name"] == "json_pretty"


def test_observation_buffer_ignores_non_structured_failures() -> None:
    buffer = CapabilityObservationBuffer()

    assert buffer.add_result({"ok": False, "error": "plain failure"}) is False
    assert buffer.add_result({"ok": True, "result": "done"}) is False
    assert buffer.requirements() == ()


def test_observation_buffer_honors_min_count() -> None:
    buffer = CapabilityObservationBuffer()
    buffer.add_result({"error_type": "unknown_tool", "original_tool_name": "missing_tool"})

    assert buffer.requirements(min_count=2) == ()

    buffer.add_result({"error_type": "unknown_tool", "original_tool_name": "missing_tool"})
    assert len(buffer.requirements(min_count=2)) == 1
