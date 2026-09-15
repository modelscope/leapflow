# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for proposal-defined plugin behavior checks."""
from __future__ import annotations

from typing import Any

import pytest

from leapflow.domain.plugin_proposal import BehaviorTestCase
from leapflow.learning.plugin_behavior_tests import run_plugin_behavior_tests
from leapflow.plugins.protocol import ToolMetadata


class _Plugin:
    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="echo",
                description="Echo a message.",
                parameters_schema={"type": "object", "properties": {}},
                handler=self._echo,
                x_leapflow={"category": "test", "risk_level": "read_only"},
            )
        ]

    async def _echo(self, message: str = "", **kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "message": message, "extra": "allowed"}


@pytest.mark.asyncio
async def test_behavior_tests_pass_on_expected_subset() -> None:
    ok, error, observations = await run_plugin_behavior_tests(
        _Plugin(),
        (
            BehaviorTestCase.create(
                "echo",
                arguments={"message": "hi"},
                expected_subset={"ok": True, "message": "hi"},
            ),
        ),
    )

    assert ok is True
    assert error == ""
    assert observations[0]["result"]["extra"] == "allowed"


@pytest.mark.asyncio
async def test_behavior_tests_fail_on_missing_tool() -> None:
    ok, error, _observations = await run_plugin_behavior_tests(
        _Plugin(),
        (BehaviorTestCase.create("missing", expected_subset={"ok": True}),),
    )

    assert ok is False
    assert "not exposed" in error


@pytest.mark.asyncio
async def test_behavior_tests_fail_on_mismatched_subset() -> None:
    ok, error, _observations = await run_plugin_behavior_tests(
        _Plugin(),
        (BehaviorTestCase.create("echo", arguments={"message": "hi"}, expected_subset={"message": "bye"}),),
    )

    assert ok is False
    assert "expected message" in error
