# Copyright (c) Alibaba, Inc. and its affiliates.
"""Behavior test execution for generated/profile plugins."""
from __future__ import annotations

import asyncio
from typing import Any

from leapflow.domain.plugin_proposal import BehaviorTestCase
from leapflow.plugins.handler_invocation import invoke_tool_handler


async def run_plugin_behavior_tests(
    plugin: Any,
    test_cases: tuple[BehaviorTestCase, ...],
    *,
    timeout_s: float = 5.0,
) -> tuple[bool, str, list[dict[str, Any]]]:
    """Run proposal-defined behavior tests against a loaded plugin instance.

    Tests assert that the handler result contains an expected subset. This keeps
    cases robust to extra diagnostic fields while still verifying behavior.
    """
    if not test_cases:
        return True, "", []
    metadata_by_name = {tool.name: tool for tool in plugin.tools}
    observations: list[dict[str, Any]] = []
    for index, case in enumerate(test_cases):
        tool = metadata_by_name.get(case.tool_name)
        if tool is None:
            return False, f"behavior test {index}: tool {case.tool_name!r} not exposed", observations
        args = dict(case.arguments)
        expected = dict(case.expected_subset)
        try:
            result = await asyncio.wait_for(invoke_tool_handler(tool.handler, args), timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001 - plugin behavior failure is test failure
            return False, f"behavior test {index}: handler raised {type(exc).__name__}: {exc}", observations
        observations.append({"tool_name": case.tool_name, "arguments": args, "result": result})
        if not isinstance(result, dict):
            return False, f"behavior test {index}: result is not a dict", observations
        for key, expected_value in expected.items():
            if result.get(key) != expected_value:
                return (
                    False,
                    f"behavior test {index}: expected {key}={expected_value!r}, got {result.get(key)!r}",
                    observations,
                )
    return True, "", observations
