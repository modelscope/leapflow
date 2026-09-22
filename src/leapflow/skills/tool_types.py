# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared domain types for skill tool execution.

Contains the lightweight data classes and protocols used across the skill
subsystem — tool definitions, parsed tool calls, step outputs, and the
execution port contract. These types are intentionally decoupled from any
executor implementation so that modules like ``action_policy``,
``semantic_schema``, and ``tool_dispatch_engine`` can import them without
pulling in the legacy ReAct executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolDefinition:
    """Schema for one available tool — injected into the LLM system prompt.

    Traits:
        mutates_state: Tool changes observable state -> clears dedup cache.
        counts_as_progress: Tool represents forward progress toward the goal
            -> triggers completion HINT. Defaults to mutates_state.
            Set False for timing/polling tools (wait, wait_until_stable).
    """

    name: str
    description: str
    parameters: Dict[str, str]
    mutates_state: bool = False
    counts_as_progress: bool | None = None

    @property
    def is_progress(self) -> bool:
        if self.counts_as_progress is not None:
            return self.counts_as_progress
        return self.mutates_state


@dataclass(frozen=True)
class ToolCall:
    """Parsed tool invocation from LLM output."""

    name: str
    params: Dict[str, Any]


@dataclass
class StepOutput:
    """Result of executing one instruction step."""

    ok: bool
    result: str = ""
    error: str = ""
    tool_calls_made: int = 0
    goal_complete: bool = False


@runtime_checkable
class ExecutionPort(Protocol):
    """Minimal execution interface (matches vsi.ports.ExecutionPort)."""

    async def perform_file_op(self, op: str, params: Dict[str, Any]) -> Dict[str, Any]: ...
    async def exec_shell(self, command: str) -> Dict[str, Any]: ...
    async def launch_app(
        self, app_id: str, urls: Optional[List[str]] = None
    ) -> Dict[str, Any]: ...
    async def perform_ui_action(
        self, node_id: str, action: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]: ...
