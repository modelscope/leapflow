"""Tool concurrency policy — determines which tool calls can execute in parallel.

Design principle: stateless/read-only tools are safe to parallelize;
stateful/mutating tools must execute sequentially to avoid race conditions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Represents a single tool call from LLM output."""

    id: str
    name: str
    arguments: dict


@runtime_checkable
class ToolConcurrencyPolicy(Protocol):
    """Determines which tool calls can execute concurrently."""

    def partition(
        self, tool_calls: Sequence[ToolCall]
    ) -> Tuple[list[ToolCall], list[ToolCall]]:
        """Split tool calls into (concurrent_group, sequential_group).

        Concurrent group executes via asyncio.gather.
        Sequential group executes one by one after the concurrent group.
        """
        ...


class DefaultConcurrencyPolicy:
    """Default policy: read-only tools parallelize, stateful tools serialize.

    Classification is prefix-based and configurable.
    """

    def __init__(
        self,
        *,
        stateful_prefixes: frozenset[str] | None = None,
        always_sequential: frozenset[str] | None = None,
    ) -> None:
        self._stateful_prefixes = stateful_prefixes or frozenset(
            {
                "file.write",
                "file.delete",
                "file.rename",
                "file.create",
                "shell.",
                "batch_",
                "clipboard.set",
            }
        )
        self._always_sequential = always_sequential or frozenset()

    def partition(
        self, tool_calls: Sequence[ToolCall]
    ) -> Tuple[list[ToolCall], list[ToolCall]]:
        if len(tool_calls) <= 1:
            return list(tool_calls), []

        concurrent: list[ToolCall] = []
        sequential: list[ToolCall] = []

        for tc in tool_calls:
            if self._is_stateful(tc.name):
                sequential.append(tc)
            else:
                concurrent.append(tc)

        logger.debug(
            "tool_concurrency.partition concurrent=%d sequential=%d",
            len(concurrent),
            len(sequential),
        )
        return concurrent, sequential

    def _is_stateful(self, tool_name: str) -> bool:
        if tool_name in self._always_sequential:
            return True
        return any(tool_name.startswith(prefix) for prefix in self._stateful_prefixes)
