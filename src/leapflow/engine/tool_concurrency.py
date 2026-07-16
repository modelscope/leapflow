"""Tool concurrency policy — determines which tool calls can execute in parallel.

Design (inspired by hermes tool_dispatch_helpers):
- Three-tier classification: always-parallel, path-scoped, always-sequential
- Path overlap detection prevents concurrent writes to same file subtree
- MCP tools get parallel safety from registry metadata
- Configurable via constructor injection (OCP)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Protocol, Sequence, Tuple, runtime_checkable

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


# Tools that are always safe to parallelize (pure reads, no side effects).
# Names MUST match actual tool names from registry_bootstrap.py:
#   gp_file_read, gp_file_list, gp_text_search, gp_skills_list, gp_skill_view,
#   gp_time_get, gp_env_info, plus their unprefixed aliases.
_DEFAULT_PARALLEL_SAFE: FrozenSet[str] = frozenset({
    "gp_file_read", "gp_file_list", "gp_text_search",
    "gp_skills_list", "gp_skill_view",
    "gp_time_get", "gp_env_info",
    "gp_web_search", "gp_web_extract",
    "gp_session_search", "gp_memory_search",
    "file_read", "file_list", "text_search",
    "skills_list", "skill_view",
    "time_get", "env_info",
    "web_search", "web_extract",
    "session_search", "memory_search",
})

# Tools where parallelism depends on non-overlapping file paths
_DEFAULT_PATH_SCOPED: FrozenSet[str] = frozenset({
    "gp_file_write", "gp_text_replace",
    "file_write", "text_replace",
})

# Tools that must never run in parallel (side effects, user interaction, shell,
# or external platform authorization boundaries where one failure can block the
# rest of the turn).
_DEFAULT_NEVER_PARALLEL: FrozenSet[str] = frozenset({
    "gp_shell_run", "shell_run",
    "gp_scm_sync", "scm_sync",
    "clarify", "gp_clarify",
    "delegate_task", "gp_delegate_task",
    "memory_add", "gp_memory_add",
    "platform_action", "gp_platform_action",
    "platform_connect", "gp_platform_connect",
    "gateway_send", "gp_gateway_send",
    "gateway_connect", "gp_gateway_connect",
    "hub_push", "gp_hub_push",
    "hub_pull", "gp_hub_pull",
    "hub_sync", "gp_hub_sync",
})


def _paths_overlap(left: str, right: str) -> bool:
    """Check if two paths share a common subtree (one is prefix of the other).

    Does not resolve symlinks — paths may not exist yet (writes).
    """
    if not left or not right:
        return False
    left_norm = os.path.normpath(left)
    right_norm = os.path.normpath(right)
    if left_norm == right_norm:
        return True
    return (
        right_norm.startswith(left_norm + os.sep)
        or left_norm.startswith(right_norm + os.sep)
    )


def _extract_path(arguments: Dict[str, Any]) -> str:
    """Extract path from tool call arguments."""
    for key in ("path", "file_path", "target", "filename"):
        val = arguments.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


class DefaultConcurrencyPolicy:
    """Three-tier concurrency policy with path overlap detection.

    Tier 1: Always-parallel tools (pure reads)
    Tier 2: Path-scoped tools (parallel iff paths don't overlap)
    Tier 3: Always-sequential (stateful/interactive)

    MCP tools are parallel-safe unless registered otherwise.
    """

    def __init__(
        self,
        *,
        parallel_safe: FrozenSet[str] | None = None,
        path_scoped: FrozenSet[str] | None = None,
        never_parallel: FrozenSet[str] | None = None,
        stateful_prefixes: FrozenSet[str] | None = None,
        mcp_parallel_safe: FrozenSet[str] | None = None,
    ) -> None:
        self._parallel_safe = parallel_safe or _DEFAULT_PARALLEL_SAFE
        self._path_scoped = path_scoped or _DEFAULT_PATH_SCOPED
        self._never_parallel = never_parallel or _DEFAULT_NEVER_PARALLEL
        self._stateful_prefixes = stateful_prefixes or frozenset({
            "shell", "batch_", "clipboard",
        })
        self._mcp_parallel_safe = mcp_parallel_safe or frozenset()

    def partition(
        self, tool_calls: Sequence[ToolCall]
    ) -> Tuple[list[ToolCall], list[ToolCall]]:
        if len(tool_calls) <= 1:
            return list(tool_calls), []

        concurrent: list[ToolCall] = []
        sequential: list[ToolCall] = []
        concurrent_paths: list[str] = []

        for tc in tool_calls:
            if tc.name in self._never_parallel:
                sequential.append(tc)
                continue

            if tc.name in self._parallel_safe:
                concurrent.append(tc)
                continue

            if tc.name.startswith("mcp_"):
                if tc.name in self._mcp_parallel_safe or self._is_mcp_read(tc):
                    concurrent.append(tc)
                else:
                    sequential.append(tc)
                continue

            if tc.name in self._path_scoped:
                path = _extract_path(tc.arguments)
                if path and any(_paths_overlap(path, ep) for ep in concurrent_paths):
                    sequential.append(tc)
                else:
                    concurrent.append(tc)
                    if path:
                        concurrent_paths.append(path)
                continue

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
        if tool_name in self._never_parallel:
            return True
        return any(tool_name.startswith(prefix) for prefix in self._stateful_prefixes)

    @staticmethod
    def _is_mcp_read(tc: ToolCall) -> bool:
        """Heuristic: MCP tools with 'read', 'get', 'list', 'search' are likely read-only."""
        name_lower = tc.name.lower()
        return any(verb in name_lower for verb in ("read", "get", "list", "search", "fetch", "query"))
