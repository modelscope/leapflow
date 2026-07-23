"""Tool concurrency policy — metadata-driven parallel/sequential partitioning.

Parallel-safety is derived from the SAME registry metadata that already drives
the idempotency ledger and the side-effect batch-stop gate
(``execution_policy_for``), rather than from a hardcoded tool-name list:

- ``read_only``            -> parallel (pure reads never conflict)
- ``mutating_idempotent``  -> path-scoped: parallel iff its file path does not
                              overlap another concurrent write in the batch
- ``mutating_once`` / ``external_side_effect`` -> sequential
- unknown / unregistered   -> sequential (conservative default; a new tool never
                              auto-parallelizes just because it is unlisted)

A tool's concurrency behavior therefore follows from its declared ``x_leapflow``
metadata, keeping this in lockstep with approval, idempotency, and batch-stop —
one source of truth, and it generalizes to any new tool (including MCP tools
that declare their metadata) with no name enumeration to maintain.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence, Tuple, runtime_checkable

from leapflow.engine.tool_execution import execution_policy_for

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


# ``spec_lookup(tool_name) -> spec | None`` returns the registry metadata object
# (a ``ToolSpec`` with risk_level / mutates_state / effect_scope /
# idempotency_scope) for a normalized tool name, or None when unregistered.
SpecLookup = Callable[[str], Any]


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


def _extract_path(arguments: dict[str, Any]) -> str:
    """Extract path from tool call arguments."""
    for key in ("path", "file_path", "target", "filename"):
        val = arguments.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


class DefaultConcurrencyPolicy:
    """Metadata-driven concurrency partition (see module docstring).

    ``spec_lookup`` is injected (dependency inversion) so this module never
    imports the registry directly; when it is absent or returns None the tool is
    treated as sequential, so parallelism is strictly opt-in via declared
    read-only / path-scoped-idempotent metadata.
    """

    def __init__(self, *, spec_lookup: Optional[SpecLookup] = None) -> None:
        self._spec_lookup = spec_lookup

    def partition(
        self, tool_calls: Sequence[ToolCall]
    ) -> Tuple[list[ToolCall], list[ToolCall]]:
        if len(tool_calls) <= 1:
            return list(tool_calls), []

        concurrent: list[ToolCall] = []
        sequential: list[ToolCall] = []
        concurrent_paths: list[str] = []

        for tc in tool_calls:
            policy = self._policy_for(tc.name)
            if policy == "read_only":
                concurrent.append(tc)
            elif policy == "mutating_idempotent":
                # Path-scoped: safe to parallelize only when this write's path
                # does not overlap another write already in the concurrent group.
                path = _extract_path(tc.arguments)
                if path and not any(_paths_overlap(path, ep) for ep in concurrent_paths):
                    concurrent.append(tc)
                    concurrent_paths.append(path)
                else:
                    sequential.append(tc)
            else:
                # mutating_once / external_side_effect / unknown -> sequential.
                sequential.append(tc)

        logger.debug(
            "tool_concurrency.partition concurrent=%d sequential=%d",
            len(concurrent),
            len(sequential),
        )
        return concurrent, sequential

    def _policy_for(self, name: str) -> Optional[str]:
        """Return the tool's execution policy from registry metadata.

        Returns None for an unregistered / unresolvable tool, which the caller
        treats as sequential (conservative default).
        """
        if self._spec_lookup is None:
            return None
        try:
            spec = self._spec_lookup(name)
        except Exception:  # a lookup failure must never break tool dispatch
            logger.debug("tool_concurrency: spec_lookup failed for %s", name, exc_info=True)
            return None
        if spec is None:
            return None
        return execution_policy_for(name, spec)
