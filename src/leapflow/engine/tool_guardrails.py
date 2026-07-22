"""Tool loop guardrails — detect and halt repeated failures, stagnation, and loops.

Monitors tool execution patterns during the agent loop and emits warnings
or halt signals when it detects:
1. Consecutive identical tool calls (exact argument match → loop detection)
2. Monotonic failure streaks exceeding threshold
3. Token burn without progress (stagnation — total tokens spent vs actions completed)
4. Single tool domination (one tool used > N times consecutively)

These guards prevent runaway agent loops that burn tokens without progress.
All thresholds are configurable via Settings (no hardcoded limits).

Implements the Guard Protocol so engine can use any guard implementation (DIP).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardrailViolation:
    """Describes a guardrail check result."""
    violated: bool
    reason: str = ""
    severity: str = "warning"  # "warning" | "halt"
    suggestion: str = ""


@runtime_checkable
class ToolLoopGuard(Protocol):
    """Protocol for tool loop guardrails (DIP)."""

    def check(self, history: List[Dict[str, Any]]) -> GuardrailViolation: ...
    def reset(self) -> None: ...


class RepetitionGuard:
    """Detect exact-duplicate tool calls (same name + same arguments hash).

    Triggers when the same tool call appears N+ times consecutively.
    """

    def __init__(self, *, max_repeats: int = 3) -> None:
        self._max_repeats = max_repeats
        self._recent_hashes: List[str] = []

    def check(self, history: List[Dict[str, Any]]) -> GuardrailViolation:
        tool_msgs = [
            m for m in history
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        if not tool_msgs:
            return GuardrailViolation(violated=False)

        hashes: List[str] = []
        for msg in tool_msgs[-self._max_repeats * 2:]:
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {})
                key = f"{fn.get('name', '')}:{fn.get('arguments', '')}"
                hashes.append(hashlib.md5(key.encode()).hexdigest()[:12])

        if len(hashes) >= self._max_repeats:
            tail = hashes[-self._max_repeats:]
            if len(set(tail)) == 1:
                return GuardrailViolation(
                    violated=True,
                    reason=f"Identical tool call repeated {self._max_repeats} times",
                    severity="halt",
                    suggestion="Try a different approach or provide the final answer.",
                )

        return GuardrailViolation(violated=False)

    def reset(self) -> None:
        self._recent_hashes.clear()


class StagnationGuard:
    """Detect token burn without forward progress.

    Measures the ratio of tool results containing 'ok: true' vs total
    tool results in the last N messages. Triggers if success rate drops
    below threshold.
    """

    def __init__(self, *, window: int = 10, min_success_rate: float = 0.2) -> None:
        self._window = window
        self._min_rate = min_success_rate

    def check(self, history: List[Dict[str, Any]]) -> GuardrailViolation:
        tool_results = [
            m for m in history[-self._window * 3:]
            if self._is_tool_result_message(m)
        ]
        if len(tool_results) < self._window:
            return GuardrailViolation(violated=False)

        recent = tool_results[-self._window:]
        successes = 0
        for msg in recent:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            if self._is_success_result(content):
                successes += 1

        rate = successes / len(recent)
        if rate < self._min_rate:
            return GuardrailViolation(
                violated=True,
                reason=f"Low tool success rate ({rate:.0%}) in last {len(recent)} calls",
                severity="warning",
                suggestion="Most recent tool calls are failing. Reassess your approach.",
            )

        return GuardrailViolation(violated=False)

    @staticmethod
    def _is_tool_result_message(msg: Dict[str, Any]) -> bool:
        """Whether a message is a genuine tool result (not injected context).

        Only native ``tool`` messages and text-mode ``Tool result (...)`` user
        messages count. Injected user/system context (live signals, research
        ledger, convergence/cost notices, memory) is excluded so the success
        rate is not diluted by non-tool messages on a context-heavy long task.
        """
        role = msg.get("role")
        if role == "tool":
            return True
        if role == "user":
            content = msg.get("content", "")
            return isinstance(content, str) and content.lstrip().startswith("Tool result (")
        return False

    @staticmethod
    def _is_success_result(content: str) -> bool:
        """Detect tool success from native tool JSON or text-mode tool results."""
        if '"ok": true' in content or '"ok":true' in content:
            return True
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and parsed.get("ok") is True:
                return True
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return False

    def reset(self) -> None:
        pass


class DominationGuard:
    """Detect single-tool domination (same tool called N+ times without variety).

    Prevents the agent from fixating on a single tool when multiple are available.
    """

    def __init__(self, *, max_consecutive_same: int = 5) -> None:
        self._threshold = max_consecutive_same

    def check(self, history: List[Dict[str, Any]]) -> GuardrailViolation:
        recent_tools: List[str] = []
        for msg in history:
            if msg.get("role") == "assistant":
                for tc in (msg.get("tool_calls") or []):
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    if name:
                        recent_tools.append(name)

        if len(recent_tools) < self._threshold:
            return GuardrailViolation(violated=False)

        tail = recent_tools[-self._threshold:]
        if len(set(tail)) == 1:
            return GuardrailViolation(
                violated=True,
                reason=f"Tool '{tail[0]}' used {self._threshold} times consecutively",
                severity="warning",
                suggestion="Consider using a different tool or providing the answer directly.",
            )

        return GuardrailViolation(violated=False)

    def reset(self) -> None:
        pass


class CompositeGuardrail:
    """Composite of multiple guards — runs all, returns first halt or worst warning."""

    def __init__(
        self,
        guards: Optional[List[ToolLoopGuard]] = None,
        *,
        max_repeats: int = 3,
        stagnation_window: int = 10,
        min_success_rate: float = 0.2,
        max_consecutive_same: int = 5,
    ) -> None:
        self._guards: List[ToolLoopGuard] = guards or [
            RepetitionGuard(max_repeats=max_repeats),
            StagnationGuard(window=stagnation_window, min_success_rate=min_success_rate),
            DominationGuard(max_consecutive_same=max_consecutive_same),
        ]

    def check(self, history: List[Dict[str, Any]]) -> GuardrailViolation:
        worst: Optional[GuardrailViolation] = None
        for guard in self._guards:
            result = guard.check(history)
            if result.violated:
                if result.severity == "halt":
                    return result
                if worst is None or worst.severity == "warning":
                    worst = result
        return worst or GuardrailViolation(violated=False)

    def reset(self) -> None:
        for guard in self._guards:
            guard.reset()
