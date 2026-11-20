# Copyright (c) Alibaba, Inc. and its affiliates.
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

import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from leapflow.engine.tools.tool_execution import (
    TIER_REPAIRED,
    TIER_SUCCEEDED,
    exit_code_from,
    result_tier,
)

logger = logging.getLogger(__name__)

# Volatile values make raw result hashes unsuitable for a retry guard: screenshots
# use random temp paths, errors carry traceback line numbers, and some drivers add
# timestamps or base64. Normalize only those unstable fragments so semantically
# distinct calls still produce distinct fingerprints.
_TRACEBACK_LINE_RE = re.compile(r"\bline\s+\d+\b", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(r"\b1\d{9,12}(?:\.\d+)?\b")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b", re.IGNORECASE)
_SCREENSHOT_TEMP_RE = re.compile(
    r"(?:/[^\s'\"]+)?/leapflow_screenshot_[0-9a-f]{8}\.png", re.IGNORECASE
)
_BASE64_RE = re.compile(r"\b[A-Za-z0-9+/]{80,}={0,2}\b")


def _normalize_failure_value(value: Any, *, key: str = "") -> Any:
    """Return a deterministic representation with known volatile data removed."""
    key_lower = key.lower()
    if any(token in key_lower for token in ("base64", "image", "traceback")):
        return "<redacted>"
    if key_lower in {"timestamp", "ts", "created_at", "updated_at", "observed_at"}:
        return "<timestamp>"
    if isinstance(value, dict):
        return {
            str(item_key): _normalize_failure_value(item_value, key=str(item_key))
            for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_failure_value(item, key=key) for item in value]
    if not isinstance(value, str):
        return value
    normalized = " ".join(value.split())
    normalized = _TRACEBACK_LINE_RE.sub("line <n>", normalized)
    normalized = _TIMESTAMP_RE.sub("<timestamp>", normalized)
    normalized = _UUID_RE.sub("<uuid>", normalized)
    normalized = _SCREENSHOT_TEMP_RE.sub("<screenshot_path>", normalized)
    return _BASE64_RE.sub("<base64>", normalized)


@dataclass(frozen=True)
class FailureFingerprint:
    """Stable identity for one failed tool action, independent of volatile output."""

    tool_name: str
    canonical_arguments: str
    failure_code: str
    error_type: str
    returncode: int | None
    normalized_error: str

    @classmethod
    def from_result(
        cls,
        tool_name: str,
        arguments: Dict[str, Any] | None,
        result: Dict[str, Any],
    ) -> "FailureFingerprint":
        normalized_arguments = _normalize_failure_value(arguments or {})
        return cls(
            tool_name=str(tool_name or "").removeprefix("gp_"),
            canonical_arguments=json.dumps(
                normalized_arguments, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")
            ),
            failure_code=str(result.get("failure_code") or ""),
            error_type=str(result.get("error_type") or ""),
            returncode=exit_code_from(result),
            normalized_error=str(_normalize_failure_value(result.get("error") or "")),
        )

    @property
    def digest(self) -> str:
        payload = {
            "tool_name": self.tool_name,
            "arguments": self.canonical_arguments,
            "failure_code": self.failure_code,
            "error_type": self.error_type,
            "returncode": self.returncode,
            "error": self.normalized_error,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class GuardrailViolation:
    """Describes a guardrail check result."""
    violated: bool
    reason: str = ""
    severity: str = "warning"  # "warning" | "halt"
    suggestion: str = ""
    # A halt the engine must honour regardless of the global stall marker.
    # Used for definitionally zero progress (identical call/result loops) and
    # resource ceilings. A productive soft ceiling can still be extended, but
    # the coarse research/governance marker cannot suppress a terminal verdict.
    progress_independent: bool = False
    # The halt may be lifted by raising a soft ceiling, because the window
    # that reached it was productive. The engine grants the extension only
    # when real resource budgets also allow it; otherwise the halt stands.
    extendable: bool = False


@runtime_checkable
class ToolLoopGuard(Protocol):
    """Protocol for tool loop guardrails (DIP)."""

    def check(self, history: List[Dict[str, Any]]) -> GuardrailViolation: ...
    def reset(self) -> None: ...


class RepetitionGuard:
    """Detect a no-progress loop: the same tool call returning the same result.

    Triggers when a tool call with identical name + arguments *and* an identical
    result appears N+ times consecutively. The result is part of the signature
    on purpose: a call that returns a *changing* value each time (legitimate
    polling of a sensor, a queue, a build status) is genuine progress and must
    not be flagged, whereas a call that keeps returning the *same* value is zero
    information gain no matter what the global progress heuristic believes. That
    is why the resulting halt is ``progress_independent`` — it is safe to honour
    without consulting the stall marker.
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

        # Correlate each native tool call with the result it produced so the
        # signature reflects information gain, not just intent. Failed results
        # use ``FailureFingerprint`` rather than their raw JSON: ephemeral paths,
        # base64, timestamps, and traceback line numbers must not turn one stuck
        # action into an apparently new failure every round.
        results_by_id: Dict[str, Any] = {}
        for m in history:
            if m.get("role") != "tool":
                continue
            content = m.get("content", "")
            if not isinstance(content, str):
                results_by_id[str(m.get("tool_call_id", ""))] = ""
                continue
            try:
                parsed = json.loads(content)
            except (TypeError, ValueError):
                parsed = content
            results_by_id[str(m.get("tool_call_id", ""))] = parsed

        hashes: List[str] = []
        for msg in tool_msgs[-self._max_repeats * 2:]:
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {})
                raw_arguments = fn.get("arguments", "{}")
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                except (TypeError, ValueError):
                    arguments = {}
                result = results_by_id.get(str(tc.get("id", "")), "")
                if isinstance(result, dict) and result.get("ok") is False:
                    fingerprint = FailureFingerprint.from_result(
                        str(fn.get("name", "")),
                        arguments if isinstance(arguments, dict) else {},
                        result,
                    )
                    hashes.append(fingerprint.digest[:12])
                    continue
                key = json.dumps(
                    {
                        "tool_name": fn.get("name", ""),
                        "arguments": _normalize_failure_value(arguments),
                        "result": _normalize_failure_value(result),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                    separators=(",", ":"),
                )
                hashes.append(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12])

        if len(hashes) >= self._max_repeats:
            tail = hashes[-self._max_repeats:]
            if len(set(tail)) == 1:
                return GuardrailViolation(
                    violated=True,
                    reason=(
                        f"Identical tool call returned the same result "
                        f"{self._max_repeats} times"
                    ),
                    severity="halt",
                    suggestion="Try a different approach or provide the final answer.",
                    progress_independent=True,
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

    def __init__(self, *, max_consecutive_same: int = 5,
                 argument_scoped_tools: tuple[str, ...] = ()) -> None:
        self._threshold = max_consecutive_same
        self._argument_scoped = frozenset(argument_scoped_tools)

    def check(self, history: List[Dict[str, Any]]) -> GuardrailViolation:
        recent_tools: List[tuple[str, str]] = []
        for msg in history:
            if msg.get("role") == "assistant":
                for tc in (msg.get("tool_calls") or []):
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    if name:
                        scope = ''
                        if name.removeprefix('gp_') in self._argument_scoped:
                            raw = fn.get('arguments', {})
                            try:
                                args = json.loads(raw) if isinstance(raw, str) else raw
                            except (TypeError, ValueError):
                                args = raw
                            scope = json.dumps(args, sort_keys=True, default=str, separators=(',', ':'))
                        recent_tools.append((name, scope))

        if len(recent_tools) < self._threshold:
            return GuardrailViolation(violated=False)

        tail = recent_tools[-self._threshold:]
        if len(set(tail)) == 1:
            return GuardrailViolation(
                violated=True,
                reason=f"Tool '{tail[0][0]}' used {self._threshold} times consecutively with the same operation scope",
                severity="warning",
                suggestion="Consider using a different tool or providing the answer directly.",
            )

        return GuardrailViolation(violated=False)

    def reset(self) -> None:
        pass


class TurnCapGuard:
    """Bound tool invocations within a single agent turn: soft cap, hard cap.

    Counts only the tool calls added *during the current turn*, not calls from
    prior turns that may be present in the conversation history.  A deferred
    baseline is captured on the first ``check()`` after ``reset()`` so that
    pre-existing calls in ``assembly.prior_turns`` are excluded.

    The engine must call ``reset()`` at each turn boundary (done inside
    ``_begin_turn_context``) and prime the baseline by calling ``check()``
    on the initial message list before any tool calls are executed.  The
    resulting halt is ``progress_independent`` because an unbounded turn is a
    resource hazard regardless of whether the task is making headway.

    Reaching the soft cap (``max_calls``) yields an ``extendable`` halt when
    the calls since the last checkpoint were productive -- mostly successful
    and mostly distinct -- so a long multi-step task (e.g. a physical
    manipulation sequence) is not cut off mid-way while a spinning one still
    stops. ``hard_max_calls`` is never extended; equal to ``max_calls`` (the
    default) it restores the flat-cap behaviour.
    """

    def __init__(
        self,
        *,
        max_calls: int = 50,
        hard_max_calls: Optional[int] = None,
        min_window_success: float = 0.5,
        min_window_distinct: float = 0.5,
    ) -> None:
        self._max_calls = max_calls
        self._hard_max_calls = max(max_calls, hard_max_calls or max_calls)
        self._min_window_success = min_window_success
        self._min_window_distinct = min_window_distinct
        self._cap = max_calls
        # Deferred baseline: set on the first check() after reset() to the
        # number of pre-existing tool calls in the conversation history.
        self._baseline: Optional[int] = None
        # Absolute call index where the window judged at the next cap starts.
        self._window_start: Optional[int] = None
        self._last_total = 0

    @property
    def cap(self) -> int:
        """The ceiling currently in force for this turn."""
        return self._cap

    @staticmethod
    def _call_records(
        history: List[Dict[str, Any]],
    ) -> tuple[List[tuple[str, str, Any]], Dict[str, Any]]:
        """Correlate native records and text-mode call/result pairs without mutation."""
        from leapflow.skills.tool_call_parser import parse_tool_call

        calls: List[tuple[str, str, Any]] = []
        results: Dict[str, Any] = {}
        pending_text_id: str | None = None
        for index, msg in enumerate(history):
            content = msg.get("content", "")
            if msg.get("role") == "assistant":
                pending_text_id = None
                native_calls = msg.get("tool_calls") or []
                for tc in native_calls:
                    fn = tc.get("function", {}) or {}
                    raw = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw) if isinstance(raw, str) else raw
                    except (TypeError, ValueError):
                        args = raw
                    calls.append((str(tc.get("id", "")), str(fn.get("name", "")), args))
                if not native_calls and isinstance(content, str):
                    text_call = parse_tool_call(content)
                    if text_call is not None:
                        pending_text_id = f"text-history-{index}"
                        calls.append((pending_text_id, text_call.name, text_call.params))
                continue
            if msg.get("role") == "tool":
                call_id = str(msg.get("tool_call_id", ""))
            elif pending_text_id is not None and StagnationGuard._is_tool_result_message(msg):
                call_id = pending_text_id
                content = content.partition("\n")[2]
                pending_text_id = None
            else:
                continue
            try:
                results[call_id] = json.loads(content) if isinstance(content, str) else content
            except (TypeError, ValueError):
                results[call_id] = content
        return calls, results

    @classmethod
    def _count_calls(cls, history: List[Dict[str, Any]]) -> int:
        return len(cls._call_records(history)[0])

    def check(self, history: List[Dict[str, Any]]) -> GuardrailViolation:
        total = self._count_calls(history)
        self._last_total = total
        if self._baseline is None:
            # First check this turn: snapshot the count of pre-existing calls
            # so only calls added after this point count towards the cap.
            self._baseline = total
            self._window_start = total
            return GuardrailViolation(violated=False)
        calls_this_turn = total - self._baseline
        if calls_this_turn >= self._cap:
            extendable = (
                calls_this_turn < self._hard_max_calls
                and self._cap < self._hard_max_calls
                and self._window_productive(history)
            )
            limit = self._hard_max_calls if calls_this_turn >= self._hard_max_calls else self._cap
            return GuardrailViolation(
                violated=True,
                severity="halt",
                progress_independent=True,
                extendable=extendable,
                reason=(
                    f"Turn tool-call cap reached ({calls_this_turn}/{limit})"
                ),
                suggestion=(
                    "Provide the best answer with the information gathered so far."
                ),
            )
        return GuardrailViolation(violated=False)

    def extend(self, step: int) -> int:
        """Raise the current cap by ``step`` (bounded by the hard cap).

        The calls made so far become the closed window; the next cap is judged
        only on calls made after this point, so one productive stretch cannot
        carry a later spinning one.
        """
        reached = max(self._cap, self._last_total - (self._baseline or 0))
        self._cap = min(self._hard_max_calls, reached + max(1, int(step)))
        self._window_start = self._last_total
        return self._cap

    def _window_productive(self, history: List[Dict[str, Any]]) -> bool:
        """Whether the calls since the last checkpoint mostly succeeded and differed."""
        calls, results = self._call_records(history)
        window = calls[self._window_start or 0:]
        if not window:
            return False
        succeeded = sum(
            1 for call_id, _, _ in window
            if call_id in results
            and not (isinstance(results[call_id], dict) and results[call_id].get("ok") is False)
            and result_tier(results[call_id]) in (TIER_SUCCEEDED, TIER_REPAIRED)
        )
        distinct = len({
            (name, json.dumps(args, sort_keys=True, default=str, separators=(",", ":")))
            for _, name, args in window
        })
        return (
            succeeded / len(window) >= self._min_window_success
            and distinct / len(window) >= self._min_window_distinct
        )

    def reset(self) -> None:
        """Clear the per-turn baseline and restore the soft cap."""
        self._baseline = None
        self._window_start = None
        self._last_total = 0
        self._cap = self._max_calls


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
        argument_scoped_tools: tuple[str, ...] = (),
        max_calls_per_turn: int = 50,
        max_calls_per_turn_hard: Optional[int] = None,
    ) -> None:
        self._guards: List[ToolLoopGuard] = guards or [
            RepetitionGuard(max_repeats=max_repeats),
            StagnationGuard(window=stagnation_window, min_success_rate=min_success_rate),
            DominationGuard(max_consecutive_same=max_consecutive_same, argument_scoped_tools=argument_scoped_tools),
            TurnCapGuard(max_calls=max_calls_per_turn, hard_max_calls=max_calls_per_turn_hard),
        ]

    def fresh(self) -> "CompositeGuardrail":
        """Return an independent copy: same configuration, cleared state.

        Guards carry per-turn state (e.g. ``TurnCapGuard``'s baseline), so each
        session engine needs its own instance; sharing one lets a turn starting
        in one session reset the baseline another session is counting against.
        """
        guards: List[ToolLoopGuard] = [copy.deepcopy(guard) for guard in self._guards]
        for guard in guards:
            guard.reset()
        return CompositeGuardrail(guards=guards)

    def extend_turn_cap(self, step: int) -> Optional[int]:
        """Raise the turn tool-call cap; returns the new cap, or None if absent."""
        for guard in self._guards:
            if isinstance(guard, TurnCapGuard):
                return guard.extend(step)
        return None

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
