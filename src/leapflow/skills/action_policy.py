"""Tool-level action policy — human-in-the-loop gate for the ReAct executor.

Intercepts tool calls between LLM output parsing and platform execution,
classifying risk via composable rules and pausing for human approval when
the verdict is ASK.

Architecture:
    ToolCall → PolicyEngine.evaluate(rules) → Verdict
                                                ↓ ASK
                                           IOProvider.prompt → allow/deny

Design:
    - Open/Closed: new rules = new class, no engine modification
    - Zero-cost when disabled: None policy → direct dispatch
    - Data-driven: rules inspect tool params via patterns, not hardcoded names
    - i18n-aware: send-detection includes CJK labels
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from leapflow.skills.tool_executor import ToolCall


class Verdict(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyDecision:
    verdict: Verdict
    reason: str = ""
    tool_name: str = ""
    summary: str = ""


@dataclass
class PolicyContext:
    skill_name: str = ""
    iteration: int = 0
    history: List[str] = field(default_factory=list)


@runtime_checkable
class PolicyRule(Protocol):
    def check(self, call: ToolCall, context: PolicyContext) -> Optional[PolicyDecision]: ...


class PolicyEngine:
    """Evaluates composable rules. First ASK/DENY wins; all None → ALLOW."""

    def __init__(self, rules: Optional[List[PolicyRule]] = None) -> None:
        self._rules: List[PolicyRule] = rules or []

    def add_rule(self, rule: PolicyRule) -> None:
        self._rules.append(rule)

    async def evaluate(self, call: ToolCall, context: PolicyContext) -> PolicyDecision:
        for rule in self._rules:
            decision = rule.check(call, context)
            if decision is not None:
                return decision
        return PolicyDecision(verdict=Verdict.ALLOW)


# ═══════════════════════════════════════════════════════════════════════
# Built-in rules
# ═══════════════════════════════════════════════════════════════════════

_SEND_LABELS = re.compile(
    r"(?i)\b(send|submit|post|publish|confirm|reply|forward)\b"
    r"|发送|提交|确认|回复|转发"
)

_SEND_SHORTCUTS = frozenset({
    "enter", "return", "cmd+enter", "ctrl+enter",
    "cmd+return", "ctrl+return",
})


def _normalize_keys(keys: str) -> str:
    """Normalize shortcut string: lowercase, remove spaces around '+'."""
    return "+".join(p.strip() for p in keys.lower().split("+"))


class SendActionRule:
    """Detects "send/submit" semantics — clicking send buttons or pressing enter."""

    def check(self, call: ToolCall, context: PolicyContext) -> Optional[PolicyDecision]:
        if call.name == "click":
            selector = call.params.get("selector", "")
            if _SEND_LABELS.search(selector):
                return PolicyDecision(
                    verdict=Verdict.ASK,
                    reason="send_action",
                    tool_name=call.name,
                    summary=f"Click: {selector}",
                )

        if call.name == "shortcut":
            keys = _normalize_keys(call.params.get("keys", ""))
            if keys in _SEND_SHORTCUTS:
                return PolicyDecision(
                    verdict=Verdict.ASK,
                    reason="send_shortcut",
                    tool_name=call.name,
                    summary=f"Shortcut: {keys}",
                )

        return None


_CONTENT_MIN_LENGTH = 10


class ContentInputRule:
    """Flags text input that will be visible to others (non-trivial content)."""

    def __init__(self, min_length: int = _CONTENT_MIN_LENGTH) -> None:
        self._min_length = min_length

    def check(self, call: ToolCall, context: PolicyContext) -> Optional[PolicyDecision]:
        if call.name != "type_text":
            return None

        text = call.params.get("text", "")
        if len(text) >= self._min_length:
            preview = text[:50] + ("..." if len(text) > 50 else "")
            return PolicyDecision(
                verdict=Verdict.ASK,
                reason="content_input",
                tool_name=call.name,
                summary=f"Type: \"{preview}\"",
            )
        return None


_DESTRUCTIVE_PATTERNS = re.compile(
    r"(?i)\b(rm\s+-r|rmdir|del\s+/|format\s|drop\s|truncate\s|"
    r"shutdown|reboot|kill\s+-9|pkill|dd\s+if=)"
)


class DestructiveShellRule:
    """Flags shell commands with destructive patterns."""

    def check(self, call: ToolCall, context: PolicyContext) -> Optional[PolicyDecision]:
        if call.name not in ("shell", "file_delete"):
            return None

        if call.name == "file_delete":
            path = call.params.get("path", "")
            return PolicyDecision(
                verdict=Verdict.ASK,
                reason="file_delete",
                tool_name=call.name,
                summary=f"Delete: {path}",
            )

        command = call.params.get("command", "")
        if _DESTRUCTIVE_PATTERNS.search(command):
            preview = command[:80] + ("..." if len(command) > 80 else "")
            return PolicyDecision(
                verdict=Verdict.ASK,
                reason="destructive_shell",
                tool_name=call.name,
                summary=f"Shell: {preview}",
            )
        return None


_NETWORK_PATTERNS = re.compile(
    r"(?i)\b(curl|wget|ssh|scp|rsync|nc\s|ncat|"
    r"python\s+-m\s+http|ngrok)\b"
)


class ExternalReachRule:
    """Flags actions that reach external systems (URLs, network commands)."""

    def check(self, call: ToolCall, context: PolicyContext) -> Optional[PolicyDecision]:
        if call.name == "open_url":
            url = call.params.get("url", "")
            return PolicyDecision(
                verdict=Verdict.ASK,
                reason="external_url",
                tool_name=call.name,
                summary=f"Open URL: {url}",
            )

        if call.name == "shell":
            command = call.params.get("command", "")
            if _NETWORK_PATTERNS.search(command):
                preview = command[:80] + ("..." if len(command) > 80 else "")
                return PolicyDecision(
                    verdict=Verdict.ASK,
                    reason="network_command",
                    tool_name=call.name,
                    summary=f"Shell (network): {preview}",
                )

        return None


def default_rules() -> List[PolicyRule]:
    """Standard rule set for non-AUTO confirmation levels."""
    return [
        SendActionRule(),
        ContentInputRule(),
        DestructiveShellRule(),
        ExternalReachRule(),
    ]
