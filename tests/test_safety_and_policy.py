"""Scenario-based tests for safety: confirmation levels, action policy, sandbox."""

from __future__ import annotations

from typing import Optional

import pytest

from conftest import make_skill
from leapflow.engine.confirmation import ConfirmationHandler, ConfirmLevel
from leapflow.skills.action_policy import (
    PolicyContext,
    PolicyDecision,
    PolicyEngine,
    Verdict,
    default_rules,
)
from leapflow.skills.sandbox import SandboxedNamespace
from leapflow.skills.tool_executor import ToolCall


# ═══════════════════════════════════════════════════════════════════
# Confirmation levels
# ═══════════════════════════════════════════════════════════════════


def test_confirm_level_v1_requires_step():
    handler = ConfirmationHandler()
    skill = make_skill(version=1, confidence=0.9)
    assert handler.determine_level(skill) == ConfirmLevel.STEP


def test_confirm_level_low_confidence_requires_step():
    handler = ConfirmationHandler()
    skill = make_skill(version=3, confidence=0.5)
    assert handler.determine_level(skill) == ConfirmLevel.STEP


def test_confirm_level_destructive_requires_confirm():
    handler = ConfirmationHandler()
    skill = make_skill(
        version=2,
        confidence=0.7,
        description="delete files from disk",
    )
    assert handler.determine_level(skill) == ConfirmLevel.CONFIRM


def test_confirm_level_mature_skill_auto():
    handler = ConfirmationHandler()
    skill = make_skill(version=3, confidence=0.9)
    assert handler.determine_level(skill) == ConfirmLevel.AUTO


def test_confirm_level_override_wins():
    handler = ConfirmationHandler()
    skill = make_skill(version=3, confidence=0.9)
    assert (
        handler.determine_level(skill, override=ConfirmLevel.STEP)
        == ConfirmLevel.STEP
    )


# ═══════════════════════════════════════════════════════════════════
# Action policy
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_policy_safe_action_allowed():
    engine = PolicyEngine(default_rules())
    ctx = PolicyContext(skill_name="test", iteration=0)
    call = ToolCall(name="observe_ui", params={})
    decision = await engine.evaluate(call, ctx)
    assert decision.verdict == Verdict.ALLOW


@pytest.mark.asyncio
async def test_policy_send_action_requires_ask():
    engine = PolicyEngine(default_rules())
    ctx = PolicyContext(skill_name="test", iteration=0)
    call = ToolCall(name="click", params={"selector": "AXButton[label=Send]"})
    decision = await engine.evaluate(call, ctx)
    assert decision.verdict == Verdict.ASK
    assert decision.reason == "send_action"


class _DenyAllRule:
    def check(self, call: ToolCall, context: PolicyContext) -> Optional[PolicyDecision]:
        return PolicyDecision(verdict=Verdict.DENY, reason="blocked")


@pytest.mark.asyncio
async def test_policy_custom_rule_injection():
    engine = PolicyEngine(default_rules())
    engine.add_rule(_DenyAllRule())
    ctx = PolicyContext(skill_name="test", iteration=0)
    call = ToolCall(name="observe_ui", params={})
    decision = await engine.evaluate(call, ctx)
    assert decision.verdict == Verdict.DENY
    assert decision.reason == "blocked"


# ═══════════════════════════════════════════════════════════════════
# Sandboxed skill execution
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_sandbox_safe_execution():
    code = """
async def greet(execution, perception, **params):
    name = params.get("name", "world")
    return {"ok": True, "result": f"Hello, {name}!"}
"""
    fn = SandboxedNamespace.compile_skill(code, "greet")
    result = await fn(None, None, name="Alice")
    assert result == {"ok": True, "result": "Hello, Alice!"}


def test_sandbox_blocks_dangerous_builtins():
    ns = SandboxedNamespace.create()
    builtins = ns["__builtins__"]
    for name in ("open", "eval", "exec", "compile", "getattr", "setattr", "delattr"):
        assert name not in builtins


@pytest.mark.asyncio
async def test_sandbox_complex_skill_with_json():
    code = """
async def summarize(execution, perception, **params):
    items = params.get("items", [])
    return {"ok": True, "payload": json.dumps({"count": len(items), "items": items})}
"""
    fn = SandboxedNamespace.compile_skill(code, "summarize")
    result = await fn(None, None, items=["a", "b", "c"])
    assert result["ok"] is True
    assert '"count": 3' in result["payload"]
    assert '"items": ["a", "b", "c"]' in result["payload"]
