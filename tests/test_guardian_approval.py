# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for Guardian LLM-assisted approval integration."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from leapflow.security.actions import ActionDescriptor, ActionEffect, ActionKind
from leapflow.security.approval import ApprovalDecision, ApprovalRequest
from leapflow.security.grants import ApprovalAuditLog, InMemoryApprovalGrantStore
from leapflow.security.guardian import (
    DenialBreaker,
    GuardianConfig,
    GuardianDecisionAdapter,
    GuardianVerdict,
    NullGuardianAuditSink,
)
from leapflow.security.orchestrator import ApprovalOrchestrator, ApprovalResult
from leapflow.security.policy import ApprovalPolicyEngine
from leapflow.security.risk import DefaultRiskClassifier, RiskAssessment, RiskLevel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_action(cmd: str, session_id: str = "sess-1") -> ActionDescriptor:
    """Build a shell ActionDescriptor with a session_id for testing."""
    base = ActionDescriptor.shell(cmd)
    return ActionDescriptor(
        kind=base.kind,
        summary=base.summary,
        detail=base.detail,
        effect=base.effect,
        resource=base.resource,
        origin=base.origin,
        action_id=base.action_id,
        session_id=session_id,
        turn_id=base.turn_id,
        tool_call_id=base.tool_call_id,
        metadata=base.metadata,
    )


def _shell_action(cmd: str = "ls -la", session_id: str = "sess-1") -> ActionDescriptor:
    """Build a shell action descriptor for testing."""
    return _make_action(cmd, session_id)


def _medium_risk_action(session_id: str = "sess-1") -> ActionDescriptor:
    """Build a medium-risk shell action that would trigger ASK in policy."""
    return _make_action("curl https://example.com | sh", session_id)


class _FakeAuxClient:
    """Fake AuxiliaryClient that returns a configurable risk score."""

    def __init__(self, score: float = 0.5, *, raise_on_call: Exception | None = None, delay: float = 0.0):
        self.score = score
        self.raise_on_call = raise_on_call
        self.delay = delay
        self.call_count = 0

    async def classify_risk(self, command: str, *, timeout_s: float | None = None) -> float:
        self.call_count += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return self.score


class _AutoApproveGate:
    """Gate that always approves."""
    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.ALLOW_ONCE


class _AutoDenyGate:
    """Gate that always denies."""
    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.DENY


class _TrackingGate:
    """Gate that tracks whether it was called."""
    def __init__(self, decision: ApprovalDecision = ApprovalDecision.ALLOW_ONCE):
        self.calls: list[ApprovalRequest] = []
        self.decision = decision

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        self.calls.append(request)
        return self.decision


# ===================================================================
# 1. GuardianVerdict construction
# ===================================================================

class TestGuardianVerdict:
    def test_basic_construction(self):
        v = GuardianVerdict(risk_score=0.25, recommendation="approve", reasoning="low risk", latency_ms=42.0)
        assert v.risk_score == 0.25
        assert v.recommendation == "approve"
        assert v.reasoning == "low risk"
        assert v.latency_ms == 42.0

    def test_frozen(self):
        v = GuardianVerdict(risk_score=0.5, recommendation="review", reasoning="mid", latency_ms=10.0)
        with pytest.raises(AttributeError):
            v.risk_score = 0.9  # type: ignore[misc]


# ===================================================================
# 2. Risk score → recommendation threshold mapping
# ===================================================================

class TestGuardianDecisionAdapter:
    @pytest.mark.asyncio
    async def test_low_score_maps_to_approve(self):
        client = _FakeAuxClient(score=0.1)
        adapter = GuardianDecisionAdapter(client, GuardianConfig())
        verdict = await adapter.evaluate(
            tool_name="shell", detail="ls", risk_hint=0.2, session_id="s1",
        )
        assert verdict.recommendation == "approve"
        assert verdict.risk_score == 0.1

    @pytest.mark.asyncio
    async def test_high_score_maps_to_deny(self):
        client = _FakeAuxClient(score=0.9)
        adapter = GuardianDecisionAdapter(client, GuardianConfig())
        verdict = await adapter.evaluate(
            tool_name="shell", detail="rm -rf /", risk_hint=0.9, session_id="s1",
        )
        assert verdict.recommendation == "deny"
        assert verdict.risk_score == 0.9

    @pytest.mark.asyncio
    async def test_mid_score_maps_to_review(self):
        client = _FakeAuxClient(score=0.5)
        adapter = GuardianDecisionAdapter(client, GuardianConfig())
        verdict = await adapter.evaluate(
            tool_name="shell", detail="pip install foo", risk_hint=0.5, session_id="s1",
        )
        assert verdict.recommendation == "review"

    @pytest.mark.asyncio
    async def test_boundary_approve(self):
        """Score exactly at threshold → approve."""
        client = _FakeAuxClient(score=0.3)
        adapter = GuardianDecisionAdapter(client, GuardianConfig(risk_threshold_auto_approve=0.3))
        verdict = await adapter.evaluate(
            tool_name="shell", detail="echo hi", risk_hint=0.1, session_id="s1",
        )
        assert verdict.recommendation == "approve"

    @pytest.mark.asyncio
    async def test_boundary_deny(self):
        """Score exactly at deny threshold → deny."""
        client = _FakeAuxClient(score=0.8)
        adapter = GuardianDecisionAdapter(client, GuardianConfig(risk_threshold_auto_deny=0.8))
        verdict = await adapter.evaluate(
            tool_name="shell", detail="rm -rf /tmp", risk_hint=0.8, session_id="s1",
        )
        assert verdict.recommendation == "deny"

    @pytest.mark.asyncio
    async def test_custom_thresholds(self):
        """Custom thresholds shift the mapping."""
        client = _FakeAuxClient(score=0.4)
        config = GuardianConfig(risk_threshold_auto_approve=0.5, risk_threshold_auto_deny=0.9)
        adapter = GuardianDecisionAdapter(client, config)
        verdict = await adapter.evaluate(
            tool_name="shell", detail="echo hi", risk_hint=0.2, session_id="s1",
        )
        assert verdict.recommendation == "approve"  # 0.4 <= 0.5

    @pytest.mark.asyncio
    async def test_timeout_returns_review(self):
        """LLM timeout degrades to 'review'."""
        client = _FakeAuxClient(score=0.1, delay=20.0)
        config = GuardianConfig(timeout_seconds=0.05)
        adapter = GuardianDecisionAdapter(client, config)
        verdict = await adapter.evaluate(
            tool_name="shell", detail="echo hi", risk_hint=0.2, session_id="s1",
        )
        assert verdict.recommendation == "review"
        assert "timed out" in verdict.reasoning.lower()

    @pytest.mark.asyncio
    async def test_error_returns_review(self):
        """LLM error degrades to 'review'."""
        client = _FakeAuxClient(raise_on_call=RuntimeError("model unavailable"))
        adapter = GuardianDecisionAdapter(client, GuardianConfig())
        verdict = await adapter.evaluate(
            tool_name="shell", detail="echo hi", risk_hint=0.2, session_id="s1",
        )
        assert verdict.recommendation == "review"
        assert "error" in verdict.reasoning.lower()

    @pytest.mark.asyncio
    async def test_audit_sink_called(self):
        """Audit sink receives the decision record."""
        client = _FakeAuxClient(score=0.2)
        sink = AsyncMock()
        adapter = GuardianDecisionAdapter(client, GuardianConfig(), audit_sink=sink)
        await adapter.evaluate(
            tool_name="shell", detail="echo test", risk_hint=0.1, session_id="s-audit",
        )
        sink.record_guardian_decision.assert_awaited_once()
        call_kwargs = sink.record_guardian_decision.call_args.kwargs
        assert call_kwargs["session_id"] == "s-audit"
        assert call_kwargs["tool_name"] == "shell"
        assert call_kwargs["risk_score"] == 0.2

    @pytest.mark.asyncio
    async def test_audit_sink_failure_does_not_break(self):
        """Audit write failure does not affect the verdict."""
        client = _FakeAuxClient(score=0.2)
        sink = AsyncMock(side_effect=RuntimeError("db gone"))
        adapter = GuardianDecisionAdapter(client, GuardianConfig(), audit_sink=sink)
        verdict = await adapter.evaluate(
            tool_name="shell", detail="echo test", risk_hint=0.1, session_id="s1",
        )
        assert verdict.recommendation == "approve"


# ===================================================================
# 3. DenialBreaker
# ===================================================================

class TestDenialBreaker:
    def test_not_tripped_initially(self):
        b = DenialBreaker(max_consecutive_denials=3)
        assert not b.is_tripped()

    def test_trips_after_max_denials(self):
        b = DenialBreaker(max_consecutive_denials=3)
        b.record_denial()
        assert not b.is_tripped()
        b.record_denial()
        assert not b.is_tripped()
        result = b.record_denial()
        assert result is True
        assert b.is_tripped()

    def test_approval_resets_counter(self):
        b = DenialBreaker(max_consecutive_denials=3)
        b.record_denial()
        b.record_denial()
        b.record_approval()
        b.record_denial()
        b.record_denial()
        assert not b.is_tripped()

    def test_approval_does_not_untrip(self):
        """Once tripped, stays tripped until explicit reset."""
        b = DenialBreaker(max_consecutive_denials=2)
        b.record_denial()
        b.record_denial()
        assert b.is_tripped()
        b.record_approval()
        assert b.is_tripped()  # still tripped

    def test_full_reset(self):
        b = DenialBreaker(max_consecutive_denials=2)
        b.record_denial()
        b.record_denial()
        assert b.is_tripped()
        b.reset()
        assert not b.is_tripped()

    def test_single_denial_limit(self):
        b = DenialBreaker(max_consecutive_denials=1)
        result = b.record_denial()
        assert result is True
        assert b.is_tripped()


# ===================================================================
# 4. GuardianConfig
# ===================================================================

class TestGuardianConfig:
    def test_defaults(self):
        cfg = GuardianConfig()
        assert cfg.mode == "hybrid"
        assert cfg.risk_threshold_auto_approve == 0.3
        assert cfg.risk_threshold_auto_deny == 0.8
        assert cfg.max_consecutive_denials == 3
        assert cfg.timeout_seconds == 10.0

    def test_frozen(self):
        cfg = GuardianConfig()
        with pytest.raises(AttributeError):
            cfg.mode = "static_only"  # type: ignore[misc]


# ===================================================================
# 5. Hybrid mode: static rules + LLM cooperation
# ===================================================================

class TestOrchestratorGuardianIntegration:
    @pytest.mark.asyncio
    async def test_guardian_auto_approve_skips_human(self):
        """Low LLM score → auto-approve without human prompt."""
        client = _FakeAuxClient(score=0.1)
        guardian = GuardianDecisionAdapter(client, GuardianConfig())
        gate = _TrackingGate()
        orch = ApprovalOrchestrator(gate, guardian=guardian)
        action = _medium_risk_action()
        result = await orch.evaluate(action)
        assert result.approved
        assert "guardian" in result.reason
        assert len(gate.calls) == 0  # human never prompted

    @pytest.mark.asyncio
    async def test_guardian_auto_deny_skips_human(self):
        """High LLM score → auto-deny without human prompt."""
        client = _FakeAuxClient(score=0.95)
        guardian = GuardianDecisionAdapter(client, GuardianConfig())
        gate = _TrackingGate()
        orch = ApprovalOrchestrator(gate, guardian=guardian)
        action = _medium_risk_action()
        result = await orch.evaluate(action)
        assert not result.approved
        assert "guardian" in result.reason
        assert len(gate.calls) == 0

    @pytest.mark.asyncio
    async def test_guardian_review_falls_through_to_human(self):
        """Mid LLM score → human prompt."""
        client = _FakeAuxClient(score=0.5)
        guardian = GuardianDecisionAdapter(client, GuardianConfig())
        gate = _TrackingGate(ApprovalDecision.ALLOW_ONCE)
        orch = ApprovalOrchestrator(gate, guardian=guardian)
        action = _medium_risk_action()
        result = await orch.evaluate(action)
        assert result.approved
        assert len(gate.calls) == 1  # human was prompted

    @pytest.mark.asyncio
    async def test_static_only_mode_ignores_guardian(self):
        """static_only mode: Guardian never fires."""
        client = _FakeAuxClient(score=0.1)  # would auto-approve
        config = GuardianConfig(mode="static_only")
        guardian = GuardianDecisionAdapter(client, config)
        gate = _TrackingGate(ApprovalDecision.DENY)
        orch = ApprovalOrchestrator(gate, guardian=guardian, guardian_config=config)
        action = _medium_risk_action()
        result = await orch.evaluate(action)
        # Should fall through to human (which denies)
        assert not result.approved
        assert len(gate.calls) == 1
        assert client.call_count == 0

    @pytest.mark.asyncio
    async def test_hybrid_mode_degrades_on_timeout(self):
        """hybrid mode: LLM timeout → falls through to human."""
        client = _FakeAuxClient(score=0.1, delay=20.0)
        config = GuardianConfig(mode="hybrid", timeout_seconds=0.05)
        guardian = GuardianDecisionAdapter(client, config)
        gate = _TrackingGate(ApprovalDecision.ALLOW_ONCE)
        orch = ApprovalOrchestrator(gate, guardian=guardian, guardian_config=config)
        action = _medium_risk_action()
        result = await orch.evaluate(action)
        assert result.approved
        assert len(gate.calls) == 1  # human prompted as fallback

    @pytest.mark.asyncio
    async def test_llm_assisted_mode_degrades_on_error(self):
        """llm_assisted mode: LLM error → falls through to human."""
        client = _FakeAuxClient(raise_on_call=RuntimeError("boom"))
        config = GuardianConfig(mode="llm_assisted")
        guardian = GuardianDecisionAdapter(client, config)
        gate = _TrackingGate(ApprovalDecision.ALLOW_ONCE)
        orch = ApprovalOrchestrator(gate, guardian=guardian, guardian_config=config)
        action = _medium_risk_action()
        result = await orch.evaluate(action)
        assert result.approved
        assert len(gate.calls) == 1

    @pytest.mark.asyncio
    async def test_no_guardian_works_as_before(self):
        """No guardian injected → orchestrator behaves identically to before."""
        gate = _TrackingGate(ApprovalDecision.ALLOW_ONCE)
        orch = ApprovalOrchestrator(gate)
        action = _medium_risk_action()
        result = await orch.evaluate(action)
        assert result.approved
        assert len(gate.calls) == 1

    @pytest.mark.asyncio
    async def test_policy_allow_bypasses_guardian(self):
        """Low-risk action auto-allowed by policy never reaches Guardian."""
        client = _FakeAuxClient(score=0.9)  # would deny if called
        guardian = GuardianDecisionAdapter(client, GuardianConfig())
        gate = _TrackingGate()
        orch = ApprovalOrchestrator(gate, guardian=guardian)
        # low risk action
        action = _shell_action("echo hello")
        result = await orch.evaluate(action)
        assert result.approved
        assert client.call_count == 0

    @pytest.mark.asyncio
    async def test_hardline_deny_bypasses_guardian(self):
        """Hardline/CRITICAL action denied by policy never reaches Guardian."""
        client = _FakeAuxClient(score=0.01)  # would approve if called
        guardian = GuardianDecisionAdapter(client, GuardianConfig())
        gate = _TrackingGate()
        orch = ApprovalOrchestrator(gate, guardian=guardian)
        action = _shell_action("rm -rf /")
        result = await orch.evaluate(action)
        assert not result.approved
        assert client.call_count == 0


# ===================================================================
# 6. DenialBreaker in orchestrator
# ===================================================================

class TestOrchestratorDenialBreaker:
    @pytest.mark.asyncio
    async def test_breaker_trips_after_consecutive_denials(self):
        """After N consecutive denials, breaker fast-denies without prompt."""
        config = GuardianConfig(max_consecutive_denials=2)
        gate = _TrackingGate(ApprovalDecision.DENY)
        orch = ApprovalOrchestrator(gate, guardian_config=config)
        action = _medium_risk_action()

        # First denial
        r1 = await orch.evaluate(action)
        assert not r1.approved
        assert len(gate.calls) == 1

        # Second denial (trips breaker)
        r2 = await orch.evaluate(action)
        assert not r2.approved
        assert len(gate.calls) == 2

        # Third — breaker kicks in, no prompt
        r3 = await orch.evaluate(action)
        assert not r3.approved
        assert r3.reason == "consecutive denial limit reached"
        assert len(gate.calls) == 2  # gate not called again

    @pytest.mark.asyncio
    async def test_breaker_reset_between_turns(self):
        config = GuardianConfig(max_consecutive_denials=2)
        gate = _TrackingGate(ApprovalDecision.DENY)
        orch = ApprovalOrchestrator(gate, guardian_config=config)
        action = _medium_risk_action()

        await orch.evaluate(action)
        await orch.evaluate(action)
        # breaker tripped
        assert orch.denial_breaker.is_tripped()

        orch.reset_turn()
        assert not orch.denial_breaker.is_tripped()


# ===================================================================
# 7. Audit table write (DuckDB)
# ===================================================================

class TestAuditTableWrite:
    @pytest.mark.asyncio
    async def test_audit_record_written(self):
        """Guardian writes to the audit sink on each evaluation."""
        sink = AsyncMock()
        client = _FakeAuxClient(score=0.2)
        adapter = GuardianDecisionAdapter(client, GuardianConfig(), audit_sink=sink)
        await adapter.evaluate(
            tool_name="shell_run",
            detail="echo hello",
            risk_hint=0.15,
            session_id="s-audit-1",
        )
        sink.record_guardian_decision.assert_awaited_once()
        kwargs = sink.record_guardian_decision.call_args.kwargs
        assert kwargs["session_id"] == "s-audit-1"
        assert kwargs["tool_name"] == "shell_run"
        assert kwargs["risk_score"] == 0.2
        assert kwargs["recommendation"] == "approve"


# ===================================================================
# 8. Config-driven mode switching
# ===================================================================

class TestConfigModeSwitching:
    @pytest.mark.asyncio
    async def test_static_only_never_calls_llm(self):
        client = _FakeAuxClient(score=0.1)
        config = GuardianConfig(mode="static_only")
        guardian = GuardianDecisionAdapter(client, config)
        gate = _TrackingGate(ApprovalDecision.ALLOW_ONCE)
        orch = ApprovalOrchestrator(gate, guardian=guardian, guardian_config=config)
        await orch.evaluate(_medium_risk_action())
        assert client.call_count == 0

    @pytest.mark.asyncio
    async def test_llm_assisted_calls_llm(self):
        client = _FakeAuxClient(score=0.1)
        config = GuardianConfig(mode="llm_assisted")
        guardian = GuardianDecisionAdapter(client, config)
        gate = _TrackingGate()
        orch = ApprovalOrchestrator(gate, guardian=guardian, guardian_config=config)
        result = await orch.evaluate(_medium_risk_action())
        assert result.approved
        assert client.call_count == 1
        assert len(gate.calls) == 0  # auto-approved by guardian

    @pytest.mark.asyncio
    async def test_hybrid_calls_llm(self):
        client = _FakeAuxClient(score=0.1)
        config = GuardianConfig(mode="hybrid")
        guardian = GuardianDecisionAdapter(client, config)
        gate = _TrackingGate()
        orch = ApprovalOrchestrator(gate, guardian=guardian, guardian_config=config)
        result = await orch.evaluate(_medium_risk_action())
        assert result.approved
        assert client.call_count == 1
        assert len(gate.calls) == 0


# ===================================================================
# 9. Schema migration (approval_decisions table)
# ===================================================================

class TestApprovalDecisionsSchema:
    def test_migration_registered(self):
        from leapflow.storage.schema import CURRENT_SCHEMA_VERSION, MIGRATIONS
        assert CURRENT_SCHEMA_VERSION == 10
        m10 = [m for m in MIGRATIONS if m.version == 10]
        assert len(m10) == 1
        assert "guardian" in m10[0].name.lower() or "approval" in m10[0].name.lower()

    def test_migration_idempotent(self):
        """The migration can run twice without error."""
        import duckdb
        conn = duckdb.connect(":memory:")
        from leapflow.storage.schema import MIGRATIONS
        m10 = [m for m in MIGRATIONS if m.version == 10][0]
        m10.apply(conn)
        m10.apply(conn)  # idempotent
        # verify table exists
        result = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'approval_decisions' ORDER BY ordinal_position"
        ).fetchall()
        columns = [r[0] for r in result]
        assert "session_id" in columns
        assert "tool_name" in columns
        assert "risk_score" in columns
        assert "recommendation" in columns
        assert "decision" in columns
        assert "reasoning" in columns
        conn.close()


# ===================================================================
# 10. NullGuardianAuditSink
# ===================================================================

class TestNullAuditSink:
    @pytest.mark.asyncio
    async def test_no_op(self):
        sink = NullGuardianAuditSink()
        # should not raise
        await sink.record_guardian_decision(
            session_id="s", tool_name="t", risk_score=0.0,
            recommendation="approve", decision="approve",
            reasoning="ok", latency_ms=1.0, metadata={},
        )
