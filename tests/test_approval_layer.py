from __future__ import annotations

import builtins
from pathlib import Path
import sys

import pytest

from leapflow.security.actions import ActionDescriptor
from leapflow.security.approval import ApprovalDecision, ApprovalRequest, SessionAwareGate
from leapflow.security.grants import ApprovalAuditLog, ApprovalGrant, ApprovalScope, JsonApprovalGrantStore, grant_key
from leapflow.security.orchestrator import ApprovalOrchestrator
from leapflow.security.risk import DefaultRiskClassifier, RiskAssessment, RiskLevel


class _Gate:
    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.requests = []

    async def request_approval(self, request):
        self.requests.append(request)
        return self.decision


@pytest.mark.asyncio
async def test_orchestrator_prompts_once_then_reuses_session_grant(tmp_path: Path) -> None:
    gate = _Gate(ApprovalDecision.ALLOW_SESSION)
    grants = JsonApprovalGrantStore(tmp_path / "grants.json")
    audit = ApprovalAuditLog(tmp_path / "audit.jsonl")
    orchestrator = ApprovalOrchestrator(gate, grants=grants, audit=audit)
    action = ActionDescriptor.shell("python << 'EOF'\nprint('hello')\nEOF")

    first = await orchestrator.evaluate(action)
    second = await orchestrator.evaluate(action)

    assert first.approved is True
    assert second.approved is True
    assert len(gate.requests) == 1
    assert grants.list()
    assert [entry["actor"] for entry in audit.entries] == ["user", "grant"]


@pytest.mark.asyncio
async def test_orchestrator_hardline_denies_without_prompt() -> None:
    gate = _Gate(ApprovalDecision.ALLOW_ONCE)
    orchestrator = ApprovalOrchestrator(gate)

    result = await orchestrator.evaluate(ActionDescriptor.shell("sudo reboot"))

    assert result.approved is False
    assert "hardline" in result.reason or result.risk.level == RiskLevel.CRITICAL
    assert not gate.requests


def test_default_risk_classifier_detects_heredoc() -> None:
    risk = DefaultRiskClassifier().assess(
        ActionDescriptor.shell("python << 'EOF'\nprint('install')\nEOF"),
    )

    assert risk.level == RiskLevel.HIGH
    assert "script_execution_via_heredoc" in risk.reasons
    assert risk.allow_permanent is False


def test_platform_action_risk_uses_registered_metadata() -> None:
    action = ActionDescriptor.platform_action(
        "feishu",
        "mail.search_unread",
        {"query": "urgent"},
        backend_kind="cli",
        metadata={"effect": "read", "risk_level": "high"},
    )

    risk = DefaultRiskClassifier().assess(action)

    assert risk.level == RiskLevel.HIGH
    assert risk.reasons == ("registered_platform_action",)
    assert risk.allow_permanent is False
    assert risk.metadata["backend_kind"] == "cli"


def test_approval_request_round_trips_request_id() -> None:
    from leapflow.security.approval import ApprovalRequest

    request = ApprovalRequest(
        category="shell.command",
        detail="echo hello",
        request_id="approval-1",
    )

    restored = ApprovalRequest.from_dict(request.to_dict())

    assert restored.request_id == "approval-1"
    assert restored.to_dict()["request_id"] == "approval-1"


@pytest.mark.asyncio
async def test_orchestrator_reuses_turn_grant(tmp_path: Path) -> None:
    gate = _Gate(ApprovalDecision.DENY)
    grants = JsonApprovalGrantStore(tmp_path / "grants.json")
    action = ActionDescriptor.shell("sudo ls", metadata={"test": True})
    action = ActionDescriptor.from_dict({**action.to_dict(), "session_id": "sess", "turn_id": "turn"})
    grants.put(ApprovalGrant(
        key=grant_key(action, ApprovalScope.TURN),
        scope=ApprovalScope.TURN.value,
        decision="allow",
        action_kind=action.kind,
        effect=action.effect,
        resource=action.resource,
        reason="turn_approved",
    ))
    orchestrator = ApprovalOrchestrator(gate, grants=grants)

    result = await orchestrator.evaluate(action)

    assert result.approved is True
    assert result.scope == ApprovalScope.TURN.value
    assert not gate.requests


@pytest.mark.asyncio
async def test_prompt_approval_waits_without_a_deadline(monkeypatch) -> None:
    """The prompt has no expiry, so a slow user is not auto-denied.

    ``ApprovalRequest`` deliberately carries no ``expires_at``; the previous
    design defaulted to Deny after 120s, refusing an action the user had not yet
    seen. The answer is now awaited with no deadline wrapped around it.
    """
    from leapflow.cli.approval_view import prompt_approval
    from leapflow.security.approval import ApprovalRequest

    assert not hasattr(ApprovalRequest(category="c", detail="d"), "expires_at")

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "allow_once")
    request = ApprovalRequest(category="shell.command", detail="echo hello")

    assert await prompt_approval(request) == ApprovalDecision.ALLOW_ONCE


@pytest.mark.asyncio
async def test_prompt_approval_denies_when_stdin_is_not_a_tty(monkeypatch) -> None:
    """Without a human to ask, denying beats blocking forever."""
    from leapflow.cli.approval_view import prompt_approval
    from leapflow.security.approval import ApprovalRequest

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    request = ApprovalRequest(category="shell.command", detail="echo hello")

    assert await prompt_approval(request) == ApprovalDecision.DENY


@pytest.mark.asyncio
async def test_prompt_approval_uses_plain_fallback_prompt(monkeypatch) -> None:
    from leapflow.cli import approval_view
    from leapflow.cli.approval_view import prompt_approval
    from leapflow.security.approval import ApprovalRequest

    prompts: list[str] = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(approval_view, "_render", lambda *_args, **_kwargs: None)

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "n"

    monkeypatch.setattr(builtins, "input", fake_input)

    request = ApprovalRequest(category="shell.command", detail="echo hello")

    assert await prompt_approval(request) == ApprovalDecision.DENY
    assert prompts == ["Select approval choice: "]


@pytest.mark.asyncio
async def test_orchestrator_persists_deny_always_as_session_grant(tmp_path: Path) -> None:
    gate = _Gate(ApprovalDecision.DENY_ALWAYS)
    grants = JsonApprovalGrantStore(tmp_path / "grants.json")
    audit = ApprovalAuditLog(tmp_path / "audit.jsonl")
    orchestrator = ApprovalOrchestrator(gate, grants=grants, audit=audit)
    action = ActionDescriptor.shell("python << 'EOF'\nprint('blocked')\nEOF")

    first = await orchestrator.evaluate(action)
    second = await orchestrator.evaluate(action)

    assert first.approved is False
    assert first.scope == ApprovalScope.SESSION.value
    assert second.approved is False
    assert second.reason == "user_denied"
    assert len(gate.requests) == 1
    assert [entry["actor"] for entry in audit.entries] == ["user", "grant"]
    assert [entry["scope"] for entry in audit.entries] == [
        ApprovalScope.SESSION.value,
        ApprovalScope.ONCE.value,
    ]


@pytest.mark.asyncio
async def test_orchestrator_cancel_workflow_is_denied_with_strong_message() -> None:
    gate = _Gate(ApprovalDecision.CANCEL_WORKFLOW)
    orchestrator = ApprovalOrchestrator(gate)

    result = await orchestrator.evaluate(
        ActionDescriptor.shell("python << 'EOF'\nprint('stop')\nEOF"),
    )

    assert result.approved is False
    assert result.reason == ApprovalDecision.CANCEL_WORKFLOW.value
    assert "Do not retry" in result.denial_message


@pytest.mark.asyncio
async def test_file_write_returns_gate_denial_message(tmp_path: Path) -> None:
    from leapflow.tools.file_operations import file_write
    from leapflow.plugins import get_registry
    _tool_reg = get_registry()

    class DenyingGate:
        denial_message = "BLOCKED: User denied this action. Do not retry."

        async def check(
            self,
            path: str,
            content: str,
            mode: str = "overwrite",
            sensitivity_meta: dict | None = None,
        ) -> bool:
            return False

    _tool_reg.set_file_write_gate(DenyingGate())
    try:
        result = await file_write({
            "path": str(tmp_path / "approval-output.py"),
            "content": "print('hello')",
        })
    finally:
        _tool_reg.set_file_write_gate(None)

    assert result == {
        "ok": False,
        "error": "BLOCKED: User denied this action. Do not retry.",
    }


def test_default_risk_classifier_detects_sensitive_file_read() -> None:
    risk = DefaultRiskClassifier().assess(
        ActionDescriptor.file_read(
            "/Users/example/.leapflow/.env",
            metadata={"sensitivity_category": "credential"},
        ),
    )

    assert risk.level == RiskLevel.HIGH
    assert risk.reasons == ("credential_file_read",)
    assert risk.allow_permanent is False


@pytest.mark.asyncio
async def test_sensitive_file_read_requires_approval_without_gate(tmp_path: Path) -> None:
    from leapflow.tools.file_operations import file_read
    from leapflow.plugins import get_registry
    _tool_reg = get_registry()

    target = tmp_path / ".env"
    target.write_text("API_KEY=sk-secret-value-123456\n", encoding="utf-8")
    _tool_reg.set_file_read_gate(None)

    result = await file_read({"path": str(target)})

    assert result["ok"] is False
    assert result["requires_approval"] is True
    assert result["sensitivity_category"] == "credential"


@pytest.mark.asyncio
async def test_sensitive_file_read_approval_redacts_content(tmp_path: Path) -> None:
    from leapflow.tools.file_operations import file_read
    from leapflow.plugins import get_registry
    _tool_reg = get_registry()

    class AllowingReadGate:
        denial_message = ""

        def __init__(self) -> None:
            self.calls = []

        async def check(
            self,
            path: str,
            mode: str = "raw",
            sensitivity_meta: dict | None = None,
        ) -> bool:
            self.calls.append((path, mode, dict(sensitivity_meta or {})))
            return True

    target = tmp_path / ".env"
    target.write_text("API_KEY=sk-secret-value-123456\nPUBLIC_VALUE=ok\n", encoding="utf-8")
    gate = AllowingReadGate()
    _tool_reg.set_file_read_gate(gate)
    try:
        result = await file_read({"path": str(target)})
    finally:
        _tool_reg.set_file_read_gate(None)

    assert result["ok"] is True
    assert gate.calls[0][2]["sensitivity_category"] == "credential"
    assert "sk-secret-value" not in result["content"]
    assert "«redacted:" in result["content"]
    assert result["redact_on_read"] is True


@pytest.mark.asyncio
async def test_sensitive_file_write_uses_approval_gate(tmp_path: Path) -> None:
    from leapflow.tools.file_operations import file_write
    from leapflow.plugins import get_registry
    _tool_reg = get_registry()

    class AllowingWriteGate:
        denial_message = ""

        def __init__(self) -> None:
            self.calls = []

        async def check(
            self,
            path: str,
            content: str,
            mode: str = "overwrite",
            sensitivity_meta: dict | None = None,
        ) -> bool:
            self.calls.append((path, content, mode, dict(sensitivity_meta or {})))
            return True

    target = tmp_path / ".env"
    gate = AllowingWriteGate()
    _tool_reg.set_file_write_gate(gate)
    try:
        result = await file_write({"path": str(target), "content": "API_KEY=sk-new-value-123456\n"})
    finally:
        _tool_reg.set_file_write_gate(None)

    assert result["ok"] is True
    assert target.read_text(encoding="utf-8") == "API_KEY=sk-new-value-123456\n"
    assert gate.calls[0][3]["sensitivity_category"] == "credential"


@pytest.mark.asyncio
async def test_runtime_database_read_is_hardline_blocked(tmp_path: Path) -> None:
    from leapflow.tools.file_operations import file_read
    from leapflow.plugins import get_registry
    _tool_reg = get_registry()

    class FailingGate:
        async def check(self, *_args, **_kwargs) -> bool:
            raise AssertionError("runtime database reads must not request approval")

    target = tmp_path / "leap.duckdb"
    target.write_bytes(b"not text")
    _tool_reg.set_file_read_gate(FailingGate())
    try:
        result = await file_read({"path": str(target)})
    finally:
        _tool_reg.set_file_read_gate(None)

    assert result["ok"] is False
    assert "Runtime database" in result["error"]


# ════════════════════════════════════════════════════════════════
# _bypass_all session bypass: security hardening (issue #30)
# ════════════════════════════════════════════════════════════════


def _high_risk_no_permanent() -> RiskAssessment:
    """A risk assessment representing HIGH + allow_permanent=False."""
    return RiskAssessment(
        level=RiskLevel.HIGH,
        score=0.78,
        reasons=("agent_self_modification",),
        explanation="plugin management action",
        allow_permanent=False,
    )


def _medium_risk_permanent() -> RiskAssessment:
    """A risk assessment representing MEDIUM + allow_permanent=True (default)."""
    return RiskAssessment(
        level=RiskLevel.MEDIUM,
        score=0.5,
        reasons=("ordinary_shell_command",),
        explanation="low-risk shell command",
    )


def _high_risk_permanent() -> RiskAssessment:
    """HIGH + allow_permanent=True, e.g. an in-envelope hardware write."""
    return RiskAssessment(
        level=RiskLevel.HIGH,
        score=0.7,
        reasons=("device_dispense",),
        explanation="in-envelope hardware write",
        allow_permanent=True,
    )


@pytest.mark.asyncio
async def test_bypass_all_does_not_auto_approve_high_no_permanent() -> None:
    """_bypass_all must not auto-approve HIGH+allow_permanent=False actions.

    This is the core of the _bypass_all privilege-escalation fix: a session
    bypass earned from a low-risk approval must not silently extend to
    plugin installs, external sends, or other actions the risk classifier
    marked as non-reusable.
    """
    delegate = _Gate(ApprovalDecision.ALLOW_ONCE)
    gate = SessionAwareGate(delegate)
    # Arm the bypass.
    gate._bypass_all = True

    request = ApprovalRequest(
        category="platform.action",
        detail="plugin install",
        risk=_high_risk_no_permanent(),
        choices=("allow_once", "allow_session", "deny"),
        default_choice="deny",
    )
    decision = await gate.request_approval(request)

    # The delegate must have been consulted -- bypass did not fire.
    assert len(delegate.requests) == 1
    assert decision == ApprovalDecision.ALLOW_ONCE


@pytest.mark.asyncio
async def test_bypass_all_still_auto_approves_low_and_medium_risk() -> None:
    """Regression guard: _bypass_all must keep working for safe actions."""
    delegate = _Gate(ApprovalDecision.DENY)
    gate = SessionAwareGate(delegate)
    gate._bypass_all = True

    for risk in (_medium_risk_permanent(), None):
        request = ApprovalRequest(
            category="shell.command",
            detail="echo hello",
            risk=risk,
            choices=("allow_once", "allow_session", "deny"),
        )
        decision = await gate.request_approval(request)
        assert decision == ApprovalDecision.ALLOW

    # Delegate was never consulted.
    assert delegate.requests == []


@pytest.mark.asyncio
async def test_bypass_all_auto_approves_high_with_allow_permanent() -> None:
    """HIGH + allow_permanent=True (e.g. hardware) is still bypassed.

    The fix gates only on the *combination* of high risk and non-reusable
    consent, so hardware writes that declare allow_permanent=True are
    unaffected.
    """
    delegate = _Gate(ApprovalDecision.DENY)
    gate = SessionAwareGate(delegate)
    gate._bypass_all = True

    request = ApprovalRequest(
        category="device.dispense",
        detail="aspirate 10 uL",
        risk=_high_risk_permanent(),
        choices=("allow_once", "allow_session", "allow_all_session",
                 "allow_always", "deny"),
    )
    decision = await gate.request_approval(request)

    assert decision == ApprovalDecision.ALLOW
    assert delegate.requests == []


@pytest.mark.asyncio
async def test_choices_exclude_allow_all_session_when_not_permanent() -> None:
    """allow_all_session must not be offered for non-reusable actions."""
    choices_restricted = ApprovalOrchestrator._choices(allow_permanent=False)
    choices_full = ApprovalOrchestrator._choices(allow_permanent=True)

    assert "allow_all_session" not in choices_restricted
    assert "allow_always" not in choices_restricted
    assert "allow_all_session" in choices_full
    assert "allow_always" in choices_full
    # Core choices are always present.
    assert "allow_once" in choices_restricted
    assert "allow_session" in choices_restricted
    assert "deny" in choices_restricted


@pytest.mark.asyncio
async def test_delegate_decision_outside_choices_falls_back_to_deny() -> None:
    """A delegate returning an un-offered choice is fail-closed to deny.

    This covers both a UI bug and a spoofed response: neither should be
    honoured.
    """
    delegate = _Gate(ApprovalDecision.ALLOW_ALL_SESSION)
    gate = SessionAwareGate(delegate)

    request = ApprovalRequest(
        category="platform.action",
        detail="external send",
        risk=_high_risk_no_permanent(),
        choices=("allow_once", "allow_session", "deny", "deny_always"),
        default_choice="deny",
    )
    decision = await gate.request_approval(request)

    # The decision was out of choices → fell back to deny.
    assert decision == ApprovalDecision.DENY
    # The delegate was called (bypass was not armed).
    assert len(delegate.requests) == 1
    # And the bypass flag must NOT have been armed.
    assert gate._bypass_all is False


@pytest.mark.asyncio
async def test_bypass_all_and_choices_validation_combined() -> None:
    """Full chain: bypass armed → HIGH non-reusable → fallthrough → delegate
    returns out-of-choices → denied.

    Exercises all three fixes together as defence-in-depth.
    """
    delegate = _Gate(ApprovalDecision.ALLOW_ALL_SESSION)
    gate = SessionAwareGate(delegate)
    gate._bypass_all = True

    request = ApprovalRequest(
        category="gateway.send",
        detail="send message to Slack",
        risk=RiskAssessment(
            level=RiskLevel.HIGH,
            score=0.72,
            reasons=("external_message_send",),
            explanation="external platform send",
            allow_permanent=False,
        ),
        choices=("allow_once", "allow_session", "deny", "deny_always"),
        default_choice="deny",
    )
    decision = await gate.request_approval(request)

    # Fix 1: bypass fell through (HIGH + !allow_permanent).
    # Fix 3: delegate returned ALLOW_ALL_SESSION not in choices → deny.
    assert decision == ApprovalDecision.DENY
    assert len(delegate.requests) == 1
    assert gate._bypass_all is True  # not disarmed, still set from before


@pytest.mark.asyncio
async def test_orchestrator_high_no_permanent_through_full_chain() -> None:
    """End-to-end: orchestrator + SessionAwareGate for a plugin_management action.

    Verifies the orchestrator builds the right choices and the gate enforces
    them when a delegate tries to escalate.
    """
    # Delegate always tries ALLOW_ALL_SESSION -- a realistic UI misconfig.
    delegate = _Gate(ApprovalDecision.ALLOW_ALL_SESSION)
    gate = SessionAwareGate(delegate)
    orchestrator = ApprovalOrchestrator(gate)

    action = ActionDescriptor.platform_action(
        "plugin_management",
        "plugin.install",
        {"package": "demo"},
        backend_kind="local",
    )

    result = await orchestrator.evaluate(action)

    # The risk classifier forces HIGH + allow_permanent=False.
    assert result.risk.level == RiskLevel.HIGH
    assert result.risk.allow_permanent is False
    # The delegate returned ALLOW_ALL_SESSION which was not in choices → deny.
    assert result.approved is False
    assert gate._bypass_all is False


# ════════════════════════════════════════════════════════════════
# Irreversible / external-output physical writes under bypass (issue #34)
# ════════════════════════════════════════════════════════════════


def _irreversible_hardware_write() -> RiskAssessment:
    """An irreversible physical write: HIGH + allow_permanent=False.

    Mirrors what ``HardwareRiskClassifier._tier_for`` now emits for an
    irreversible ACTUATE or any DISPENSE -- material leaving the device
    cannot be un-dispensed, so reusable consent is withheld.
    """
    return RiskAssessment(
        level=RiskLevel.HIGH,
        score=0.8,
        reasons=("device_dispense", "irreversible"),
        explanation="in-envelope dispense; the effect cannot be undone",
        allow_permanent=False,
    )


def _critical_no_permanent() -> RiskAssessment:
    """A CRITICAL + allow_permanent=False assessment (e.g. a hardline write)."""
    return RiskAssessment(
        level=RiskLevel.CRITICAL,
        score=1.0,
        reasons=("unresolvable_device",),
        explanation="a command that cannot be described",
        allow_permanent=False,
    )


@pytest.mark.asyncio
async def test_bypass_all_does_not_auto_approve_irreversible_hardware_write() -> None:
    """An irreversible physical write must fall through, not be blanket-approved.

    Before #34, ``_tier_for`` marked in-envelope physical writes
    allow_permanent=True, so an irreversible DISPENSE reached the gate as
    HIGH+allow_permanent=True and was silently authorised by a session-wide
    bypass earned from a lower-risk approval. With the tightening it arrives
    as HIGH+allow_permanent=False, so the bypass must fall through to the
    delegate for per-invocation consent.
    """
    delegate = _Gate(ApprovalDecision.ALLOW_ONCE)
    gate = SessionAwareGate(delegate)
    gate._bypass_all = True

    request = ApprovalRequest(
        category="device.dispense",
        detail="aspirate 10 uL",
        risk=_irreversible_hardware_write(),
        choices=("allow_once", "allow_session", "deny"),
        default_choice="deny",
    )
    decision = await gate.request_approval(request)

    # The delegate was consulted -- the bypass did not fire.
    assert len(delegate.requests) == 1
    assert decision == ApprovalDecision.ALLOW_ONCE


@pytest.mark.asyncio
async def test_bypass_all_does_not_auto_approve_critical_no_permanent() -> None:
    """CRITICAL + allow_permanent=False must fall through under a session bypass.

    The fallthrough covers both HIGH and CRITICAL; this guards the CRITICAL
    arm so a session-wide bypass cannot blanket-approve, for example, a write
    that could only be classified as an unresolvable/hardline command.
    """
    delegate = _Gate(ApprovalDecision.ALLOW_ONCE)
    gate = SessionAwareGate(delegate)
    gate._bypass_all = True

    request = ApprovalRequest(
        category="device.actuate",
        detail="unresolvable device command",
        risk=_critical_no_permanent(),
        choices=("allow_once", "allow_session", "deny"),
        default_choice="deny",
    )
    decision = await gate.request_approval(request)

    assert len(delegate.requests) == 1
    assert decision == ApprovalDecision.ALLOW_ONCE
