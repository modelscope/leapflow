from __future__ import annotations

import builtins
from pathlib import Path
import sys
import time

import pytest

from leapflow.security.actions import ActionDescriptor
from leapflow.security.approval import ApprovalDecision
from leapflow.security.grants import ApprovalAuditLog, ApprovalGrant, ApprovalScope, JsonApprovalGrantStore, grant_key
from leapflow.security.orchestrator import ApprovalOrchestrator
from leapflow.security.risk import DefaultRiskClassifier, RiskLevel


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
async def test_prompt_approval_expired_request_denies(monkeypatch) -> None:
    from leapflow.cli.approval_view import prompt_approval
    from leapflow.security.approval import ApprovalRequest

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    request = ApprovalRequest(
        category="shell.command",
        detail="echo hello",
        expires_at=time.time() - 1,
    )

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
    from leapflow.tools.registry_bootstrap import set_file_write_gate

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

    set_file_write_gate(DenyingGate())
    try:
        result = await file_write({
            "path": str(tmp_path / "approval-output.py"),
            "content": "print('hello')",
        })
    finally:
        set_file_write_gate(None)

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
    from leapflow.tools.registry_bootstrap import set_file_read_gate

    target = tmp_path / ".env"
    target.write_text("API_KEY=sk-secret-value-123456\n", encoding="utf-8")
    set_file_read_gate(None)

    result = await file_read({"path": str(target)})

    assert result["ok"] is False
    assert result["requires_approval"] is True
    assert result["sensitivity_category"] == "credential"


@pytest.mark.asyncio
async def test_sensitive_file_read_approval_redacts_content(tmp_path: Path) -> None:
    from leapflow.tools.file_operations import file_read
    from leapflow.tools.registry_bootstrap import set_file_read_gate

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
    set_file_read_gate(gate)
    try:
        result = await file_read({"path": str(target)})
    finally:
        set_file_read_gate(None)

    assert result["ok"] is True
    assert gate.calls[0][2]["sensitivity_category"] == "credential"
    assert "sk-secret-value" not in result["content"]
    assert "«redacted:" in result["content"]
    assert result["redact_on_read"] is True


@pytest.mark.asyncio
async def test_sensitive_file_write_uses_approval_gate(tmp_path: Path) -> None:
    from leapflow.tools.file_operations import file_write
    from leapflow.tools.registry_bootstrap import set_file_write_gate

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
    set_file_write_gate(gate)
    try:
        result = await file_write({"path": str(target), "content": "API_KEY=sk-new-value-123456\n"})
    finally:
        set_file_write_gate(None)

    assert result["ok"] is True
    assert target.read_text(encoding="utf-8") == "API_KEY=sk-new-value-123456\n"
    assert gate.calls[0][3]["sensitivity_category"] == "credential"


@pytest.mark.asyncio
async def test_runtime_database_read_is_hardline_blocked(tmp_path: Path) -> None:
    from leapflow.tools.file_operations import file_read
    from leapflow.tools.registry_bootstrap import set_file_read_gate

    class FailingGate:
        async def check(self, *_args, **_kwargs) -> bool:
            raise AssertionError("runtime database reads must not request approval")

    target = tmp_path / "leap.duckdb"
    target.write_bytes(b"not text")
    set_file_read_gate(FailingGate())
    try:
        result = await file_read({"path": str(target)})
    finally:
        set_file_read_gate(None)

    assert result["ok"] is False
    assert "Runtime database" in result["error"]
