# Copyright (c) Alibaba, Inc. and its affiliates.
"""Every path-oriented tool asks before crossing the workspace boundary.

The refusal text has always said "Approval is required to access paths outside
the workspace", but only ``shell_run`` ever opened a prompt. The other eleven
call sites returned the refusal directly, so ``file_list`` and ``code_search``
refused in ~39ms with a message promising an approval that never came, and
ignored a session-wide "Allow ALL" that the shell honoured.

These tests pin the three properties that were broken:
  1. every entry point routes through the approval gate,
  2. every entry point keeps the approval chain observable under bypass,
  3. the escape is risk-classified so the policy engine always asks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from leapflow.security.actions import ActionKind
from leapflow.security.risk import DefaultRiskClassifier, RiskLevel
from leapflow.tools.execution_context import (
    ToolExecutionContext,
    require_workspace_access,
    reset_tool_context,
    set_tool_context,
)


class RecordingOrchestrator:
    """Approval orchestrator double that records what it was asked to approve."""

    def __init__(self, approved: bool, denial_message: str = "") -> None:
        self._approved = approved
        self._denial_message = denial_message
        self.actions: list[Any] = []

    async def evaluate(self, action: Any) -> Any:
        self.actions.append(action)
        approved = self._approved
        denial_message = self._denial_message

        class Result:
            pass

        result = Result()
        result.approved = approved
        result.denial_message = denial_message
        return result


class BrokenOrchestrator:
    async def evaluate(self, action: Any) -> Any:
        raise RuntimeError("gate exploded")


def _context(tmp_path: Path, orchestrator: Any = None, *, bypass: bool = False):
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    return ToolExecutionContext.from_strings(
        workspace_root=str(workspace),
        session_id="sess-1",
        approval_bypass=bypass,
        orchestrator=orchestrator,
    )


# ── every entry point asks ────────────────────────────────────────────────────

# (module, handler, params-builder, expected effect). Every tool that can reach a
# path outside the workspace belongs here; a new one that forgets to gate shows
# up as a missing prompt rather than as a silent refusal in production.
_ENTRY_POINTS = [
    ("file_operations", "file_list", lambda p: {"path": str(p)}, "read"),
    ("file_operations", "file_read", lambda p: {"path": str(p / "a.txt")}, "read"),
    ("file_operations", "file_write", lambda p: {"path": str(p / "a.txt"), "content": "x"}, "write"),
    ("file_operations", "code_search", lambda p: {"pattern": "x", "path": str(p)}, "read"),
    ("file_operations", "file_find", lambda p: {"glob": "*.py", "path": str(p)}, "read"),
    ("file_operations", "edit_file", lambda p: {"path": str(p / "a.txt"), "original_text": "a", "new_text": "b"}, "write"),
    ("repo_map", "repo_map", lambda p: {"path": str(p)}, "read"),
    ("dev_tools", "test_run", lambda p: {"cwd": str(p)}, "execute"),
    ("dev_tools", "lint_check", lambda p: {"cwd": str(p)}, "execute"),
    ("terminal_session", "terminal_open", lambda p: {"cwd": str(p)}, "execute"),
    ("scm_tools", "scm_sync", lambda p: {"action": "status", "cwd": str(p)}, "write"),
    ("shell_tools", "shell_run", lambda p: {"command": f"cat {p}/secret"}, "execute"),
]


def _handler(module_name: str, attr: str):
    import importlib

    module = importlib.import_module(f"leapflow.tools.{module_name}")
    return getattr(module, attr)


@pytest.fixture(autouse=True)
def _enable_terminal_sessions(monkeypatch):
    """Persistent terminals are opt-in; without this they refuse before the gate."""
    import leapflow.tools.terminal_session as terminal_session

    monkeypatch.setattr(terminal_session, "_ENABLED", True, raising=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "attr", "build_params", "expected_effect"),
    _ENTRY_POINTS,
    ids=[f"{m}.{a}" for m, a, _, _ in _ENTRY_POINTS],
)
async def test_entry_point_asks_before_leaving_the_workspace(
    tmp_path, module_name, attr, build_params, expected_effect,
) -> None:
    """The gate must be consulted, and the declared effect must reach it."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.txt").write_text("a", encoding="utf-8")

    gate = RecordingOrchestrator(approved=False, denial_message="denied by test")
    token = set_tool_context(_context(tmp_path, gate))
    try:
        result = await _handler(module_name, attr)(build_params(outside))
    finally:
        reset_tool_context(token)

    assert len(gate.actions) == 1, f"{attr} never asked for approval"
    action = gate.actions[0]
    assert action.kind == ActionKind.WORKSPACE_ESCAPE.value
    assert action.effect == expected_effect
    assert result["ok"] is False
    # The gate's own wording reaches the caller, not a generic scope error.
    assert result["error"] == "denied by test"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "attr", "build_params", "expected_effect"),
    _ENTRY_POINTS,
    ids=[f"{m}.{a}" for m, a, _, _ in _ENTRY_POINTS],
)
async def test_entry_point_uses_unified_approval_chain_under_bypass(
    tmp_path, module_name, attr, build_params, expected_effect,
) -> None:
    """Bypass is decided by the orchestrator rather than a tool-local shortcut."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.txt").write_text("a", encoding="utf-8")

    gate = RecordingOrchestrator(approved=True)
    token = set_tool_context(_context(tmp_path, gate, bypass=True))
    try:
        result = await _handler(module_name, attr)(build_params(outside))
    finally:
        reset_tool_context(token)

    assert len(gate.actions) == 1, f"{attr} bypassed the unified approval chain"
    assert result.get("error_type") != "outside_workspace"


@pytest.mark.asyncio
async def test_approval_lets_the_operation_through(tmp_path) -> None:
    """An approved escape proceeds; the refusal is not returned anyway."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "note.txt").write_text("hello", encoding="utf-8")

    gate = RecordingOrchestrator(approved=True)
    token = set_tool_context(_context(tmp_path, gate))
    try:
        from leapflow.tools.file_operations import file_read

        result = await file_read({"path": str(outside / "note.txt")})
    finally:
        reset_tool_context(token)

    assert len(gate.actions) == 1
    assert result["ok"] is True
    assert "hello" in result["content"]


# ── fail closed ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_missing_orchestrator_refuses(tmp_path) -> None:
    """No gate means no consent, so the escape is refused rather than allowed."""
    outside = tmp_path / "outside"
    outside.mkdir()

    token = set_tool_context(_context(tmp_path, orchestrator=None))
    try:
        result = await require_workspace_access(outside, operation="file_list")
    finally:
        reset_tool_context(token)

    assert result is not None
    assert result["error_type"] == "outside_workspace"


@pytest.mark.asyncio
async def test_broken_orchestrator_refuses(tmp_path) -> None:
    """A gate that raises must not become an open door."""
    outside = tmp_path / "outside"
    outside.mkdir()

    token = set_tool_context(_context(tmp_path, BrokenOrchestrator()))
    try:
        result = await require_workspace_access(outside, operation="file_list")
    finally:
        reset_tool_context(token)

    assert result is not None
    assert result["error_type"] == "outside_workspace"


@pytest.mark.asyncio
async def test_paths_inside_the_workspace_are_not_gated(tmp_path) -> None:
    gate = RecordingOrchestrator(approved=False)
    ctx = _context(tmp_path, gate)
    token = set_tool_context(ctx)
    try:
        result = await require_workspace_access(
            ctx.workspace_root / "src", operation="file_list",
        )
    finally:
        reset_tool_context(token)

    assert result is None
    assert gate.actions == []



# ── the escape is always risk-classified above the auto-allow floor ──────────

def test_workspace_escape_is_never_auto_allowed() -> None:
    """The policy engine auto-allows LOW risk, so the escape must never be LOW.

    Classifying the escape by the target file's own sensitivity would let an
    ordinary file in another project score low and be permitted with no prompt,
    which is strictly worse than the refusal it replaced.
    """
    from leapflow.security.actions import ActionDescriptor
    from leapflow.security.policy import ApprovalPolicyEngine, PolicyVerdict

    classifier = DefaultRiskClassifier()
    policy = ApprovalPolicyEngine()

    for effect in ("read", "write", "execute"):
        action = ActionDescriptor.workspace_escape(
            "/elsewhere/ordinary.txt", operation="file_list", effect=effect,
        )
        risk = classifier.assess(action)
        assert risk.level is not RiskLevel.LOW, effect
        assert risk.score >= 0.35, effect
        assert policy.evaluate(action, risk).verdict == PolicyVerdict.ASK, effect


def test_mutating_escape_outranks_a_read_and_refuses_permanent_grants() -> None:
    """Listing a sibling repo must not be weighed like writing into it."""
    from leapflow.security.actions import ActionDescriptor

    classifier = DefaultRiskClassifier()
    read = classifier.assess(
        ActionDescriptor.workspace_escape("/elsewhere", operation="file_list", effect="read")
    )
    write = classifier.assess(
        ActionDescriptor.workspace_escape("/elsewhere", operation="file_write", effect="write")
    )

    assert write.score > read.score
    assert write.level == RiskLevel.HIGH
    assert write.allow_permanent is False
