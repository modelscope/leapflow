"""Guards for the config capability and the path boundaries around it.

The scenario these lock down: a user asks to change ``llm.model`` from inside an
arbitrary workspace. Before the ``config_*`` tools existed the model had only
``file_read`` / ``shell_run``, so it guessed at ``~/.leapflow/...`` and the
workspace sandbox refused every attempt — with a message that wrongly implied
approval could lift the boundary.

Covered:
- config tools work by key and never depend on a workspace/tool context
- reads reach the model by default; writes stay behind disclosure and approval
- a write is visible to an immediate read-back (no stale singleton)
- credentials never echo into the transcript
- unknown keys are self-correctable in the same turn
- the sandbox refusal is honest about approval and points at the config tools
- the workspace manifest never lands inside the LeapFlow home
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from leapflow.layout import build_layout
from leapflow.tools import config_tools


@pytest.fixture()
def cfg_home(monkeypatch, tmp_path):
    """Point LeapFlow at a temp home, reset the settings singleton, allow writes.

    ``config_set`` fails closed without an approval gate, so tests that exercise a
    write install a permissive one here; the gate's own behavior is covered
    separately below.
    """
    import leapflow.config as config_module

    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "home"))
    monkeypatch.setattr(config_module, "_settings_instance", None, raising=False)
    config_tools.set_config_context(None)
    config_tools.set_config_approval_gate(_AllowGate())
    yield tmp_path / "home"
    monkeypatch.setattr(config_module, "_settings_instance", None, raising=False)
    config_tools.set_config_context(None)
    config_tools.set_config_approval_gate(None)


class _Result:
    """Mirrors the fields of security.orchestrator.ApprovalResult that we read."""

    def __init__(self, approved: bool, denial_message: str = "") -> None:
        self.approved = approved
        self.denial_message = denial_message
        self.reason = denial_message


class _AllowGate:
    """Approves every config write and records the action it was asked about.

    Implements ``evaluate(ActionDescriptor)`` — the orchestrator's real interface.
    An earlier version of this fake accepted the shell-style ``check(...)``
    signature, which let the tests pass while the production call raised
    TypeError against the actual orchestrator.
    """

    def __init__(self) -> None:
        self.actions: list[Any] = []

    async def evaluate(self, action):
        self.actions.append(action)
        return _Result(True)


class _DenyGate:
    def __init__(self, message: str = "denied for test") -> None:
        self.message = message

    async def evaluate(self, action):
        return _Result(False, self.message)


class _ReloadingContext:
    """Minimal stand-in for the runtime Context's hot-reload contract."""

    def __init__(self) -> None:
        from leapflow.config import load_config

        self.settings = load_config()
        self.reloads = 0

    def reload_runtime_config_if_changed(self, *, force: bool = False) -> bool:
        from leapflow.config import load_config

        self.settings = load_config()
        self.reloads += 1
        return True


# ── Reading and writing by key, never by path ────────────────────────────


def test_config_list_reports_categories_and_bounded_fields(cfg_home) -> None:
    result = asyncio.run(config_tools.config_list_handler({"limit": 5}))

    assert result["ok"] is True
    assert result["returned"] == 5 and result["total"] > 5
    assert result["truncated"] is True
    assert "LLM Provider" in result["categories"]


def test_config_list_rejects_unknown_category_with_the_valid_set(cfg_home) -> None:
    """A wrong filter must be recoverable, not a dead end."""
    result = asyncio.run(config_tools.config_list_handler({"category": "Nope"}))

    assert result["ok"] is False and result["retryable"] is True
    assert result["available_categories"]


def test_config_get_exposes_hot_reload_semantics(cfg_home) -> None:
    """Without this the user changes a restart-required field and sees no effect."""
    result = asyncio.run(config_tools.config_get_handler({"key": "daemon.log_level"}))

    assert result["ok"] is True
    assert result["hot_reload"] == "restart-required"
    assert result["description"]


def test_config_set_changes_the_value_and_read_back_agrees(cfg_home) -> None:
    """The original scenario: switch the model and confirm it took."""
    config_tools.set_config_context(_ReloadingContext())

    written = asyncio.run(
        config_tools.config_set_handler({"key": "llm.model", "value": "qwen3.8-max"})
    )
    read_back = asyncio.run(config_tools.config_get_handler({"key": "llm.model"}))

    assert written["ok"] is True
    assert written["changed_keys"] == ["llm.model"]
    assert written["session_reloaded"] is True
    # The whole point: a stale singleton here reads as a failed write.
    assert read_back["value"] == "qwen3.8-max"


def test_config_set_reports_restart_requirement(cfg_home) -> None:
    result = asyncio.run(
        config_tools.config_set_handler({"key": "daemon.log_level", "value": "DEBUG"})
    )

    assert result["ok"] is True
    assert result["restart_required"] is True
    assert "restart" in result["next_step"].lower()


def test_config_set_never_echoes_a_secret(cfg_home) -> None:
    """A credential in the tool result would land in the transcript."""
    result = asyncio.run(
        config_tools.config_set_handler({"key": "llm.api_key", "value": "sk-must-not-echo"})
    )

    assert result["ok"] is True
    assert "value" not in result
    assert "sk-must-not-echo" not in str(result)


def test_config_set_requires_key_and_value(cfg_home) -> None:
    missing_value = asyncio.run(config_tools.config_set_handler({"key": "llm.model"}))
    missing_key = asyncio.run(config_tools.config_set_handler({"value": "x"}))

    assert missing_value["ok"] is False and missing_value["retryable"] is True
    assert missing_key["ok"] is False and missing_key["retryable"] is True


@pytest.mark.parametrize(
    ("typo", "expected"),
    [
        ("llm.modle", "llm.model"),          # transposed letters
        ("daemon.loglevel", "daemon.log_level"),  # dropped separator
        ("llm.api-key", "llm.api_key"),      # hyphen instead of underscore
        ("model", "llm.model"),              # bare last segment
    ],
)
def test_unknown_key_suggests_the_real_one(cfg_home, typo: str, expected: str) -> None:
    """Self-correction in the same turn is what prevents a fallback to probing."""
    result = asyncio.run(config_tools.config_get_handler({"key": typo}))

    assert result["ok"] is False and result["retryable"] is True
    assert expected in result["did_you_mean"]


def test_config_tools_ignore_the_workspace_boundary(cfg_home) -> None:
    """They take keys, so an unrelated workspace context must not affect them.

    This is why the sandbox never has to be relaxed for configuration.
    """
    from leapflow.tools.execution_context import (
        ToolExecutionContext,
        reset_tool_context,
        set_tool_context,
    )

    ctx = ToolExecutionContext.from_strings(workspace_root=str(cfg_home.parent / "elsewhere"))
    token = set_tool_context(ctx)
    try:
        result = asyncio.run(config_tools.config_get_handler({"key": "llm.model"}))
    finally:
        reset_tool_context(token)

    assert result["ok"] is True


# ── Sandbox refusal: honest, and pointing somewhere useful ───────────────


def test_sandbox_refusal_does_not_promise_approval(cfg_home) -> None:
    """The boundary is a hard gate; claiming approval helps sends users in circles."""
    from leapflow.tools.execution_context import (
        ToolExecutionContext,
        reset_tool_context,
        set_tool_context,
        workspace_scope_refusal,
    )

    token = set_tool_context(
        ToolExecutionContext.from_strings(workspace_root=str(cfg_home.parent / "ws"))
    )
    try:
        error = workspace_scope_refusal(cfg_home / "config" / "user.yaml", operation="file_read")
    finally:
        reset_tool_context(token)

    assert error is not None
    assert "Approval is required" in error["error"]


def test_sandbox_refusal_redirects_config_paths_to_the_tools(cfg_home) -> None:
    """A refusal that names the right capability ends the guessing loop."""
    from leapflow.tools.execution_context import (
        ToolExecutionContext,
        reset_tool_context,
        set_tool_context,
        workspace_scope_refusal,
    )

    build_layout(cfg_home).ensure(profile_id="default")
    token = set_tool_context(
        ToolExecutionContext.from_strings(workspace_root=str(cfg_home.parent / "ws"))
    )
    try:
        config_error = workspace_scope_refusal(
            cfg_home / "config" / "user.yaml", operation="file_read"
        )
        vault_error = workspace_scope_refusal(
            cfg_home / "secrets" / "vault.key", operation="file_read"
        )
        plain_error = workspace_scope_refusal(
            cfg_home.parent / "unrelated" / "notes.txt", operation="file_read"
        )
    finally:
        reset_tool_context(token)

    assert "config_set" in config_error["error"]
    assert "config_set" in vault_error["error"]
    # An ordinary outside-workspace path gets no config advice.
    assert "config_set" not in plain_error["error"]


# ── Workspace manifest must not land in the LeapFlow home ────────────────


def test_manifest_is_skipped_when_workspace_is_the_leapflow_home(tmp_path) -> None:
    """Writing here would drop a workspace marker beside config/ and secrets/."""
    home = tmp_path / ".leapflow"
    layout = build_layout(home)

    written = layout.write_workspace_manifest(tmp_path)

    assert written is None
    assert not (home / "workspace.yaml").exists()


def test_manifest_is_written_for_a_normal_workspace(tmp_path) -> None:
    layout = build_layout(tmp_path / "leap-home")
    workspace = tmp_path / "project"
    workspace.mkdir()

    written = layout.write_workspace_manifest(workspace)

    assert written == workspace / ".leapflow" / "workspace.yaml"
    assert Path(written).exists()


# ── Writes are gated: the model must not be able to unsupervise itself ─────


def _guardrail_is_on() -> bool:
    """Whether the guardrail is still enabled.

    Compares loosely on purpose: the effective value surfaces as a string
    (``'true'``) rather than a bool, and what matters here is only that the write
    did not switch it off.
    """
    value = asyncio.run(config_tools.config_get_handler({"key": "guardrail.enabled"}))["value"]
    return str(value).strip().lower() in {"true", "1", "yes"}


def test_config_write_is_denied_without_an_approval_gate(cfg_home) -> None:
    """Fail closed and stop the agent turn rather than spending its loop budget.

    ``requires_approval`` only informs capability disclosure. A denied write must
    also carry the engine's explicit hard-stop signal; otherwise the model retries
    an action that can never obtain consent until the reasoning budget is exhausted.
    """
    from leapflow.security.permission_failures import is_permission_hard_stop_payload

    config_tools.set_config_approval_gate(None)

    result = asyncio.run(
        config_tools.config_set_handler({"key": "guardrail.enabled", "value": False})
    )

    assert result["ok"] is False
    assert result["requires_approval"] is True
    assert result["failure_code"] == "approval_denied"
    assert result["blocks_approval"] is True
    assert result.get("failure_class") is None
    assert is_permission_hard_stop_payload(result) is True
    assert _guardrail_is_on(), "a denied write must not reach disk"


def test_config_write_honors_gate_denial(cfg_home) -> None:
    config_tools.set_config_approval_gate(_DenyGate())

    result = asyncio.run(
        config_tools.config_set_handler({"key": "llm.model", "value": "sneaky"})
    )

    assert result["ok"] is False
    assert "denied for test" in result["error"]
    assert asyncio.run(config_tools.config_get_handler({"key": "llm.model"}))["value"] != "sneaky"


def test_gate_sees_the_key_but_never_the_value(cfg_home) -> None:
    """An approval prompt and its audit trail must not carry a credential."""
    gate = _AllowGate()
    config_tools.set_config_approval_gate(gate)

    asyncio.run(
        config_tools.config_set_handler({"key": "llm.api_key", "value": "sk-secret-value"})
    )

    assert gate.actions
    action = gate.actions[-1]
    assert action.resource == "llm.api_key"
    assert action.kind == "runtime.configure"
    assert action.metadata["secret"] is True
    assert "sk-secret-value" not in f"{action.summary}{action.detail}{action.metadata}"


def test_gate_is_called_through_its_real_interface(cfg_home) -> None:
    """Regression: the tool must use evaluate(ActionDescriptor), not check().

    ApprovalOrchestrator.check() takes a single command string, so calling it with
    the file-write signature raised TypeError at runtime while a fake accepting
    that signature kept the tests green.
    """
    from leapflow.security.orchestrator import ApprovalOrchestrator

    assert hasattr(ApprovalOrchestrator, "evaluate")
    gate = _AllowGate()
    config_tools.set_config_approval_gate(gate)

    result = asyncio.run(
        config_tools.config_set_handler({"key": "llm.model", "value": "qwen3.8-max"})
    )

    assert result["ok"] is True
    assert gate.actions, "the gate must be consulted via evaluate()"


def test_a_gate_with_only_the_shell_signature_is_denied(cfg_home) -> None:
    """A gate lacking evaluate() must fail closed, not silently allow."""

    class _ShellOnlyGate:
        async def check(self, command):
            return True

    config_tools.set_config_approval_gate(_ShellOnlyGate())

    result = asyncio.run(
        config_tools.config_set_handler({"key": "guardrail.enabled", "value": False})
    )

    assert result["ok"] is False
    assert _guardrail_is_on()


def test_write_works_against_the_real_orchestrator(cfg_home) -> None:
    """End-to-end through the production ApprovalOrchestrator, not a fake.

    This is the check the fakes could not make: the tool previously called the
    shell-oriented ``check(path, content, mode, meta)``, which the orchestrator
    does not accept, so every config_set failed with a TypeError at runtime while
    a signature-matching fake kept the suite green. It also confirms a config
    change is classified ``runtime.configure`` and rated HIGH, so it reaches a
    human rather than being auto-allowed.
    """
    from leapflow.security.approval import ApprovalDecision
    from leapflow.security.orchestrator import ApprovalOrchestrator

    class _AutoAllow:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        async def request_approval(self, request):
            self.requests.append(request)
            return ApprovalDecision.ALLOW_ONCE

    prompt = _AutoAllow()
    config_tools.set_config_approval_gate(ApprovalOrchestrator(prompt))
    # Bind a reloading context so the read-back sees the write; without it the
    # in-process settings keep the old value (covered separately above).
    config_tools.set_config_context(_ReloadingContext())

    result = asyncio.run(
        config_tools.config_set_handler({"key": "llm.model", "value": "qwen3.8-max"})
    )

    assert result["ok"] is True, result.get("error")
    assert asyncio.run(config_tools.config_get_handler({"key": "llm.model"}))["value"] == "qwen3.8-max"
    assert len(prompt.requests) == 1, "a config change must reach the human prompt"
    assert str(prompt.requests[0].risk.level).endswith("HIGH")


def test_real_orchestrator_denial_blocks_the_write(cfg_home) -> None:
    """A declined prompt must leave the setting untouched."""
    from leapflow.security.approval import ApprovalDecision
    from leapflow.security.orchestrator import ApprovalOrchestrator

    class _AutoDeny:
        async def request_approval(self, request):
            return ApprovalDecision.DENY

    config_tools.set_config_approval_gate(ApprovalOrchestrator(_AutoDeny()))

    result = asyncio.run(
        config_tools.config_set_handler({"key": "guardrail.enabled", "value": False})
    )

    assert result["ok"] is False
    assert _guardrail_is_on()


def test_a_broken_gate_denies_rather_than_opens(cfg_home) -> None:
    """A gate that raises must not degrade into an open door."""

    class _BrokenGate:
        async def evaluate(self, action):
            raise RuntimeError("gate exploded")

    config_tools.set_config_approval_gate(_BrokenGate())

    result = asyncio.run(
        config_tools.config_set_handler({"key": "guardrail.enabled", "value": False})
    )

    assert result["ok"] is False
    assert _guardrail_is_on(), "a gate error must not let the write through"


# ── The status bar must learn about the change ─────────────────────────


def test_stream_metadata_carries_the_active_model() -> None:
    """A daemon-mode TUI caches the model at startup and only updates from this.

    Symptom this prevents: ``config_get`` reports the new model while the status
    bar still shows the old one. The change notification in the chat loop is
    change-detection based and a write has already refreshed the signature, so
    the model has to travel on ordinary metadata instead.
    """
    from types import SimpleNamespace

    from leapflow.daemon._service_helpers import engine_context_metadata

    metadata = engine_context_metadata(
        None, SimpleNamespace(llm_context_length=1_000_000, llm_model="qwen3.8-max"),
    )

    assert metadata["llm_model"] == "qwen3.8-max"
    assert metadata["llm_context_length"] == 1_000_000


def test_stream_metadata_omits_an_empty_model() -> None:
    """An unset model must not blank out whatever the bar already shows."""
    from types import SimpleNamespace

    from leapflow.daemon._service_helpers import engine_context_metadata

    metadata = engine_context_metadata(None, SimpleNamespace(llm_context_length=0, llm_model=""))

    assert "llm_model" not in metadata
