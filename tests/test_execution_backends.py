from __future__ import annotations

import asyncio
from typing import Any, Mapping

import pytest

from leapflow.gateway.adapters.common import JsonBody
from leapflow.gateway.backends.cli_backend import CliBackend
from leapflow.gateway.backends.rest_backend import RestBackend
from leapflow.gateway.connectors.protocol import ActionSpec, BackendKind


class FakeProcess:
    def __init__(self, returncode: int, stdout: bytes, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


class HangingFakeProcess:
    def __init__(self) -> None:
        self.returncode = None
        self.killed = False
        self.waited = False

    async def communicate(self) -> tuple[bytes, bytes]:
        await asyncio.sleep(60)
        return b"", b""

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.waited = True
        return int(self.returncode or 0)


@pytest.mark.asyncio
async def test_cli_backend_executes_registered_action(monkeypatch) -> None:
    captured: list[tuple[str, ...]] = []

    async def fake_exec(*argv, **_kwargs):
        captured.append(tuple(str(arg) for arg in argv))
        return FakeProcess(0, b'{"ok": true, "data": {"message_id": "om_1"}}')

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    backend = CliBackend(binary="lark-cli", profile="bot-reader", identity="bot")
    spec = ActionSpec(
        name="im.send_message",
        backend_kind=BackendKind.CLI.value,
        backend_config={"argv": ("im", "+messages-send", "--chat-id", "{chat_id}", "--text", "{text}")},
    )

    result = await backend.execute(spec, {"chat_id": "oc_1", "text": "hello"})

    assert result.ok is True
    assert result.resource_id == "om_1"
    assert captured[0] == (
        "lark-cli",
        "--profile",
        "bot-reader",
        "--as",
        "bot",
        "im",
        "+messages-send",
        "--chat-id",
        "oc_1",
        "--text",
        "hello",
    )


@pytest.mark.asyncio
async def test_cli_backend_normalizes_error(monkeypatch) -> None:
    async def fake_exec(*_argv, **_kwargs):
        return FakeProcess(1, b"", b'{"ok": false, "error": {"message": "missing scope"}}')

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    backend = CliBackend(binary="lark-cli")
    spec = ActionSpec(
        name="im.send_message",
        backend_kind=BackendKind.CLI.value,
        capability="im.message.send",
        backend_config={"argv": ("im", "+messages-send")},
    )

    result = await backend.execute(spec, {})

    assert result.ok is False
    assert result.error == "missing scope"
    assert result.failure is not None
    assert result.failure.failure_code == "missing_scope"
    assert result.failure.failure_class == "authorization"
    assert result.failure.blocks_approval is True


@pytest.mark.asyncio
async def test_cli_backend_preserves_lark_cli_permission_problem(monkeypatch) -> None:
    payload = (
        b'{"ok": false, "error": {'
        b'"type":"authorization",'
        b'"subtype":"missing_scope",'
        b'"message":"access denied for this operation",'
        b'"hint":"grant scope",'
        b'"missing_scopes":["im:message.group_msg"],'
        b'"requested_scopes":["im:message:readonly"],'
        b'"granted_scopes":[], '
        b'"identity":"bot",'
        b'"console_url":"https://console.example"'
        b'}}'
    )

    async def fake_exec(*_argv, **_kwargs):
        return FakeProcess(1, b"", payload)

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    backend = CliBackend(binary="lark-cli", profile="work", identity="bot")
    spec = ActionSpec(
        name="im.list_messages",
        backend_kind=BackendKind.CLI.value,
        capability="im.message.read",
        backend_config={"argv": ("im", "+chat-messages-list")},
    )

    result = await backend.execute(spec, {})

    assert result.ok is False
    assert result.failure is not None
    assert result.failure.failure_class == "authorization"
    assert result.failure.failure_code == "missing_scope"
    assert result.failure.recoverability == "admin_required"
    assert result.failure.retryable is False
    assert result.failure.blocks_approval is True
    assert result.failure.missing_scopes == ("im:message.group_msg",)
    assert result.failure.requested_scopes == ("im:message:readonly",)
    assert result.failure.identity == "bot"
    assert result.failure.console_url == "https://console.example"


@pytest.mark.asyncio
async def test_cli_backend_preview_uses_dry_run_template_without_execution() -> None:
    backend = CliBackend(binary="lark-cli", profile="bot-reader", identity="bot")
    spec = ActionSpec(
        name="docs.create_markdown",
        backend_kind=BackendKind.CLI.value,
        backend_config={
            "argv": ("docs", "+create", "--title", "{title}"),
            "dry_run_argv": ("docs", "+create", "--title", "{title}", "--dry-run"),
            "approval_summary": "Create Feishu document {title} as {identity} using profile {profile}.",
        },
    )

    preview = await backend.preview(spec, {"title": "Demo"})

    assert preview.ok is True
    assert preview.data["dry_run"] is True
    assert preview.data["argv"] == [
        "lark-cli", "--profile", "bot-reader", "--as", "bot",
        "docs", "+create", "--title", "Demo", "--dry-run",
    ]
    assert preview.summary == "Create Feishu document Demo as bot using profile bot-reader."


@pytest.mark.asyncio
async def test_cli_backend_status_reports_missing_binary_recovery(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _binary: None)
    backend = CliBackend(binary="missing-lark-cli", profile="work", identity="bot")

    status = await backend.status()

    assert status.ok is False
    assert status.metadata["recoverable"] is True
    assert status.metadata["binary"] == "missing-lark-cli"
    assert status.metadata["profile"] == "work"
    assert "Install 'missing-lark-cli'" in status.metadata["recovery_hint"]
    assert status.metadata["next_steps"]


@pytest.mark.asyncio
async def test_cli_backend_status_reports_auth_recovery(monkeypatch) -> None:
    async def fake_exec(*_argv, **_kwargs):
        return FakeProcess(1, b"", b'{"ok": false, "error": {"message": "missing scope im:message"}}')

    monkeypatch.setattr("shutil.which", lambda _binary: "/usr/local/bin/lark-cli")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    backend = CliBackend(binary="lark-cli", profile="work")

    status = await backend.status()

    assert status.ok is False
    assert status.metadata["binary_path"] == "/usr/local/bin/lark-cli"
    assert status.metadata["auth_status"] == "not_ready"
    assert "permission scopes" in status.metadata["recovery_hint"]
    assert "lark-cli --profile work auth login --json" in status.metadata["next_steps"]


@pytest.mark.asyncio
async def test_cli_backend_appends_explicit_output_args(monkeypatch) -> None:
    captured: list[tuple[str, ...]] = []

    async def fake_exec(*argv, **_kwargs):
        captured.append(tuple(str(arg) for arg in argv))
        return FakeProcess(0, b'{"ok": true, "data": {"id": "item-1"}}')

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    backend = CliBackend(binary="demo-cli")
    spec = ActionSpec(
        name="repo.read_file",
        backend_kind=BackendKind.CLI.value,
        backend_config={"argv": ("repo", "read"), "output_args": ("--format", "json")},
    )

    result = await backend.execute(spec, {})

    assert result.ok is True
    assert captured[0] == ("demo-cli", "repo", "read", "--format", "json")


@pytest.mark.asyncio
async def test_cli_backend_status_reports_contract_mismatch(monkeypatch) -> None:
    async def fake_exec(*_argv, **_kwargs):
        return FakeProcess(1, b"", b'unknown flag: --format')

    monkeypatch.setattr("shutil.which", lambda _binary: "/usr/local/bin/lark-cli")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    backend = CliBackend(binary="lark-cli", profile="work")

    status = await backend.status()

    assert status.ok is False
    assert status.metadata["failure_code"] == "cli_contract_mismatch"
    assert status.metadata["recoverable"] is False
    assert "lark-cli --profile work auth status --json" in status.metadata["next_steps"]


@pytest.mark.asyncio
async def test_cli_backend_kills_subprocess_on_timeout(monkeypatch) -> None:
    process = HangingFakeProcess()

    async def fake_exec(*_argv, **_kwargs):
        return process

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    backend = CliBackend(binary="lark-cli")
    spec = ActionSpec(
        name="im.send_message",
        backend_kind=BackendKind.CLI.value,
        backend_config={"argv": ("im", "+messages-send"), "timeout_s": 0.01},
    )

    result = await backend.execute(spec, {})

    assert result.ok is False
    assert "timed out" in result.error
    assert process.killed is True
    assert process.waited is True


@pytest.mark.asyncio
async def test_rest_backend_preview_is_side_effect_free() -> None:
    backend = RestBackend(base_url="https://api.example.test", token="token")
    spec = ActionSpec(
        name="repo.read_file",
        backend_kind=BackendKind.REST.value,
        backend_config={"method": "GET", "path": "/repos/{repo}/contents/{path}"},
    )

    preview = await backend.preview(spec, {"repo": "demo", "path": "README.md"})

    assert preview.ok is True
    assert preview.summary == "GET /repos/demo/contents/README.md"
    assert preview.data["backend_kind"] == "rest"


class FakeHttpClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def request_json(
        self,
        method: str,
        url: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = 10.0,
    ) -> tuple[int, JsonBody]:
        self.requests.append({
            "method": method,
            "url": url,
            "json_body": dict(json_body or {}),
            "headers": dict(headers or {}),
            "timeout_s": timeout_s,
        })
        return 200, {"id": "repo-1"}


@pytest.mark.asyncio
async def test_rest_backend_executes_registered_action() -> None:
    http = FakeHttpClient()
    backend = RestBackend(base_url="https://api.example.test", token="token", http_client=http)
    spec = ActionSpec(
        name="repo.read_file",
        backend_kind=BackendKind.REST.value,
        backend_config={"method": "POST", "path": "/repos/{repo}/contents", "json_body": {"path": "{path}"}},
    )

    result = await backend.execute(spec, {"repo": "demo", "path": "README.md"})

    assert result.ok is True
    assert result.resource_id == "repo-1"
    assert http.requests[0]["url"] == "https://api.example.test/repos/demo/contents"
    assert http.requests[0]["json_body"] == {"path": "README.md"}
    assert http.requests[0]["headers"]["Authorization"] == "Bearer token"
