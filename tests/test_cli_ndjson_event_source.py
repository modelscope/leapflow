"""Tests for CliNdjsonEventSource — generic CLI NDJSON subprocess manager."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from leapflow.gateway.connectors.event_sources import CliEventSourceConfig, CliNdjsonEventSource


def _config(**overrides: Any) -> CliEventSourceConfig:
    defaults = {
        "binary": "fake-cli",
        "args": ("event", "consume", "test.event"),
        "platform_id": "test",
        "ready_pattern": r"\[event\] ready",
        "error_pattern": r"\[error\]",
        "ready_timeout_s": 2.0,
        "restart_backoff_base_s": 0.01,
        "max_restart_backoff_s": 0.05,
        "max_restarts": 3,
    }
    defaults.update(overrides)
    return CliEventSourceConfig(**defaults)


class FakeStreamReader:
    """Simulates asyncio.StreamReader for testing."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)
        self._index = 0

    async def readline(self) -> bytes:
        if self._index < len(self._lines):
            line = self._lines[self._index]
            self._index += 1
            return line
        return b""


class FakeStreamWriter:
    """Simulates asyncio.StreamWriter for testing."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def is_closing(self) -> bool:
        return self.closed

    async def drain(self) -> None:
        pass


class FakeProcess:
    """Simulates asyncio.subprocess.Process for NDJSON event source testing."""

    def __init__(
        self,
        stdout_lines: list[bytes],
        stderr_lines: list[bytes] | None = None,
        exit_code: int = 0,
    ) -> None:
        self.stdout = FakeStreamReader(stdout_lines)
        self.stderr = FakeStreamReader(
            stderr_lines if stderr_lines is not None
            else [b"[event] ready event_key=test.event\n"],
        )
        self.stdin = FakeStreamWriter()
        self.returncode: int | None = None
        self._exit_code = exit_code

    async def wait(self) -> int:
        self.returncode = self._exit_code
        return self._exit_code


@pytest.mark.asyncio
async def test_events_yields_parsed_ndjson(monkeypatch: Any) -> None:
    """Valid NDJSON lines are parsed into BackendEvent objects."""
    process = FakeProcess(
        stdout_lines=[
            b'{"event_id":"e1","type":"im.message.receive_v1","content":"hello"}\n',
            b'{"event_id":"e2","type":"im.message.receive_v1","content":"world"}\n',
        ],
    )

    async def fake_exec(*_argv: Any, **_kw: Any) -> FakeProcess:
        return process

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    source = CliNdjsonEventSource(_config())
    status = await source.start()
    assert status.ok is True

    events = []
    async for event in source.events():
        events.append(event)
        if len(events) >= 2:
            break

    await source.stop()

    assert len(events) == 2
    assert events[0].event_id == "e1"
    assert events[0].event_type == "im.message.receive_v1"
    assert events[0].platform_id == "test"
    assert events[0].payload["content"] == "hello"
    assert events[1].event_id == "e2"


@pytest.mark.asyncio
async def test_malformed_lines_skipped(monkeypatch: Any) -> None:
    """Non-JSON lines and empty lines are silently skipped."""
    process = FakeProcess(
        stdout_lines=[
            b"not json\n",
            b"\n",
            b'{"event_id":"e1","type":"test.event"}\n',
        ],
    )

    async def fake_exec(*_argv: Any, **_kw: Any) -> FakeProcess:
        return process

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    source = CliNdjsonEventSource(_config())
    await source.start()

    events = []
    async for event in source.events():
        events.append(event)

    assert len(events) == 1
    assert events[0].event_id == "e1"


@pytest.mark.asyncio
async def test_ready_timeout_returns_not_ok(monkeypatch: Any) -> None:
    """If stderr ready marker not received within timeout, start returns ok=False."""
    process = FakeProcess(
        stdout_lines=[],
        stderr_lines=[b"some other log\n"],
    )

    async def fake_exec(*_argv: Any, **_kw: Any) -> FakeProcess:
        return process

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    source = CliNdjsonEventSource(_config(ready_timeout_s=0.1))
    status = await source.start()
    assert status.ok is False
    assert "ready" in status.detail.lower()


@pytest.mark.asyncio
async def test_binary_not_found_returns_not_ok(monkeypatch: Any) -> None:
    """FileNotFoundError from spawn returns ok=False with descriptive detail."""
    async def fake_exec(*_argv: Any, **_kw: Any) -> FakeProcess:
        raise FileNotFoundError("fake-cli not found")

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    source = CliNdjsonEventSource(_config())
    status = await source.start()
    assert status.ok is False
    assert "not found" in status.detail.lower()


@pytest.mark.asyncio
async def test_stderr_json_error_detected(monkeypatch: Any) -> None:
    """Structured JSON error on stderr is captured before ready."""
    error_json = b'{"ok":false,"error":{"type":"authentication","message":"missing token","hint":"run auth login"}}\n'
    process = FakeProcess(
        stdout_lines=[],
        stderr_lines=[error_json],
    )

    async def fake_exec(*_argv: Any, **_kw: Any) -> FakeProcess:
        return process

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    source = CliNdjsonEventSource(_config(ready_timeout_s=0.5))
    status = await source.start()
    assert status.ok is False
    assert "missing token" in status.detail


@pytest.mark.asyncio
async def test_stop_is_idempotent(monkeypatch: Any) -> None:
    """Calling stop multiple times does not raise."""
    process = FakeProcess(stdout_lines=[])

    async def fake_exec(*_argv: Any, **_kw: Any) -> FakeProcess:
        return process

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    source = CliNdjsonEventSource(_config())
    await source.start()
    status1 = await source.stop()
    status2 = await source.stop()
    assert status1.ok is True
    assert status2.ok is True


@pytest.mark.asyncio
async def test_status_reports_running_state(monkeypatch: Any) -> None:
    """Status correctly reflects running/stopped state."""
    process = FakeProcess(
        stdout_lines=[b'{"event_id":"e1","type":"test"}\n'],
    )

    async def fake_exec(*_argv: Any, **_kw: Any) -> FakeProcess:
        return process

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    source = CliNdjsonEventSource(_config())

    status_before = await source.status()
    assert status_before.ok is False

    await source.start()
    status_running = await source.status()
    assert status_running.ok is True
    assert status_running.metadata["started"] is True

    await source.stop()
    status_stopped = await source.status()
    assert status_stopped.metadata["started"] is False


@pytest.mark.asyncio
async def test_event_type_from_type_field(monkeypatch: Any) -> None:
    """lark-cli uses 'type' field; fallback to 'event_type'."""
    process = FakeProcess(
        stdout_lines=[
            b'{"event_id":"e1","type":"im.message.receive_v1"}\n',
            b'{"event_id":"e2","event_type":"custom.event"}\n',
        ],
    )

    async def fake_exec(*_argv: Any, **_kw: Any) -> FakeProcess:
        return process

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    source = CliNdjsonEventSource(_config())
    await source.start()

    events = []
    async for event in source.events():
        events.append(event)

    assert events[0].event_type == "im.message.receive_v1"
    assert events[1].event_type == "custom.event"
