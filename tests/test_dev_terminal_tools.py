# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for P2 tools: test_run, lint_check (shell_run wrappers) and the
terminal_session lifecycle (opt-in persistent shells).

test_run/lint_check are exercised with a mocked shell_run for deterministic
output parsing; terminal_session uses a real /bin/sh with timing-tolerant reads.
"""
from __future__ import annotations

import asyncio
import sys

import pytest

from leapflow.tools import dev_tools
from leapflow.tools import terminal_session as ts
from leapflow.tools.dev_tools import lint_check, set_dev_commands
from leapflow.tools.terminal_session import (
    terminal_close,
    terminal_list,
    terminal_open,
    terminal_read,
    terminal_send,
)


def _run(coro):
    return asyncio.run(coro)


def _fake_shell(returncode: int, stdout: str = "", stderr: str = ""):
    async def _sh(params):
        return {"ok": returncode == 0, "returncode": returncode, "stdout": stdout, "stderr": stderr}
    return _sh


# ── test_run ─────────────────────────────────────────────────────────

def test_test_run_pytest_parse_failure(tmp_path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    set_dev_commands()
    monkeypatch.setattr(dev_tools, "shell_run", _fake_shell(1, stdout="FAILED tests/test_x.py::test_a\n3 passed, 1 failed in 0.5s"))
    result = _run(dev_tools.test_run({"cwd": str(tmp_path)}))
    assert result["ok"] is True and result["framework"] == "pytest"
    assert result["passed"] == 3 and result["failed"] == 1 and result["success"] is False
    assert any("test_x" in f for f in result["failures"])
    assert "pytest" in result["command"]


def test_test_run_pytest_success(tmp_path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    set_dev_commands()
    monkeypatch.setattr(dev_tools, "shell_run", _fake_shell(0, stdout="10 passed in 1.0s"))
    result = _run(dev_tools.test_run({"cwd": str(tmp_path)}))
    assert result["ok"] is True and result["success"] is True and result["passed"] == 10


def test_test_run_no_command_detected(tmp_path) -> None:
    set_dev_commands()
    result = _run(dev_tools.test_run({"cwd": str(tmp_path)}))
    assert result["ok"] is False and result["failure_code"] == "no_test_command"


def test_test_run_config_override(tmp_path, monkeypatch) -> None:
    set_dev_commands(test_command="mytest --run")
    captured: dict = {}

    async def _sh(params):
        captured["cmd"] = params["command"]
        return {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(dev_tools, "shell_run", _sh)
    try:
        result = _run(dev_tools.test_run({"cwd": str(tmp_path)}))
    finally:
        set_dev_commands()
    assert result["ok"] is True and captured["cmd"] == "mytest --run"


def test_test_run_runner_error(tmp_path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    set_dev_commands()

    async def _sh(params):
        return {"ok": False, "error": "Command timed out after 120s"}

    monkeypatch.setattr(dev_tools, "shell_run", _sh)
    result = _run(dev_tools.test_run({"cwd": str(tmp_path)}))
    assert result["ok"] is False and result["failure_code"] == "runner_error"


# ── lint_check ───────────────────────────────────────────────────────

def test_lint_check_clean(tmp_path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    set_dev_commands()
    monkeypatch.setattr(dev_tools, "shell_run", _fake_shell(0, stdout="All checks passed!"))
    result = _run(lint_check({"cwd": str(tmp_path)}))
    assert result["ok"] is True and result["clean"] is True and result["issue_count"] == 0


def test_lint_check_reports_issue_count(tmp_path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    set_dev_commands()
    monkeypatch.setattr(dev_tools, "shell_run", _fake_shell(1, stdout="x.py:1:1: F401 unused\nFound 3 errors."))
    result = _run(lint_check({"cwd": str(tmp_path)}))
    assert result["ok"] is True and result["clean"] is False and result["issue_count"] == 3


# ── terminal_session ─────────────────────────────────────────────────

@pytest.mark.parametrize("command", [
    "leap daemon stop", "leap daemon restart --force",
    "nohup sh -c 'sleep 5; leap daemon restart' & disown",
    "setsid leap daemon stop", "pkill -f leapd", "killall leapd",
    "kill -TERM 4242", 'kill "$LEAPFLOW_HOST_DAEMON_PID"',
    "kill -9 ${LEAPFLOW_HOST_DAEMON_PID}", "kill $(cat leapd.pid)",
])
def test_shell_self_termination_is_typed_and_never_executed(command, monkeypatch) -> None:
    from leapflow.tools.shell_tools import shell_run
    monkeypatch.setenv("LEAPFLOW_HOST_DAEMON_PID", "4242")
    async def forbidden(*args, **kwargs):
        raise AssertionError("blocked command must never spawn")
    monkeypatch.setattr(asyncio, "create_subprocess_shell", forbidden)
    result = _run(shell_run({"command": command}))
    assert result["ok"] is False
    assert result["failure_code"] == "self_termination_blocked"
    assert result["failure_class"] == "safety_policy"
    assert result["side_effect_state"] == "none"
    assert result["retryable"] is False
    assert "hw_fleet_reconnect" in result["next_step"]


@pytest.mark.parametrize("command", ["leap daemon status", "kill -0 4242", "kill 9999", "echo healthy"])
def test_shell_guard_preserves_safe_commands(command, monkeypatch) -> None:
    from leapflow.tools.shell_tools import _is_self_termination
    monkeypatch.setenv("LEAPFLOW_HOST_DAEMON_PID", "4242")
    assert not _is_self_termination(command)


def test_terminal_initial_and_send_commands_share_guard(monkeypatch) -> None:
    from types import SimpleNamespace
    monkeypatch.setenv("LEAPFLOW_HOST_DAEMON_PID", "4242")
    ts.set_terminal_sessions_enabled(True)
    try:
        opened = _run(terminal_open({"command": "leap daemon restart --force"}))
        assert opened["failure_code"] == "self_termination_blocked"
        monkeypatch.setattr(ts, "_get", lambda _: SimpleNamespace(alive=lambda: True))
        sent = _run(terminal_send({"session_id": "guard-probe", "input": "kill 4242"}))
        assert sent["failure_code"] == "self_termination_blocked"
        assert sent["side_effect_state"] == "none"
    finally:
        ts.set_terminal_sessions_enabled(False)


def test_terminal_disabled_by_default() -> None:
    ts.set_terminal_sessions_enabled(False)
    result = _run(terminal_open({}))
    assert result["ok"] is False and result["failure_code"] == "disabled"


def test_terminal_session_lifecycle() -> None:
    ts.set_terminal_sessions_enabled(True)
    sid = ""
    try:
        opened = _run(terminal_open({}))
        assert opened["ok"] is True, opened
        sid = opened["session_id"]

        # PowerShell cold-starts slowly (banner + profile + first prompt),
        # especially under parallel xdist load; widen the capture budget there.
        send_wait, read_wait, reads = (2.0, 1.0, 8) if sys.platform == "win32" else (0.6, 0.4, 5)
        sent = _run(terminal_send({"session_id": sid, "input": "echo hello-term", "wait": send_wait}))
        assert sent["ok"] is True
        output = sent["output"]
        for _ in range(reads):
            if "hello-term" in output:
                break
            output += _run(terminal_read({"session_id": sid, "wait": read_wait}))["output"]
        assert "hello-term" in output

        listed = _run(terminal_list({}))
        assert any(s["session_id"] == sid for s in listed["sessions"])

        closed = _run(terminal_close({"session_id": sid}))
        assert closed["ok"] is True and closed["closed"] is True
        sid = ""

        after = _run(terminal_send({"session_id": sid, "input": "x"}))
        assert after["ok"] is False and after["failure_code"] == "session_not_found"
    finally:
        if sid:
            _run(terminal_close({"session_id": sid}))
        ts.set_terminal_sessions_enabled(False)


def test_terminal_send_unknown_session() -> None:
    ts.set_terminal_sessions_enabled(True)
    try:
        result = _run(terminal_send({"session_id": "does-not-exist", "input": "x"}))
        assert result["ok"] is False and result["failure_code"] == "session_not_found"
    finally:
        ts.set_terminal_sessions_enabled(False)


def test_terminal_max_sessions(monkeypatch) -> None:
    ts.set_terminal_sessions_enabled(True)
    monkeypatch.setattr(ts, "_MAX_SESSIONS", 1)
    opened = _run(terminal_open({}))
    try:
        assert opened["ok"] is True, opened
        second = _run(terminal_open({}))
        assert second["ok"] is False and second["failure_code"] == "too_many_sessions"
    finally:
        _run(terminal_close({"session_id": opened.get("session_id", "")}))
        ts.set_terminal_sessions_enabled(False)
