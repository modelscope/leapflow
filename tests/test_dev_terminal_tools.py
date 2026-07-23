"""Tests for P2 tools: test_run, lint_check (shell_run wrappers) and the
terminal_session lifecycle (opt-in persistent shells).

test_run/lint_check are exercised with a mocked shell_run for deterministic
output parsing; terminal_session uses a real /bin/sh with timing-tolerant reads.
"""
from __future__ import annotations

import asyncio

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

def test_terminal_disabled_by_default() -> None:
    ts.set_terminal_sessions_enabled(False)
    result = _run(terminal_open({}))
    assert result["ok"] is False and result["failure_code"] == "disabled"


def test_terminal_session_lifecycle() -> None:
    ts.set_terminal_sessions_enabled(True)
    try:
        opened = _run(terminal_open({"shell": "/bin/sh"}))
        assert opened["ok"] is True
        sid = opened["session_id"]

        sent = _run(terminal_send({"session_id": sid, "input": "echo hello-term", "wait": 0.6}))
        assert sent["ok"] is True
        output = sent["output"]
        for _ in range(5):
            if "hello-term" in output:
                break
            output += _run(terminal_read({"session_id": sid, "wait": 0.4}))["output"]
        assert "hello-term" in output

        listed = _run(terminal_list({}))
        assert any(s["session_id"] == sid for s in listed["sessions"])

        closed = _run(terminal_close({"session_id": sid}))
        assert closed["ok"] is True and closed["closed"] is True

        after = _run(terminal_send({"session_id": sid, "input": "x"}))
        assert after["ok"] is False and after["failure_code"] == "session_not_found"
    finally:
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
    opened = _run(terminal_open({"shell": "/bin/sh"}))
    try:
        assert opened["ok"] is True
        second = _run(terminal_open({"shell": "/bin/sh"}))
        assert second["ok"] is False and second["failure_code"] == "too_many_sessions"
    finally:
        _run(terminal_close({"session_id": opened["session_id"]}))
        ts.set_terminal_sessions_enabled(False)
