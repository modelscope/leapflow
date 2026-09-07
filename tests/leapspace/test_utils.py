"""Tests for leapspace.app_space.utils: check() lines and path conventions."""

import asyncio
import platform

import pytest

pytest.importorskip("cua_sandbox")  # leapspace extra only (utils imports cua_sandbox)

from leapspace.app_space.utils import (
    check,
    get_image_venv_python,
    get_sandbox_state_dir,
    load_action,
)


def test_check_pass_line(capsys):
    assert check("reply-sent", True, "found") is True
    assert capsys.readouterr().out == "PASS reply-sent: found\n"


def test_check_fail_line_without_detail(capsys):
    assert check("badge-cleared", False) is False
    assert capsys.readouterr().out == "FAIL badge-cleared: \n"


def test_check_accumulates_with_and():
    ok = True
    ok &= check("a", True)
    ok &= check("b", False)
    ok &= check("c", True)
    assert ok is False


def test_state_dir_from_system_name_on_host():
    assert str(get_sandbox_state_dir(False, "linux")) == "/tmp/leapspace"


def test_state_dir_requires_system_on_host():
    with pytest.raises(ValueError, match="system"):
        get_sandbox_state_dir(False)


def test_state_dir_detects_running_os_in_sandbox(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    assert str(get_sandbox_state_dir(True)) == "C:\\ProgramData\\leapspace"


def test_image_venv_python_pins_linux_venv():
    # `make space-sync` is a bare `uv sync`, so the venv is .venv
    assert get_image_venv_python("linux") == "/opt/leapflow/.venv/bin/python"


def test_image_venv_python_undefined_for_other_systems():
    with pytest.raises(NotImplementedError, match="macos"):
        get_image_venv_python("macos")


ACTION = '''\
import sys

async def reference(actor):
    return "stim"

def expect(state_root="/tmp/leapspace"):
    return 0

if __name__ == "__main__":
    sys.exit(expect())
'''


def test_load_action_returns_reference_and_expect(tmp_path):
    (tmp_path / "action.py").write_text(ACTION)
    reference, expect = load_action(tmp_path / "action.py")
    assert asyncio.iscoroutinefunction(reference)
    assert expect() == 0


def test_load_action_leaves_main_guard_inert(tmp_path):
    # a load that ran the guard would raise SystemExit(expect()) here
    (tmp_path / "action.py").write_text(
        ACTION.replace("return 0", "return 2")
    )
    load_action(tmp_path / "action.py")
