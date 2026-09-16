# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for action_utils — dispatch contract and lazy-import safety."""

import subprocess
import sys

from leapspace.app_space import action_utils


def test_module_imports_without_x11_or_atspi():
    # every function owns its heavy imports lazily, so the module must import
    # under any interpreter (host test env has neither pyatspi nor a display)
    assert callable(action_utils.unmap_overlay)
    assert callable(action_utils.dump_elements)
    assert callable(action_utils.type_text)


def test_main_without_function_prints_usage(capsys):
    assert action_utils.main([]) == 2
    assert "usage" in capsys.readouterr().err


def test_main_rejects_unknown_function(capsys):
    assert action_utils.main(["bogus"]) == 2
    assert "unknown function 'bogus'" in capsys.readouterr().err


def test_main_dispatches_by_function_name(monkeypatch):
    calls = []
    monkeypatch.setattr(action_utils, "type_text", lambda text: calls.append(text))
    assert action_utils.main(["type_text", "hi"]) == 0
    assert calls == ["hi"]


def test_python_m_roundtrip_rejects_unknown_function():
    # the actor's production invocation form is `python -m`; verify the
    # __main__ guard dispatches the same way
    result = subprocess.run(
        [sys.executable, "-m", "leapspace.app_space.action_utils", "bogus"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unknown function" in result.stderr
