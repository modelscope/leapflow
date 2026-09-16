# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for TUI tool-audit rendering fidelity.

These pin the three defects observed in a real session: a parallel batch printed
one line instead of two, that line showed a sibling call's arguments, and the
failure text shown was the head of a traceback rather than the exception it ended
with.
"""

from __future__ import annotations

import time

from leapflow.cli.tui_app.stream import StreamRenderer
from leapflow.engine.engine import _tool_args_metadata, _tool_result_metadata

CMD = (
    'curl -s "https://query1.finance.yahoo.com/v8/finance/chart/BABA" 2>/dev/null | python3 -c "\n'
    "import json, sys\ndata = json.load(sys.stdin)\n\""
)
TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "<string>", line 3, in <module>\n'
    "    data = json.load(sys.stdin)\n"
    '  File ".../json/__init__.py", line 298, in load\n'
    "    return loads(fp.read(),\n"
    '  File ".../json/decoder.py", line 363, in raw_decode\n'
    '    raise JSONDecodeError("Expecting value", s, err.value) from None\n'
    "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)\n"
)
SHELL_FAILURE = {
    "ok": False,
    "returncode": 1,
    "stdout": "",
    "stderr": TRACEBACK,
    "error": TRACEBACK[-800:],
    "execution_policy": "external_side_effect",
}


class _CapturingConsole:
    """Collects printed lines as plain text and ignores everything else."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, renderable="") -> None:
        self.lines.append(getattr(renderable, "plain", str(renderable)))

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def _render_batch() -> tuple[_CapturingConsole, StreamRenderer]:
    """Replay the recovered batch: shell_run(command=...) then time_get({})."""
    console = _CapturingConsole()
    renderer = StreamRenderer(console)
    renderer.start()
    renderer.tool_started(
        "shell_run",
        _tool_args_metadata("shell_run", {"command": CMD, "timeout": 15}, tool_call_id="call_shell"),
    )
    renderer.tool_started("time_get", _tool_args_metadata("time_get", {}, tool_call_id="call_time"))
    renderer.tool_finished(
        "shell_run",
        metadata=_tool_result_metadata(
            "shell_run", {"command": CMD}, SHELL_FAILURE, tool_call_id="call_shell"
        ),
    )
    renderer.tool_finished(
        "time_get",
        metadata=_tool_result_metadata(
            "time_get", {}, {"ok": True, "human": "2026-08-05 10:41:07"}, tool_call_id="call_time"
        ),
    )
    return console, renderer


def test_every_tool_in_a_batch_prints_its_own_line() -> None:
    """A sibling's result must not disappear.

    With a single shared slot the second completion found the timer already
    cleared and printed nothing, so a batch of two tools reported one.
    """
    console, renderer = _render_batch()

    assert len(console.lines) == 2
    assert console.lines[0].strip().startswith("✗ shell_run")
    assert "time_get" in console.lines[1]
    assert renderer.tool_count == 2


def test_failed_line_shows_its_own_command_not_a_siblings_arguments() -> None:
    """The audit line must describe the call it belongs to.

    Previously the detail came from whichever start ran last, so a failing
    shell_run was labelled with `time_get`'s empty argument dict.
    """
    console, _ = _render_batch()
    shell_line = console.lines[0]

    assert "$ curl -s" in shell_line
    assert "{}" not in shell_line


def test_failed_line_shows_the_exception_and_exit_code() -> None:
    """The cause is the last line of a traceback, so the preview must reach it."""
    console, _ = _render_batch()
    shell_line = console.lines[0]

    assert "JSONDecodeError" in shell_line
    assert "exit=1" in shell_line


def test_single_call_path_without_ids_still_renders() -> None:
    """Callers that emit no tool_call_id must keep working."""
    console = _CapturingConsole()
    renderer = StreamRenderer(console)
    renderer.start()
    renderer.tool_started("file_read")
    renderer.tool_finished("file_read", metadata={"ok": True, "normalized_tool_name": "file_read"})

    assert len(console.lines) == 1
    assert "file_read" in console.lines[0]
    assert renderer.tool_count == 1


def test_repeated_same_name_calls_pair_oldest_first() -> None:
    """Two calls to one tool without ids must produce two lines, not one."""
    console = _CapturingConsole()
    renderer = StreamRenderer(console)
    renderer.start()
    renderer.tool_started("shell_run")
    renderer.tool_started("shell_run")
    renderer.tool_finished("shell_run", metadata={"ok": True, "normalized_tool_name": "shell_run"})
    renderer.tool_finished("shell_run", metadata={"ok": True, "normalized_tool_name": "shell_run"})

    assert len(console.lines) == 2
    assert renderer.tool_count == 2


def test_durations_are_attributed_per_call() -> None:
    """Elapsed time must come from the call's own start, not the batch's last."""
    console = _CapturingConsole()
    renderer = StreamRenderer(console)
    renderer.start()
    renderer.tool_started("slow_tool", {"tool_call_id": "a", "normalized_tool_name": "slow_tool"})
    time.sleep(0.05)
    renderer.tool_started("fast_tool", {"tool_call_id": "b", "normalized_tool_name": "fast_tool"})
    renderer.tool_finished("fast_tool", metadata={"ok": True, "normalized_tool_name": "fast_tool", "tool_call_id": "b"})
    renderer.tool_finished("slow_tool", metadata={"ok": True, "normalized_tool_name": "slow_tool", "tool_call_id": "a"})

    names = [name for name, _duration in renderer._tool_history]
    durations = dict(renderer._tool_history)
    assert names == ["fast_tool", "slow_tool"]
    assert durations["slow_tool"] > durations["fast_tool"]


def test_spinner_reports_batch_depth() -> None:
    """Two running calls must not look like one."""
    renderer = StreamRenderer(_CapturingConsole())
    renderer.start()
    first = renderer.tool_started("shell_run", {"tool_call_id": "a", "normalized_tool_name": "shell_run"})
    second = renderer.tool_started("time_get", {"tool_call_id": "b", "normalized_tool_name": "time_get"})

    assert first.endswith("shell_run")
    assert second.endswith("+1")


def test_hidden_tools_release_their_slot() -> None:
    """A ui_hidden completion prints nothing and leaves no stale in-flight entry."""
    console = _CapturingConsole()
    renderer = StreamRenderer(console)
    renderer.start()
    renderer.tool_started("skipped", {"tool_call_id": "a", "normalized_tool_name": "skipped"})
    renderer.tool_finished(
        "skipped",
        metadata={"ok": True, "normalized_tool_name": "skipped", "tool_call_id": "a", "ui_hidden": True},
    )

    assert console.lines == []
    assert renderer.tool_count == 0
    assert renderer._active_tools == {}


def test_exit_code_is_read_under_either_key_name() -> None:
    """Shell tools emit `returncode`; evidence and UI read `exit_code`."""
    from leapflow.engine.context_control import ToolEvidenceBuilder
    from leapflow.engine.tool_execution import exit_code_from

    assert exit_code_from({"returncode": 2}) == 2
    assert exit_code_from({"exit_code": 3}) == 3
    assert exit_code_from({"exit_code": None, "returncode": 4}) == 4
    assert exit_code_from({"ok": True}) is None
    # Booleans are not exit codes even though bool is an int subclass.
    assert exit_code_from({"returncode": True}) is None

    evidence = ToolEvidenceBuilder(max_content_chars=200).build(
        "shell_run", {}, {"ok": True, "returncode": 0, "stdout": "done", "stderr": ""}
    )
    assert evidence["exit_code"] == 0

    metadata = _tool_result_metadata("shell_run", {}, {"ok": False, "returncode": 1, "stderr": "boom"})
    assert metadata["exit_code"] == 1
