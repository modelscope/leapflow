from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest

import leapflow.cli.tui_app.app as app_module
from leapflow.security.approval import ApprovalDecision, ApprovalRequest
from prompt_toolkit.auto_suggest import Suggestion
from prompt_toolkit.completion import Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import to_plain_text
from leapflow.cli.tui_app.app import LeapApp, _DynamicPlaceholderProcessor
from leapflow.cli.tui_app.console import LeapConsole
from leapflow.cli.tui_app.command import TuiCommand, TuiCommandStatus
from leapflow.cli.tui_app.input import SlashCommandCompleter
from leapflow.cli.tui_app.stream import StreamRenderer, _sanitize_final_response
from leapflow.cli.tui_app.theme import _LIGHT, resolve_theme


class _FakeConsole:
    def __init__(self) -> None:
        self.cards: list[TuiCommand] = []
        self.errors: list[str] = []
        self.systems: list[str] = []
        self.warnings: list[str] = []

    def command_card(self, command: TuiCommand) -> None:
        self.cards.append(command)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def system(self, message: str) -> None:
        self.systems.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _FakeStatus:
    def __init__(self) -> None:
        self.counts: list[tuple[int, int]] = []

    def __call__(self) -> list[tuple[str, str]]:
        return []

    def update_task_counts(self, *, running: int, queued: int) -> None:
        self.counts.append((running, queued))


async def _wait_for(condition, *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def _make_app(
    on_input=None,
    *,
    commands: tuple[tuple[str, str], ...] = (),
    on_control=None,
) -> tuple[LeapApp, _FakeConsole, _FakeStatus]:
    console = _FakeConsole()
    status = _FakeStatus()
    app = LeapApp(
        console=console,
        theme=resolve_theme(_LIGHT, terminal_bg="#FFFFFF"),
        status=status,
        commands=commands,
        on_input=on_input,
        on_control=on_control,
    )
    return app, console, status


@pytest.mark.asyncio
async def test_tui_approval_modal_is_keyboard_selectable() -> None:
    app, _console, _status = _make_app()
    request = ApprovalRequest(
        category="shell.command",
        detail="python -m pip install aiohttp",
        default_choice="allow_once",
    )

    task = asyncio.create_task(app.request_approval(request))
    await _wait_for(lambda: app._approval_modal is not None)
    modal = app._approval_modal
    assert modal is not None
    assert modal.selected_index == 0

    modal.move(1)
    assert modal.selected_index == 1
    fragments = modal.fragments()
    rendered = "".join(text for _style, text in fragments)
    assert "→" not in rendered
    assert "╭" in rendered and "╮" in rendered
    assert "╰" in rendered and "╯" in rendered
    assert "▸ 2. Allow for this session" in rendered

    modal.choose_selected()
    assert await task == ApprovalDecision.ALLOW_SESSION
    assert app._approval_modal is None


@pytest.mark.asyncio
async def test_tui_approval_modal_preserves_frame_and_choices_in_small_height() -> None:
    request = ApprovalRequest(
        category="shell.command",
        detail="python3 -m pip install aiohttp requests --user --quiet 2>&1\n" * 5,
        choices=("allow_once", "allow_session", "allow_always", "deny"),
        display={
            "summary": "Run shell command",
            "reason": "This shell command has side effects or reaches external systems.",
        },
    )
    modal = app_module.ApprovalModal.create(request)

    rendered = "".join(text for _style, text in modal.fragments())
    lines = [line for line in rendered.splitlines() if line]

    assert lines[0].startswith("╭") and lines[0].endswith("╮")
    assert lines[-1].startswith("╰") and lines[-1].endswith("╯")
    assert "1. Allow once" in rendered
    assert "2. Allow for this session" in rendered
    assert "3. Add to permanent allowlist" in rendered
    assert "4. Deny" in rendered


@pytest.mark.asyncio
async def test_tui_approval_modal_constrained_height_preserves_choices() -> None:
    request = ApprovalRequest(
        category="shell.command",
        detail="python3 -m pip install aiohttp requests --user --quiet 2>&1\n" * 10,
        choices=("allow_once", "allow_session", "deny"),
        display={
            "summary": "Run a potentially dangerous shell command",
            "reason": "This shell command has side effects or reaches external systems.",
        },
    )
    modal = app_module.ApprovalModal.create(request)

    unconstrained = modal.line_count()
    constrained_limit = 12
    assert unconstrained > constrained_limit

    rendered = "".join(text for _style, text in modal.fragments(max_lines=constrained_limit))
    lines = [line for line in rendered.splitlines() if line]

    assert lines[0].startswith("╭") and lines[0].endswith("╮")
    assert lines[-1].startswith("╰") and lines[-1].endswith("╯")
    assert "1. Allow once" in rendered
    assert "2. Allow for this session" in rendered
    assert "3. Deny" in rendered
    assert len(lines) <= constrained_limit

    constrained_count = modal.line_count(max_lines=constrained_limit)
    assert constrained_count <= constrained_limit
    assert constrained_count == len(lines)


@pytest.mark.asyncio
async def test_tui_approval_modal_extremely_tight_height_still_shows_choices() -> None:
    request = ApprovalRequest(
        category="shell.command",
        detail="rm -rf /important",
        choices=("allow_once", "deny"),
        display={"summary": "Run shell command", "reason": "Dangerous command"},
    )
    modal = app_module.ApprovalModal.create(request)

    rendered = "".join(text for _style, text in modal.fragments(max_lines=7))
    lines = [line for line in rendered.splitlines() if line]

    assert lines[0].startswith("╭") and lines[0].endswith("╮")
    assert lines[-1].startswith("╰") and lines[-1].endswith("╯")
    assert "1. Allow once" in rendered
    assert "2. Deny" in rendered


@pytest.mark.asyncio
async def test_tui_approval_modal_shortcuts_and_details() -> None:
    app, _console, _status = _make_app()
    request = ApprovalRequest(
        category="gateway_send",
        detail="send external message\n" * 8,
        choices=("allow_once", "show_details", "deny"),
    )

    task = asyncio.create_task(app.request_approval(request))
    await _wait_for(lambda: app._approval_modal is not None)
    modal = app._approval_modal
    assert modal is not None

    assert modal.choose_text("2") is True
    assert modal.show_details is True
    assert task.done() is False
    assert modal.choose_text("n") is True
    assert await task == ApprovalDecision.DENY


def test_submit_text_assigns_ids_and_keeps_input_editable() -> None:
    app, console, status = _make_app()

    first = app.submit_text("first command")
    second = app.submit_text("second command")

    assert first.id == 1
    assert second.id == 2
    assert app._pending_input.qsize() == 2
    assert status.counts[-1] == (0, 2)
    assert [card.status for card in console.cards] == [TuiCommandStatus.QUEUED]
    assert app._input_area.buffer.read_only() is False


def test_submit_text_rejects_empty_commands() -> None:
    app, console, status = _make_app()

    with pytest.raises(ValueError):
        app.submit_text("  \n  ")

    assert app._pending_input.qsize() == 0
    assert console.cards == []
    assert status.counts == []


def test_slash_completer_shows_commands_and_descriptions() -> None:
    completer = SlashCommandCompleter((
        ("help", "Show available commands"),
        ("teach start", "Start teaching mode"),
    ))

    completions = list(completer.get_completions(Document("/", 1), None))

    assert [completion.text for completion in completions] == ["/help", "/teach start"]
    assert [to_plain_text(completion.display) for completion in completions] == ["/help", "/teach start"]
    assert to_plain_text(completions[0].display_meta) == "Show available commands"


def test_slash_completer_filters_multi_word_commands() -> None:
    completer = SlashCommandCompleter((
        ("teach start", "Start teaching mode"),
        ("teach stop", "Stop and distill skill"),
        ("tools", "List available tools"),
    ))

    completions = list(completer.get_completions(Document("/teach s", len("/teach s")), None))

    assert [completion.text for completion in completions] == ["/teach start", "/teach stop"]
    assert all(completion.start_position == -len("/teach s") for completion in completions)


def test_slash_completer_does_not_pollute_natural_language() -> None:
    completer = SlashCommandCompleter((("help", "Show available commands"),))

    assert list(completer.get_completions(Document("please help", len("please help")), None)) == []


def test_tab_accepts_selected_completion() -> None:
    app, _console, _status = _make_app()
    buffer = app._input_area.buffer
    buffer.completer = None
    buffer.auto_suggest = None
    buffer.text = "/he"
    buffer.cursor_position = len(buffer.text)
    completion = Completion("/help", start_position=-len(buffer.text))
    buffer._set_completions([completion])

    app._accept_or_start_completion(buffer)

    assert buffer.text == "/help"
    assert buffer.cursor_position == len("/help")


def test_escape_closes_visible_completion_menu() -> None:
    app, _console, _status = _make_app()
    buffer = app._input_area.buffer
    buffer.completer = None
    buffer.auto_suggest = None
    buffer.text = "/h"
    buffer.cursor_position = len(buffer.text)
    buffer._set_completions([Completion("/help", start_position=-len(buffer.text))])

    assert app._close_completion(buffer) is True

    assert buffer.complete_state is None


def test_down_arrow_navigates_visible_completion_menu() -> None:
    app, _console, _status = _make_app()
    buffer = app._input_area.buffer
    buffer.completer = None
    buffer.auto_suggest = None
    buffer.text = "/h"
    buffer.cursor_position = len(buffer.text)
    buffer._set_completions([
        Completion("/help", start_position=-len(buffer.text)),
        Completion("/host", start_position=-len(buffer.text)),
    ])

    app._completion_next_or_cursor_down(buffer)
    first = buffer.complete_state.current_completion
    app._completion_next_or_cursor_down(buffer)
    second = buffer.complete_state.current_completion

    assert first is not None and first.text == "/help"
    assert second is not None and second.text == "/host"


def test_history_navigation_restores_unsubmitted_draft() -> None:
    app, _console, _status = _make_app()
    buffer = app._input_area.buffer
    buffer.completer = None
    buffer.auto_suggest = None
    buffer.text = "你是谁"
    buffer.cursor_position = len(buffer.text)
    buffer._working_lines.appendleft("上一条消息")
    buffer.working_index += 1

    app._completion_previous_or_cursor_up(buffer)

    assert buffer.text == "上一条消息"
    assert buffer.cursor_position == len("上一条消息")

    app._completion_next_or_cursor_down(buffer)

    assert buffer.text == "你是谁"
    assert buffer.cursor_position == len("你是谁")


def test_up_arrow_keeps_multiline_cursor_navigation_before_history() -> None:
    app, _console, _status = _make_app()
    buffer = app._input_area.buffer
    buffer.completer = None
    buffer.auto_suggest = None
    buffer.text = "第一行\n第二行"
    buffer._working_lines.appendleft("上一条消息")
    buffer.working_index += 1
    buffer.cursor_position = len(buffer.text)
    working_index = buffer.working_index

    app._completion_previous_or_cursor_up(buffer)

    assert buffer.text == "第一行\n第二行"
    assert buffer.working_index == working_index
    assert buffer.document.cursor_position_row == 0


def test_leap_app_exposes_completion_menu_styles() -> None:
    app, _console, _status = _make_app(commands=(("help", "Show available commands"),))
    style = app._build_style()

    assert style.get_attrs_for_style_str("class:completion-menu.completion") is not None
    assert style.get_attrs_for_style_str("class:completion-menu.meta.completion.current") is not None
    assert app._input_area.buffer.completer is not None


@pytest.mark.asyncio
async def test_tab_accepts_visible_auto_suggestion() -> None:
    app, _console, _status = _make_app()
    buffer = app._input_area.buffer
    buffer.text = "你能"
    buffer.cursor_position = len(buffer.text)
    buffer.suggestion = Suggestion("帮我找一找 hub 上有啥 skill 么？")

    app._accept_or_start_completion(buffer)

    assert buffer.text == "你能帮我找一找 hub 上有啥 skill 么？"


@pytest.mark.asyncio
async def test_right_arrow_accepts_suggestion_only_at_end() -> None:
    app, _console, _status = _make_app()
    buffer = app._input_area.buffer
    buffer.text = "abc"
    buffer.cursor_position = 1
    buffer.suggestion = Suggestion("def")

    app._move_right_or_accept_suggestion(buffer)

    assert buffer.text == "abc"
    assert buffer.cursor_position == 2

    buffer.cursor_position = len(buffer.text)
    app._move_right_or_accept_suggestion(buffer)

    assert buffer.text == "abcdef"
    assert buffer.cursor_position == len("abcdef")


def test_command_card_keeps_elapsed_in_title_not_body(monkeypatch) -> None:
    class CapturingConsole:
        width = 100

        def __init__(self) -> None:
            self.rendered = []

        def print(self, renderable) -> None:
            self.rendered.append(renderable)

    capture = CapturingConsole()
    leap_console = LeapConsole(resolve_theme(_LIGHT, terminal_bg="#FFFFFF"))
    monkeypatch.setattr(leap_console, "_console", capture)

    command = TuiCommand.create(command_id=1, text="summarize the current project layout")
    command = command.mark_running().mark_done()
    leap_console.command_card(command)

    panel = capture.rendered[0]
    title = getattr(panel.title, "plain", str(panel.title))
    body = getattr(panel.renderable, "plain", str(panel.renderable))
    assert "done" in title
    assert "elapsed:" not in body
    assert "summarize the current project layout" in body


def test_response_label_merges_done_command_status(monkeypatch) -> None:
    class CapturingConsole:
        width = 100

        def __init__(self) -> None:
            self.rendered = []

        def print(self, renderable) -> None:
            self.rendered.append(renderable)

    capture = CapturingConsole()
    leap_console = LeapConsole(resolve_theme(_LIGHT, terminal_bg="#FFFFFF"))
    monkeypatch.setattr(leap_console, "_console", capture)

    command = TuiCommand.create(command_id=1, text="连接飞书并查看可用的群组")
    command = command.mark_running().mark_done()

    leap_console.response_label(16.5, tool_count=3, command=command)

    assert len(capture.rendered) == 1
    assert capture.rendered[0].plain == " |--  LEAP  #1 done  16.5s  3 tools"


def test_stream_renderer_prints_tool_command_and_success_preview() -> None:
    class CaptureConsole:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def print(self, renderable) -> None:
            self.lines.append(getattr(renderable, "plain", str(renderable)))

        def thinking(self, text: str) -> None:
            pass

        def markdown(self, text: str, *, indent: int = 0, margin_top: int = 0) -> None:
            pass

        def response_label(self, elapsed_s: float, *, tool_count: int = 0, command=None) -> None:
            pass

        def newline(self) -> None:
            pass

    console = CaptureConsole()
    renderer = StreamRenderer(console)  # type: ignore[arg-type]
    renderer.start()

    renderer.tool_started("shell_run", metadata={"command": "python -V"})
    renderer.tool_finished("shell_run", metadata={"ok": True, "stdout_preview": "Python 3.13.0"})

    assert len(console.lines) == 1
    line = console.lines[0]
    assert "💻 shell_run" in line
    assert "$ python -V" in line
    assert "→ Python 3.13.0" in line
    assert " | " in line


def test_stream_renderer_prints_normalized_tool_name_with_alias_hint() -> None:
    class CaptureConsole:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def print(self, renderable) -> None:
            self.lines.append(getattr(renderable, "plain", str(renderable)))

        def thinking(self, text: str) -> None:
            pass

        def markdown(self, text: str, *, indent: int = 0, margin_top: int = 0) -> None:
            pass

        def response_label(self, elapsed_s: float, *, tool_count: int = 0, command=None) -> None:
            pass

        def newline(self) -> None:
            pass

    console = CaptureConsole()
    renderer = StreamRenderer(console)  # type: ignore[arg-type]
    renderer.start()
    metadata = {
        "ok": True,
        "path": "/tmp/demo",
        "original_tool_name": "list_directory",
        "normalized_tool_name": "file_list",
    }

    spinner = renderer.tool_started("list_directory", metadata=metadata)
    renderer.tool_finished("list_directory", metadata=metadata)

    assert spinner == "📁 file_list"
    assert len(console.lines) == 1
    line = console.lines[0]
    assert "📁 file_list" in line
    assert "alias=list_directory" in line
    assert "list_directory  path=" not in line


def test_stream_renderer_prints_tool_failure_exit_and_stderr() -> None:
    class CaptureConsole:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def print(self, renderable) -> None:
            self.lines.append(getattr(renderable, "plain", str(renderable)))

        def thinking(self, text: str) -> None:
            pass

        def markdown(self, text: str, *, indent: int = 0, margin_top: int = 0) -> None:
            pass

        def response_label(self, elapsed_s: float, *, tool_count: int = 0, command=None) -> None:
            pass

        def newline(self) -> None:
            pass

    console = CaptureConsole()
    renderer = StreamRenderer(console)  # type: ignore[arg-type]
    renderer.start()

    renderer.tool_started("shell_run", metadata={"command": "false"})
    renderer.tool_finished(
        "shell_run",
        metadata={"ok": False, "exit_code": 1, "stderr_preview": "permission denied"},
    )

    assert len(console.lines) == 1
    line = console.lines[0]
    assert "✗ shell_run" in line
    assert "$ false" in line
    assert "→ exit=1 permission denied" in line
    assert " | " in line


def test_stream_renderer_prints_context_evidence_metadata() -> None:
    class CaptureConsole:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def print(self, renderable) -> None:
            self.lines.append(getattr(renderable, "plain", str(renderable)))

        def thinking(self, text: str) -> None:
            pass

        def markdown(self, text: str, *, indent: int = 0, margin_top: int = 0) -> None:
            pass

        def response_label(self, elapsed_s: float, *, tool_count: int = 0, command=None) -> None:
            pass

        def newline(self) -> None:
            pass

    console = CaptureConsole()
    renderer = StreamRenderer(console)  # type: ignore[arg-type]
    renderer.start()

    renderer.tool_started("file_read", metadata={"path": "/tmp/sample.py"})
    renderer.tool_finished(
        "file_read",
        metadata={
            "ok": True,
            "path": "/tmp/sample.py",
            "mode": "symbols",
            "context_evidence": True,
            "tool_truncated": True,
            "repeat_read": True,
            "read_count": 3,
        },
    )

    assert len(console.lines) == 1
    line = console.lines[0]
    assert "📄 file_read" in line
    assert "path=/tmp/sample.py" in line
    assert " | " in line
    assert "symbols" not in line
    assert "truncated" not in line
    assert "repeat×3" not in line
    assert "evidence" not in line
    assert "repeat-read" not in line


def test_stream_renderer_prints_compression_and_posture_metadata() -> None:
    class CaptureConsole:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def print(self, renderable) -> None:
            self.lines.append(getattr(renderable, "plain", str(renderable)))

        def thinking(self, text: str) -> None:
            pass

        def markdown(self, text: str, *, indent: int = 0, margin_top: int = 0) -> None:
            pass

        def response_label(self, elapsed_s: float, *, tool_count: int = 0, command=None) -> None:
            pass

        def newline(self) -> None:
            pass

    console = CaptureConsole()
    renderer = StreamRenderer(console)  # type: ignore[arg-type]
    renderer.start()

    renderer.tool_started("shell_run", metadata={"command": "pytest"})
    renderer.tool_finished(
        "shell_run",
        metadata={
            "ok": True,
            "stdout_preview": "passed",
            "context_evidence": True,
            "compression_stages": ["trim", "summarize"],
            "compression_savings_ratio": 0.42,
            "compression_reason": "threshold-triggered",
            "context_posture": "research",
            "context_guidance": "maintain research ledger and synthesize findings",
            "disclosure_level": "expanded",
            "disclosure_reason": "tier1: continuity(shell)",
        },
    )

    assert len(console.lines) == 1
    line = console.lines[0]
    assert "💻 shell_run" in line
    assert "$ pytest" in line
    assert "→ passed" in line
    assert " | " in line
    assert "research" not in line
    assert "evidence" not in line
    assert "compressed" not in line
    assert "saved≈" not in line
    assert "threshold-triggered" not in line
    assert "maintain research ledger" not in line
    assert "selected capabilities matched" not in line


def test_stream_renderer_suppresses_structured_tool_blobs_and_compacts_paths() -> None:
    class CaptureConsole:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def print(self, renderable) -> None:
            self.lines.append(getattr(renderable, "plain", str(renderable)))

        def thinking(self, text: str) -> None:
            pass

        def markdown(self, text: str, *, indent: int = 0, margin_top: int = 0) -> None:
            pass

        def response_label(self, elapsed_s: float, *, tool_count: int = 0, command=None) -> None:
            pass

        def newline(self) -> None:
            pass

    console = CaptureConsole()
    renderer = StreamRenderer(console)  # type: ignore[arg-type]
    renderer.start()
    long_path = "/very/long/path/that/should/not/dominate/the/tool/log/with/many/nested/segments/src/leapflow"

    renderer.tool_started("file_list", metadata={"path": long_path})
    renderer.tool_finished(
        "file_list",
        metadata={
            "ok": True,
            "path": long_path,
            "result_preview": '{"ok": true, "entries": [{"name": "a"}, {"name": "b"}]}',
            "context_evidence": True,
            "disclosure_level": "full",
            "disclosure_reason": "task requires broad execution context",
        },
    )

    assert len(console.lines) == 1
    line = console.lines[0]
    assert "📁 file_list" in line
    assert "path=…/segments/src/leapflow" in line
    assert " | " in line
    assert long_path not in line
    assert "ok" not in line
    assert "disclosure" not in line
    assert "evidence" not in line
    assert "entries" not in line
    assert "task requires broad" not in line


def test_final_response_adds_copyable_url_for_markdown_links() -> None:
    answer = _sanitize_final_response(
        "请打开 [点击申请权限](https://open.feishu.cn/app/cli_xxx/auth) 后继续。"
    )

    assert "[点击申请权限](https://open.feishu.cn/app/cli_xxx/auth)" in answer
    assert "复制链接：https://open.feishu.cn/app/cli_xxx/auth" in answer


def test_stream_renderer_prints_permission_recovery_card() -> None:
    class CaptureConsole:
        def __init__(self) -> None:
            self.lines: list[str] = []
            self.cards: list[dict[str, object]] = []

        def print(self, renderable) -> None:
            self.lines.append(getattr(renderable, "plain", str(renderable)))

        def permission_recovery_card(self, metadata: dict[str, object]) -> None:
            self.cards.append(metadata)

        def thinking(self, text: str) -> None:
            pass

        def markdown(self, text: str, *, indent: int = 0, margin_top: int = 0) -> None:
            pass

        def response_label(self, elapsed_s: float, *, tool_count: int = 0, command=None) -> None:
            pass

        def newline(self) -> None:
            pass

    console = CaptureConsole()
    renderer = StreamRenderer(console)  # type: ignore[arg-type]
    renderer.start()
    metadata = {
        "ok": False,
        "platform": "feishu",
        "action": "im.list_chats",
        "capability": "im.chat.read",
        "failure_class": "authorization",
        "failure_code": "missing_scope",
        "missing_scopes": ["im:chat:read"],
        "scope_relation": "all_required",
        "scope_source": "authoritative",
        "console_url": "https://open.feishu.cn/app/cli_xxx/auth",
        "recovery_hint": "Grant the missing scope in the developer console.",
    }

    renderer.tool_started("platform_action", metadata={"platform": "feishu", "action": "im.list_chats"})
    renderer.tool_finished("platform_action", metadata=metadata)

    assert len(console.lines) == 1
    assert "✗ platform_action" in console.lines[0]
    assert len(console.cards) == 1
    assert console.cards[0]["console_url"] == "https://open.feishu.cn/app/cli_xxx/auth"
    assert console.cards[0]["missing_scopes"] == ["im:chat:read"]


@pytest.mark.asyncio
async def test_process_loop_does_not_duplicate_done_card_when_response_label_owns_status() -> None:
    holder: dict[str, LeapApp] = {}

    async def on_input(_text: str) -> None:
        completed = holder["app"].complete_active_command_in_response()
        assert completed is not None
        assert completed.status is TuiCommandStatus.DONE

    app, console, status = _make_app(on_input=on_input)
    holder["app"] = app
    app.submit_text("successful command")

    worker = asyncio.create_task(app._process_loop())
    try:
        await _wait_for(lambda: app.active_command is None and app._pending_input.qsize() == 0)
    finally:
        app._should_exit = True
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker

    assert [(card.id, card.status) for card in console.cards] == [
        (1, TuiCommandStatus.RUNNING),
    ]
    assert status.counts[-1] == (0, 0)


@pytest.mark.asyncio
async def test_process_loop_marks_failed_commands_and_recovers_counts() -> None:
    async def on_input(text: str) -> None:
        raise RuntimeError(f"boom from {text}")

    app, console, status = _make_app(on_input=on_input)
    app.submit_text("failing command")

    worker = asyncio.create_task(app._process_loop())
    try:
        await _wait_for(
            lambda: len(console.cards) == 2
            and console.cards[-1].status is TuiCommandStatus.FAILED
        )
    finally:
        app._should_exit = True
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker

    assert [(card.id, card.status) for card in console.cards] == [
        (1, TuiCommandStatus.RUNNING),
        (1, TuiCommandStatus.FAILED),
    ]
    assert "RuntimeError: boom from failing command" in console.cards[-1].error
    assert console.errors == ["boom from failing command"]
    assert status.counts[-1] == (0, 0)


def test_failed_command_error_is_single_line_and_truncated() -> None:
    command = TuiCommand.create(command_id=1, text="demo")
    failed = command.mark_failed("line1\n" + "x" * 400)

    assert "\n" not in failed.error
    assert len(failed.error) == 240
    assert failed.error.endswith("…")


def test_cancelled_and_skipped_commands_are_terminal() -> None:
    command = TuiCommand.create(command_id=1, text="long task").mark_running()

    cancelled = command.mark_cancelled("user pressed cancel")
    skipped = command.mark_skipped("user skipped")

    assert cancelled.status is TuiCommandStatus.CANCELLED
    assert cancelled.error == "user pressed cancel"
    assert cancelled.finished_at > 0
    assert skipped.status is TuiCommandStatus.SKIPPED
    assert skipped.error == "user skipped"
    assert skipped.finished_at > 0


def test_placeholder_processor_indents_hint_after_prompt_space() -> None:
    processor = _DynamicPlaceholderProcessor(
        lambda: "Queue paused · /resume continue",
        lambda: [("class:prompt", "❯ ")],
    )
    empty_input = SimpleNamespace(
        document=SimpleNamespace(text=""),
        lineno=0,
        fragments=[],
    )

    transformed = processor.apply_transformation(empty_input)

    assert transformed.fragments == [
        ("class:prompt", "❯ "),
        ("class:placeholder", "  "),
        ("class:placeholder", "Queue paused · /resume continue"),
    ]
    assert sum(text.count("❯") for _style, text in transformed.fragments) == 1
    assert transformed.source_to_display(0) == len("❯ ")
    assert transformed.display_to_source(len("❯   Queue paused · /resume continue")) == 0


def test_prompt_prefix_is_preserved_with_placeholder_hint() -> None:
    app, _console, _status = _make_app()
    processor = _DynamicPlaceholderProcessor(
        app._placeholder_text,
        app._prompt_fragments,
    )
    empty_input = SimpleNamespace(
        document=SimpleNamespace(text=""),
        lineno=0,
        fragments=[],
    )

    transformed = processor.apply_transformation(empty_input)

    assert transformed.fragments == [
        ("class:prompt", "❯ "),
        ("class:placeholder", "  "),
        ("class:placeholder", app._placeholder_text()),
    ]
    assert sum(text.count("❯") for _style, text in transformed.fragments) == 1
    assert transformed.source_to_display(0) == len("❯ ")


def test_placeholder_processor_hides_hint_after_user_input() -> None:
    processor = _DynamicPlaceholderProcessor(
        lambda: "Ask LeapFlow…",
        lambda: [("class:prompt", "❯ ")],
    )
    typed_input = SimpleNamespace(
        document=SimpleNamespace(text="hello"),
        lineno=0,
        fragments=[("", "hello")],
    )

    transformed = processor.apply_transformation(typed_input)

    assert transformed.fragments == [("class:prompt", "❯ "), ("", "hello")]
    assert transformed.source_to_display(0) == len("❯ ")
    assert transformed.display_to_source(len("❯ ") + 1) == 1


def test_control_commands_are_handled_without_queueing() -> None:
    handled: list[str] = []

    def on_control(text: str) -> bool:
        handled.append(text)
        return text == "/cancel"

    app, _console, status = _make_app(on_control=on_control)

    command = app.submit_text("/cancel")

    assert command.id == 0
    assert command.status is TuiCommandStatus.DONE
    assert handled == ["/cancel"]
    assert app._pending_input.qsize() == 0
    assert status.counts == []


@pytest.mark.asyncio
async def test_pause_queue_holds_and_resume_runs_pending_command() -> None:
    processed: list[str] = []

    async def on_input(text: str) -> None:
        processed.append(text)

    app, console, status = _make_app(on_input=on_input)
    app.pause_queue()
    app.submit_text("held command")
    worker = asyncio.create_task(app._process_loop())
    try:
        await asyncio.sleep(0.05)
        assert processed == []
        assert app._pending_input.qsize() == 1
        assert app.queue_paused is True

        app.resume_queue()
        await _wait_for(lambda: processed == ["held command"])
    finally:
        app._should_exit = True
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker

    assert [card.status for card in console.cards] == [
        TuiCommandStatus.QUEUED,
        TuiCommandStatus.RUNNING,
        TuiCommandStatus.DONE,
    ]
    assert status.counts[-1] == (0, 0)


@pytest.mark.asyncio
async def test_cancel_active_command_continues_queued_work() -> None:
    started = asyncio.Event()
    processed: list[str] = []

    async def on_input(text: str) -> None:
        processed.append(text)
        if text == "first command":
            started.set()
            await asyncio.sleep(10)

    app, console, status = _make_app(on_input=on_input)
    app.submit_text("first command")
    app.submit_text("second command")
    worker = asyncio.create_task(app._process_loop())
    try:
        await _wait_for(lambda: started.is_set())
        cancelled = app.request_cancel_active("cancelled in test")
        assert cancelled is not None
        await _wait_for(lambda: processed == ["first command", "second command"])
    finally:
        app._should_exit = True
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker

    rendered = [(card.id, card.status) for card in console.cards]
    assert rendered == [
        (2, TuiCommandStatus.QUEUED),
        (1, TuiCommandStatus.RUNNING),
        (1, TuiCommandStatus.CANCELLED),
        (2, TuiCommandStatus.RUNNING),
        (2, TuiCommandStatus.DONE),
    ]
    assert console.cards[2].error == "cancelled in test"
    assert status.counts[-1] == (0, 0)


def test_queue_drop_and_clear_render_skipped_commands() -> None:
    app, console, status = _make_app()
    app.submit_text("first")
    app.submit_text("second")
    app.submit_text("third")

    dropped = app.drop_queued_command(2, "not needed")
    cleared = app.clear_queued_commands("clear rest")

    assert dropped is not None
    assert dropped.id == 2
    assert [command.id for command in cleared] == [1, 3]
    assert app._pending_input.qsize() == 0
    assert [card.status for card in console.cards] == [
        TuiCommandStatus.QUEUED,
        TuiCommandStatus.QUEUED,
        TuiCommandStatus.SKIPPED,
        TuiCommandStatus.SKIPPED,
        TuiCommandStatus.SKIPPED,
    ]
    assert status.counts[-1] == (0, 0)


def test_task_control_commands_are_registered_for_completion() -> None:
    from leapflow.cli.commands.registry import completion_entries, resolve_command

    entries = dict(completion_entries())

    assert resolve_command("cancel") is not None
    assert resolve_command("skip") is not None
    assert resolve_command("teach skip") is not None
    assert resolve_command("stop") is None
    assert entries["cancel"] == "Cancel the currently running task"
    assert entries["queue"] == "Show or clear queued tasks"


@pytest.mark.asyncio
async def test_teach_skip_command_marks_noise_steps() -> None:
    from leapflow.cli.commands.interactive import _handle_teach

    class FakeSession:
        def __init__(self) -> None:
            self.skipped = 0

        def mark_skip(self, count: int) -> int:
            self.skipped = count
            return count

    session = FakeSession()
    ctx = SimpleNamespace(session=session)
    console = _FakeConsole()

    handled = await _handle_teach(ctx, console, "teach skip 3", learning=False)
    legacy_handled = await _handle_teach(ctx, console, "skip 2", learning=True)

    assert handled is True
    assert legacy_handled is False
    assert session.skipped == 3
    assert console.systems == ["Marked 3 step(s) as noise."]


class _FakeBuffer:
    def __init__(self) -> None:
        self.text = ""

    def insert_text(self, text: str) -> None:
        self.text += text


def test_large_paste_is_compacted_but_submits_full_text() -> None:
    app, _console, _status = _make_app()
    buffer = _FakeBuffer()
    pasted = "line\n" * 2_000

    app._insert_paste_text(buffer, pasted)

    assert pasted not in buffer.text
    assert "pasted block #1" in buffer.text
    assert "full text will be submitted" in buffer.text
    assert buffer.text.isascii()
    command = app.submit_text(f"Please summarize:\n{buffer.text}")

    assert command.text == f"Please summarize:\n{pasted}".strip()
    assert app._paste_blocks == {}


@pytest.mark.asyncio
async def test_buffer_insert_compacts_large_english_paste() -> None:
    app, _console, _status = _make_app()
    pasted = "long english line\n" * 400

    app._input_area.buffer.insert_text(pasted)

    visible = app._input_area.buffer.text
    assert pasted not in visible
    assert visible.startswith("[pasted block #1:")
    assert "full text will be submitted" in visible
    assert len(visible) < 120
    assert app.submit_text(visible).text == pasted.strip()


@pytest.mark.asyncio
async def test_buffer_insert_compacts_large_chinese_paste_with_ascii_marker() -> None:
    app, _console, _status = _make_app()
    pasted = "这是一段用于验证中文大段粘贴不会直接渲染的内容。\n" * 300

    app._input_area.buffer.insert_text(pasted)

    visible = app._input_area.buffer.text
    assert pasted not in visible
    assert "这是一段" not in visible
    assert visible.startswith("[pasted block #1:")
    assert visible.isascii()
    assert len(visible) < 120
    assert app.submit_text(visible).text == pasted.strip()


@pytest.mark.asyncio
async def test_fragmented_chinese_paste_compacts_and_submits_full_text() -> None:
    app, _console, _status = _make_app()
    pasted = "经济活动达到最低点，经济增长理论，索洛增长模型。" * 80

    for index in range(0, len(pasted), 18):
        app._input_area.buffer.insert_text(pasted[index:index + 18])

    visible = app._input_area.buffer.text
    assert pasted not in visible
    assert "经济活动" not in visible
    assert visible.startswith("[pasted block #1:")
    assert visible.isascii()
    assert len(visible) < 120
    assert app.submit_text(visible).text == pasted.strip()


@pytest.mark.asyncio
async def test_fragmented_english_single_line_paste_compacts() -> None:
    app, _console, _status = _make_app()
    pasted = "capital accumulation and productivity growth " * 80

    for index in range(0, len(pasted), 16):
        app._input_area.buffer.insert_text(pasted[index:index + 16])

    visible = app._input_area.buffer.text
    assert pasted not in visible
    assert visible.startswith("[pasted block #1:")
    assert visible.isascii()
    assert app.submit_text(visible).text == pasted.strip()


@pytest.mark.asyncio
async def test_control_character_paste_is_compacted_and_sanitized() -> None:
    app, _console, _status = _make_app()
    pasted = "normal\rtext\x1b[31mred\x1b[0m\x00tail\u202edone"

    app._input_area.buffer.insert_text(pasted)

    visible = app._input_area.buffer.text
    assert visible.startswith("[pasted block #1:")
    assert visible.isascii()
    assert "\x1b" not in visible
    command = app.submit_text(visible)
    assert command.text == "normal\ntextredtaildone"


@pytest.mark.asyncio
async def test_fragmented_paste_window_expiry_keeps_followup_typing_visible(monkeypatch) -> None:
    app, _console, _status = _make_app()
    clock = 100.0

    def fake_monotonic() -> float:
        return clock

    monkeypatch.setattr(app_module.time, "monotonic", fake_monotonic)
    pasted = "fragmented paste " * 80
    for index in range(0, len(pasted), 20):
        app._input_area.buffer.insert_text(pasted[index:index + 20])

    visible = app._input_area.buffer.text
    assert visible.startswith("[pasted block #1:")

    clock = 101.0
    app._input_area.buffer.insert_text(" follow-up")

    assert app._input_area.buffer.text == f"{visible} follow-up"
    assert app.submit_text(app._input_area.buffer.text).text == f"{pasted} follow-up".strip()


def test_small_paste_stays_inline() -> None:
    app, _console, _status = _make_app()
    buffer = _FakeBuffer()
    pasted = "short pasted text"

    app._insert_paste_text(buffer, pasted)

    assert buffer.text == pasted
    assert app._paste_blocks == {}


def test_submit_clears_stale_compacted_paste_when_marker_removed() -> None:
    app, _console, _status = _make_app()
    buffer = _FakeBuffer()
    app._insert_paste_text(buffer, "line\n" * 2_000)

    command = app.submit_text("marker was deleted by user")

    assert command.text == "marker was deleted by user"
    assert app._paste_blocks == {}


@pytest.mark.asyncio
async def test_process_loop_runs_submitted_commands_serially() -> None:
    processed: list[str] = []

    async def on_input(text: str) -> None:
        processed.append(text)
        await asyncio.sleep(0.01)

    app, console, status = _make_app(on_input=on_input)
    app.submit_text("first command")
    app.submit_text("second command")

    worker = asyncio.create_task(app._process_loop())
    try:
        await _wait_for(
            lambda: processed == ["first command", "second command"]
            and len(console.cards) == 5
        )
    finally:
        app._should_exit = True
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker

    rendered = [(card.id, card.status) for card in console.cards]
    assert rendered == [
        (2, TuiCommandStatus.QUEUED),
        (1, TuiCommandStatus.RUNNING),
        (1, TuiCommandStatus.DONE),
        (2, TuiCommandStatus.RUNNING),
        (2, TuiCommandStatus.DONE),
    ]
    assert status.counts[-1] == (0, 0)
