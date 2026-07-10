from __future__ import annotations

from dataclasses import dataclass

from leapflow.cli.tui_app.session_summary import (
    SessionExitStats,
    build_exit_summary_lines,
    format_duration,
    summarize_messages,
)
from leapflow.cli.tui_app.stream import StreamRenderer


@dataclass(frozen=True)
class _Message:
    role: str
    tool_name: str | None = None
    tool_calls_json: str | None = None


class _Console:
    def __init__(self) -> None:
        self.markdown_calls: list[dict[str, object]] = []
        self.thinking_calls: list[str] = []
        self.labels: list[tuple[float, int]] = []
        self.answer_labels = 0
        self.lines = 0

    def markdown(
        self,
        text: str,
        *,
        indent: int = 0,
        margin_top: int = 0,
        margin_bottom: int = 0,
    ) -> None:
        self.markdown_calls.append({
            "text": text,
            "indent": indent,
            "margin_top": margin_top,
            "margin_bottom": margin_bottom,
        })

    def thinking(self, text: str) -> None:
        self.thinking_calls.append(text)

    def answer_label(self) -> None:
        self.answer_labels += 1

    def response_label(self, elapsed_s: float, *, tool_count: int = 0) -> None:
        self.labels.append((elapsed_s, tool_count))

    def newline(self) -> None:
        self.lines += 1

    def print(self, *args, **kwargs) -> None:
        return None


def test_format_duration_compacts_common_ranges() -> None:
    assert format_duration(11.9) == "11s"
    assert format_duration(61) == "1m 1s"
    assert format_duration(3661) == "1h 1m 1s"


def test_build_exit_summary_lines_matches_resume_shape() -> None:
    lines = build_exit_summary_lines(
        session_id="abc123",
        duration_s=11,
        message_count=2,
        user_messages=1,
        tool_calls=0,
    )

    assert lines == [
        "Resume this session with:",
        "  leap --resume abc123",
        "",
        "Session:        abc123",
        "Duration:       11s",
        "Messages:       2 (1 user, 0 tool calls)",
    ]


def test_empty_session_summary_degrades_to_goodbye() -> None:
    assert build_exit_summary_lines(
        session_id="",
        duration_s=0,
        message_count=0,
        user_messages=0,
        tool_calls=0,
    ) == ["Goodbye!"]


def test_volatile_session_summary_does_not_offer_resume() -> None:
    lines = build_exit_summary_lines(
        session_id="volatile123",
        duration_s=5,
        message_count=2,
        user_messages=1,
        tool_calls=0,
        resumable=False,
    )

    assert lines == [
        "Session not saved:",
        "  This window used volatile storage because the primary database was locked.",
        "",
        "Session:        volatile123",
        "Duration:       5s",
        "Messages:       2 (1 user, 0 tool calls)",
    ]


def test_summarize_messages_counts_tool_call_shapes() -> None:
    total, users, tools = summarize_messages([
        _Message("user"),
        _Message("assistant", tool_calls_json="[]"),
        _Message("tool", tool_name="shell"),
    ])

    assert total == 3
    assert users == 1
    assert tools == 2


def test_session_exit_stats_tracks_fallback_counts() -> None:
    stats = SessionExitStats()
    stats.record_user_message()
    stats.record_assistant_message()
    stats.record_tool_calls(2)
    stats.record_tool_calls(-1)

    assert stats.message_count == 2
    assert stats.user_messages == 1
    assert stats.tool_calls == 2
    assert stats.duration_s >= 0


def test_stream_renderer_exposes_output_without_private_access() -> None:
    renderer = StreamRenderer(_Console())
    renderer.start()
    assert renderer.has_output is False

    renderer.feed("hello")
    assert renderer.has_output is True


def test_stream_renderer_spaces_and_indents_final_response_only() -> None:
    console = _Console()
    renderer = StreamRenderer(console)
    renderer.start()

    renderer.feed_thinking("internal reasoning")
    renderer.feed("final **answer**")
    renderer.finish()

    assert console.thinking_calls == ["internal reasoning"]
    assert console.markdown_calls == [{
        "text": "final **answer**",
        "indent": 4,
        "margin_top": 1,
        "margin_bottom": 0,
    }]
    assert len(console.labels) == 1
    assert console.answer_labels == 1
    assert console.lines == 1


def test_stream_renderer_suppresses_synthetic_round_thinking() -> None:
    console = _Console()
    renderer = StreamRenderer(console)
    renderer.start()

    renderer.feed_thinking("round 1")
    renderer.feed_thinking("round 2round 3")
    assert renderer.has_output is False

    renderer.feed("final answer")
    renderer.finish()

    assert console.thinking_calls == []
    assert console.markdown_calls[0]["text"] == "final answer"


def test_stream_renderer_keeps_meaningful_thinking_lines() -> None:
    console = _Console()
    renderer = StreamRenderer(console)
    renderer.start()

    renderer.feed_thinking("round 1")
    renderer.feed_thinking("Reading relevant files")
    renderer.feed_thinking("round 2\nSynthesizing findings")
    renderer.feed("final answer")
    renderer.finish()

    assert console.thinking_calls == ["Reading relevant files\nSynthesizing findings"]


def test_stream_renderer_sanitizes_tool_protocol_from_final_answer() -> None:
    console = _Console()
    renderer = StreamRenderer(console)
    renderer.start()

    renderer.feed("""
  · skills_list  {}
  ✓ skills_list  4ms  ok [disclosure=indexed_capabilities]

```json
{"name": "skills_list", "arguments": {}}
```

目前可用能力如下：

{"name": "skills_list", "arguments": {"query": "", "source": "local"}}

- 文件操作
- Shell 命令执行
""")
    renderer.finish()

    text = str(console.markdown_calls[0]["text"])
    assert "skills_list" not in text
    assert "arguments" not in text
    assert "目前可用能力如下" in text
    assert "文件操作" in text
    assert console.answer_labels == 1


def test_stream_renderer_keeps_regular_json_examples() -> None:
    console = _Console()
    renderer = StreamRenderer(console)
    renderer.start()

    renderer.feed("""
配置示例：

```json
{"theme": "dark", "enabled": true}
```
""")
    renderer.finish()

    text = str(console.markdown_calls[0]["text"])
    assert '"theme": "dark"' in text
    assert '"enabled": true' in text


def test_global_resume_routes_to_interactive(monkeypatch) -> None:
    from leapflow.cli import cli

    captured = {}

    async def fake_daemon_main(args):
        captured["command"] = args.command
        captured["resume"] = args.resume
        return 0

    monkeypatch.setattr(cli, "_async_daemon_main", fake_daemon_main)

    assert cli.main(["--resume", "abc123"]) == 0
    assert captured == {"command": "interactive", "resume": "abc123"}
