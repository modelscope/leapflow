"""Streaming LLM output renderer — Application-compatible.

Accumulates streaming token deltas, tracks tool call state, and
renders the final response via Rich Console.  No ``rich.Live`` —
all output flows through ``patch_stdout`` to appear above the
Application layout.

During streaming, the caller updates ``LeapApp.spinner_text`` for
visual feedback (e.g. tool name + elapsed timer).  On finish, the
accumulated response is rendered as Rich Markdown.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from rich.text import Text

_FINAL_RESPONSE_INDENT_SPACES = 4
_FINAL_RESPONSE_MARGIN_TOP = 1
_TOOL_DETAIL_LIMIT = 180


def _metadata_text(metadata: dict[str, Any] | None, key: str) -> str:
    if not metadata:
        return ""
    value = metadata.get(key)
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _truncate_detail(text: str, *, limit: int = _TOOL_DETAIL_LIMIT) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _tool_action_detail(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    command = _metadata_text(metadata, "command") or _metadata_text(metadata, "cmd")
    if command:
        return f"$ {_truncate_detail(command)}"
    path = _metadata_text(metadata, "path")
    pattern = _metadata_text(metadata, "pattern")
    if path and pattern:
        return f"path={_truncate_detail(path)} pattern={_truncate_detail(pattern, limit=80)}"
    if path:
        return f"path={_truncate_detail(path)}"
    query = _metadata_text(metadata, "query")
    if query:
        return f"query={_truncate_detail(query)}"
    url = _metadata_text(metadata, "url")
    if url:
        return f"url={_truncate_detail(url)}"
    return _truncate_detail(_metadata_text(metadata, "args_summary"))


def _tool_context_detail(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    parts: list[str] = []
    mode = _metadata_text(metadata, "mode")
    if mode and mode != "raw":
        parts.append(f"mode={mode}")
    if metadata.get("tool_truncated"):
        parts.append("truncated")
    stages = metadata.get("compression_stages")
    if isinstance(stages, list) and stages:
        stage_text = "+".join(str(stage) for stage in stages[:3])
        parts.append(f"compressed={stage_text}")
        savings = metadata.get("compression_savings_ratio")
        if isinstance(savings, (int, float)) and savings > 0:
            parts.append(f"saved≈{int(savings * 100)}%")
        reason = _metadata_text(metadata, "compression_reason")
        if reason:
            parts.append(reason)
    posture = _metadata_text(metadata, "context_posture")
    if posture and posture != "baseline":
        parts.append(posture)
    guidance = _metadata_text(metadata, "context_guidance")
    if guidance:
        parts.append(_truncate_detail(guidance, limit=72))
    read_count = metadata.get("read_count")
    if metadata.get("repeat_read") and read_count is not None:
        parts.append(f"repeat-read x{read_count}")
    elif metadata.get("context_evidence"):
        parts.append("evidence")
    return ", ".join(parts)


def _tool_result_detail(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    if metadata.get("ok") is False:
        exit_code = metadata.get("exit_code")
        prefix = f"exit={exit_code} " if exit_code is not None else ""
        detail = (
            _metadata_text(metadata, "stderr_preview")
            or _metadata_text(metadata, "error_preview")
            or _metadata_text(metadata, "result_preview")
        )
        base = _truncate_detail(prefix + detail) if detail or prefix else "failed"
        context = _tool_context_detail(metadata)
        return f"{base} ({context})" if context else base
    detail = (
        _metadata_text(metadata, "stdout_preview")
        or _metadata_text(metadata, "content_preview")
        or _metadata_text(metadata, "output_preview")
        or _metadata_text(metadata, "result_preview")
    )
    if detail:
        base = _truncate_detail(detail)
    elif metadata.get("path"):
        base = f"path={_truncate_detail(str(metadata['path']))}"
    else:
        base = "ok (no output)"
    context = _tool_context_detail(metadata)
    return f"{base} ({context})" if context else base


if TYPE_CHECKING:
    from leapflow.cli.tui_app.console import LeapConsole


def _format_elapsed(seconds: float) -> str:
    """Format elapsed time as a compact human-readable string."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}m{secs:.0f}s"


class StreamRenderer:
    """Accumulates streaming output and renders on finish.

    Usage::

        renderer = StreamRenderer(console)
        renderer.start()
        renderer.feed("Hello ")
        renderer.feed("**world**!")
        spinner = renderer.tool_started("shell")   # returns "⚡ shell"
        renderer.tool_finished("shell")             # prints ✓ line
        renderer.finish()                           # prints markdown + label
    """

    def __init__(self, console: "LeapConsole") -> None:
        self._console = console
        self._buffer: str = ""
        self._thinking_buffer: str = ""
        self._start_time: float = 0.0
        self._tool_start_time: float = 0.0
        self._active_tool: str = ""
        self._tool_history: list[tuple[str, float]] = []

    @property
    def text(self) -> str:
        """Full accumulated response text."""
        return self._buffer

    @property
    def has_output(self) -> bool:
        return bool(self._buffer.strip() or self._thinking_buffer.strip())

    @property
    def elapsed(self) -> float:
        """Seconds since streaming started."""
        return time.monotonic() - self._start_time if self._start_time else 0.0

    @property
    def tool_count(self) -> int:
        return len(self._tool_history)

    def start(self) -> None:
        """Begin a new streaming session."""
        self._buffer = ""
        self._thinking_buffer = ""
        self._active_tool = ""
        self._tool_history = []
        self._start_time = time.monotonic()
        self._tool_start_time = 0.0

    def feed(self, chunk: str) -> None:
        """Append a text chunk to the response buffer."""
        self._buffer += chunk

    def feed_thinking(self, chunk: str) -> None:
        """Append a thinking/reasoning chunk."""
        self._thinking_buffer += chunk

    def tool_started(self, name: str, metadata: dict[str, Any] | None = None) -> str:
        """Mark a tool call as started. Returns spinner text for LeapApp."""
        self._active_tool = name
        self._tool_start_time = time.monotonic()
        detail = _tool_action_detail(metadata)
        line = Text()
        line.append("  ⚡ ", style="leap.accent")
        line.append(name, style="bold")
        if detail:
            line.append("  ", style="dim")
            line.append(detail, style="leap.dim")
        self._console.print(line)
        return f"⚡ {name}"

    def tool_finished(
        self,
        name: str = "",
        output: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark a tool call as finished; print completion line immediately."""
        tool_name = name or self._active_tool
        if tool_name and self._tool_start_time > 0:
            duration = time.monotonic() - self._tool_start_time
            self._tool_history.append((tool_name, duration))
            ok = metadata.get("ok", True) if metadata else True
            line = Text()
            line.append("  ✓ " if ok else "  ✗ ", style="leap.success" if ok else "leap.error")
            line.append(tool_name, style="bold")
            line.append(f"  {_format_elapsed(duration)}", style="dim")
            detail = _tool_result_detail(metadata) or _truncate_detail(output)
            if detail:
                line.append("  ", style="dim")
                line.append(detail, style="leap.dim" if ok else "leap.error")
            self._console.print(line)
        self._active_tool = ""
        self._tool_start_time = 0.0

    def finish(self) -> None:
        """Render all accumulated content to the console."""
        if self._thinking_buffer.strip():
            self._console.thinking(self._thinking_buffer)

        if self._buffer.strip():
            self._console.markdown(
                self._buffer,
                indent=_FINAL_RESPONSE_INDENT_SPACES,
                margin_top=_FINAL_RESPONSE_MARGIN_TOP,
            )

        self._console.response_label(self.elapsed, tool_count=self.tool_count)
        self._console.newline()
