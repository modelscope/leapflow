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

import json
import os
import re
import time
from typing import TYPE_CHECKING, Any

from rich.text import Text

_FINAL_RESPONSE_INDENT_SPACES = 4
_FINAL_RESPONSE_MARGIN_TOP = 1
_TOOL_INPUT_LIMIT = 96
_TOOL_OUTPUT_LIMIT = 96
_TOOL_PATH_LIMIT = 72
_TOOL_CONTEXT_TAG_LIMIT = 3
_SYNTHETIC_THINKING_ROUND_RE = re.compile(r"round\s*\d+", re.IGNORECASE)
_FENCED_BLOCK_RE = re.compile(r"```(?P<lang>[\w+-]*)\s*\n(?P<body>.*?)\n```", re.DOTALL)
_TOOL_AUDIT_LINE_RE = re.compile(
    r"^\s*[·✓✗]\s+[A-Za-z_][\w.-]*(?:\s|$).*",
    re.MULTILINE,
)
_JSON_DECODER = json.JSONDecoder()


def _is_tool_protocol_payload(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("name"), str)
        and isinstance(value.get("arguments"), dict)
    )


def _strip_tool_protocol_fences(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        body = match.group("body").strip()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return match.group(0)
        return "" if _is_tool_protocol_payload(payload) else match.group(0)

    return _FENCED_BLOCK_RE.sub(replace, text)


def _strip_tool_protocol_json_objects(text: str) -> str:
    pieces: list[str] = []
    cursor = 0
    while True:
        start = text.find("{", cursor)
        if start < 0:
            pieces.append(text[cursor:])
            break
        pieces.append(text[cursor:start])
        try:
            payload, end = _JSON_DECODER.raw_decode(text[start:])
        except json.JSONDecodeError:
            pieces.append(text[start:start + 1])
            cursor = start + 1
            continue
        absolute_end = start + end
        if _is_tool_protocol_payload(payload):
            cursor = absolute_end
            while cursor < len(text) and text[cursor] in " \t\r":
                cursor += 1
            if cursor < len(text) and text[cursor] == "\n":
                cursor += 1
            continue
        pieces.append(text[start:absolute_end])
        cursor = absolute_end
    return "".join(pieces)


def _collapse_blank_lines(text: str) -> str:
    lines = text.splitlines()
    collapsed: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip():
            blank_count = 0
            collapsed.append(line.rstrip())
            continue
        blank_count += 1
        if blank_count <= 1:
            collapsed.append("")
    return "\n".join(collapsed).strip()


def _sanitize_final_response(text: str) -> str:
    """Remove leaked tool protocol artifacts from user-facing final answers."""
    without_fences = _strip_tool_protocol_fences(text)
    without_objects = _strip_tool_protocol_json_objects(without_fences)
    without_audit_lines = _TOOL_AUDIT_LINE_RE.sub("", without_objects)
    return _collapse_blank_lines(without_audit_lines)


def _normalize_thinking_text(text: str) -> str:
    without_round_markers = _SYNTHETIC_THINKING_ROUND_RE.sub(" ", text)
    lines = [" ".join(line.split()) for line in without_round_markers.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _metadata_text(metadata: dict[str, Any] | None, key: str) -> str:
    if not metadata:
        return ""
    value = metadata.get(key)
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _truncate_detail(text: str, *, limit: int = _TOOL_OUTPUT_LIMIT) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _compact_path(path: str, *, limit: int = _TOOL_PATH_LIMIT) -> str:
    expanded_home = os.path.expanduser("~")
    compact = path.replace(expanded_home, "~", 1) if path.startswith(expanded_home) else path
    if len(compact) <= limit:
        return compact
    parts = compact.split("/")
    if len(parts) > 3:
        compact = "…/" + "/".join(parts[-3:])
    if len(compact) <= limit:
        return compact
    return "…" + compact[-(limit - 1):]


def _looks_like_structured_blob(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def _context_tags(metadata: dict[str, Any] | None) -> list[str]:
    if not metadata:
        return []
    tags: list[str] = []
    posture = _metadata_text(metadata, "context_posture")
    if posture and posture != "baseline":
        tags.append(posture)
    mode = _metadata_text(metadata, "mode")
    if mode and mode not in {"raw", posture}:
        tags.append(mode)
    disclosure = _metadata_text(metadata, "disclosure_level")
    if disclosure and disclosure not in {"selected_tools", "minimal"}:
        tags.append(f"disclosure={disclosure}")
    if metadata.get("tool_truncated"):
        tags.append("truncated")
    read_count = metadata.get("read_count")
    if metadata.get("repeat_read") and read_count is not None:
        tags.append(f"repeat×{read_count}")
    elif metadata.get("context_evidence"):
        tags.append("evidence")
    if metadata.get("compression_stages"):
        tags.append("compressed")
    deduped: list[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped[:_TOOL_CONTEXT_TAG_LIMIT]


def _tool_action_detail(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    command = _metadata_text(metadata, "command") or _metadata_text(metadata, "cmd")
    if command:
        return f"$ {_truncate_detail(command, limit=_TOOL_INPUT_LIMIT)}"
    path = _metadata_text(metadata, "path")
    pattern = _metadata_text(metadata, "pattern")
    if path and pattern:
        return f"path={_compact_path(path)} pattern={_truncate_detail(pattern, limit=48)}"
    if path:
        return f"path={_compact_path(path)}"
    query = _metadata_text(metadata, "query")
    if query:
        return f"query={_truncate_detail(query, limit=_TOOL_INPUT_LIMIT)}"
    url = _metadata_text(metadata, "url")
    if url:
        return f"url={_truncate_detail(url, limit=_TOOL_INPUT_LIMIT)}"
    return _truncate_detail(_metadata_text(metadata, "args_summary"), limit=_TOOL_INPUT_LIMIT)


def _tool_context_detail(metadata: dict[str, Any] | None) -> str:
    tags = _context_tags(metadata)
    return f"[{' · '.join(tags)}]" if tags else ""

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
        base = _truncate_detail(prefix + detail, limit=_TOOL_OUTPUT_LIMIT) if detail or prefix else "failed"
        context = _tool_context_detail(metadata)
        return f"{base} {context}" if context else base
    detail = (
        _metadata_text(metadata, "stdout_preview")
        or _metadata_text(metadata, "content_preview")
        or _metadata_text(metadata, "output_preview")
        or _metadata_text(metadata, "result_preview")
    )
    if detail and not _looks_like_structured_blob(detail):
        base = _truncate_detail(detail, limit=_TOOL_OUTPUT_LIMIT)
    else:
        base = "ok"
    context = _tool_context_detail(metadata)
    return f"{base} {context}" if context else base


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
        """Append meaningful thinking/reasoning text."""
        text = _normalize_thinking_text(chunk)
        if not text:
            return
        if self._thinking_buffer and not self._thinking_buffer.endswith("\n"):
            self._thinking_buffer += "\n"
        self._thinking_buffer += text

    def tool_started(self, name: str, metadata: dict[str, Any] | None = None) -> str:
        """Mark a tool call as started. Returns spinner text for LeapApp."""
        self._active_tool = name
        self._tool_start_time = time.monotonic()
        detail = _tool_action_detail(metadata)
        line = Text()
        line.append("  · ", style="leap.tool")
        line.append(name, style="leap.tool_name")
        if detail:
            line.append("  ", style="leap.tool")
            line.append(detail, style="leap.tool")
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
            status_style = "leap.tool" if ok else "leap.error"
            name_style = "leap.tool_name" if ok else "leap.error"
            line.append("  ✓ " if ok else "  ✗ ", style=status_style)
            line.append(tool_name, style=name_style)
            line.append(f"  {_format_elapsed(duration)}", style="leap.tool")
            detail = _tool_result_detail(metadata) or _truncate_detail(output, limit=_TOOL_OUTPUT_LIMIT)
            if detail:
                line.append("  ", style="leap.tool")
                line.append(detail, style="leap.tool" if ok else "leap.error")
            self._console.print(line)
        self._active_tool = ""
        self._tool_start_time = 0.0

    def finish(self) -> None:
        """Render all accumulated content to the console."""
        if self._thinking_buffer.strip():
            self._console.thinking(self._thinking_buffer)

        answer = _sanitize_final_response(self._buffer)
        if answer:
            answer_label = getattr(self._console, "answer_label", None)
            if callable(answer_label):
                answer_label()
            self._console.markdown(
                answer,
                indent=_FINAL_RESPONSE_INDENT_SPACES,
                margin_top=_FINAL_RESPONSE_MARGIN_TOP,
            )

        self._console.response_label(self.elapsed, tool_count=self.tool_count)
        self._console.newline()
