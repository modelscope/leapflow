"""Chat subcommand — single-turn conversational mode with rich output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from leapflow.cli.helpers import require_initialized
from leapflow.engine import StreamEvent

if TYPE_CHECKING:
    from leapflow.cli.context import Context


async def cmd_chat(ctx: "Context", prompt: str, thinking: bool) -> int:
    require_initialized(ctx)

    from leapflow.cli.tui_app import detect_theme, LeapConsole, StreamRenderer

    theme = detect_theme()
    console = LeapConsole(theme)

    renderer = StreamRenderer(console.raw, theme)
    renderer.start()

    try:
        async for event in ctx.engine.run_stream(prompt, enable_thinking=thinking):
            if isinstance(event, str):
                renderer.feed(event)
            elif event.type == "chunk":
                renderer.feed(event.content)
            elif event.type == "thinking":
                renderer.feed_thinking(event.content)
            elif event.type == "tool_start":
                renderer.tool_started(event.content)
            elif event.type == "tool_complete":
                renderer.tool_finished(event.content)
            elif event.type == "final" and not renderer.text:
                renderer.feed(event.content)
    finally:
        elapsed = renderer.elapsed
        tool_count = len(renderer._tool_history)
        renderer.finish()

    final_text = renderer.text.strip()
    if final_text:
        console.markdown(final_text)
    console.response_label(elapsed, tool_count=tool_count)
    console.print()

    return 0
