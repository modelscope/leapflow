"""Chat subcommand — single-turn conversational mode."""

from __future__ import annotations

from typing import TYPE_CHECKING

from leapflow.cli.helpers import require_initialized
from leapflow.engine import StreamEvent

if TYPE_CHECKING:
    from leapflow.cli.context import Context


async def cmd_chat(ctx: "Context", prompt: str, thinking: bool) -> int:
    require_initialized(ctx)
    streamed = False
    async for event in ctx.engine.run_stream(prompt, enable_thinking=thinking):
        if isinstance(event, str):
            print(event, end="", flush=True)
        elif event.type == "chunk":
            print(event.content, end="", flush=True)
            streamed = True
        elif event.type == "final" and not streamed:
            print(event.content, end="", flush=True)
        # tool_call events are silently consumed
    print()
    return 0
