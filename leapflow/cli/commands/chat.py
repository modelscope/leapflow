"""Chat subcommand — single-turn conversational mode."""

from __future__ import annotations

from typing import TYPE_CHECKING

from leapflow.cli.helpers import require_initialized

if TYPE_CHECKING:
    from leapflow.cli.context import Context


async def cmd_chat(ctx: "Context", prompt: str, thinking: bool) -> int:
    require_initialized(ctx)
    async for chunk in ctx.engine.run_stream(prompt, enable_thinking=thinking):
        print(chunk, end="", flush=True)
    print()
    return 0
