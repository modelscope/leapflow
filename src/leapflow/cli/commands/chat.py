"""Chat subcommand — single-turn conversational mode with rich output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

from leapflow.cli.helpers import require_initialized

if TYPE_CHECKING:
    from leapflow.cli.context import Context
    from leapflow.daemon.client import DaemonClient


ApprovalResolver = Callable[[str, str], Awaitable[object]]


async def render_chat_stream(
    events: AsyncIterator[object],
    approval_resolver: ApprovalResolver | None = None,
) -> int:
    """Render a stream of engine events to the terminal."""
    from leapflow.cli.tui_app import detect_theme, LeapConsole, StreamRenderer

    theme = detect_theme()
    console = LeapConsole(theme)

    renderer = StreamRenderer(console)
    renderer.start()

    try:
        async for event in events:
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
            elif event.type == "error":
                renderer.feed(event.content)
            elif event.type == "approval_request":
                await _handle_approval_event(event, approval_resolver)
    finally:
        renderer.finish()

    return 0


async def _handle_approval_event(event: Any, approval_resolver: ApprovalResolver | None) -> None:
    if approval_resolver is None:
        return
    from leapflow.cli.approval_view import prompt_approval
    from leapflow.security.approval import ApprovalDecision, ApprovalRequest

    metadata = event.metadata or {}
    payload = metadata.get("approval")
    if not isinstance(payload, dict):
        return
    pending_id = str(payload.get("pending_id") or "")
    if not pending_id:
        return
    request = ApprovalRequest.from_dict(payload)
    decision = await prompt_approval(request)
    value = decision.value if isinstance(decision, ApprovalDecision) else str(decision)
    await approval_resolver(pending_id, value)


async def cmd_chat_daemon(client: "DaemonClient", prompt: str, thinking: bool) -> int:
    """Single-turn conversational mode backed by leapd."""
    return await render_chat_stream(
        client.engine_chat(prompt, enable_thinking=thinking),
        lambda pending_id, decision: client.approval_resolve(pending_id, decision),
    )


async def cmd_chat(ctx: "Context", prompt: str, thinking: bool) -> int:
    require_initialized(ctx)
    return await render_chat_stream(
        ctx.engine.run_stream(prompt, enable_thinking=thinking)
    )
