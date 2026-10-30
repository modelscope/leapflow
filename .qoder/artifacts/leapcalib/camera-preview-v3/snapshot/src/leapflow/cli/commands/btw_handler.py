# Copyright (c) Alibaba, Inc. and its affiliates.
"""Handler for ``/btw`` (side question) slash command.

Kept in its own file to avoid inflating ``slash_handlers.py`` (>3700 lines).
The handler creates a :class:`SideQuestionFiber`, streams the LLM response
through the existing ``StreamRenderer``, and cleans up.

Both in-process and daemon code paths funnel here:

- **In-process** (``interactive.py``): called directly as
  ``await handle_btw(ctx, console, args)``.
- **Daemon** (``command_execute``): called via the ``btw`` branch in
  ``command_execute``, which returns a streaming payload.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from leapflow.cli.context import Context
    from leapflow.cli.tui_app.console import LeapConsole

logger = logging.getLogger(__name__)


async def handle_btw(
    ctx: "Context",
    console: "LeapConsole",
    args: str,
) -> None:
    """Execute a ``/btw`` side question with streaming output.

    The question is answered by the same LLM provider as the main session
    but in complete conversation isolation: no messages are written to the
    parent session's history, and no tool calls are made.

    Parameters
    ----------
    ctx:
        CLI context with engine and settings.
    console:
        TUI console for rendering output.
    args:
        The side question text (everything after ``/btw ``).
    """
    question = args.strip()
    if not question:
        console.warning("Usage: /btw <question>  — ask a quick side question")
        return

    engine = ctx.engine
    if engine is None:
        console.warning("No active engine — send a message first, then use /btw.")
        return

    from leapflow.engine.side_question import SideQuestionConfig, SideQuestionFiber

    parent_session_id = getattr(engine, "_current_session_id", "") or ""
    config = SideQuestionConfig(
        question=question,
        parent_session_id=parent_session_id,
    )
    fiber = SideQuestionFiber(engine, config)

    # Stream the response through the existing renderer
    from leapflow.cli.tui_app.stream import StreamRenderer

    renderer = StreamRenderer(console)
    renderer.start()
    try:
        async for chunk in fiber.run_stream():
            renderer.feed(chunk)
    except Exception as exc:
        logger.warning("/btw streaming failed: %s", exc, exc_info=True)
        console.warning(f"Side question failed: {exc}")
        return
    finally:
        renderer.finish()


async def build_btw_payload(
    ctx: "Context",
    args: str,
) -> Dict[str, Any]:
    """Build a side-question payload for daemon-mode execution.

    Unlike most ``command_execute`` payloads, ``/btw`` runs a full LLM
    call and returns the answer inline (the question is too lightweight to
    justify the full engine chat stream machinery).

    Returns a dict compatible with ``render_command_payload``.
    """
    question = args.strip()
    if not question:
        return {"ok": False, "message": "Usage: /btw <question>"}

    engine = ctx.engine
    if engine is None:
        return {"ok": False, "message": "No active engine — send a message first."}

    from leapflow.engine.side_question import SideQuestionConfig, SideQuestionFiber

    parent_session_id = getattr(engine, "_current_session_id", "") or ""
    config = SideQuestionConfig(
        question=question,
        parent_session_id=parent_session_id,
    )
    fiber = SideQuestionFiber(engine, config)

    try:
        answer = await fiber.run()
    except Exception as exc:
        logger.warning("/btw daemon execution failed: %s", exc, exc_info=True)
        return {"ok": False, "message": f"Side question failed: {exc}"}

    return {
        "ok": True,
        "view": "btw",
        "question": question,
        "answer": answer,
        "fiber_id": config.fiber_id,
        "parent_session_id": parent_session_id,
    }
