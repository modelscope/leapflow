"""Built-in clipboard manager skill."""

from __future__ import annotations

import logging

from leapflow.platform.protocol import HostRpc, Methods
from leapflow.llm.base import LLMProvider
from leapflow.llm.message_builder import build_system_message, build_user_message_text
from leapflow.memory.providers.semantic import SemanticMemoryProvider
from leapflow.memory.providers.working import WorkingMemoryProvider

logger = logging.getLogger(__name__)


async def run(
    rpc: HostRpc,
    llm: LLMProvider,
    wm: WorkingMemoryProvider,
    lt: SemanticMemoryProvider,
    *,
    user_goal: str,
) -> str:
    """Summarize clipboard contents and store a durable memory row."""
    state = await rpc.call(Methods.CLIPBOARD_GET, {})
    text = str(state.get("text", "")).strip()
    if not text:
        msg = "Clipboard is empty."
        wm.remember_event("clipboard", msg, {"user_goal": user_goal})
        return msg

    messages = [
        build_system_message("Summarize the clipboard text in 3-6 bullet points. Be factual."),
        build_user_message_text(f"Goal context: {user_goal}\nClipboard:\n{text}"),
    ]
    resp = await llm.achat(messages, stream=False, enable_thinking=False)
    summary = (resp.content or "").strip()
    lt.insert_raw("clipboard_summary", summary, metadata={"chars": len(text), "goal": user_goal})
    wm.remember_event("clipboard_summary", summary, {"chars": len(text)})
    logger.info("clipboard.summary stored (lt+wm)")
    return summary
