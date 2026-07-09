"""Gateway message router — per-session LLM processing for inbound platform messages.

Sits between ``GatewayServer`` (message ingress) and the LLM/tool layer
(response generation).  Each external session gets independent message
history so concurrent conversations on different platforms don't collide
with each other or the interactive CLI.

Module boundary
~~~~~~~~~~~~~~~
Depends on ``leapflow.llm`` (LLM provider interface) and optionally on
tool handlers, but **not** on ``AgentEngine`` or ``cli/``.  ``Context``
is the sole point that wires all dependencies.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

from leapflow.gateway.protocol import (
    InboundMessage,
    MessageSource,
    OutboundContent,
    SendTarget,
)

logger = logging.getLogger(__name__)

SendFn = Callable[[MessageSource, str], Coroutine[Any, Any, None]]


class GatewayRouter:
    """Routes inbound gateway messages through LLM with per-session history.

    Parameters
    ----------
    llm
        Any object implementing ``achat(messages, *, stream=False, **kw)``.
    system_prompt
        System message prepended to every new session.
    send_fn
        ``async (source, reply_text) -> None`` — called to deliver the
        LLM response back to the originating conversation.
    max_history
        Maximum messages retained per session before tail-trimming.
    """

    def __init__(
        self,
        *,
        llm: Any,
        system_prompt: str = "",
        send_fn: SendFn,
        max_history: int = 50,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt
        self._send_fn = send_fn
        self._max_history = max_history
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    async def handle_message(
        self,
        message: InboundMessage,
        session_key: str,
    ) -> None:
        """Process an inbound message: add to history → LLM call → reply.

        Serialises per-session to prevent interleaving within one chat.
        """
        lock = self._locks.setdefault(session_key, asyncio.Lock())
        async with lock:
            await self._process(message, session_key)

    async def _process(
        self,
        message: InboundMessage,
        session_key: str,
    ) -> None:
        history = self._sessions.get(session_key)
        if history is None:
            history = []
            if self._system_prompt:
                history.append({"role": "system", "content": self._system_prompt})
            self._sessions[session_key] = history

        history.append({"role": "user", "content": message.text})
        self._trim_history(history)

        try:
            resp = await self._llm.achat(history, stream=False)
            reply = (resp.content or "").strip()
        except Exception:
            logger.error(
                "LLM call failed for gateway session %s",
                session_key,
                exc_info=True,
            )
            return

        if not reply:
            return

        history.append({"role": "assistant", "content": reply})
        try:
            await self._send_fn(message.source, reply)
        except Exception:
            logger.error(
                "Failed to send reply for session %s",
                session_key,
                exc_info=True,
            )

    def _trim_history(self, history: List[Dict[str, Any]]) -> None:
        """Keep history under budget by discarding oldest non-system messages."""
        if len(history) <= self._max_history:
            return
        has_system = history and history[0].get("role") == "system"
        keep = self._max_history - (1 if has_system else 0)
        if has_system:
            history[1:] = history[-keep:]
        else:
            history[:] = history[-keep:]

    def clear_session(self, session_key: str) -> None:
        """Remove all state for a session (called on disconnect/timeout)."""
        self._sessions.pop(session_key, None)
        self._locks.pop(session_key, None)

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)
