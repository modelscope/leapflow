"""Gateway message router — per-session LLM processing for inbound platform messages.

Sits between ``GatewayServer`` (message ingress) and the LLM/tool layer
(response generation).  Each external session gets independent message
history so concurrent conversations on different platforms don't collide
with each other or the interactive CLI.

Tool support
~~~~~~~~~~~~
The router can optionally execute a **restricted** set of safe tools
(read-only: memory_search, time_get, skills_list, etc.) during inbound
message processing.  Dangerous tools (shell_run, file_write, delegate_task)
are excluded by default — configurable via ``allowed_tools``.

Module boundary
~~~~~~~~~~~~~~~
Depends on ``leapflow.llm`` (LLM provider interface) and optionally on
tool handlers, but **not** on ``AgentEngine`` or ``cli/``.  ``Context``
is the sole point that wires all dependencies.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from leapflow.gateway.protocol import InboundMessage, MessageSource
from leapflow.tools.name_resolver import ToolRegistry


@runtime_checkable
class SessionPersistence(Protocol):
    """Minimal persistence interface for gateway sessions.

    Satisfied by ``DuckDBConversationStore`` without requiring
    a direct import — dependency inversion via structural subtyping.
    """

    def create_session(self, session_id: str, *, title: str = "", **kwargs: Any) -> Any: ...
    def get_session(self, session_id: str) -> Any: ...
    def append_message(self, session_id: str, role: str, content: str, **kwargs: Any) -> Any: ...
    def get_messages(self, session_id: str, *, limit: int = 100, active_only: bool = True) -> list[Any]: ...

logger = logging.getLogger(__name__)

SendFn = Callable[[MessageSource, str], Coroutine[Any, Any, None]]

SAFE_TOOLS: frozenset[str] = frozenset({
    "memory_search", "memory_add",
    "time_get", "env_info",
    "skills_list", "skill_view",
    "text_search",
    "gateway_connect", "gateway_send",
    "platform_connect", "platform_action",
})


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
    tool_definitions
        OpenAI-format tool schemas.  Filtered to ``allowed_tools`` on init.
    tool_handlers
        ``name → async handler`` mapping.  Filtered to ``allowed_tools``.
    allowed_tools
        Frozenset of tool names permitted for gateway sessions.
        Defaults to ``SAFE_TOOLS`` (read-only, no shell/file-write).
    max_history
        Maximum messages retained per session before tail-trimming.
    max_tool_rounds
        Maximum number of LLM→tool→LLM rounds before forcing a text reply.
    """

    def __init__(
        self,
        *,
        llm: Any,
        system_prompt: str = "",
        send_fn: SendFn,
        tool_definitions: Sequence[Dict[str, Any]] = (),
        tool_handlers: Optional[Dict[str, Any]] = None,
        allowed_tools: frozenset[str] = SAFE_TOOLS,
        max_history: int = 50,
        max_tool_rounds: int = 3,
        persistence: Optional[SessionPersistence] = None,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt
        self._send_fn = send_fn
        self._max_history = max_history
        self._max_tool_rounds = max_tool_rounds
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._persistence = persistence

        self._tool_defs = [
            td for td in tool_definitions
            if td.get("function", {}).get("name", "") in allowed_tools
        ]
        all_handlers = tool_handlers or {}
        self._tool_handlers = {
            k: v for k, v in all_handlers.items() if k in allowed_tools
        }
        self._tool_registry = ToolRegistry.from_definitions(self._tool_defs, self._tool_handlers)

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
            history = self._init_session(session_key, message)
            self._sessions[session_key] = history

        history.append({"role": "user", "content": message.text})
        self._persist_message(session_key, "user", message.text)
        self._trim_history(history)

        reply = await self._llm_with_tools(history)
        if not reply:
            return

        history.append({"role": "assistant", "content": reply})
        self._persist_message(session_key, "assistant", reply)
        try:
            await self._send_fn(message.source, reply)
        except Exception:
            logger.error(
                "Failed to send reply for session %s",
                session_key,
                exc_info=True,
            )

    def _init_session(
        self,
        session_key: str,
        message: InboundMessage,
    ) -> List[Dict[str, Any]]:
        """Initialize a session, restoring history from persistence if available."""
        history: List[Dict[str, Any]] = []
        if self._system_prompt:
            history.append({"role": "system", "content": self._system_prompt})

        if self._persistence is not None:
            try:
                existing = self._persistence.get_session(session_key)
                if existing is not None:
                    stored = self._persistence.get_messages(
                        session_key, limit=self._max_history,
                    )
                    for msg in stored:
                        role = getattr(msg, "role", "")
                        content = getattr(msg, "content", "")
                        if role in ("user", "assistant") and content:
                            history.append({"role": role, "content": content})
                    if stored:
                        logger.info(
                            "Restored %d messages for session %s",
                            len(stored), session_key[:30],
                        )
                else:
                    source = message.source
                    title = f"{source.platform}:{source.chat_type}:{source.chat_id}"
                    self._persistence.create_session(
                        session_key,
                        title=title[:200],
                        source=f"gateway:{source.platform}",
                    )
            except Exception:
                logger.debug("Session persistence error for %s", session_key, exc_info=True)

        return history

    def _persist_message(self, session_key: str, role: str, content: str) -> None:
        """Append a message to persistent storage (fire-and-forget)."""
        if self._persistence is None:
            return
        try:
            self._persistence.append_message(session_key, role, content)
        except Exception:
            logger.debug("Failed to persist %s message for %s", role, session_key, exc_info=True)

    async def _llm_with_tools(
        self,
        history: List[Dict[str, Any]],
    ) -> str:
        """Call LLM with optional tool loop (bounded rounds)."""
        llm_kwargs: Dict[str, Any] = {}
        if self._tool_defs:
            llm_kwargs["tools"] = self._tool_defs

        for _round in range(self._max_tool_rounds + 1):
            try:
                resp = await self._llm.achat(history, stream=False, **llm_kwargs)
            except Exception:
                logger.error("LLM call failed in gateway router", exc_info=True)
                return ""

            tool_calls = getattr(resp, "tool_calls", None)
            if not tool_calls:
                return (resp.content or "").strip()

            for tc in tool_calls:
                tool_args = getattr(tc, "arguments", {})
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}
                if not isinstance(tool_args, dict):
                    tool_args = {}

                resolution = self._tool_registry.resolve(tc.name, tool_args)
                handler = (
                    self._tool_handlers.get(resolution.normalized_name)
                    if resolution.auto_executable and resolution.normalized_name
                    else None
                )
                if handler is None:
                    unknown = self._tool_registry.unknown_result(resolution)
                    history.append({
                        "role": "tool",
                        "tool_call_id": getattr(tc, "id", ""),
                        "content": json.dumps(unknown, default=str, ensure_ascii=False),
                    })
                    continue
                try:
                    result = await asyncio.wait_for(handler(tool_args), timeout=30)
                    result_text = json.dumps(result, default=str, ensure_ascii=False)[:2000]
                except asyncio.TimeoutError:
                    result_text = f"Tool '{tc.name}' timed out"
                except Exception as exc:
                    result_text = f"Tool error: {type(exc).__name__}"

                history.append({
                    "role": "tool",
                    "tool_call_id": getattr(tc, "id", ""),
                    "content": result_text,
                })

        logger.warning("Gateway router: max tool rounds exceeded")
        return ""

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

    async def handle_callback(
        self,
        callback: Any,
        session_key: str,
    ) -> None:
        """Process an inbound callback: inject action context into session, LLM call, reply.

        Callbacks (card button clicks, form submissions) carry action_value
        that provides context for the LLM to generate an appropriate response.
        """
        lock = self._locks.setdefault(session_key, asyncio.Lock())
        async with lock:
            from leapflow.gateway.connectors.protocol import InboundCallback

            if not isinstance(callback, InboundCallback):
                return
            action_desc = (
                f"[Card action: {callback.action_type}] "
                f"value={callback.action_value}"
            )
            history = self._sessions.get(session_key)
            if history is None:
                history = []
                if self._system_prompt:
                    history.append({"role": "system", "content": self._system_prompt})
                self._sessions[session_key] = history

            history.append({"role": "user", "content": action_desc})
            self._persist_message(session_key, "user", action_desc)
            self._trim_history(history)

            reply = await self._llm_with_tools(history)
            if not reply:
                return

            history.append({"role": "assistant", "content": reply})
            self._persist_message(session_key, "assistant", reply)
            try:
                await self._send_fn(callback.source, reply)
            except Exception:
                logger.error(
                    "Failed to send callback reply for session %s",
                    session_key,
                    exc_info=True,
                )

    def clear_session(self, session_key: str) -> None:
        """Remove all state for a session (called on disconnect/timeout)."""
        self._sessions.pop(session_key, None)
        self._locks.pop(session_key, None)

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)
