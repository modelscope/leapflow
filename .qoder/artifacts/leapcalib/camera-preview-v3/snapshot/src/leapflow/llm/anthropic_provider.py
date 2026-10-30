# Copyright (c) Alibaba, Inc. and its affiliates.
"""Native Anthropic Messages API provider.

Implements ``LLMProvider`` using the official ``anthropic`` Python SDK.  Supports:
- Synchronous and asynchronous chat (``achat``, ``achat_stream``)
- Explicit ``cache_control`` breakpoint pass-through (for AnthropicCacheStrategy)
- Usage parsing with ``cache_creation_input_tokens`` / ``cache_read_input_tokens``
- ``base_url`` override (e.g. ``https://api.deepseek.com/anthropic``)
- Graceful degradation when the ``anthropic`` SDK is not installed

The module is safe to import even without the SDK installed — all SDK references
are isolated inside ``AnthropicChat`` construction / method bodies and guarded
by a top-level availability flag.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, AsyncIterator, Dict, List

from leapflow.llm.base import ChunkCallback, LLMChatResponse, LLMProvider, ToolCallInfo

logger = logging.getLogger(__name__)

# ── SDK availability gate ──────────────────────────────────────────────────
_ANTHROPIC_AVAILABLE = False
try:
    import anthropic as _anthropic_sdk  # noqa: F401

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _anthropic_sdk = None  # type: ignore[assignment]


def is_anthropic_available() -> bool:
    """Return True when the ``anthropic`` SDK is importable."""
    return _ANTHROPIC_AVAILABLE


# ── Retryable error types (populated only when SDK is present) ─────────────
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = ()
if _ANTHROPIC_AVAILABLE:
    _RETRYABLE_ERRORS = (
        _anthropic_sdk.APIConnectionError,
        _anthropic_sdk.APITimeoutError,
        _anthropic_sdk.InternalServerError,
        _anthropic_sdk.RateLimitError,
    )


class AnthropicChatResponse(LLMChatResponse):
    """Concrete response type returned by :class:`AnthropicChat`."""


# ── Message conversion helpers ─────────────────────────────────────────────

def _convert_messages(messages: List[Dict[str, Any]]) -> tuple[
    str | List[Dict[str, Any]], List[Dict[str, Any]]
]:
    """Split LeapFlow message list into Anthropic ``system`` and ``messages``.

    Returns:
        (system_content, conversation_messages) where system_content is either
        a plain string or a list of content-block dicts (when cache_control is
        present), and conversation_messages are user/assistant turns.
    """
    system_parts: List[Dict[str, Any]] = []
    conversation: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            content = msg.get("content", "")
            cache_ctrl = msg.get("cache_control")
            if isinstance(content, list):
                # Already structured content blocks — pass through as-is.
                system_parts.extend(content)
            elif cache_ctrl:
                system_parts.append({
                    "type": "text",
                    "text": str(content),
                    "cache_control": cache_ctrl,
                })
            else:
                system_parts.append({"type": "text", "text": str(content)})
        elif role == "tool":
            # Map OpenAI-format tool result → Anthropic tool_result content block.
            conversation.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": str(msg.get("content", "")),
                }],
            })
        else:
            converted_msg: Dict[str, Any] = {"role": role}
            content = msg.get("content", "")
            cache_ctrl = msg.get("cache_control")
            if isinstance(content, list):
                # Structured content blocks — pass through.
                converted_msg["content"] = content
            elif cache_ctrl:
                converted_msg["content"] = [{
                    "type": "text",
                    "text": str(content),
                    "cache_control": cache_ctrl,
                }]
            else:
                converted_msg["content"] = str(content)
            conversation.append(converted_msg)

    # Anthropic requires alternating user/assistant. Merge consecutive same-role
    # messages to avoid API errors.
    merged: List[Dict[str, Any]] = []
    for msg in conversation:
        if merged and merged[-1]["role"] == msg["role"]:
            # Merge content into the previous message.
            prev_content = merged[-1]["content"]
            new_content = msg["content"]
            if isinstance(prev_content, str) and isinstance(new_content, str):
                merged[-1]["content"] = prev_content + "\n" + new_content
            else:
                # Convert to block format for merging.
                if isinstance(prev_content, str):
                    prev_content = [{"type": "text", "text": prev_content}]
                if isinstance(new_content, str):
                    new_content = [{"type": "text", "text": new_content}]
                merged[-1]["content"] = prev_content + new_content
        else:
            merged.append(msg)

    # Build system param: plain string when no cache markers, else block list.
    if not system_parts:
        system_content: str | List[Dict[str, Any]] = ""
    elif len(system_parts) == 1 and "cache_control" not in system_parts[0]:
        system_content = system_parts[0].get("text", "")
    else:
        system_content = system_parts

    return system_content, merged


def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-format tool definitions to Anthropic tool format."""
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool.get("function", {})
            entry: Dict[str, Any] = {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
            # Preserve cache_control if present on the original tool definition.
            if "cache_control" in tool:
                entry["cache_control"] = tool["cache_control"]
            converted.append(entry)
    return converted


def _parse_usage(usage: Any) -> Dict[str, int]:
    """Extract token counts from an Anthropic usage object.

    Mapping:
    - ``input_tokens`` → ``prompt_tokens``
    - ``output_tokens`` → ``completion_tokens``
    - ``cache_read_input_tokens`` → ``cached_tokens`` (cache hit count)
    - Preserves original Anthropic-specific fields for regression telemetry.
    """
    result: Dict[str, int] = {}
    if usage is None:
        return result

    input_t = getattr(usage, "input_tokens", None)
    if isinstance(input_t, int):
        result["prompt_tokens"] = input_t

    output_t = getattr(usage, "output_tokens", None)
    if isinstance(output_t, int):
        result["completion_tokens"] = output_t

    if isinstance(input_t, int) and isinstance(output_t, int):
        result["total_tokens"] = input_t + output_t

    # Anthropic cache fields
    cache_read = getattr(usage, "cache_read_input_tokens", None)
    if isinstance(cache_read, int):
        result["cache_read_input_tokens"] = cache_read
        result["cached_tokens"] = cache_read  # unified key

    cache_create = getattr(usage, "cache_creation_input_tokens", None)
    if isinstance(cache_create, int):
        result["cache_creation_input_tokens"] = cache_create

    return result


class AnthropicChat(LLMProvider):
    """Native Anthropic Messages API client with retries and streaming.

    Args:
        api_key: Anthropic API key.
        model: Model identifier (e.g. ``claude-sonnet-4-20250514``).
        base_url: Optional API base URL override.
        max_retries: Maximum retry attempts for transient errors.
        timeout_s: Per-request timeout in seconds.
        max_tokens: Maximum response tokens (Anthropic requires this explicitly).
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        max_retries: int = 3,
        timeout_s: float = 180.0,
        max_tokens: int = 8192,
    ) -> None:
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic SDK is not installed. "
                "Install it with: pip install anthropic"
            )

        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout_s,
            "max_retries": 0,  # We own retry logic.
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._sync = _anthropic_sdk.Anthropic(**client_kwargs)
        self._async = _anthropic_sdk.AsyncAnthropic(**client_kwargs)
        self._model = model
        self._max_retries = max(1, int(max_retries))
        self._max_tokens = max_tokens
        self._base_url = base_url or "https://api.anthropic.com"
        logger.info(
            "Anthropic provider initialized: model=%s base_url=%s",
            model, self._base_url,
        )

    @property
    def model(self) -> str:
        return self._model

    async def _sleep_backoff(self, attempt: int) -> None:
        base = 0.75 * (2 ** attempt)
        jitter = random.random() * 0.35
        await asyncio.sleep(base + jitter)

    # ── Main entry points ──────────────────────────────────────────────────

    async def achat(
        self,
        messages: List[Dict[str, Any]],
        *,
        stream: bool = True,
        enable_thinking: bool = False,
        on_chunk: ChunkCallback = None,
        **kwargs: Any,
    ) -> AnthropicChatResponse:
        last_err: BaseException | None = None
        for attempt in range(self._max_retries):
            try:
                if stream:
                    return await self._achat_stream_collapsed(
                        messages,
                        enable_thinking=enable_thinking,
                        on_chunk=on_chunk,
                        **kwargs,
                    )
                return await self._achat_nonstream(
                    messages, enable_thinking=enable_thinking, **kwargs,
                )
            except _RETRYABLE_ERRORS as exc:
                last_err = exc
                if attempt >= self._max_retries - 1:
                    logger.warning(
                        "Anthropic request failed after %d attempts: %s",
                        self._max_retries, exc,
                    )
                    break
                logger.debug(
                    "Anthropic retry %d/%d: %s",
                    attempt + 1, self._max_retries, exc,
                )
                await self._sleep_backoff(attempt)
            except Exception:
                raise
        assert last_err is not None
        raise last_err

    async def achat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        create_kwargs = self._build_create_kwargs(
            messages, enable_thinking=enable_thinking, **kwargs,
        )
        last_err: BaseException | None = None
        for attempt in range(self._max_retries):
            try:
                async with self._async.messages.stream(**create_kwargs) as stream:
                    async for text in stream.text_stream:
                        yield text
                return
            except _RETRYABLE_ERRORS as exc:
                last_err = exc
                if attempt >= self._max_retries - 1:
                    logger.warning(
                        "Anthropic stream failed after %d attempts: %s",
                        self._max_retries, exc,
                    )
                    break
                logger.debug(
                    "Anthropic stream retry %d/%d: %s",
                    attempt + 1, self._max_retries, exc,
                )
                await self._sleep_backoff(attempt)
        assert last_err is not None
        raise last_err

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_create_kwargs(
        self,
        messages: List[Dict[str, Any]],
        *,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build kwargs dict for ``messages.create`` / ``messages.stream``."""
        system_content, conversation = _convert_messages(messages)

        create_kwargs: Dict[str, Any] = {
            "model": kwargs.pop("model", self._model),
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
            "messages": conversation,
        }
        if system_content:
            create_kwargs["system"] = system_content

        # Tool definitions
        tools_raw = kwargs.pop("tools", None)
        if tools_raw:
            create_kwargs["tools"] = _convert_tools(tools_raw)

        # Anthropic extended thinking (beta)
        if enable_thinking:
            create_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": kwargs.pop("thinking_budget", 4096),
            }

        # Pass remaining kwargs through (e.g. temperature, top_p).
        for k, v in kwargs.items():
            if k not in create_kwargs:
                create_kwargs[k] = v

        return create_kwargs

    async def _achat_nonstream(
        self,
        messages: List[Dict[str, Any]],
        *,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> AnthropicChatResponse:
        create_kwargs = self._build_create_kwargs(
            messages, enable_thinking=enable_thinking, **kwargs,
        )
        t0 = time.monotonic()
        resp = await self._async.messages.create(**create_kwargs)
        dt_ms = int((time.monotonic() - t0) * 1000)

        text_parts: List[str] = []
        thinking_parts: List[str] = []
        tool_calls: List[ToolCallInfo] = []

        for block in resp.content:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif block_type == "thinking":
                thinking_parts.append(getattr(block, "thinking", ""))
            elif block_type == "tool_use":
                tool_calls.append(ToolCallInfo(
                    id=getattr(block, "id", ""),
                    name=getattr(block, "name", ""),
                    arguments=getattr(block, "input", {}),
                ))

        usage = _parse_usage(getattr(resp, "usage", None))
        usage["latency_ms"] = dt_ms

        return AnthropicChatResponse(
            content="".join(text_parts),
            role="assistant",
            usage=usage,
            model=getattr(resp, "model", self._model),
            finish_reason=getattr(resp, "stop_reason", None),
            thinking_content="".join(thinking_parts) if thinking_parts else None,
            tool_calls=tool_calls,
        )

    async def _achat_stream_collapsed(
        self,
        messages: List[Dict[str, Any]],
        *,
        enable_thinking: bool = False,
        on_chunk: ChunkCallback = None,
        **kwargs: Any,
    ) -> AnthropicChatResponse:
        create_kwargs = self._build_create_kwargs(
            messages, enable_thinking=enable_thinking, **kwargs,
        )
        t0 = time.monotonic()

        text_parts: List[str] = []
        thinking_parts: List[str] = []
        tool_calls: List[ToolCallInfo] = []
        usage: Dict[str, int] = {}
        model_name: str | None = None
        stop_reason: str | None = None

        # Accumulate tool_use blocks from streaming events.
        _current_tool: Dict[str, Any] | None = None
        _tool_json_parts: List[str] = []

        async with self._async.messages.stream(**create_kwargs) as stream:
            async for event in stream:
                event_type = getattr(event, "type", "")

                if event_type == "message_start":
                    msg = getattr(event, "message", None)
                    if msg:
                        model_name = getattr(msg, "model", None)
                        u = getattr(msg, "usage", None)
                        if u:
                            usage.update(_parse_usage(u))

                elif event_type == "content_block_start":
                    cb = getattr(event, "content_block", None)
                    if cb and getattr(cb, "type", "") == "tool_use":
                        _current_tool = {
                            "id": getattr(cb, "id", ""),
                            "name": getattr(cb, "name", ""),
                        }
                        _tool_json_parts = []

                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        delta_type = getattr(delta, "type", "")
                        if delta_type == "text_delta":
                            text = getattr(delta, "text", "")
                            if text:
                                text_parts.append(text)
                                if on_chunk is not None:
                                    on_chunk(text)
                        elif delta_type == "thinking_delta":
                            thinking = getattr(delta, "thinking", "")
                            if thinking:
                                thinking_parts.append(thinking)
                        elif delta_type == "input_json_delta":
                            partial = getattr(delta, "partial_json", "")
                            if partial:
                                _tool_json_parts.append(partial)

                elif event_type == "content_block_stop":
                    if _current_tool is not None:
                        raw_json = "".join(_tool_json_parts)
                        try:
                            args = json.loads(raw_json) if raw_json else {}
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        tool_calls.append(ToolCallInfo(
                            id=_current_tool["id"],
                            name=_current_tool["name"],
                            arguments=args,
                        ))
                        _current_tool = None
                        _tool_json_parts = []

                elif event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        sr = getattr(delta, "stop_reason", None)
                        if sr:
                            stop_reason = sr
                    u = getattr(event, "usage", None)
                    if u:
                        usage.update(_parse_usage(u))

        dt_ms = int((time.monotonic() - t0) * 1000)
        usage.setdefault("latency_ms", dt_ms)

        return AnthropicChatResponse(
            content="".join(text_parts),
            role="assistant",
            usage=usage,
            model=model_name or self._model,
            finish_reason=stop_reason,
            thinking_content="".join(thinking_parts) if thinking_parts else None,
            tool_calls=tool_calls,
        )
