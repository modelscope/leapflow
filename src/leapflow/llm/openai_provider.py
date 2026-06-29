"""OpenAI-compatible chat client with provider profiles, retries, and streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import openai
from openai import AsyncOpenAI, OpenAI

from leapflow.llm.base import ChunkCallback, LLMChatResponse, LLMProvider, ToolCallInfo

logger = logging.getLogger(__name__)

_RETRYABLE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.RateLimitError,
    openai.NotFoundError,
)


@dataclass(frozen=True)
class _ProviderProfile:
    """Provider-specific behavior flags."""

    name: str
    stream_options: bool = True
    thinking_param: Optional[str] = None
    thinking_content_field: Optional[str] = None


_PROVIDERS: Dict[str, _ProviderProfile] = {
    "openai": _ProviderProfile("openai"),
    "azure": _ProviderProfile("azure"),
    "anthropic_proxy": _ProviderProfile("anthropic_proxy"),
    "dashscope": _ProviderProfile(
        "dashscope",
        thinking_param="enable_thinking",
        thinking_content_field="reasoning_content",
    ),
    "deepseek": _ProviderProfile(
        "deepseek",
        thinking_param="enable_thinking",
        thinking_content_field="reasoning_content",
    ),
    "groq": _ProviderProfile("groq"),
    "generic": _ProviderProfile("generic"),
}


def _normalize_host(url: str) -> str:
    u = url.lower().strip()
    u = re.sub(r"^https?://", "", u)
    u = u.split("/")[0]
    return u


def detect_provider_from_base_url(base_url: str) -> str:
    """Infer a coarse provider key from an OpenAI-compatible `base_url`.

    Args:
        base_url: Root or versioned API base URL.

    Returns:
        Provider key used to select request shaping behavior.
    """
    host = _normalize_host(base_url)
    if "dashscope" in host or "aliyuncs" in host:
        return "dashscope"
    if "deepseek" in host:
        return "deepseek"
    if "groq.com" in host:
        return "groq"
    if "openai.azure.com" in host or ".azure.com" in host:
        return "azure"
    if "openai.com" in host:
        return "openai"
    return "generic"


def _profile_for(provider: Optional[str], base_url: str) -> _ProviderProfile:
    key = (provider or "").strip().lower()
    if not key:
        key = detect_provider_from_base_url(base_url)
    return _PROVIDERS.get(key, _PROVIDERS["generic"])


class OpenAIChatResponse(LLMChatResponse):
    """Concrete chat response type used by :class:`OpenAIChat`."""


class OpenAIChat(LLMProvider):
    """OpenAI SDK client with sync/async, streaming, thinking hooks, and retries."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        max_retries: int = 3,
        provider: Optional[str] = None,
        timeout_s: float = 180.0,
    ) -> None:
        timeout = httpx.Timeout(
            connect=30.0,
            read=timeout_s,
            write=30.0,
            pool=30.0,
        )
        self._sync = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._async = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._model = model
        self._max_retries = max(1, int(max_retries))
        self._base_url = base_url
        self._profile = _profile_for(provider, base_url)
        logger.info("LLM provider profile selected: %s", self._profile.name)

    @property
    def model(self) -> str:
        return self._model

    def _thinking_kwargs(self, enable_thinking: bool) -> Dict[str, Any]:
        if not enable_thinking or not self._profile.thinking_param:
            return {}
        return {"extra_body": {self._profile.thinking_param: True}}

    def _stream_options(self) -> Dict[str, Any]:
        if not self._profile.stream_options:
            return {}
        return {"stream_options": {"include_usage": True}}

    def _extract_text_and_thinking(self, message: Any) -> tuple[str, Optional[str]]:
        content = getattr(message, "content", None) or ""
        thinking: Optional[str] = None
        if self._profile.thinking_content_field:
            thinking = getattr(message, self._profile.thinking_content_field, None)
        if thinking is None:
            reasoning = getattr(message, "reasoning_content", None)
            if isinstance(reasoning, str):
                thinking = reasoning
        return content, thinking

    async def _sleep_backoff(self, attempt: int) -> None:
        base = 0.75 * (2**attempt)
        jitter = random.random() * 0.35
        await asyncio.sleep(base + jitter)

    async def achat(
        self,
        messages: List[Dict[str, Any]],
        *,
        stream: bool = True,
        enable_thinking: bool = False,
        on_chunk: ChunkCallback = None,
        **kwargs: Any,
    ) -> OpenAIChatResponse:
        last_err: Optional[BaseException] = None
        for attempt in range(self._max_retries):
            try:
                if stream:
                    return await self._achat_stream_collapsed(
                        messages, enable_thinking=enable_thinking,
                        on_chunk=on_chunk, **kwargs,
                    )
                return await self._achat_nonstream(
                    messages, enable_thinking=enable_thinking, **kwargs
                )
            except _RETRYABLE_ERRORS as exc:
                last_err = exc
                if attempt >= self._max_retries - 1:
                    logger.warning("LLM request failed after %d attempts: %s", self._max_retries, exc)
                    break
                logger.debug("LLM retry %d/%d: %s", attempt + 1, self._max_retries, exc)
                await self._sleep_backoff(attempt)
            except Exception:
                raise
        assert last_err is not None
        raise last_err

    async def _achat_nonstream(
        self,
        messages: List[Dict[str, Any]],
        *,
        enable_thinking: bool,
        **kwargs: Any,
    ) -> OpenAIChatResponse:
        create_kwargs = {
            "model": kwargs.pop("model", self._model),
            "messages": messages,
            **self._thinking_kwargs(enable_thinking),
            **kwargs,
        }
        t0 = time.monotonic()
        resp = await self._async.chat.completions.create(**create_kwargs)
        dt_ms = int((time.monotonic() - t0) * 1000)
        choice = resp.choices[0]
        msg = choice.message
        text, thinking = self._extract_text_and_thinking(msg)
        usage_map: Dict[str, int] = {}
        u = getattr(resp, "usage", None)
        if u is not None:
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                v = getattr(u, k, None)
                if isinstance(v, int):
                    usage_map[k] = v
        usage_map["latency_ms"] = dt_ms

        # Extract native tool_calls from the response message
        tool_calls_raw = getattr(msg, "tool_calls", None) or []
        tool_calls_parsed: list[ToolCallInfo] = []
        for tc in tool_calls_raw:
            tc_id = getattr(tc, "id", "") or ""
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            fn_name = getattr(fn, "name", "") or ""
            fn_args_raw = getattr(fn, "arguments", "{}") or "{}"
            try:
                fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
            except (json.JSONDecodeError, TypeError):
                fn_args = {}
            tool_calls_parsed.append(ToolCallInfo(id=tc_id, name=fn_name, arguments=fn_args))

        return OpenAIChatResponse(
            content=text or "",
            role=getattr(msg, "role", "assistant") or "assistant",
            usage=usage_map,
            model=getattr(resp, "model", None),
            finish_reason=getattr(choice, "finish_reason", None),
            thinking_content=thinking,
            tool_calls=tool_calls_parsed,
        )

    async def _achat_stream_collapsed(
        self,
        messages: List[Dict[str, Any]],
        *,
        enable_thinking: bool,
        on_chunk: ChunkCallback = None,
        **kwargs: Any,
    ) -> OpenAIChatResponse:
        create_kwargs = {
            "model": kwargs.pop("model", self._model),
            "messages": messages,
            "stream": True,
            **self._stream_options(),
            **self._thinking_kwargs(enable_thinking),
            **kwargs,
        }
        t0 = time.monotonic()
        stream = await self._async.chat.completions.create(**create_kwargs)

        parts: List[str] = []
        thinking_parts: List[str] = []
        finish_reason: Optional[str] = None
        model_name: Optional[str] = None
        usage_map: Dict[str, int] = {}

        async for chunk in stream:
            if getattr(chunk, "model", None):
                model_name = chunk.model
            ch0 = chunk.choices[0] if chunk.choices else None
            if ch0 is None:
                continue
            if getattr(ch0, "finish_reason", None):
                finish_reason = ch0.finish_reason
            delta = ch0.delta
            d_content = getattr(delta, "content", None)
            if d_content:
                parts.append(d_content)
                if on_chunk is not None:
                    on_chunk(d_content)
            field = self._profile.thinking_content_field or "reasoning_content"
            d_think = getattr(delta, field, None)
            if isinstance(d_think, str) and d_think:
                thinking_parts.append(d_think)
            u = getattr(chunk, "usage", None)
            if u is not None:
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v = getattr(u, k, None)
                    if isinstance(v, int):
                        usage_map[k] = v

        dt_ms = int((time.monotonic() - t0) * 1000)
        usage_map.setdefault("latency_ms", dt_ms)
        thinking = "".join(thinking_parts) if thinking_parts else None

        # Extract native tool_calls from final streamed chunk (if available)
        # Note: tool_calls in streaming arrive as incremental deltas;
        # the OpenAI SDK accumulates them on the final chunk's message.
        # For streaming mode, we accumulate tool_call deltas.
        tool_calls_parsed: list[ToolCallInfo] = []
        # tool_calls are NOT reliably available from stream chunks in all providers;
        # this is handled by the non-stream path. Streaming + tool_calls should use
        # stream=False mode (which engine._unified_tool_loop already does).

        return OpenAIChatResponse(
            content="".join(parts),
            usage=usage_map,
            model=model_name,
            finish_reason=finish_reason,
            thinking_content=thinking,
            tool_calls=tool_calls_parsed,
        )

    async def achat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        create_kwargs = {
            "model": kwargs.pop("model", self._model),
            "messages": messages,
            "stream": True,
            **self._stream_options(),
            **self._thinking_kwargs(enable_thinking),
            **kwargs,
        }

        last_err: Optional[BaseException] = None
        for attempt in range(self._max_retries):
            try:
                stream = await self._async.chat.completions.create(**create_kwargs)
                async for chunk in stream:
                    ch0 = chunk.choices[0] if chunk.choices else None
                    if ch0 is None:
                        continue
                    delta = ch0.delta
                    d_content = getattr(delta, "content", None)
                    if d_content:
                        yield d_content
                return
            except _RETRYABLE_ERRORS as exc:
                last_err = exc
                if attempt >= self._max_retries - 1:
                    logger.warning("LLM stream failed after %d attempts: %s", self._max_retries, exc)
                    break
                logger.debug("LLM stream retry %d/%d: %s", attempt + 1, self._max_retries, exc)
                await self._sleep_backoff(attempt)
        assert last_err is not None
        raise last_err

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        stream: bool = True,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> OpenAIChatResponse:
        """Synchronous chat with retry (uses the synchronous OpenAI client)."""

        last_err: Optional[BaseException] = None
        for attempt in range(self._max_retries):
            try:
                if stream:
                    return self._chat_stream_collapsed(
                        messages, enable_thinking=enable_thinking, **kwargs
                    )
                return self._chat_nonstream(
                    messages, enable_thinking=enable_thinking, **kwargs
                )
            except _RETRYABLE_ERRORS as exc:
                last_err = exc
                if attempt >= self._max_retries - 1:
                    logger.warning("LLM request failed after %d attempts: %s", self._max_retries, exc)
                    break
                logger.debug("LLM retry %d/%d: %s", attempt + 1, self._max_retries, exc)
                time.sleep(0.75 * (2**attempt) + random.random() * 0.35)
            except Exception:
                raise
        assert last_err is not None
        raise last_err

    def _chat_nonstream(
        self,
        messages: List[Dict[str, Any]],
        *,
        enable_thinking: bool,
        **kwargs: Any,
    ) -> OpenAIChatResponse:
        create_kwargs = {
            "model": kwargs.pop("model", self._model),
            "messages": messages,
            **self._thinking_kwargs(enable_thinking),
            **kwargs,
        }
        t0 = time.monotonic()
        resp = self._sync.chat.completions.create(**create_kwargs)
        dt_ms = int((time.monotonic() - t0) * 1000)
        choice = resp.choices[0]
        msg = choice.message
        text, thinking = self._extract_text_and_thinking(msg)
        usage_map: Dict[str, int] = {}
        u = getattr(resp, "usage", None)
        if u is not None:
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                v = getattr(u, k, None)
                if isinstance(v, int):
                    usage_map[k] = v
        usage_map["latency_ms"] = dt_ms

        # Extract native tool_calls from the response message (sync path)
        tool_calls_raw = getattr(msg, "tool_calls", None) or []
        tool_calls_parsed: list[ToolCallInfo] = []
        for tc in tool_calls_raw:
            tc_id = getattr(tc, "id", "") or ""
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            fn_name = getattr(fn, "name", "") or ""
            fn_args_raw = getattr(fn, "arguments", "{}") or "{}"
            try:
                fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
            except (json.JSONDecodeError, TypeError):
                fn_args = {}
            tool_calls_parsed.append(ToolCallInfo(id=tc_id, name=fn_name, arguments=fn_args))

        return OpenAIChatResponse(
            content=text or "",
            role=getattr(msg, "role", "assistant") or "assistant",
            usage=usage_map,
            model=getattr(resp, "model", None),
            finish_reason=getattr(choice, "finish_reason", None),
            thinking_content=thinking,
            tool_calls=tool_calls_parsed,
        )

    def _chat_stream_collapsed(
        self,
        messages: List[Dict[str, Any]],
        *,
        enable_thinking: bool,
        **kwargs: Any,
    ) -> OpenAIChatResponse:
        create_kwargs = {
            "model": kwargs.pop("model", self._model),
            "messages": messages,
            "stream": True,
            **self._stream_options(),
            **self._thinking_kwargs(enable_thinking),
            **kwargs,
        }
        t0 = time.monotonic()
        stream = self._sync.chat.completions.create(**create_kwargs)

        parts: List[str] = []
        thinking_parts: List[str] = []
        finish_reason: Optional[str] = None
        model_name: Optional[str] = None
        usage_map: Dict[str, int] = {}

        for chunk in stream:
            if getattr(chunk, "model", None):
                model_name = chunk.model
            ch0 = chunk.choices[0] if chunk.choices else None
            if ch0 is None:
                continue
            if getattr(ch0, "finish_reason", None):
                finish_reason = ch0.finish_reason
            delta = ch0.delta
            d_content = getattr(delta, "content", None)
            if d_content:
                parts.append(d_content)
            field = self._profile.thinking_content_field or "reasoning_content"
            d_think = getattr(delta, field, None)
            if isinstance(d_think, str) and d_think:
                thinking_parts.append(d_think)
            u = getattr(chunk, "usage", None)
            if u is not None:
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v = getattr(u, k, None)
                    if isinstance(v, int):
                        usage_map[k] = v

        dt_ms = int((time.monotonic() - t0) * 1000)
        usage_map.setdefault("latency_ms", dt_ms)
        thinking = "".join(thinking_parts) if thinking_parts else None
        return OpenAIChatResponse(
            content="".join(parts),
            usage=usage_map,
            model=model_name,
            finish_reason=finish_reason,
            thinking_content=thinking,
        )
