# Copyright (c) Alibaba, Inc. and its affiliates.
"""Built-in LLM provider plugins.

Contains plugin wrappers for providers that ship with LeapFlow.
These satisfy the LLMProviderPlugin protocol and are registered
by LLMProviderRegistry.discover_builtin().
"""
from __future__ import annotations

from typing import Any, Dict, List

from leapflow.llm.base import LLMProvider


class OpenAICompatiblePlugin:
    """Plugin for all OpenAI-compatible providers.

    Wraps the existing OpenAIChat implementation, which already supports
    OpenAI, Azure, DeepSeek, Dashscope, Groq, and any generic
    OpenAI-format API. The provider auto-detects the backend from the
    base_url and adjusts behavior (stream_options, thinking params, etc.).

    Config keys:
        api_key: str — API key (required)
        base_url: str — API endpoint URL (required)
        model: str — Model identifier (required)
        max_retries: int — Retry count (default: 3)
        timeout_s: float — Request timeout seconds (default: 180.0)
        provider: str — Force a specific provider profile
                        (openai/azure/deepseek/dashscope/groq/generic)
    """

    @property
    def provider_id(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI-Compatible (OpenAI, Azure, DeepSeek, Dashscope, Groq, etc.)"

    @property
    def supported_models(self) -> List[str]:
        return [
            "gpt-4o*",
            "gpt-4-turbo*",
            "gpt-4.1*",
            "gpt-3.5*",
            "o1-*", "o3-*", "o4-*",
            "claude-*",
            "deepseek-*",
            "qwen-*",
        ]

    @property
    def capabilities(self) -> Dict[str, Any]:
        return {
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": True,
            "supports_thinking": True,
            "credential_rotation": True,
            "cache_type": "auto_prefix",
            "cache_usage_fields": ["prompt_cache_hit_tokens", "cached_tokens"],
        }

    def create_provider(self, config: Dict[str, Any]) -> LLMProvider:
        """Create an OpenAIChat instance from config dict.

        Args:
            config: Must include 'api_key', 'base_url', 'model'.
                    Optional: 'max_retries', 'timeout_s', 'provider'.

        Returns:
            Configured OpenAIChat instance.

        Raises:
            ValueError: If required keys are missing.
        """
        from leapflow.llm.openai_provider import OpenAIChat

        api_key = config.get("api_key")
        base_url = config.get("base_url")
        model = config.get("model")

        if not api_key:
            raise ValueError("OpenAI-compatible provider requires 'api_key' in config")
        if not base_url:
            raise ValueError("OpenAI-compatible provider requires 'base_url' in config")
        if not model:
            raise ValueError("OpenAI-compatible provider requires 'model' in config")

        return OpenAIChat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_retries=int(config.get("max_retries", 3)),
            timeout_s=float(config.get("timeout_s", 180.0)),
            provider=config.get("provider"),
        )


# Module-level singleton for auto-discovery and reload support.
plugin = OpenAICompatiblePlugin()
